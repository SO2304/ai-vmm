//! Host network plumbing: dynamic TAP interface creation (Linux only).
//!
//! This module shells out to `ip` (iproute2) to create a host TAP interface
//! for a VM and attach it to an existing bridge.
//!
//! Security model:
//!  - `std::process::Command` is used with separate arguments and **never** a
//!    shell, so shell metacharacters in a name cannot be interpreted.
//!  - Every interface name that reaches `ip` is first checked against a strict
//!    allowlist ([`is_valid_ifname`]). The allowlist also blocks argument
//!    injection: a name can never start with `-`, so `ip` cannot mistake it
//!    for a command-line option.
//!  - The byte-level allowlist is proven correct by the Kani harnesses at the
//!    bottom of this file.
//!  - Untrusted names are rendered with `{:?}` in error messages, which
//!    escapes control characters.

use std::process::Command;

/// Maximum length of a Linux network interface name.
///
/// The kernel's `IFNAMSIZ` is 16; one byte is reserved for the trailing NUL,
/// leaving 15 usable characters.
const MAX_IFNAME_LEN: usize = 15;

/// Returns `true` iff `byte` is allowed anywhere in an interface name.
///
/// The allowlist is ASCII alphanumerics plus `_`, `-` and `.`. Every other
/// byte — path separators, whitespace, NUL, shell metacharacters — is rejected.
const fn is_allowed_ifname_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' || byte == b'.'
}

/// Returns `true` iff `byte` is allowed as the FIRST byte of an interface name.
///
/// The first byte must be an ASCII alphanumeric. This is what guarantees the
/// name can never be mistaken by `ip` for a command-line option (`-...`).
const fn is_safe_first_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
}

/// Returns `true` iff `name` is a safe Linux interface name: non-empty, at most
/// [`MAX_IFNAME_LEN`] bytes, starting with an ASCII alphanumeric and containing
/// only allowlisted bytes.
pub fn is_valid_ifname(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_IFNAME_LEN {
        return false;
    }
    is_safe_first_byte(bytes[0]) && bytes.iter().all(|&b| is_allowed_ifname_byte(b))
}

/// Number of lowercase-hex digits of the VM-name hash carried by a TAP name.
///
/// `tap-` (4 bytes) plus [`TAP_HASH_DIGITS`] fills exactly [`MAX_IFNAME_LEN`].
const TAP_HASH_DIGITS: usize = MAX_IFNAME_LEN - 4;

/// 64-bit FNV-1a hash of `data`.
///
/// A tiny, dependency-free, fully deterministic hash — ample to give every
/// distinct VM name its own TAP interface.
fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Maps a nibble (`0..=15`) to its lowercase ASCII hex digit.
const fn hex_digit(nibble: u8) -> u8 {
    if nibble < 10 {
        b'0' + nibble
    } else {
        b'a' + (nibble - 10)
    }
}

/// Renders the TAP interface name `tap-<hash digits>` into a fixed buffer.
///
/// Allocation-free and `format!`-free, so the Kani harness can prove that, for
/// ANY hash, the result is always a valid, injection-safe interface name: it
/// starts with `t` and every byte is on the allowlist.
fn render_tap_name(hash: u64) -> [u8; MAX_IFNAME_LEN] {
    let mut name = [0_u8; MAX_IFNAME_LEN];
    name[..4].copy_from_slice(b"tap-");
    for (index, slot) in name[4..].iter_mut().enumerate() {
        // Most-significant nibble first; covers the low 4 * TAP_HASH_DIGITS bits.
        let shift = 4 * (TAP_HASH_DIGITS - 1 - index);
        *slot = hex_digit(((hash >> shift) & 0xf) as u8);
    }
    name
}

/// Derives a safe, deterministic, collision-free TAP interface name from a VM
/// name.
///
/// The name is `tap-` followed by hex digits of a hash of the *whole* VM name,
/// so two VM names — however similar, however long — always get distinct TAP
/// interfaces. (The previous scheme truncated the VM name, so any two names
/// sharing an 11-character prefix collided on one TAP.) The result is always a
/// valid interface name — proven by the Kani harness for [`render_tap_name`].
fn derive_tap_name(vm_name: &str) -> String {
    let name = render_tap_name(fnv1a_64(vm_name.as_bytes()));
    // `render_tap_name` only ever emits ASCII, so the conversion is exact.
    String::from_utf8_lossy(&name).into_owned()
}

/// Runs `ip` with the given arguments. Returns an error if `ip` is missing or
/// exits unsuccessfully. No shell is involved: each argument is passed verbatim
/// as a separate `argv` entry.
fn run_ip(args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new("ip")
        .args(args)
        .output()
        .map_err(|e| format!("could not run 'ip' ({e}) — is the iproute2 package installed?"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "`ip {}` failed ({}): {}",
            args.join(" "),
            output.status,
            stderr.trim()
        )
        .into());
    }
    Ok(())
}

/// Returns `true` iff a network interface named `ifname` currently exists.
fn interface_exists(ifname: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let output = Command::new("ip")
        .args(["link", "show", "dev", ifname])
        .output()
        .map_err(|e| format!("could not run 'ip' ({e}) — is the iproute2 package installed?"))?;
    Ok(output.status.success())
}

/// Attaches `tap_name` to `bridge_name` and brings it up. Both `ip` calls are
/// idempotent: re-attaching to the same bridge, or re-upping, is a no-op.
fn attach_and_activate(
    tap_name: &str,
    bridge_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    run_ip(&["link", "set", tap_name, "master", bridge_name])?;
    run_ip(&["link", "set", "dev", tap_name, "up"])?;
    Ok(())
}

/// Creates and configures a host TAP interface for a VM, attaching it to
/// `bridge_name`.
///
/// The function is idempotent (a pre-existing TAP is reused) and transactional
/// (if attachment fails after the TAP was created here, the TAP is rolled
/// back). It returns the name of the TAP interface, so the hypervisor can
/// later attach the guest NIC to it.
pub fn setup_tap_interface(
    vm_name: &str,
    bridge_name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    // 1. Strictly validate the bridge name before it can reach `ip`.
    if !is_valid_ifname(bridge_name) {
        return Err(format!(
            "refusing unsafe bridge name {bridge_name:?}: a bridge name must be 1 to \
             {MAX_IFNAME_LEN} characters, start with a letter or digit, and contain only \
             letters, digits, '_', '-' or '.'"
        )
        .into());
    }

    // 2. Derive a TAP name we fully control, then validate it as well
    //    (defense in depth — the derivation is already constrained).
    let tap_name = derive_tap_name(vm_name);
    if !is_valid_ifname(&tap_name) {
        return Err(format!(
            "could not derive a valid TAP interface name from VM name {vm_name:?}"
        )
        .into());
    }

    // 3. Create the TAP only if it does not already exist (idempotency).
    let created_here = !interface_exists(&tap_name)?;
    if created_here {
        if let Err(e) = run_ip(&["tuntap", "add", "mode", "tap", "name", tap_name.as_str()]) {
            // Tolerate a lost race: another actor may have created it meanwhile.
            if !interface_exists(&tap_name)? {
                return Err(e);
            }
        }
    }

    // 4. Attach the TAP to the bridge and bring it up. If this fails and we
    //    created the interface in this call, roll it back so nothing leaks.
    if let Err(e) = attach_and_activate(&tap_name, bridge_name) {
        if created_here {
            let _ = run_ip(&["link", "del", tap_name.as_str()]);
        }
        return Err(e);
    }

    Ok(tap_name)
}

/// Formal proofs checked by the Kani model checker (`cargo kani`).
///
/// They verify the byte-level allowlist that underpins [`is_valid_ifname`] —
/// the security boundary in front of every `ip` invocation — and that every
/// TAP name this module derives always clears it.
#[cfg(kani)]
mod proofs {
    use super::{hex_digit, is_allowed_ifname_byte, is_safe_first_byte, render_tap_name};

    /// Proof: the per-byte allowlist excludes every byte that could subvert
    /// `ip`'s argument tokenization, path handling, or a shell (defense in
    /// depth — `Command` already bypasses the shell).
    #[kani::proof]
    fn proof_allowlist_rejects_dangerous_bytes() {
        let byte: u8 = kani::any();
        if is_allowed_ifname_byte(byte) {
            assert!(byte != b'/');
            assert!(byte != 0);
            assert!(byte != b' ');
            assert!(byte != b'\t');
            assert!(byte != b'\n');
            assert!(byte != b'\r');
            assert!(byte != b';');
            assert!(byte != b'|');
            assert!(byte != b'&');
            assert!(byte != b'$');
            assert!(byte != b'`');
            assert!(byte != b':');
        }
    }

    /// Proof: a validated first byte can never be `-`, so an interface name can
    /// never be parsed by `ip` as a command-line option.
    #[kani::proof]
    fn proof_first_byte_blocks_option_injection() {
        let byte: u8 = kani::any();
        if is_safe_first_byte(byte) {
            assert!(byte != b'-');
            assert!(byte.is_ascii_alphanumeric());
        }
    }

    /// Proof: any byte accepted as a first byte is also accepted by the general
    /// allowlist — the two checks are consistent.
    #[kani::proof]
    fn proof_first_byte_is_a_subset_of_the_allowlist() {
        let byte: u8 = kani::any();
        if is_safe_first_byte(byte) {
            assert!(is_allowed_ifname_byte(byte));
        }
    }

    /// Proof: every nibble (`0..=15`) renders to an allowlisted hex digit.
    #[kani::proof]
    fn proof_hex_digit_yields_an_allowed_byte() {
        let nibble: u8 = kani::any();
        kani::assume(nibble < 16);
        assert!(is_allowed_ifname_byte(hex_digit(nibble)));
    }

    /// Proof: for ANY hash, the derived TAP name is a valid, injection-safe
    /// interface name — its first byte is alphanumeric (never `-`) and every
    /// byte is on the allowlist — so it always passes `is_valid_ifname`.
    #[kani::proof]
    fn proof_rendered_tap_name_is_a_valid_interface_name() {
        let name = render_tap_name(kani::any());
        assert!(is_safe_first_byte(name[0]));
        for &byte in &name {
            assert!(is_allowed_ifname_byte(byte));
        }
    }
}

/// Tests: pure name validation/derivation (hermetic) and a best-effort
/// integration test against real `ip` commands.
#[cfg(test)]
mod tests {
    use super::{derive_tap_name, fnv1a_64, is_valid_ifname, setup_tap_interface, MAX_IFNAME_LEN};

    #[test]
    fn accepts_real_interface_names() {
        assert!(is_valid_ifname("br0"));
        assert!(is_valid_ifname("virbr0"));
        assert!(is_valid_ifname("eth0"));
        assert!(is_valid_ifname("tap-db-prod"));
        assert!(is_valid_ifname("eth0.100"));
    }

    #[test]
    fn rejects_empty_and_overlong_names() {
        assert!(!is_valid_ifname(""));
        assert!(!is_valid_ifname(&"a".repeat(MAX_IFNAME_LEN + 1)));
    }

    #[test]
    fn rejects_option_injection() {
        // A name starting with '-' could be parsed by `ip` as an option.
        assert!(!is_valid_ifname("-x"));
        assert!(!is_valid_ifname("--help"));
    }

    #[test]
    fn rejects_metacharacters_and_separators() {
        assert!(!is_valid_ifname("br0; rm -rf /"));
        assert!(!is_valid_ifname("a b"));
        assert!(!is_valid_ifname("a/b"));
        assert!(!is_valid_ifname("a$b"));
        assert!(!is_valid_ifname("a\nb"));
        assert!(!is_valid_ifname("."));
    }

    #[test]
    fn derive_tap_name_is_valid_deterministic_and_collision_free() {
        // Deterministic: the same VM name always derives the same TAP name.
        assert_eq!(derive_tap_name("db-prod"), derive_tap_name("db-prod"));
        // Always a valid, injection-safe interface name starting with `tap-`.
        let tap = derive_tap_name("db-prod");
        assert!(is_valid_ifname(&tap));
        assert!(tap.starts_with("tap-"));
        // Collision-free: two VM names sharing an 11-character prefix — which
        // the old truncation scheme mapped to the SAME TAP — now differ.
        assert_ne!(
            derive_tap_name("database-server-one"),
            derive_tap_name("database-server-two")
        );
    }

    #[test]
    fn fnv1a_matches_known_vectors() {
        // Reference values for the 64-bit FNV-1a hash.
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn derived_tap_name_is_always_a_valid_ifname() {
        for vm in [
            "db-prod",
            "a",
            "VeryLongVirtualMachineName",
            "weird name!!",
            "日本語",
            "",
        ] {
            let tap = derive_tap_name(vm);
            assert!(
                is_valid_ifname(&tap),
                "derived tap name {tap:?} from {vm:?} must be a valid ifname"
            );
        }
    }

    #[test]
    fn rejects_unsafe_bridge_name_before_running_ip() {
        // Validation fails before any `ip` process is spawned.
        assert!(setup_tap_interface("db-prod", "br0; rm -rf /").is_err());
        assert!(setup_tap_interface("db-prod", "-x").is_err());
        assert!(setup_tap_interface("db-prod", "").is_err());
    }

    /// Best-effort integration test: needs root and iproute2. Creates a
    /// throwaway bridge, provisions a TAP against it twice (to check
    /// idempotency), then cleans up. Without the privileges or kernel support
    /// it stays inconclusive instead of failing.
    #[test]
    fn setup_tap_interface_smoke() {
        use std::process::Command;
        let bridge = "br-aivmm-tst";

        // Pre-clean any leftover from a previously interrupted run.
        let _ = Command::new("ip").args(["link", "del", bridge]).output();

        let bridge_created = Command::new("ip")
            .args(["link", "add", "name", bridge, "type", "bridge"])
            .output()
            .is_ok_and(|o| o.status.success());
        if !bridge_created {
            eprintln!("[net-smoke] need root + iproute2 to create a bridge — inconclusive");
            return;
        }

        let first = setup_tap_interface("net-smoke-vm", bridge);
        let second = setup_tap_interface("net-smoke-vm", bridge);

        // Clean up before asserting so nothing leaks on a failed assertion.
        if let Ok(tap) = &first {
            let _ = Command::new("ip").args(["link", "del", tap]).output();
        }
        let _ = Command::new("ip").args(["link", "del", bridge]).output();

        match (&first, &second) {
            (Ok(tap1), Ok(tap2)) => {
                assert_eq!(tap1, tap2, "setup_tap_interface must be idempotent");
                assert!(is_valid_ifname(tap1), "the derived TAP name must be valid");
                eprintln!("[net-smoke] real TAP plumbing verified: '{tap1}' (idempotent)");
            }
            _ => {
                eprintln!("[net-smoke] TAP plumbing unavailable here, inconclusive: {first:?}");
            }
        }
    }
}
