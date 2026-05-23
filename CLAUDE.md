# CLAUDE.md — `ai-vmm`

**Strict** guide for every Claude Code session working on this repository.
These rules are binding, not advisory.

## 1. Project

`ai-vmm` is an **agentic hypervisor**. A control plane translates
natural-language intents into physical resource allocations through KVM.
A reasoning model provides the **reasoning** — the hosted Anthropic Claude
API, or any local OpenAI-compatible server (Ollama, vLLM) for a fully
air-gapped deployment; `rust-vmm` (`kvm-ioctls`, `kvm-bindings`, `vm-memory`)
performs the **native allocation**.

## 2. Architecture

| File | Responsibility |
|---|---|
| `src/lib.rs` | Library crate root; re-exports `agent`, `config`, `vmm` so the binary and `tests/` share one code base. |
| `src/main.rs` | `clap` CLI, subcommand dispatch, Tokio runtime. |
| `src/config.rs` | BYOK credentials + AI-provider settings, key validation, Kani proofs. |
| `src/registry.rs` | Declarative VM ledger — persists applied plans to `vms.toml`; Kani proofs. |
| `src/server.rs` | HTTP control-plane API (`ai-vmm serve`) — `axum`; bearer auth, Kani proofs. |
| `src/agent/mod.rs` | Provider-agnostic LLM client (Anthropic + OpenAI), bounded tool-use loop. |
| `src/agent/prompts.rs` | System prompt + provider-neutral tool schemas. |
| `src/vmm/mod.rs` | Native KVM provisioning, pure validation, Kani proofs. |
| `src/vmm/network.rs` | Host TAP interface plumbing via `ip` (Linux only). |
| `src/vmm/boot.rs` | Direct Kernel Boot — kernel load, zero page, long-mode vCPU setup (Linux only). |
| `src/vmm/run.rs` | VMM run loop + serial-console UART emulation (Linux only). |
| `src/vmm/mmio.rs` | MMIO device bus — routes guest MMIO accesses to devices (Linux only). |
| `src/vmm/storage.rs` | virtio-mmio block device (Linux only). |
| `src/vmm/mptable.rs` | Intel MP table — guest CPU enumeration for SMP (Linux only). |
| `tests/live_os_boot_test.rs` | Live integration test — boots a real `vmlinux` + `rootfs.ext4` through the full pipeline; skips cleanly when the images are absent. |

Flow: `auth login → config::save_api_key`, then `run → config::load_api_key →
agent::ask_agent → ExecutionPlan → operator approval → vmm::provision_kvm_machine_native`
(Terraform-style plan / review / apply).

## 3. Commands

```sh
# Build & quality gates
cargo build
cargo clippy -- -D warnings    # strict lint — MUST pass with no error
cargo fmt                      # formatting
cargo kani                     # formal proofs — MUST pass

# Usage
ai-vmm auth login              # store your Anthropic API key once (BYOK)
ai-vmm run "<intent>"          # plan a request, review it, apply after approval
ai-vmm run "<intent>" --yes    # same, non-interactive (scripts / automation)
ai-vmm provision --name db --vcpus 2 --memory 2G  # headless: no AI, no prompt
ai-vmm serve                   # HTTP control-plane API (planning + ledger)
```

## 4. STRICT rules

1. **English only.** All code, identifiers, comments, doc comments, error
   messages and documentation are written in English.
2. **Cross-platform.** The crate MUST compile on Linux **and** Windows/macOS.
   KVM is Linux-only: every KVM call stays behind
   `#[cfg(target_os = "linux")]`. Never remove that gating, and never move the
   KVM dependencies out of the `[target.'cfg(...)']` section of `Cargo.toml`.
3. **Zero warnings.** `cargo clippy -- -D warnings` must pass cleanly.
4. **No placeholders.** No `// ... rest of code`, no `todo!()`, no function
   left empty.
5. **Idiomatic error handling.** `Result` + `?` everywhere. No `unwrap()` or
   `expect()` on a fallible production path (tests may use them).
6. **Bounded `unsafe`.** Allowed only for KVM ioctls
   (`set_user_memory_region`). Every `unsafe` block carries a `// SAFETY:`
   comment justifying the upheld invariant.
7. **Mandatory validation.** Every argument coming from Claude (`tool_use`)
   MUST go through `vmm::validate_spec` before any hardware contact. Claude
   reasons; it never sets the bounds.
8. **Proofs kept current & verifiable validation.** Any change to
   `validate_spec`, `memory_bytes`, `VmSpec`, `SpecError` or the
   `MAX_*`/`MIN_*` constants MUST come with an updated set of Kani harnesses
   in `src/vmm/mod.rs`; `cargo kani` must stay green (6/6 harnesses). Code
   reached by a Kani harness stays **allocation-free and `format!`-free**: the
   std formatting machinery blows up CBMC's model-checking time. Validation
   errors go through the `SpecError` enum with `&'static str` messages, never
   formatted `String`s.
9. **Secrets.** The Anthropic API key follows a BYOK model: stored once via
   `ai-vmm auth login` in a per-user `credentials.toml`. On Unix that file is
   `0600` and its directory `0700`. The key is never hard-coded, never logged,
   never committed; `Config` deliberately does not derive `Debug` so the
   secret cannot leak through a debug print.
10. **Model.** The default Anthropic model is the `ANTHROPIC_MODEL` constant
    in `src/agent/mod.rs` (a configured `model` overrides it). Never
    reintroduce a retired model (e.g. `claude-3-opus-20240229`).

## 5. Runtime requirements

- For the hosted Anthropic provider: an API key stored via `ai-vmm auth login`
  (BYOK), or the `ANTHROPIC_API_KEY` environment variable (overrides the file,
  intended for CI).
- For an air-gapped deployment: set `AI_VMM_PROVIDER=openai` plus
  `AI_VMM_BASE_URL` / `AI_VMM_MODEL` (or the matching `credentials.toml`
  fields) to target a local OpenAI-compatible server such as Ollama or vLLM —
  no API key, no network egress required.
- Real provisioning: a **Linux** host, `/dev/kvm` present, the user in the
  `kvm` group (or root).
- A guest kernel (`vmlinux` ELF) and a root-filesystem image. The kernel path
  resolves from `$AI_VMM_KERNEL`, the `kernel_path` entry of `credentials.toml`,
  or `./vmlinux`; the disk from the request / `--disk`, `$AI_VMM_DISK`, or the
  `disk_path` entry. A VM with no resolvable disk is refused, with guidance,
  before boot — never a kernel panic.
- Formal verification: `cargo install --locked kani-verifier` then
  `cargo kani setup`.

## 6. Security

- The control plane trusts Claude for **reasoning**, never for **bounds**: the
  limits are enforced in Rust and **proven** by Kani.
- Guest memory is an anonymous private `mmap`, hence isolated from the host.
- Every external input (API response, tool arguments) is treated as
  untrusted: explicit typed conversion, clean failure on any invalid value.
- Network plumbing shells out to `ip` via `std::process::Command` — never a
  shell. Every interface name is checked against a strict allowlist before
  use; a validated name can never start with `-`, which blocks argument
  injection into `ip`. The byte-level allowlist is proven by Kani.
- `ai-vmm auth login` reads the key with terminal echo disabled (interactive)
  or straight from stdin (piped/CI), so the secret is never echoed to the
  screen or the terminal scrollback.

## 7. MVP known limitations

- The agentic tool-use loop is bounded at `MAX_TURNS` round-trips: a request
  that would need more registry deliberation is rejected, not looped.
- Multi-vCPU VMs are supported: an Intel MP table enumerates the processors and
  each vCPU runs on its own host thread, sharing the MMIO bus and guest memory.
  There is no vCPU pinning and no live CPU hotplug.
- Direct Kernel Boot needs a `vmlinux` ELF kernel and an `ext4` root-filesystem
  image, supplied by the operator (see §5); `ai-vmm` does not ship them.
- The virtio-blk datapath services read, write, flush and get-id requests
  over a single virtqueue — descriptor-chain parsing, real block I/O,
  used-ring updates and the completion irqfd — and always completes a
  request, so the guest driver never blocks. It is validated against a real
  guest kernel driver: the Firecracker kernel mounts its ext4 root over it.
  It stays an MVP, though: one virtqueue, no `DISCARD`/`WRITE_ZEROES`.
- The MMIO bus uses a local `MmioDevice` trait, not `vm-device`'s
  `MutDeviceMmio`: the rust-vmm `virtio-blk`/`virtio-queue`/`virtio-device`
  crates do not form a buildable graph with this stack and provide no turnkey
  `VirtioBlk`, so they are deliberately not depended on.
- A single tool is exposed: `provision_kvm_machine`.
- The `ANTHROPIC_API_KEY` override is read from the process environment, which
  is less protected than the `0600` file; prefer `auth login` outside CI.
- Host networking creates a TAP and attaches it to the bridge; attaching the
  guest NIC to that TAP (opening the tap fd) is still deferred.
- TAP creation needs root / `CAP_NET_ADMIN`.
- The plan / approve / apply flow can only be exercised end-to-end with a live
  Anthropic API key; tests cover the validation and the apply target directly.
- Heavyweight virtualization features — live migration, memory snapshots,
  multi-node clustering, GPU PCI passthrough — are out of scope by design.
  The declarative VM ledger (§9.4) is the lightweight substitute: every VM is
  fully described by its plan, so the plan *is* the snapshot — re-applying a
  stored plan re-materialises the VM with no guest-memory image to ship.

## 8. Cross-platform verification

The KVM code only compiles and runs on Linux. Development happens on Windows;
the native layer is validated through **WSL Ubuntu** (which exposes a working
`/dev/kvm`).

```sh
# From WSL Ubuntu, separate target dir so the Windows build is not clobbered:
cd /mnt/c/Users/pc/Desktop/z.rs
export CARGO_TARGET_DIR=/tmp/zrs-target
cargo clippy --all-targets -- -D warnings   # strict lint
cargo test -- --nocapture                   # includes real KVM provisioning
CARGO_TARGET_DIR=/tmp/zrs-kani cargo kani    # formal proofs
```

Known-good baseline, never to be regressed:

| Check | Windows | Linux (WSL) |
|---|---|---|
| `cargo clippy -- -D warnings` | 0 warnings | 0 warnings |
| `cargo test` | 44/44 | 83/83 (incl. live Linux-kernel boot, serial console, virtio-blk datapath, `ip`) |
| `cargo kani` | — | 42/42 harnesses proven |

## 9. Roadmap

Ordered next steps; each item must keep the §4 rules and the §8 baseline green.

1. **Close the agentic loop — done.** The agent runs a bounded multi-turn
   tool-use loop: it executes `list_vms` / `inspect_vm` locally and feeds the
   `tool_result` back to the model before it commits to a plan. It speaks two
   back ends with no vendor SDK — the hosted Anthropic API and any local
   OpenAI-compatible server (Ollama, vLLM) — for air-gapped deployments.
2. **Keep VMs alive.** Return a `RunningVm` struct that owns the `VmFd`, the
   `GuestMemoryMmap` and the `Vec<VcpuFd>` instead of dropping them at scope
   end; track running VMs in an in-memory registry.
3. **Boot a real OS end to end — done.** The hypervisor boots a real,
   unmodified Linux kernel from a `./vmlinux` to a userspace `/sbin/init`:
   with the Firecracker kernel and an Ubuntu 18.04 `./rootfs.ext4`, the guest
   mounts its ext4 root over the virtio-blk MMIO datapath and starts systemd
   as PID 1 — driven by `tests/live_os_boot_test.rs`, which skips cleanly when
   the images are absent. Reaching this took six host-side fixes:
   `KVM_SET_CPUID2`, `KVM_SET_TSS_ADDR`, `KVM_PIT_SPEAKER_DUMMY`, a two-region
   e820 map, `KVM_SET_LAPIC` virtual-wire mode (`LINT0=ExtINT`, so the PIT
   timer interrupt reaches the guest), and a virtio-blk datapath that answers
   `GET_ID`/`FLUSH` and always completes a request. Known caveat: under nested
   virtualization (a KVM guest inside WSL2) the kernel's `calibrate_APIC_clock`
   is timing-sensitive and occasionally hangs before root mount (~1 run in 4);
   on real hardware or a single-level KVM host the boot is deterministic.
4. **Fleet visibility — partly done.** A declarative VM ledger
   (`src/registry.rs`) records every applied plan to a per-user `vms.toml`;
   `ai-vmm list` / `inspect` / `forget` manage it from the CLI (Terraform-style
   `state list` / `show` / `rm`). The ledger is bounded (oldest records roll
   off, Kani-proven) and the plan is the snapshot — re-applying a stored plan
   re-materialises a VM with no guest-memory image to ship. The registry is
   also exposed to the model as the `list_vms` and `inspect_vm` tools, so a
   clone/modify intent is grounded in the real recorded specs. Remaining:
   track live state — a VM is for now a ledger record, not a supervised
   running process.
5. **Host capability checks — done.** A VM's vCPU count is reconciled against
   the host's `KVM_CAP_MAX_VCPUS` (`effective_vcpu_cap`), and its RAM against
   the host's `MemAvailable` from `/proc/meminfo` (`effective_memory_cap`,
   keeping `HOST_RESERVE_MB` free for the host and the VMM itself). Both are
   Kani-proven and both reject an impossible request before any hardware is
   touched; an unreadable `/proc/meminfo` falls back to the architectural
   `MAX_MEMORY_MB` bound rather than blocking.
6. **HTTP control plane.** Expose `ask_agent` behind an `axum` endpoint with
   authentication, structured logging and metrics.
7. **CI — done.** A GitHub Actions workflow (`.github/workflows/ci.yml`) runs
   `cargo fmt --check`, `cargo clippy -- -D warnings` and `cargo test` on Linux
   and Windows, and `cargo kani` on Linux — blocking merges on the §8 baseline.
8. **Privilege separation for networking.** TAP/bridge setup needs
   `CAP_NET_ADMIN`; move it behind a minimal privileged helper (or a
   pre-created TAP pool) so the main process can drop privileges. (The TAP
   already has a collision-free, hash-derived name — `derive_tap_name` /
   `render_tap_name` — proven a valid, injection-safe interface name by Kani.)
9. **Zeroize the key in memory.** Hold the API key in a wiped-on-drop type
   (e.g. `zeroize` / `secrecy`) so it does not linger in process memory or
   core dumps after use.
```
