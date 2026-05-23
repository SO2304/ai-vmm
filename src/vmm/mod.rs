//! Hardware execution layer: native KVM provisioning.
//!
//! This module holds:
//!  1. Pure validation logic (`validate_spec`, `memory_bytes`, `normalize_bridge`)
//!     that is formally verifiable.
//!  2. The `provision_kvm_machine_native` function, which talks to the host KVM
//!     API (Linux only).
//!  3. The `network` submodule (Linux only) — host TAP interface plumbing.
//!  4. The `boot` submodule (Linux only) — Direct Kernel Boot.
//!  5. The `run` submodule (Linux only) — the VMM run loop and serial console.
//!  6. The `mmio` submodule (Linux only) — the MMIO device bus.
//!  7. The `storage` submodule (Linux only) — the virtio-mmio block device.
//!  8. Formal Kani proofs (`#[cfg(kani)]`) that show the validation rejects
//!     every out-of-bounds configuration and that the memory-size computation
//!     never overflows.
//!
//! On platforms without KVM (Windows, macOS) the crate still compiles: only
//! `provision_kvm_machine_native` then returns an explicit error.

// The network plumbing relies on Linux's `ip` (iproute2) and TAP interfaces;
// it is therefore compiled for Linux targets only.
#[cfg(target_os = "linux")]
pub mod network;

// Direct Kernel Boot uses KVM and the rust-vmm kernel loader; Linux only.
#[cfg(target_os = "linux")]
pub mod boot;

// The VMM run loop drives `KVM_RUN`; Linux only.
#[cfg(target_os = "linux")]
pub mod run;

// The MMIO device bus; Linux only (used only by the KVM run loop).
#[cfg(target_os = "linux")]
pub mod mmio;

// The virtio-mmio block device; Linux only.
#[cfg(target_os = "linux")]
pub mod storage;

// The Intel MP table — guest CPU enumeration for SMP; Linux only.
#[cfg(target_os = "linux")]
pub mod mptable;

/// Maximum number of vCPUs allowed per VM (common KVM limit).
pub const MAX_VCPUS: u32 = 255;

/// Minimum memory allowed for a VM, in mebibytes.
pub const MIN_MEMORY_MB: u32 = 8;

/// Maximum memory allowed for a VM, in mebibytes (1 TiB).
pub const MAX_MEMORY_MB: u32 = 1024 * 1024;

/// Guest-physical base address of the virtio-mmio block device.
#[cfg(target_os = "linux")]
const VIRTIO_BLK_MMIO_BASE: u64 = 0xd000_0000;
/// Size of the virtio-mmio device's register window.
#[cfg(target_os = "linux")]
const VIRTIO_BLK_MMIO_SIZE: u64 = 0x1000;
/// Guest interrupt line (GSI) for the virtio-mmio block device.
#[cfg(target_os = "linux")]
const VIRTIO_BLK_GSI: u32 = 5;

/// Validated specification of a virtual machine.
///
/// A value of this type can only be obtained through [`validate_spec`], so its
/// existence proves that `vcpus` and `memory_mb` are within bounds and that
/// `network_bridge` / `disk_image_path`, when present, are non-empty trimmed
/// values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmSpec {
    /// Number of virtual cores, guaranteed within `1..=MAX_VCPUS`.
    pub vcpus: u32,
    /// Memory in mebibytes, guaranteed within `MIN_MEMORY_MB..=MAX_MEMORY_MB`.
    pub memory_mb: u32,
    /// Host bridge interface to attach the VM to, or `None` for an isolated VM.
    pub network_bridge: Option<String>,
    /// Root-filesystem disk image to attach as a virtio-blk device, if any.
    pub disk_image_path: Option<String>,
}

/// Reason a specification was rejected by [`validate_spec`].
///
/// Deliberately a "flat" error type: no allocation, no dynamic formatting. It
/// therefore stays trivial to verify with Kani — the standard library's
/// `format!` machinery blows up CBMC's model-checking time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecError {
    /// The requested vCPU count is zero.
    ZeroVcpu,
    /// The requested vCPU count exceeds [`MAX_VCPUS`].
    TooManyVcpus,
    /// The requested memory is below [`MIN_MEMORY_MB`].
    MemoryTooLow,
    /// The requested memory exceeds [`MAX_MEMORY_MB`].
    MemoryTooHigh,
}

impl SpecError {
    /// Readable, static message describing the rejection cause.
    pub const fn as_str(self) -> &'static str {
        match self {
            SpecError::ZeroVcpu => "vCPU count must be at least 1",
            SpecError::TooManyVcpus => "vCPU count exceeds the allowed limit",
            SpecError::MemoryTooLow => "requested memory is below the allowed minimum",
            SpecError::MemoryTooHigh => "requested memory exceeds the allowed maximum",
        }
    }
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for SpecError {}

/// Normalizes a requested bridge name.
///
/// Surrounding whitespace is trimmed; an empty name or the literal `"none"`
/// (any case) is treated as "no bridge" and yields `None`.
pub fn normalize_bridge(network_bridge: Option<&str>) -> Option<String> {
    let trimmed = network_bridge?.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Normalizes an optional filesystem path: trims it, and maps an empty string
/// to `None`.
fn normalize_path(path: Option<&str>) -> Option<String> {
    let trimmed = path?.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Validates a resource request and returns a safe [`VmSpec`].
///
/// The vCPU/memory bounds checks are pure, deterministic and allocation-free,
/// which makes them verifiable by Kani. The `network_bridge` and
/// `disk_image_path` are normalized (trimmed, empties dropped).
pub fn validate_spec(
    vcpus: u32,
    memory_mb: u32,
    network_bridge: Option<&str>,
    disk_image_path: Option<&str>,
) -> Result<VmSpec, SpecError> {
    if vcpus == 0 {
        return Err(SpecError::ZeroVcpu);
    }
    if vcpus > MAX_VCPUS {
        return Err(SpecError::TooManyVcpus);
    }
    if memory_mb < MIN_MEMORY_MB {
        return Err(SpecError::MemoryTooLow);
    }
    if memory_mb > MAX_MEMORY_MB {
        return Err(SpecError::MemoryTooHigh);
    }
    Ok(VmSpec {
        vcpus,
        memory_mb,
        network_bridge: normalize_bridge(network_bridge),
        disk_image_path: normalize_path(disk_image_path),
    })
}

/// Converts an amount of memory expressed in mebibytes into bytes.
///
/// The computation is done in `u64` to rule out any overflow, then cast to
/// `usize`. Because `memory_mb` is a multiple of 1 MiB, the result is always
/// aligned on a 4096-byte page.
pub fn memory_bytes(memory_mb: u32) -> usize {
    (memory_mb as u64 * 1024 * 1024) as usize
}

/// Multiplies a memory `value` by a unit `factor`, yielding mebibytes — or
/// `None` if the product would not fit in `u32`.
///
/// Used by [`parse_memory_mb`] to reject an oversized request rather than
/// silently wrapping it to a wrong, smaller size. `proof_scaled_mib_is_exact`
/// proves the result, when present, is exactly `value * factor`.
fn scaled_mib(value: u64, factor: u32) -> Option<u32> {
    value
        .checked_mul(u64::from(factor))
        .and_then(|total| u32::try_from(total).ok())
}

/// Parses a human memory size into mebibytes.
///
/// Accepts a plain integer (already MiB) or one carrying a binary unit suffix
/// — `M`/`MiB`/`MB`, `G`/`GiB`/`GB`, `T`/`TiB`/`TB`, case-insensitive — so
/// `2G` is 2048 MiB. This gives the headless `provision` path the same unit
/// fidelity the natural-language path already has. A malformed string, or a
/// value that would overflow, is a clean error — never a panic or a wrap.
pub fn parse_memory_mb(spec: &str) -> Result<u32, Box<dyn std::error::Error>> {
    let lower = spec.trim().to_ascii_lowercase();
    // Tolerate a trailing `b`/`ib` so `2g`, `2gb` and `2gib` are all accepted.
    let body = lower.strip_suffix('b').unwrap_or(&lower);
    let body = body.strip_suffix('i').unwrap_or(body);
    let (digits, factor) = if let Some(n) = body.strip_suffix('t') {
        (n, 1024 * 1024)
    } else if let Some(n) = body.strip_suffix('g') {
        (n, 1024)
    } else if let Some(n) = body.strip_suffix('m') {
        (n, 1)
    } else {
        (body, 1)
    };
    let value: u64 = digits
        .trim()
        .parse()
        .map_err(|_| format!("invalid memory size {spec:?} — use e.g. 2048, 512M or 2G"))?;
    scaled_mib(value, factor).ok_or_else(|| format!("memory size {spec:?} is too large").into())
}

/// Returns the largest vCPU count this hypervisor will provision on a host
/// whose KVM reports `host_max` as its maximum: the tighter of that figure and
/// the architectural ceiling [`MAX_VCPUS`].
#[cfg(target_os = "linux")]
fn effective_vcpu_cap(host_max: u32) -> u32 {
    host_max.min(MAX_VCPUS)
}

/// RAM, in MiB, kept free for the host kernel and the VMM process itself; a VM
/// is never granted memory that would eat into this reserve.
#[cfg(target_os = "linux")]
const HOST_RESERVE_MB: u32 = 256;

/// Returns the largest VM memory, in MiB, this host can safely grant given
/// `host_available_mb` (the host's `MemAvailable`): the architectural ceiling
/// [`MAX_MEMORY_MB`], further capped so [`HOST_RESERVE_MB`] always stays free.
#[cfg(target_os = "linux")]
fn effective_memory_cap(host_available_mb: u32) -> u32 {
    host_available_mb
        .saturating_sub(HOST_RESERVE_MB)
        .min(MAX_MEMORY_MB)
}

/// Reads the host's available memory, in MiB, from `/proc/meminfo`.
///
/// `MemAvailable` is the kernel's own estimate of the memory free for new
/// allocations — it accounts for reclaimable cache — and is reported in
/// kibibytes.
#[cfg(target_os = "linux")]
fn read_host_available_memory_mb() -> Result<u32, Box<dyn std::error::Error>> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kib: u64 = rest
                .split_whitespace()
                .next()
                .ok_or("malformed MemAvailable line in /proc/meminfo")?
                .parse()?;
            return Ok((kib / 1024).min(u64::from(u32::MAX)) as u32);
        }
    }
    Err("MemAvailable not found in /proc/meminfo".into())
}

/// Removes a host TAP interface when the provisioning function returns — by
/// any path.
///
/// Created right after the TAP is set up, it is dropped when
/// `provision_kvm_machine_native` returns: whether the guest exited cleanly,
/// the operator asked for a graceful shutdown, or a later step failed with `?`.
/// A TAP is therefore never orphaned — the previous Ctrl+C-only cleanup left
/// one behind on every normal exit, and on a mid-pipeline failure.
#[cfg(target_os = "linux")]
struct TapGuard(Option<String>);

#[cfg(target_os = "linux")]
impl Drop for TapGuard {
    fn drop(&mut self) {
        if let Some(tap) = &self.0 {
            let _ = std::process::Command::new("ip")
                .args(["link", "del", tap])
                .output();
            eprintln!("[vmm] removed host TAP interface '{tap}'.");
        }
    }
}

/// Natively provisions a VM on the Linux host through the KVM API and boots it.
///
/// Steps: open `/dev/kvm`, create the VM, set up its in-kernel interrupt
/// routing, allocate and register guest memory, create the vCPUs, set up the
/// host TAP interface, perform a Direct Kernel Boot, attach the MMIO device bus
/// (with a virtio-blk device when a disk image is present), then run every vCPU
/// — each on its own host thread, sharing the MMIO bus and guest memory. The
/// run loops block until the guest exits, so on a successful boot this function
/// only returns once the guest has shut down.
///
/// `stop` is a caller-owned flag: raising it (typically from a Ctrl+C handler)
/// makes every vCPU wind down and the function return — a graceful shutdown.
#[cfg(target_os = "linux")]
pub fn provision_kvm_machine_native(
    vcpus: u32,
    memory_mb: u32,
    vm_name: &str,
    network_bridge: Option<&str>,
    disk_image_path: Option<&str>,
    kernel_path: &str,
    stop: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<String, Box<dyn std::error::Error>> {
    use kvm_bindings::{
        kvm_pit_config, kvm_userspace_memory_region, KVM_API_VERSION, KVM_MAX_CPUID_ENTRIES,
        KVM_PIT_SPEAKER_DUMMY,
    };
    use kvm_ioctls::{Kvm, VcpuFd};
    use std::sync::{Arc, Mutex};
    use vm_memory::{GuestAddress, GuestMemory, GuestMemoryMmap};

    if vm_name.trim().is_empty() {
        return Err("the VM name must not be empty".into());
    }

    // Single check point: no unvalidated value reaches KVM.
    let spec = validate_spec(vcpus, memory_mb, network_bridge, disk_image_path)?;
    let mem_size = memory_bytes(spec.memory_mb);

    // Host memory capability: reject — before the guest-memory mmap — a VM
    // whose RAM would not fit this host. Best-effort: if `/proc/meminfo`
    // cannot be read, the architectural `MAX_MEMORY_MB` bound enforced by
    // `validate_spec` still applies and provisioning proceeds.
    match read_host_available_memory_mb() {
        Ok(available_mb) => {
            let memory_cap = effective_memory_cap(available_mb);
            if spec.memory_mb > memory_cap {
                return Err(format!(
                    "requested {} MiB of RAM but this host can safely grant at most \
                     {} MiB ({} MiB available, {} MiB reserved for the host)",
                    spec.memory_mb, memory_cap, available_mb, HOST_RESERVE_MB
                )
                .into());
            }
        }
        Err(e) => eprintln!("[vmm] host memory capability check skipped: {e}"),
    }

    // 1. Open the /dev/kvm hypervisor device, then confirm this host's KVM can
    //    run the requested number of vCPUs. `validate_spec` has already
    //    enforced the architectural ceiling; this rejects, up front, a request
    //    that exceeds *this host's* reported maximum, rather than failing
    //    partway through the `create_vcpu` loop.
    let kvm = Kvm::new()?;
    let vcpu_cap = effective_vcpu_cap(kvm.get_max_vcpus() as u32);
    if spec.vcpus > vcpu_cap {
        return Err(format!(
            "requested {} vCPU(s) but this host's KVM supports at most {}",
            spec.vcpus, vcpu_cap
        )
        .into());
    }

    // 2. Check the KVM API version exposed by the kernel.
    let api_version = kvm.get_api_version();
    if api_version != KVM_API_VERSION as i32 {
        return Err(format!(
            "incompatible KVM API version: expected {KVM_API_VERSION}, got {api_version}"
        )
        .into());
    }

    // 3. Create the virtual machine.
    let vm_fd = kvm.create_vm()?;

    // 3a. Reserve the KVM TSS region. `KVM_SET_TSS_ADDR` defines a three-page
    //     area, clear of every memory slot, that Intel VT-x needs to virtualize
    //     guest state; KVM documents it as required on Intel hosts. The address
    //     sits high in the 32-bit space, well above guest RAM and the virtio
    //     MMIO window.
    const KVM_TSS_ADDRESS: usize = 0xfffb_d000;
    vm_fd
        .set_tss_address(KVM_TSS_ADDRESS)
        .map_err(|e| format!("failed to set the KVM TSS address: {e}"))?;

    // 4. Interrupt routing. This MUST happen after `create_vm` and *before*
    //    any `create_vcpu` call: KVM builds each vCPU's local APIC from the
    //    VM's interrupt-controller model, so that model has to exist first.
    //    The in-kernel IRQCHIP also routes the virtio-blk irqfd registered in
    //    step 10, and the PIT keeps the kernel's timer calibration alive.
    //
    //    `KVM_PIT_SPEAKER_DUMMY` makes KVM emulate the speaker/control port
    //    `0x61` in-kernel. Without it, `0x61` exits to userspace, the PIT
    //    channel-2 gate the guest sets there is dropped, channel 2 never
    //    counts, and the kernel's `quick_pit_calibrate` TSC calibration spins
    //    forever — the guest hangs before printing a single line.
    vm_fd
        .create_irq_chip()
        .map_err(|e| format!("failed to create the in-kernel IRQCHIP: {e}"))?;
    let pit_config = kvm_pit_config {
        flags: KVM_PIT_SPEAKER_DUMMY,
        ..Default::default()
    };
    vm_fd
        .create_pit2(pit_config)
        .map_err(|e| format!("failed to create the in-kernel PIT (i8254): {e}"))?;

    // 5. Allocate a guest memory region: an anonymous, private mmap, hence
    //    fully isolated from the rest of the system.
    let guest_memory = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), mem_size)])
        .map_err(|e| format!("guest memory allocation failed: {e}"))?;
    let host_addr = guest_memory
        .get_host_address(GuestAddress(0))
        .map_err(|e| format!("guest memory host address not found: {e}"))?;

    // 6. Register the memory region with KVM (slot 0).
    let mem_region = kvm_userspace_memory_region {
        slot: 0,
        guest_phys_addr: 0,
        memory_size: mem_size as u64,
        userspace_addr: host_addr as u64,
        flags: 0,
    };
    // SAFETY: `host_addr` and `mem_size` come from a valid mmap owned by
    // `guest_memory`, which stays alive until the end of this function. The
    // region is therefore mapped and accessible for the whole lifetime of the
    // VmFd, which satisfies the `set_user_memory_region` contract.
    unsafe { vm_fd.set_user_memory_region(mem_region)? };

    // 7. Create the exact number of vCPUs requested. Each vCPU is given the
    //    host's supported CPUID: without `KVM_SET_CPUID2` the guest reads an
    //    all-zero CPUID and a real kernel triple-faults during early CPU setup,
    //    before it can emit a single console byte.
    let supported_cpuid = kvm
        .get_supported_cpuid(KVM_MAX_CPUID_ENTRIES)
        .map_err(|e| format!("failed to query the host's supported CPUID: {e}"))?;
    let mut vcpu_handles: Vec<VcpuFd> = Vec::with_capacity(spec.vcpus as usize);
    for index in 0..spec.vcpus {
        let vcpu = vm_fd.create_vcpu(index as u64)?;
        vcpu.set_cpuid2(&supported_cpuid)
            .map_err(|e| format!("failed to set guest CPUID on vCPU {index}: {e}"))?;
        vcpu_handles.push(vcpu);
    }

    // 8. Network configuration: when a bridge is requested, create a host TAP
    //    interface and attach it to that bridge.
    let tap_name: Option<String> = match spec.network_bridge.as_deref() {
        Some(bridge) => {
            println!("[vmm] Network: provisioning host TAP for bridge '{bridge}'...");
            let tap = network::setup_tap_interface(vm_name, bridge)?;
            println!("[vmm] Network: host TAP '{tap}' attached to bridge '{bridge}'.");
            Some(tap)
        }
        None => {
            println!("[vmm] Network: no bridge requested, VM stays isolated.");
            None
        }
    };
    // The TAP, if any, is now owned by a guard that removes it on every return
    // path — clean exit, graceful shutdown, or a later step failing — so it is
    // never orphaned.
    let _tap_guard = TapGuard(tap_name);

    // 9. Direct Kernel Boot: load the kernel, build the page tables, put every
    //    vCPU into 64-bit long mode, and build the boot-protocol zero page.
    if !std::path::Path::new(kernel_path).exists() {
        return Err(format!(
            "kernel image '{kernel_path}' not found. Point ai-vmm at a vmlinux (ELF) \
             kernel: set the AI_VMM_KERNEL environment variable, add a `kernel_path` \
             entry to credentials.toml, or place a 'vmlinux' file in the working \
             directory. The README explains how to obtain a kernel image."
        )
        .into());
    }
    println!("[vmm] Direct Kernel Boot: loading kernel image '{kernel_path}'...");
    let entry_point = boot::load_kernel(&guest_memory, kernel_path)?;
    boot::setup_boot_memory(&guest_memory)?;
    for vcpu in &vcpu_handles {
        boot::setup_vcpu_registers(vcpu, entry_point)?;
        boot::configure_lapic(vcpu)?;
    }
    boot::setup_zero_page(&guest_memory, mem_size, spec.disk_image_path.is_some())?;

    // 9a. SMP enumeration. Direct Kernel Boot hands the guest no ACPI or BIOS,
    //     so without an MP table the kernel cannot discover the application
    //     processors and runs on the bootstrap processor alone — whatever the
    //     vCPU count. An MP table is written only for a multi-vCPU VM; a
    //     single-vCPU boot keeps the proven, table-free minimal path.
    if spec.vcpus > 1 {
        mptable::setup_mptable(&guest_memory, spec.vcpus)?;
        println!(
            "[vmm] SMP: MP table written — {} processors enumerated for the guest.",
            spec.vcpus
        );
    }

    // 10. MMIO device bus. When a root-filesystem image was found, attach a
    //     virtio-mmio block device and wire its interrupt EventFd to KVM as an
    //     irqfd on GSI VIRTIO_BLK_GSI — the in-kernel IRQCHIP from step 4
    //     routes it. The guest finds the device via the `virtio_mmio.device=`
    //     kernel command-line parameter set in `boot::setup_zero_page`.
    let mut mmio_bus = mmio::MmioBus::new();
    match spec.disk_image_path.as_deref() {
        Some(disk) => {
            println!("[vmm] Storage: attaching virtio-blk backed by '{disk}'...");
            let block = storage::create_block_device(disk)?;
            vm_fd
                .register_irqfd(block.irq_eventfd(), VIRTIO_BLK_GSI)
                .map_err(|e| format!("failed to register the virtio-blk irqfd: {e}"))?;
            mmio_bus.register(VIRTIO_BLK_MMIO_BASE, VIRTIO_BLK_MMIO_SIZE, Box::new(block));
            println!(
                "[vmm] Storage: virtio-blk on the MMIO bus at {VIRTIO_BLK_MMIO_BASE:#x} \
                 (IRQ {VIRTIO_BLK_GSI})."
            );
        }
        None => {
            println!("[vmm] Storage: no disk image requested — booting without a block device.");
        }
    }

    // 11. Run the VM. The bootstrap processor (vCPU 0) drives the boot on this
    //     thread; each application processor (vCPU 1..N) gets its own host
    //     thread and waits inside KVM for the INIT-SIPI-SIPI the guest kernel
    //     sends during SMP bring-up — KVM's in-kernel local APIC performs that
    //     handshake. Every vCPU thread shares the MMIO bus, guest memory and
    //     the caller's `stop` flag; `run_vcpu_loop` blocks until its vCPU stops
    //     and raises that flag, so when any vCPU stops — the guest exiting, a
    //     fault, or the operator raising `stop` for a graceful shutdown —
    //     every other vCPU leaves its loop too and can be joined.
    println!(
        "[vmm] 🚀 Direct Kernel Boot configured (entry point {entry_point:#x}). Starting \
         {} vCPU(s) — kernel log follows. Press Ctrl+C to stop.",
        spec.vcpus
    );
    println!("──────────────────────────── guest console ────────────────────────────");

    let guest_memory = Arc::new(guest_memory);
    let mmio_bus = Arc::new(Mutex::new(mmio_bus));

    let mut vcpus = vcpu_handles.into_iter();
    let bsp = vcpus.next().ok_or("no bootstrap vCPU was created")?;
    let mut ap_threads = Vec::new();
    for (offset, ap) in vcpus.enumerate() {
        let ap_id = offset + 1;
        let bus = Arc::clone(&mmio_bus);
        let memory = Arc::clone(&guest_memory);
        let stop_flag = Arc::clone(stop);
        ap_threads.push(std::thread::spawn(move || {
            if let Err(e) = run::run_vcpu_loop(&ap, &bus, &memory, &stop_flag) {
                eprintln!("[vmm] vCPU {ap_id} stopped with an error: {e}");
            }
        }));
    }

    let bsp_outcome = run::run_vcpu_loop(&bsp, &mmio_bus, &guest_memory, stop);

    // The bootstrap processor has stopped and `run_vcpu_loop` has raised the
    // VM-wide flag: join every application-processor thread so no vCPU outlives
    // the VM, then surface the bootstrap processor's outcome.
    for thread in ap_threads {
        let _ = thread.join();
    }
    bsp_outcome?;

    Ok(format!("VM '{vm_name}' guest has exited."))
}

/// Non-Linux variant: KVM is unavailable, allocation is refused.
///
/// The specification is still validated so that a precise, useful error
/// message can be returned to the operator.
#[cfg(not(target_os = "linux"))]
pub fn provision_kvm_machine_native(
    vcpus: u32,
    memory_mb: u32,
    vm_name: &str,
    network_bridge: Option<&str>,
    disk_image_path: Option<&str>,
    _kernel_path: &str,
    _stop: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<String, Box<dyn std::error::Error>> {
    if vm_name.trim().is_empty() {
        return Err("the VM name must not be empty".into());
    }
    let spec = validate_spec(vcpus, memory_mb, network_bridge, disk_image_path)?;
    let mem_size = memory_bytes(spec.memory_mb);
    let network_summary = match spec.network_bridge.as_deref() {
        Some(bridge) => format!("bridge '{bridge}'"),
        None => "no bridge".to_string(),
    };
    Err(format!(
        "KVM unavailable: VM '{}' ({} vCPU / {} MiB, i.e. {} bytes, {}) can only \
         be provisioned on a Linux host that exposes /dev/kvm. Current platform: {}.",
        vm_name,
        spec.vcpus,
        spec.memory_mb,
        mem_size,
        network_summary,
        std::env::consts::OS
    )
    .into())
}

/// Formal proofs checked by the Kani model checker (`cargo kani`).
///
/// These harnesses are only compiled under the model checker (`cfg(kani)`);
/// they have no effect on normal compilation or execution. They pass `None`
/// for the optional string fields so the verified path stays allocation-free.
#[cfg(kani)]
mod proofs {
    use super::{memory_bytes, validate_spec, MAX_MEMORY_MB, MAX_VCPUS, MIN_MEMORY_MB};

    /// Proof: any request for 0 vCPUs is always rejected, regardless of the
    /// amount of memory.
    #[kani::proof]
    fn proof_zero_vcpu_always_rejected() {
        let memory_mb: u32 = kani::any();
        assert!(validate_spec(0, memory_mb, None, None).is_err());
    }

    /// Proof: when `validate_spec` succeeds, the returned specification is
    /// necessarily within every allowed bound.
    #[kani::proof]
    fn proof_accepted_spec_is_within_bounds() {
        let vcpus: u32 = kani::any();
        let memory_mb: u32 = kani::any();
        if let Ok(spec) = validate_spec(vcpus, memory_mb, None, None) {
            assert!(spec.vcpus >= 1 && spec.vcpus <= MAX_VCPUS);
            assert!(spec.memory_mb >= MIN_MEMORY_MB && spec.memory_mb <= MAX_MEMORY_MB);
        }
    }

    /// Proof: any configuration strictly within bounds is accepted (no false
    /// negative in the validation).
    #[kani::proof]
    fn proof_in_bounds_spec_is_accepted() {
        let vcpus: u32 = kani::any();
        let memory_mb: u32 = kani::any();
        kani::assume(vcpus >= 1 && vcpus <= MAX_VCPUS);
        kani::assume(memory_mb >= MIN_MEMORY_MB && memory_mb <= MAX_MEMORY_MB);
        assert!(validate_spec(vcpus, memory_mb, None, None).is_ok());
    }

    /// Proof: for any validated memory amount, `memory_bytes` never overflows,
    /// stays within the expected range and yields a size aligned on a
    /// 4096-byte page — a prerequisite of `set_user_memory_region`.
    #[kani::proof]
    fn proof_memory_bytes_safe_and_page_aligned() {
        let memory_mb: u32 = kani::any();
        kani::assume(memory_mb >= MIN_MEMORY_MB && memory_mb <= MAX_MEMORY_MB);
        let bytes = memory_bytes(memory_mb);
        assert!(bytes % 4096 == 0);
        assert!(bytes >= MIN_MEMORY_MB as usize * 1024 * 1024);
        assert!(bytes <= MAX_MEMORY_MB as usize * 1024 * 1024);
    }

    /// Proof: validation is idempotent — re-validating the vCPU/memory of an
    /// already accepted specification succeeds and yields the same values.
    ///
    /// Only the integer fields are compared: a deep `==` on `VmSpec` would drag
    /// its `Option<String>` fields — and the heap/allocator machinery behind
    /// `String` — into CBMC, which is needlessly expensive to model-check.
    #[kani::proof]
    fn proof_validation_is_idempotent() {
        let vcpus: u32 = kani::any();
        let memory_mb: u32 = kani::any();
        if let Ok(spec) = validate_spec(vcpus, memory_mb, None, None) {
            let revalidated = validate_spec(spec.vcpus, spec.memory_mb, None, None);
            assert!(revalidated.is_ok());
            if let Ok(again) = revalidated {
                assert!(again.vcpus == spec.vcpus);
                assert!(again.memory_mb == spec.memory_mb);
            }
        }
    }

    /// Proof: the composed pipeline is sound — for any specification accepted
    /// by `validate_spec`, `memory_bytes` produces a non-zero, page-aligned
    /// size within the hardware-allocatable range.
    #[kani::proof]
    fn proof_validated_spec_yields_safe_memory_size() {
        let vcpus: u32 = kani::any();
        let memory_mb: u32 = kani::any();
        if let Ok(spec) = validate_spec(vcpus, memory_mb, None, None) {
            let bytes = memory_bytes(spec.memory_mb);
            assert!(bytes > 0);
            assert!(bytes % 4096 == 0);
            assert!(bytes <= MAX_MEMORY_MB as usize * 1024 * 1024);
        }
    }

    /// Proof: the effective vCPU cap never exceeds either the architectural
    /// ceiling or the host's reported maximum — whichever is tighter wins.
    #[kani::proof]
    fn proof_effective_vcpu_cap_respects_both_limits() {
        let host_max: u32 = kani::any();
        let cap = super::effective_vcpu_cap(host_max);
        assert!(cap <= MAX_VCPUS);
        assert!(cap <= host_max);
    }

    /// Proof: the effective memory cap never exceeds the architectural ceiling.
    #[kani::proof]
    fn proof_memory_cap_respects_architectural_max() {
        let available_mb: u32 = kani::any();
        assert!(super::effective_memory_cap(available_mb) <= MAX_MEMORY_MB);
    }

    /// Proof: granting the memory cap always leaves the host reserve free —
    /// the cap is either zero or small enough that `cap + reserve` still fits
    /// inside the host's available memory.
    #[kani::proof]
    fn proof_memory_cap_preserves_host_reserve() {
        let available_mb: u32 = kani::any();
        let cap = super::effective_memory_cap(available_mb);
        assert!(
            cap == 0
                || u64::from(cap) + u64::from(super::HOST_RESERVE_MB) <= u64::from(available_mb)
        );
    }

    /// Proof: the memory-unit scaler is exact or refuses — when `scaled_mib`
    /// returns a size, it is precisely `value * factor`, never a wrapped one,
    /// so the headless `provision` path can never be tricked, via overflow,
    /// into allocating a smaller VM than the operator asked for.
    #[kani::proof]
    fn proof_scaled_mib_is_exact() {
        let value: u64 = kani::any();
        let factor: u32 = kani::any();
        if let Some(mb) = super::scaled_mib(value, factor) {
            assert!(value.checked_mul(u64::from(factor)) == Some(u64::from(mb)));
        }
    }
}

/// Tests for the pure validation logic (all platforms) and the in-kernel
/// interrupt routing (Linux with `/dev/kvm`).
#[cfg(test)]
mod tests {
    use super::{
        memory_bytes, provision_kvm_machine_native, validate_spec, MAX_MEMORY_MB, MAX_VCPUS,
        MIN_MEMORY_MB,
    };

    #[test]
    fn rejects_zero_vcpu() {
        assert!(validate_spec(0, 2048, None, None).is_err());
    }

    #[test]
    fn rejects_too_many_vcpus() {
        assert!(validate_spec(MAX_VCPUS + 1, 2048, None, None).is_err());
    }

    #[test]
    fn rejects_insufficient_memory() {
        assert!(validate_spec(2, MIN_MEMORY_MB - 1, None, None).is_err());
    }

    #[test]
    fn rejects_excessive_memory() {
        assert!(validate_spec(2, MAX_MEMORY_MB + 1, None, None).is_err());
    }

    #[test]
    fn accepts_a_valid_spec() {
        let spec = validate_spec(2, 2048, None, None).expect("2 vCPU / 2048 MiB must be valid");
        assert_eq!(spec.vcpus, 2);
        assert_eq!(spec.memory_mb, 2048);
        assert_eq!(spec.network_bridge, None);
        assert_eq!(spec.disk_image_path, None);
    }

    #[test]
    fn memory_size_is_page_aligned() {
        let bytes = memory_bytes(2048);
        assert_eq!(bytes, 2048 * 1024 * 1024);
        assert_eq!(bytes % 4096, 0);
    }

    #[test]
    fn rejects_empty_vm_name() {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        assert!(
            provision_kvm_machine_native(2, 256, "   ", None, None, "./vmlinux", &stop).is_err()
        );
    }

    #[test]
    fn trims_and_keeps_named_bridge() {
        let spec = validate_spec(2, 2048, Some("  br0  "), None).expect("valid spec");
        assert_eq!(spec.network_bridge.as_deref(), Some("br0"));
    }

    #[test]
    fn normalizes_blank_bridge_to_none() {
        let spec = validate_spec(2, 2048, Some("   "), None).expect("valid spec");
        assert_eq!(spec.network_bridge, None);
    }

    #[test]
    fn treats_literal_none_bridge_as_no_bridge() {
        let spec = validate_spec(2, 2048, Some("none"), None).expect("valid spec");
        assert_eq!(spec.network_bridge, None);
    }

    #[test]
    fn trims_and_keeps_disk_image_path() {
        let spec = validate_spec(2, 2048, None, Some("  ./rootfs.ext4  ")).expect("valid spec");
        assert_eq!(spec.disk_image_path.as_deref(), Some("./rootfs.ext4"));
    }

    #[test]
    fn normalizes_blank_disk_image_path_to_none() {
        let spec = validate_spec(2, 2048, None, Some("   ")).expect("valid spec");
        assert_eq!(spec.disk_image_path, None);
    }

    /// Best-effort integration test: needs `/dev/kvm`. Verifies that the
    /// in-kernel IRQCHIP and PIT can be created on a fresh VM — the routing
    /// `provision_kvm_machine_native` sets up, in the order KVM requires
    /// (both before any vCPU exists).
    #[cfg(target_os = "linux")]
    #[test]
    fn creates_in_kernel_irqchip_and_pit() {
        use kvm_bindings::kvm_pit_config;
        use kvm_ioctls::Kvm;

        let kvm = match Kvm::new() {
            Ok(kvm) => kvm,
            Err(e) => {
                eprintln!("[irq-smoke] /dev/kvm unavailable, test inconclusive: {e}");
                return;
            }
        };
        let vm = kvm.create_vm().expect("create_vm");

        vm.create_irq_chip()
            .expect("create_irq_chip must succeed before any vCPU is created");
        let pit_config = kvm_pit_config {
            flags: 0,
            ..Default::default()
        };
        vm.create_pit2(pit_config)
            .expect("create_pit2 must succeed");

        eprintln!("[irq-smoke] in-kernel IRQCHIP and PIT created successfully");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn effective_vcpu_cap_is_the_tighter_limit() {
        use super::effective_vcpu_cap;
        assert_eq!(effective_vcpu_cap(4), 4); // host below our ceiling
        assert_eq!(effective_vcpu_cap(10_000), MAX_VCPUS); // host above -> clamped
        assert_eq!(effective_vcpu_cap(MAX_VCPUS), MAX_VCPUS);
    }

    /// Best-effort integration test: needs `/dev/kvm`. Confirms the host KVM
    /// reports a usable vCPU maximum and that the effective cap stays sane.
    #[cfg(target_os = "linux")]
    #[test]
    fn host_reports_a_usable_vcpu_maximum() {
        use super::effective_vcpu_cap;
        use kvm_ioctls::Kvm;

        let kvm = match Kvm::new() {
            Ok(kvm) => kvm,
            Err(e) => {
                eprintln!("[cap-smoke] /dev/kvm unavailable, test inconclusive: {e}");
                return;
            }
        };
        let host_max = kvm.get_max_vcpus() as u32;
        assert!(host_max >= 1, "KVM must support at least one vCPU");
        let cap = effective_vcpu_cap(host_max);
        assert!((1..=MAX_VCPUS).contains(&cap));
        eprintln!("[cap-smoke] host KVM supports up to {host_max} vCPUs (effective cap {cap})");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn effective_memory_cap_reserves_host_headroom() {
        use super::{effective_memory_cap, HOST_RESERVE_MB};
        // A roomy host: capped by the architectural maximum.
        assert_eq!(effective_memory_cap(u32::MAX), MAX_MEMORY_MB);
        // A host with no spare RAM: nothing is granted, the reserve is kept.
        assert_eq!(effective_memory_cap(HOST_RESERVE_MB), 0);
        assert_eq!(effective_memory_cap(0), 0);
        // A modest host: available memory minus the reserve.
        assert_eq!(
            effective_memory_cap(HOST_RESERVE_MB + 512),
            512.min(MAX_MEMORY_MB)
        );
    }

    /// Best-effort integration test: confirms the host's `/proc/meminfo`
    /// reports a usable amount of available memory.
    #[cfg(target_os = "linux")]
    #[test]
    fn host_reports_available_memory() {
        use super::read_host_available_memory_mb;
        let available = read_host_available_memory_mb().expect("read /proc/meminfo");
        assert!(
            available > 0,
            "a live host always reports some available memory"
        );
        eprintln!("[mem-smoke] host reports {available} MiB available");
    }

    /// A 1 TiB VM clears the architectural `MAX_MEMORY_MB` bound, but no test
    /// host has that much RAM available, so the host memory capability check
    /// must reject it before any hardware is touched.
    #[cfg(target_os = "linux")]
    #[test]
    fn provisioning_rejects_a_vm_larger_than_the_host() {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let result = provision_kvm_machine_native(
            1,
            MAX_MEMORY_MB,
            "oversized",
            None,
            None,
            "./vmlinux",
            &stop,
        );
        assert!(result.is_err(), "an oversized VM must be rejected");
    }

    #[test]
    fn parses_memory_sizes_and_rejects_bad_input() {
        use super::parse_memory_mb;
        assert_eq!(parse_memory_mb("2048").expect("plain MiB"), 2048);
        assert_eq!(parse_memory_mb("512M").expect("M suffix"), 512);
        assert_eq!(parse_memory_mb("512MiB").expect("MiB suffix"), 512);
        assert_eq!(parse_memory_mb("2G").expect("G suffix"), 2048);
        assert_eq!(parse_memory_mb("  4gb ").expect("GB suffix, padded"), 4096);
        assert_eq!(parse_memory_mb("1T").expect("T suffix"), 1024 * 1024);
        // Malformed input and an overflowing size are clean errors, not panics.
        assert!(parse_memory_mb("abc").is_err());
        assert!(parse_memory_mb("").is_err());
        assert!(parse_memory_mb("99999999T").is_err());
    }
}
