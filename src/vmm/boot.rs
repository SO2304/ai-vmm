//! Direct Kernel Boot for x86_64: load a Linux kernel image into guest memory,
//! build the boot data structures it expects, and put the vCPUs into 64-bit
//! long mode.
//!
//! # Why this is delicate
//!
//! A freshly created KVM vCPU starts in 16-bit real mode. A modern `vmlinux`
//! ELF kernel is entered at `startup_64`, which requires the CPU to ALREADY be
//! in 64-bit long mode with paging enabled, and `RSI` pointing at a populated
//! `boot_params` "zero page". Bridging that gap means configuring, before the
//! first guest instruction runs:
//!
//!  * a 4-level page table that identity-maps guest memory, referenced by `CR3`;
//!  * `CR0.PE` + `CR0.PG`, `CR4.PAE`, and `EFER.LME` + `EFER.LMA`;
//!  * a GDT with a 64-bit code segment (`CS.L = 1`) and a flat data segment;
//!  * a `boot_params` zero page carrying the e820 memory map and the address of
//!    the kernel command line.
//!
//! A single wrong bit does not produce a clean error: the CPU faults, cannot
//! handle the fault (there is no IDT), faults again, and KVM reports a triple
//! fault. The bit-packing of the page-table and GDT entries is isolated into
//! the pure functions [`pd_entry`] and [`gdt_entry`] and proven correct by the
//! Kani harnesses at the bottom of this file.
//!
//! # vmlinux vs bzImage
//!
//! We load a `vmlinux` ELF and enter it at `startup_64`. That entry uses the
//! SAME `boot_params`/zero-page convention as the 64-bit bzImage boot protocol:
//! `RSI` must hold the guest-physical address of the zero page. Only the PVH
//! entry point — which we do not use — would take a different structure.
//!
//! # Guest-physical layout
//!
//! Every boot structure is kept in the first megabyte, below the kernel, and
//! the page-aligned ones (`zero page`, page tables) sit on 4 KiB boundaries:
//!
//! ```text
//!   0x00000500  GDT       (null, 64-bit code, flat data)
//!   0x00000520  IDT       (empty placeholder — a valid pointer is required)
//!   0x00007000  zero page (boot_params: e820 map + cmdline pointer)
//!   0x00009000  PML4      (level-4 page table)
//!   0x0000a000  PDPT      (level-3 page-directory-pointer table)
//!   0x0000b000  PD        (level-2 page directory, 2 MiB pages)
//!   0x00020000  kernel command line (NUL-terminated ASCII)
//!   0x00100000  kernel image / entry point
//! ```

use kvm_bindings::{kvm_lapic_state, kvm_segment};
use kvm_ioctls::VcpuFd;
use linux_loader::loader::bootparam::{boot_e820_entry, boot_params};
use linux_loader::loader::elf::Elf;
use linux_loader::loader::KernelLoader;
use std::fs::File;
use vm_memory::{Address, Bytes, GuestAddress, GuestMemoryMmap};

// --- Guest-physical addresses of the boot data structures ------------------

/// Global Descriptor Table.
const BOOT_GDT_ADDR: u64 = 0x500;
/// Interrupt Descriptor Table (empty, but `IDTR` still needs a valid pointer).
const BOOT_IDT_ADDR: u64 = 0x520;
/// The Linux x86 boot-protocol "zero page" (`boot_params`); 4 KiB-aligned.
const ZERO_PAGE_ADDR: u64 = 0x7000;
/// Level-4 page table (PML4).
const PML4_ADDR: u64 = 0x9000;
/// Level-3 page table (page-directory-pointer table).
const PDPT_ADDR: u64 = 0xa000;
/// Level-2 page table (page directory).
const PD_ADDR: u64 = 0xb000;
/// Address of the NUL-terminated kernel command line.
const CMDLINE_ADDR: u64 = 0x2_0000;

/// Number of entries in the page directory; each entry maps a 2 MiB page, so
/// the directory identity-maps the first `512 * 2 MiB = 1 GiB` of memory.
const PD_ENTRY_COUNT: u64 = 512;

/// Number of 8-byte GDT entries written by [`setup_boot_memory`].
const GDT_ENTRY_COUNT: u64 = 3;

/// Largest kernel command line the guest's boot buffer accepts. The x86_64
/// Linux `COMMAND_LINE_SIZE` is 2048 bytes; a longer line is silently
/// truncated by the kernel — which would drop a trailing `root=` directive and
/// break the boot. `proof_kernel_cmdline_fits_guest_buffer` proves both
/// command lines below stay within this bound.
const MAX_GUEST_CMDLINE_LEN: usize = 2048;

/// Kernel command line for a diskless VM (no root filesystem attached).
///
/// `earlyprintk=ttyS0` surfaces the kernel log on COM1 from the very first
/// boot stage, before the real console registers; `console=ttyS0` then routes
/// the ongoing log to that same COM1 (both printed by the run loop's UART
/// emulation); `virtio_mmio.device=1K@0xd0000000:5` tells the guest where to
/// probe for the virtio-blk device and which IRQ it uses.
///
/// `nox2apic` and `lapic=notscdeadline` harden the local-APIC bring-up. Once
/// the MP table makes the kernel enumerate more than one CPU it follows the
/// full SMP APIC-init path: that path probes x2APIC and arms the local-APIC
/// timer in TSC-deadline mode. The guest's TSC is unstable here (no invariant
/// TSC is exposed), so TSC-deadline arming can stall; `notscdeadline` forces
/// the classic periodic APIC timer and `nox2apic` skips the x2APIC probe. Both
/// are inert on the single-CPU path, which never reaches that code.
const KERNEL_CMDLINE_DISKLESS: &str = "earlyprintk=ttyS0 console=ttyS0 reboot=k panic=1 pci=off nox2apic lapic=notscdeadline virtio_mmio.device=1K@0xd0000000:5";

/// Kernel command line for a VM with a virtio-blk root disk: the diskless line
/// plus `root=/dev/vda rw`, which mounts the root filesystem read/write from
/// the first virtio-blk device.
///
/// The `root=` directive is added *only* when a disk is attached: on a
/// diskless VM it would make the kernel panic looking for an absent
/// `/dev/vda`, so the two lines are kept strictly separate.
const KERNEL_CMDLINE_WITH_DISK: &str = "earlyprintk=ttyS0 console=ttyS0 reboot=k panic=1 pci=off nox2apic lapic=notscdeadline virtio_mmio.device=1K@0xd0000000:5 root=/dev/vda rw";

/// Returns the kernel command line for a VM, carrying the `root=` directive
/// exactly when a virtio-blk root disk is attached.
const fn kernel_cmdline(has_disk: bool) -> &'static str {
    if has_disk {
        KERNEL_CMDLINE_WITH_DISK
    } else {
        KERNEL_CMDLINE_DISKLESS
    }
}

// --- x86_64 control-register and EFER bits ---------------------------------

/// `CR0.PE` — Protected Mode Enable.
const CR0_PE: u64 = 1 << 0;
/// `CR0.PG` — Paging enable.
const CR0_PG: u64 = 1 << 31;
/// `CR4.PAE` — Physical Address Extension (mandatory for long mode).
const CR4_PAE: u64 = 1 << 5;
/// `EFER.LME` — Long Mode Enable.
const EFER_LME: u64 = 1 << 8;
/// `EFER.LMA` — Long Mode Active.
const EFER_LMA: u64 = 1 << 10;

/// `RFLAGS` with only the architecturally-reserved bit 1 set.
const RFLAGS_RESERVED_ONE: u64 = 0x0000_0002;

// --- Page-table entry flags ------------------------------------------------

/// Flags for a PML4/PDPT entry that points at the next table: present + writable.
const PT_PRESENT_WRITABLE: u64 = 0b11;
/// Flags for a page-directory entry mapping a 2 MiB page:
/// present (bit 0) + writable (bit 1) + page-size / huge page (bit 7).
const PD_HUGE_PAGE_FLAGS: u64 = 0b1000_0011;

// --- GDT descriptor flags --------------------------------------------------

/// Access byte + granularity nibble for the 64-bit code segment.
const GDT_FLAGS_CODE: u16 = 0xa09b;
/// Access byte + granularity nibble for the flat data segment.
const GDT_FLAGS_DATA: u16 = 0xc093;

// --- Linux boot-protocol constants -----------------------------------------

/// e820 entry type for usable RAM.
const E820_RAM: u32 = 1;
/// Top of the low usable-RAM region: the start of the Extended BIOS Data Area.
/// The e820 map's first region is `[0, EBDA_START)`.
const EBDA_START: u64 = 0x9_fc00;
/// Start of the high usable-RAM region, just above the legacy sub-1 MiB hole.
const HIGH_MEMORY_START: u64 = 0x10_0000;
/// `type_of_loader` value for a non-standard bootloader (a custom VMM).
const LOADER_TYPE_UNDEFINED: u8 = 0xff;
/// `boot_flag` magic at offset 0x1fe of the boot sector.
const BOOT_FLAG_MAGIC: u16 = 0xaa55;
/// `header` magic — the little-endian ASCII "HdrS" of a valid setup header.
const SETUP_HEADER_MAGIC: u32 = 0x5372_6448;

/// Builds a page-directory entry that identity-maps the 2 MiB page at `index`.
///
/// Entry `i` maps guest-physical `[i * 2 MiB, (i + 1) * 2 MiB)` to the same
/// host-physical range. The 2 MiB frame number is shifted into bits 21+, which
/// leaves bits 0..21 free for the flags — [`proof_pd_entry_keeps_flags`] proves
/// the two never collide.
const fn pd_entry(index: u64) -> u64 {
    (index << 21) | PD_HUGE_PAGE_FLAGS
}

/// Packs an x86 segment descriptor into its 64-bit GDT representation.
///
/// `flags` carries the access byte (bits 0..8) and the granularity nibble
/// (bits 12..16); `base` is the 32-bit segment base; `limit` is the 20-bit
/// segment limit. The architectural layout scatters each field across the
/// 64-bit word — [`proof_gdt_entry_encodes_full_limit`] proves the limit
/// survives that scattering intact.
const fn gdt_entry(flags: u16, base: u32, limit: u32) -> u64 {
    (((base as u64) & 0xff00_0000) << 32)
        | (((flags as u64) & 0x0000_f0ff) << 40)
        | (((limit as u64) & 0x000f_0000) << 32)
        | (((base as u64) & 0x00ff_ffff) << 16)
        | ((limit as u64) & 0x0000_ffff)
}

/// Builds the descriptor cache for the 64-bit code segment (`CS`).
///
/// `l = 1` selects 64-bit mode; `db` MUST be 0 when `l = 1`. `g = 1` makes the
/// limit 4 KiB-granular so `limit = 0xf_ffff` covers the whole address space.
fn code_segment() -> kvm_segment {
    kvm_segment {
        base: 0,
        limit: 0x000f_ffff,
        selector: 1 << 3, // GDT index 1, ring 0
        type_: 0b1011,    // code: execute/read, accessed
        present: 1,
        dpl: 0,
        db: 0,
        s: 1, // code/data descriptor (not a system descriptor)
        l: 1, // 64-bit long mode
        g: 1, // 4 KiB granularity
        avl: 0,
        unusable: 0,
        padding: 0,
    }
}

/// Builds the descriptor cache shared by every data segment (`DS`/`ES`/`SS`/...).
fn data_segment() -> kvm_segment {
    kvm_segment {
        base: 0,
        limit: 0x000f_ffff,
        selector: 2 << 3, // GDT index 2, ring 0
        type_: 0b0011,    // data: read/write, accessed
        present: 1,
        dpl: 0,
        db: 1,
        s: 1,
        l: 0,
        g: 1,
        avl: 0,
        unusable: 0,
        padding: 0,
    }
}

/// Loads a Linux kernel ELF image into guest memory.
///
/// Opens `kernel_path`, loads its `PT_LOAD` segments into `guest_memory` with
/// rust-vmm's ELF loader, and returns the kernel entry point as reported by the
/// loader — the value that [`setup_vcpu_registers`] writes into `RIP`.
pub fn load_kernel(
    guest_memory: &GuestMemoryMmap,
    kernel_path: &str,
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut kernel_image = File::open(kernel_path)
        .map_err(|e| format!("cannot open kernel image '{kernel_path}': {e}"))?;

    let loaded = Elf::load(guest_memory, None, &mut kernel_image, None)
        .map_err(|e| format!("failed to load kernel ELF '{kernel_path}': {e}"))?;

    Ok(loaded.kernel_load.raw_value())
}

/// Writes the GDT and the identity-mapping page tables into guest memory.
///
/// Called once per VM, before the vCPUs are configured: every vCPU then shares
/// these tables through the fixed addresses referenced by
/// [`setup_vcpu_registers`].
pub fn setup_boot_memory(guest_memory: &GuestMemoryMmap) -> Result<(), Box<dyn std::error::Error>> {
    // GDT: a null descriptor, a 64-bit code segment, a flat data segment.
    let gdt: [u64; 3] = [
        0,
        gdt_entry(GDT_FLAGS_CODE, 0, 0x000f_ffff),
        gdt_entry(GDT_FLAGS_DATA, 0, 0x000f_ffff),
    ];
    for (index, entry) in gdt.iter().enumerate() {
        let addr = GuestAddress(BOOT_GDT_ADDR + index as u64 * 8);
        guest_memory
            .write_obj(*entry, addr)
            .map_err(|e| format!("failed to write GDT entry {index}: {e}"))?;
    }

    // Page tables: PML4 -> PDPT -> PD, identity-mapping the first 1 GiB with
    // 2 MiB pages.
    guest_memory
        .write_obj(PDPT_ADDR | PT_PRESENT_WRITABLE, GuestAddress(PML4_ADDR))
        .map_err(|e| format!("failed to write PML4 entry: {e}"))?;
    guest_memory
        .write_obj(PD_ADDR | PT_PRESENT_WRITABLE, GuestAddress(PDPT_ADDR))
        .map_err(|e| format!("failed to write PDPT entry: {e}"))?;
    for index in 0..PD_ENTRY_COUNT {
        let addr = GuestAddress(PD_ADDR + index * 8);
        guest_memory
            .write_obj(pd_entry(index), addr)
            .map_err(|e| format!("failed to write PD entry {index}: {e}"))?;
    }

    Ok(())
}

/// Builds the Linux boot-protocol "zero page" (`boot_params`) in guest memory.
///
/// Writes the NUL-terminated kernel command line at [`CMDLINE_ADDR`], then a
/// `boot_params` structure at the page-aligned [`ZERO_PAGE_ADDR`] carrying:
///  * a minimal but valid setup header (loader type, magics, cmdline pointer);
///  * an e820 memory map with a single `E820_RAM` region covering all guest RAM.
///
/// The vCPU's `RSI` is pointed at [`ZERO_PAGE_ADDR`] by [`setup_vcpu_registers`]
/// (this function's signature has no vCPU access), which is how the kernel
/// finds this structure.
pub fn setup_zero_page(
    guest_memory: &GuestMemoryMmap,
    mem_size_bytes: usize,
    has_disk: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Write the kernel command line, NUL-terminated. The line carries the
    //    `root=` directive only when a virtio-blk disk is attached.
    let cmdline = kernel_cmdline(has_disk);
    // The line must fit the guest kernel's boot command-line buffer;
    // `proof_kernel_cmdline_fits_guest_buffer` proves both variants always do.
    debug_assert!(cmdline.len() < MAX_GUEST_CMDLINE_LEN);
    guest_memory
        .write_slice(cmdline.as_bytes(), GuestAddress(CMDLINE_ADDR))
        .map_err(|e| format!("failed to write kernel cmdline: {e}"))?;
    let nul_addr = GuestAddress(CMDLINE_ADDR + cmdline.len() as u64);
    guest_memory
        .write_obj(0_u8, nul_addr)
        .map_err(|e| format!("failed to NUL-terminate kernel cmdline: {e}"))?;

    // 2. Build the zero page (boot_params) with a minimal valid setup header.
    let mut params = boot_params::default();
    params.hdr.type_of_loader = LOADER_TYPE_UNDEFINED;
    params.hdr.boot_flag = BOOT_FLAG_MAGIC;
    params.hdr.header = SETUP_HEADER_MAGIC;
    params.hdr.cmd_line_ptr = CMDLINE_ADDR as u32;
    params.hdr.cmdline_size = cmdline.len() as u32;

    // 3. e820 memory map: two usable-RAM regions split around the legacy
    //    sub-1 MiB hole, the layout a real BIOS reports. The split matters:
    //    the guest kernel's `append_e820_table` rejects any map with fewer
    //    than two entries, so a single [0, mem) region is silently discarded
    //    and the kernel falls back to a 640 KiB default and cannot boot.
    let high_ram = (mem_size_bytes as u64).saturating_sub(HIGH_MEMORY_START);
    params.e820_table[0] = boot_e820_entry {
        addr: 0,
        size: EBDA_START,
        type_: E820_RAM,
    };
    params.e820_table[1] = boot_e820_entry {
        addr: HIGH_MEMORY_START,
        size: high_ram,
        type_: E820_RAM,
    };
    params.e820_entries = 2;

    // 4. Write the zero page to its fixed, 4 KiB-aligned guest address.
    guest_memory
        .write_obj(params, GuestAddress(ZERO_PAGE_ADDR))
        .map_err(|e| format!("failed to write boot_params zero page: {e}"))?;

    Ok(())
}

/// Configures one vCPU so it can be entered directly at a 64-bit kernel.
///
/// [`setup_boot_memory`] and [`setup_zero_page`] MUST have run first.
///
/// General-purpose registers:
///  * `RIP` is set to `entry_point`;
///  * `RSI` is set to [`ZERO_PAGE_ADDR`] — the Linux x86_64 boot protocol
///    requires `RSI` to hold the guest-physical address of `boot_params`;
///  * `RFLAGS` is set to `0x2` so interrupts stay masked until the kernel
///    installs its own IDT.
///
/// Special registers switch the CPU into 64-bit long mode: `GDTR`/`IDTR` point
/// at the tables from `setup_boot_memory`; `CS` is a 64-bit code segment; `CR3`
/// points at the PML4; and `CR4.PAE`, `CR0.PE`, `CR0.PG`, `EFER.LME`,
/// `EFER.LMA` are set — the long-mode switch.
pub fn setup_vcpu_registers(
    vcpu: &VcpuFd,
    entry_point: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    // --- General-purpose registers ---
    let mut regs = vcpu.get_regs()?;
    regs.rip = entry_point;
    regs.rsi = ZERO_PAGE_ADDR;
    regs.rflags = RFLAGS_RESERVED_ONE;
    vcpu.set_regs(&regs)?;

    // --- Special registers: switch to 64-bit long mode ---
    let mut sregs = vcpu.get_sregs()?;

    sregs.gdt.base = BOOT_GDT_ADDR;
    sregs.gdt.limit = (GDT_ENTRY_COUNT * 8 - 1) as u16;
    sregs.idt.base = BOOT_IDT_ADDR;
    sregs.idt.limit = 8 - 1;

    let code = code_segment();
    let data = data_segment();
    sregs.cs = code;
    sregs.ds = data;
    sregs.es = data;
    sregs.fs = data;
    sregs.gs = data;
    sregs.ss = data;

    sregs.cr3 = PML4_ADDR;
    sregs.cr4 |= CR4_PAE;
    sregs.cr0 |= CR0_PE | CR0_PG;
    sregs.efer |= EFER_LME | EFER_LMA;

    vcpu.set_sregs(&sregs)?;
    Ok(())
}

// --- Local-APIC virtual-wire-mode setup ------------------------------------

/// Offset of the `LVT LINT0` entry within the local-APIC register page.
const APIC_LVT0: usize = 0x350;
/// Offset of the `LVT LINT1` entry within the local-APIC register page.
const APIC_LVT1: usize = 0x360;
/// LVT delivery-mode field — bits 8..11 of an LVT entry.
const APIC_DELIVERY_MODE_MASK: u32 = 0x700;
/// LVT delivery mode: deliver as an NMI.
const APIC_MODE_NMI: u32 = 0x4;
/// LVT delivery mode: deliver an external (8259-PIC) interrupt.
const APIC_MODE_EXTINT: u32 = 0x7;

/// Reads the little-endian 32-bit APIC register at `offset` from `state`.
fn read_lapic_reg(state: &kvm_lapic_state, offset: usize) -> u32 {
    let mut bytes = [0_u8; 4];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = state.regs[offset + i] as u8;
    }
    u32::from_le_bytes(bytes)
}

/// Writes `value` as the little-endian 32-bit APIC register at `offset`.
fn write_lapic_reg(state: &mut kvm_lapic_state, offset: usize, value: u32) {
    for (i, byte) in value.to_le_bytes().into_iter().enumerate() {
        state.regs[offset + i] = byte as i8;
    }
}

/// Replaces the delivery-mode field of an LVT entry, leaving every other bit.
fn with_delivery_mode(lvt: u32, mode: u32) -> u32 {
    (lvt & !APIC_DELIVERY_MODE_MASK) | (mode << 8)
}

/// Puts the vCPU's local APIC into "virtual wire mode".
///
/// `LINT0` is set to `ExtINT` so the legacy 8259 PIC — and with it the PIT
/// timer interrupt — reaches the CPU, and `LINT1` to `NMI`. A kernel booted
/// without ACPI or MP tables relies on the BIOS having established this
/// wiring; KVM's reset LAPIC leaves the LVT entries masked, so without this
/// the PIT timer interrupt never arrives and the kernel hangs forever in
/// local-APIC clock calibration.
pub fn configure_lapic(vcpu: &VcpuFd) -> Result<(), Box<dyn std::error::Error>> {
    let mut lapic = vcpu.get_lapic()?;
    let lvt0 = read_lapic_reg(&lapic, APIC_LVT0);
    write_lapic_reg(
        &mut lapic,
        APIC_LVT0,
        with_delivery_mode(lvt0, APIC_MODE_EXTINT),
    );
    let lvt1 = read_lapic_reg(&lapic, APIC_LVT1);
    write_lapic_reg(
        &mut lapic,
        APIC_LVT1,
        with_delivery_mode(lvt1, APIC_MODE_NMI),
    );
    vcpu.set_lapic(&lapic)?;
    Ok(())
}

/// Formal proofs checked by the Kani model checker (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::{
        gdt_entry, kernel_cmdline, kvm_lapic_state, pd_entry, read_lapic_reg, with_delivery_mode,
        write_lapic_reg, APIC_DELIVERY_MODE_MASK, APIC_LVT0, APIC_MODE_EXTINT,
        MAX_GUEST_CMDLINE_LEN, PD_ENTRY_COUNT, PD_HUGE_PAGE_FLAGS,
    };

    /// Proof: every page-directory entry identity-maps its 2 MiB frame.
    #[kani::proof]
    fn proof_pd_entry_is_identity_mapped() {
        let index: u64 = kani::any();
        kani::assume(index < PD_ENTRY_COUNT);
        assert!(pd_entry(index) >> 21 == index);
    }

    /// Proof: the frame number never bleeds into the low 21 flag/offset bits.
    #[kani::proof]
    fn proof_pd_entry_keeps_flags() {
        let index: u64 = kani::any();
        kani::assume(index < PD_ENTRY_COUNT);
        assert!(pd_entry(index) & 0x001f_ffff == PD_HUGE_PAGE_FLAGS);
    }

    /// Proof: a GDT descriptor for a full 4 KiB-granular segment encodes the
    /// complete 20-bit limit, whatever the flags and base.
    #[kani::proof]
    fn proof_gdt_entry_encodes_full_limit() {
        let flags: u16 = kani::any();
        let base: u32 = kani::any();
        let entry = gdt_entry(flags, base, 0x000f_ffff);
        assert!(entry & 0x0000_ffff == 0x0000_ffff);
        assert!((entry >> 48) & 0xf == 0xf);
    }

    /// Proof: the local-APIC register codec round-trips — any `u32` written
    /// into the register page reads back unchanged. A wrong codec would
    /// silently corrupt the `LVT` entries set up for virtual wire mode.
    #[kani::proof]
    fn proof_lapic_register_codec_round_trips() {
        let value: u32 = kani::any();
        let mut state = kvm_lapic_state { regs: [0_i8; 1024] };
        write_lapic_reg(&mut state, APIC_LVT0, value);
        assert!(read_lapic_reg(&state, APIC_LVT0) == value);
    }

    /// Proof: `with_delivery_mode` writes the requested 3-bit delivery mode
    /// and preserves every other bit of the LVT entry.
    #[kani::proof]
    fn proof_with_delivery_mode_sets_only_the_mode() {
        let lvt: u32 = kani::any();
        let mode: u32 = kani::any();
        kani::assume(mode <= APIC_MODE_EXTINT);
        let result = with_delivery_mode(lvt, mode);
        assert!((result >> 8) & 0x7 == mode);
        assert!(result & !APIC_DELIVERY_MODE_MASK == lvt & !APIC_DELIVERY_MODE_MASK);
    }

    /// Proof: both kernel command lines — diskless and with-disk — fit the
    /// guest's boot command-line buffer, whatever the disk state. A line over
    /// `MAX_GUEST_CMDLINE_LEN` would be silently truncated by the kernel,
    /// dropping the trailing `root=` directive and breaking the boot.
    #[kani::proof]
    fn proof_kernel_cmdline_fits_guest_buffer() {
        let has_disk: bool = kani::any();
        assert!(kernel_cmdline(has_disk).len() < MAX_GUEST_CMDLINE_LEN);
    }
}

/// Tests: pure bit-packing, the zero-page builder, and a real long-mode boot
/// of a one-instruction program through `vcpu.run()`.
#[cfg(test)]
mod tests {
    use super::{
        gdt_entry, kernel_cmdline, load_kernel, pd_entry, setup_boot_memory, setup_vcpu_registers,
        setup_zero_page, CMDLINE_ADDR, E820_RAM, EBDA_START, GDT_FLAGS_CODE, HIGH_MEMORY_START,
        MAX_GUEST_CMDLINE_LEN, PD_HUGE_PAGE_FLAGS, ZERO_PAGE_ADDR,
    };
    use linux_loader::loader::bootparam::boot_params;
    use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

    #[test]
    fn pd_entry_identity_maps_frames() {
        assert_eq!(pd_entry(0) >> 21, 0);
        assert_eq!(pd_entry(1) >> 21, 1);
        assert_eq!(pd_entry(511) >> 21, 511);
    }

    #[test]
    fn pd_entry_keeps_flag_bits() {
        for index in [0_u64, 1, 200, 511] {
            assert_eq!(pd_entry(index) & 0x001f_ffff, PD_HUGE_PAGE_FLAGS);
        }
    }

    #[test]
    fn gdt_entry_encodes_limit() {
        let code = gdt_entry(GDT_FLAGS_CODE, 0, 0x000f_ffff);
        assert_eq!(code & 0x0000_ffff, 0x0000_ffff);
        assert_eq!((code >> 48) & 0xf, 0xf);
    }

    #[test]
    fn load_kernel_reports_missing_file() {
        let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x20_0000)])
            .expect("guest memory");
        assert!(load_kernel(&mem, "/nonexistent/ai-vmm/kernel-image").is_err());
    }

    #[test]
    fn kernel_cmdline_carries_root_directive_only_with_a_disk() {
        // A disk-backed VM mounts /dev/vda; a diskless VM must not carry a
        // `root=` directive, or the kernel panics on an absent root device.
        assert!(kernel_cmdline(true).contains("root=/dev/vda"));
        assert!(!kernel_cmdline(false).contains("root="));
        // Both lines fit the guest's boot command-line buffer.
        assert!(kernel_cmdline(true).len() < MAX_GUEST_CMDLINE_LEN);
        assert!(kernel_cmdline(false).len() < MAX_GUEST_CMDLINE_LEN);
    }

    #[test]
    fn setup_zero_page_builds_e820_and_cmdline() {
        let mem_size = 0x0400_0000_usize; // 64 MiB
        let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), mem_size)])
            .expect("guest memory");
        setup_zero_page(&mem, mem_size, true).expect("setup_zero_page");

        let params: boot_params = mem
            .read_obj(GuestAddress(ZERO_PAGE_ADDR))
            .expect("read boot_params");

        // Copy packed fields into aligned locals before asserting on them.
        let entries = params.e820_entries;
        let cmd_ptr = params.hdr.cmd_line_ptr;
        let low = params.e820_table[0];
        let high = params.e820_table[1];
        let (low_type, low_addr, low_size) = (low.type_, low.addr, low.size);
        let (high_type, high_addr, high_size) = (high.type_, high.addr, high.size);

        // The guest kernel rejects an e820 map of fewer than two entries, so
        // the map is split around the legacy sub-1 MiB hole.
        assert_eq!(entries, 2);
        assert_eq!(u64::from(cmd_ptr), CMDLINE_ADDR);
        assert_eq!(low_type, E820_RAM);
        assert_eq!(low_addr, 0);
        assert_eq!(low_size, EBDA_START);
        assert_eq!(high_type, E820_RAM);
        assert_eq!(high_addr, HIGH_MEMORY_START);
        assert_eq!(high_size, mem_size as u64 - HIGH_MEMORY_START);
    }

    /// Best-effort integration test: needs `/dev/kvm`. Writes a single `hlt`
    /// instruction into guest memory, applies the full long-mode setup, and
    /// runs the vCPU. A correct setup reaches `VcpuExit::Hlt`; a wrong
    /// descriptor or page-table bit would triple-fault instead.
    #[test]
    fn long_mode_setup_does_not_triple_fault() {
        use kvm_bindings::kvm_userspace_memory_region;
        use kvm_ioctls::{Kvm, VcpuExit};
        use vm_memory::{Bytes, GuestMemory};

        let kvm = match Kvm::new() {
            Ok(kvm) => kvm,
            Err(e) => {
                eprintln!("[boot-smoke] /dev/kvm unavailable, test inconclusive: {e}");
                return;
            }
        };
        let vm = kvm.create_vm().expect("create_vm");

        let mem_size = 2 * 1024 * 1024_usize;
        let guest_memory = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), mem_size)])
            .expect("guest memory");
        let host_addr = guest_memory
            .get_host_address(GuestAddress(0))
            .expect("host address");
        let region = kvm_userspace_memory_region {
            slot: 0,
            guest_phys_addr: 0,
            memory_size: mem_size as u64,
            userspace_addr: host_addr as u64,
            flags: 0,
        };
        // SAFETY: `host_addr` / `mem_size` describe the live `guest_memory`
        // mmap, which outlives the VM for the whole test.
        unsafe {
            vm.set_user_memory_region(region)
                .expect("set_user_memory_region");
        }

        const CODE_ADDR: u64 = 0x10_0000;
        guest_memory
            .write_obj(0xf4_u8, GuestAddress(CODE_ADDR))
            .expect("write hlt");

        setup_boot_memory(&guest_memory).expect("setup_boot_memory");

        let vcpu = vm.create_vcpu(0).expect("create_vcpu");
        setup_vcpu_registers(&vcpu, CODE_ADDR).expect("setup_vcpu_registers");

        match vcpu.run().expect("vcpu run") {
            VcpuExit::Hlt => {
                eprintln!("[boot-smoke] vCPU reached HLT in 64-bit long mode — no triple fault");
            }
            other => panic!("expected VcpuExit::Hlt, got {other:?} — long-mode setup is wrong"),
        }
    }
}
