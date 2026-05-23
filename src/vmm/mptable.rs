//! Intel MultiProcessor (MP) table — guest CPU enumeration for SMP (Linux only).
//!
//! # Why this exists
//!
//! Direct Kernel Boot hands the guest no firmware: no ACPI, no DMI, no BIOS.
//! Without one of those, the Linux kernel has no way to discover that a VM owns
//! more than one CPU — it prints `smpboot: SMP motherboard not detected`,
//! `SMP disabled`, and runs on the bootstrap processor alone, however many
//! vCPUs KVM created and however many host threads drive them.
//!
//! The Intel MultiProcessor Specification (v1.4) table is the lightest cure: a
//! 16-byte "floating pointer" the kernel finds by scanning the BIOS ROM window
//! `0xf0000..0x100000`, plus a configuration table that lists every processor,
//! the ISA bus and the local-interrupt wiring. With it present the kernel
//! enumerates the application processors and brings them up through the
//! INIT-SIPI-SIPI sequence its own trampoline handles; KVM's in-kernel local
//! APIC performs the handshake.
//!
//! # Completeness
//!
//! The table is the full configuration the kernel expects: every processor,
//! the ISA bus, the I/O APIC, the 16 legacy ISA interrupt lines and the
//! local-interrupt wiring. A partial table — processors but no I/O APIC —
//! leaves the kernel in an inconsistent "MP table found, but no I/O APIC"
//! state in which it stalls during local-APIC bring-up; the complete table
//! puts it on the ordinary symmetric-I/O path every multiprocessor PC firmware
//! provides. KVM's in-kernel I/O APIC answers at the architectural address.
//!
//! # Guest-physical layout
//!
//! Both structures live in the BIOS hole `[0x9fc00, 0x100000)` — real guest
//! RAM, but deliberately absent from the e820 map, so the kernel never
//! allocates over them:
//!
//! ```text
//!   0xf0000  MP floating pointer     (16 bytes, signature "_MP_")
//!   0xf0010  MP configuration table  (signature "PCMP": header + entries)
//! ```
//!
//! # Verifiability
//!
//! The guest kernel rejects either structure unless its bytes sum to zero
//! modulo 256. [`checksum`] computes the byte that enforces that, and the Kani
//! harness proves it always does — for any input. A second harness proves the
//! configuration table can never outgrow its 16-bit length field.

use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

// --- Guest-physical addresses ----------------------------------------------

/// Address of the 16-byte MP floating pointer. It sits at the start of the
/// BIOS ROM window the kernel scans unconditionally for the `_MP_` signature.
const MPTABLE_FLOATING_POINTER_ADDR: u64 = 0x000f_0000;
/// Address of the MP configuration table, right after the floating pointer.
const MPTABLE_CONFIG_ADDR: u64 = MPTABLE_FLOATING_POINTER_ADDR + MP_FLOATING_POINTER_LEN as u64;

// --- Structure signatures and sizes ----------------------------------------

/// Floating-pointer signature — ASCII `_MP_`.
const MP_FP_SIGNATURE: &[u8; 4] = b"_MP_";
/// Configuration-table signature — ASCII `PCMP`.
const MPC_SIGNATURE: &[u8; 4] = b"PCMP";
/// MP specification revision 1.4.
const MP_SPEC_REVISION: u8 = 4;
/// Size of the MP floating pointer, in bytes.
const MP_FLOATING_POINTER_LEN: usize = 16;
/// Size of the MP configuration-table header, in bytes.
const MPC_HEADER_LEN: u32 = 44;
/// Size of one processor entry, in bytes.
const MPC_PROCESSOR_LEN: u32 = 20;
/// Size of one bus entry, in bytes.
const MPC_BUS_LEN: u32 = 8;
/// Size of one I/O APIC entry, in bytes.
const MPC_IO_APIC_LEN: u32 = 8;
/// Size of one I/O interrupt entry, in bytes.
const MPC_IO_INT_LEN: u32 = 8;
/// Size of one local-interrupt entry, in bytes.
const MPC_LOCAL_INT_LEN: u32 = 8;
/// Number of legacy ISA interrupt lines routed through the I/O APIC.
const ISA_IRQ_COUNT: u32 = 16;
/// Number of local-interrupt entries written: ExtINT to LINT0, NMI to LINT1.
const LOCAL_INT_COUNT: u32 = 2;

// --- Configuration-table entry type codes ----------------------------------

/// Entry type: processor.
const MPC_ENTRY_PROCESSOR: u8 = 0;
/// Entry type: bus.
const MPC_ENTRY_BUS: u8 = 1;
/// Entry type: I/O APIC.
const MPC_ENTRY_IO_APIC: u8 = 2;
/// Entry type: I/O interrupt assignment.
const MPC_ENTRY_IO_INTERRUPT: u8 = 3;
/// Entry type: local interrupt assignment.
const MPC_ENTRY_LOCAL_INTERRUPT: u8 = 4;

// --- Field values ----------------------------------------------------------

/// Local-APIC register base, the architectural default.
const LAPIC_ADDR: u32 = 0xfee0_0000;
/// Local-APIC version reported for every processor entry (integrated APIC).
const LAPIC_VERSION: u8 = 0x14;
/// I/O APIC register base, the architectural default — where KVM's in-kernel
/// I/O APIC answers.
const IO_APIC_ADDR: u32 = 0xfec0_0000;
/// I/O APIC version reported in its entry (82093AA-class integrated I/O APIC).
const IO_APIC_VERSION: u8 = 0x11;
/// I/O APIC entry flag: this I/O APIC is enabled.
const IO_APIC_FLAG_ENABLED: u8 = 1;
/// Processor-entry CPU-flags bit: this processor is enabled.
const PROCESSOR_FLAG_ENABLED: u8 = 1 << 0;
/// Processor-entry CPU-flags bit: this processor is the bootstrap processor.
const PROCESSOR_FLAG_BSP: u8 = 1 << 1;
/// Interrupt entry interrupt type: a vectored interrupt (INT).
const MP_INT_TYPE_INT: u8 = 0;
/// Interrupt entry interrupt type: NMI.
const MP_INT_TYPE_NMI: u8 = 1;
/// Interrupt entry interrupt type: ExtINT (legacy 8259 PIC).
const MP_INT_TYPE_EXTINT: u8 = 3;
/// Local-interrupt destination: every local APIC.
const LOCAL_INT_DEST_ALL: u8 = 0xff;
/// OEM identifier — exactly 8 bytes, space-padded.
const OEM_ID: &[u8; 8] = b"AIVMM   ";
/// Product identifier — exactly 12 bytes, space-padded.
const PRODUCT_ID: &[u8; 12] = b"Hypervisor  ";
/// ISA bus type string — exactly 6 bytes, space-padded.
const BUS_TYPE_ISA: &[u8; 6] = b"ISA   ";

/// Returns the byte that drives `bytes` plus that byte to zero modulo 256.
///
/// Appending this value makes the whole structure's 8-bit sum zero — the exact
/// integrity condition the guest kernel checks before it will trust an MP
/// floating pointer or configuration table. Allocation-free and `format!`-free
/// so the Kani harness can prove the property for any input.
fn checksum(bytes: &[u8]) -> u8 {
    let sum = bytes.iter().fold(0_u8, |acc, &b| acc.wrapping_add(b));
    sum.wrapping_neg()
}

/// Total size of the configuration table — header plus every entry — for a VM
/// of `num_cpus` processors.
///
/// This is the value written into the 16-bit `base_table_length` header field.
/// `proof_config_table_length_fits_u16` proves it never overflows that field
/// for any allowed vCPU count.
const fn config_table_length(num_cpus: u32) -> u32 {
    MPC_HEADER_LEN
        + num_cpus * MPC_PROCESSOR_LEN
        + MPC_BUS_LEN
        + MPC_IO_APIC_LEN
        + ISA_IRQ_COUNT * MPC_IO_INT_LEN
        + LOCAL_INT_COUNT * MPC_LOCAL_INT_LEN
}

/// Builds the CPU-flags byte of a processor entry: always enabled, and marked
/// as the bootstrap processor for vCPU 0.
const fn processor_flags(is_bsp: bool) -> u8 {
    if is_bsp {
        PROCESSOR_FLAG_ENABLED | PROCESSOR_FLAG_BSP
    } else {
        PROCESSOR_FLAG_ENABLED
    }
}

/// Writes a valid MP floating pointer and configuration table into guest
/// memory, enumerating `num_cpus` processors so the guest kernel can bring up
/// the application processors.
///
/// Called once, before the vCPUs run, for a multi-vCPU VM only.
pub fn setup_mptable(
    guest_memory: &GuestMemoryMmap,
    num_cpus: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // --- Configuration table: a 44-byte header, then the entries ---
    let mut config: Vec<u8> = Vec::with_capacity(config_table_length(num_cpus) as usize);

    // Header. `base_table_length` (offset 4) and `entry_count` (offset 34) are
    // placeholders here; both are patched in once the entries are appended.
    config.extend_from_slice(MPC_SIGNATURE); //               0:  "PCMP"
    config.extend_from_slice(&[0, 0]); //                     4:  base table length
    config.push(MP_SPEC_REVISION); //                         6:  spec revision
    config.push(0); //                                        7:  checksum
    config.extend_from_slice(OEM_ID); //                      8:  OEM id (8 bytes)
    config.extend_from_slice(PRODUCT_ID); //                  16: product id (12 bytes)
    config.extend_from_slice(&0_u32.to_le_bytes()); //        28: OEM table pointer
    config.extend_from_slice(&0_u16.to_le_bytes()); //        32: OEM table size
    config.extend_from_slice(&0_u16.to_le_bytes()); //        34: entry count
    config.extend_from_slice(&LAPIC_ADDR.to_le_bytes()); //   36: local-APIC address
    config.extend_from_slice(&0_u16.to_le_bytes()); //        40: extended table length
    config.push(0); //                                        42: extended table checksum
    config.push(0); //                                        43: reserved

    let mut entry_count: u16 = 0;

    // One processor entry per vCPU. The local-APIC id equals the vCPU index;
    // vCPU 0 is the bootstrap processor.
    for cpu in 0..num_cpus {
        config.push(MPC_ENTRY_PROCESSOR);
        config.push(cpu as u8); //                            local-APIC id
        config.push(LAPIC_VERSION);
        config.push(processor_flags(cpu == 0)); //            enabled (+ BSP)
        config.extend_from_slice(&0_u32.to_le_bytes()); //    CPU signature
        config.extend_from_slice(&0_u32.to_le_bytes()); //    feature flags
        config.extend_from_slice(&[0_u8; 8]); //              reserved
        entry_count += 1;
    }

    // One ISA bus entry — the bus the legacy interrupts below belong to.
    config.push(MPC_ENTRY_BUS);
    config.push(0); //                                        bus id
    config.extend_from_slice(BUS_TYPE_ISA);
    entry_count += 1;

    // One I/O APIC entry. Its id sits just above the processor APIC ids; KVM's
    // in-kernel I/O APIC answers at the architectural 0xfec00000 base.
    let ioapic_id = num_cpus as u8;
    config.push(MPC_ENTRY_IO_APIC);
    config.push(ioapic_id);
    config.push(IO_APIC_VERSION);
    config.push(IO_APIC_FLAG_ENABLED);
    config.extend_from_slice(&IO_APIC_ADDR.to_le_bytes());
    entry_count += 1;

    // One I/O interrupt entry per legacy ISA line: ISA IRQ n is routed,
    // identity-mapped, to I/O APIC input n — the conventional PC wiring, and
    // the routing the virtio-blk irqfd on GSI 5 relies on.
    for irq in 0..ISA_IRQ_COUNT as u8 {
        config.push(MPC_ENTRY_IO_INTERRUPT);
        config.push(MP_INT_TYPE_INT);
        config.extend_from_slice(&0_u16.to_le_bytes()); //    flags (conforming)
        config.push(0); //                                    source bus id
        config.push(irq); //                                  source bus irq
        config.push(ioapic_id); //                            destination I/O APIC id
        config.push(irq); //                                  destination I/O APIC input
        entry_count += 1;
    }

    // Local-interrupt wiring, applied to every local APIC: ExtINT on LINT0 (so
    // the 8259 PIC's interrupts — including the PIT timer — reach a CPU) and
    // NMI on LINT1.
    for (interrupt_type, lapic_lintin) in [(MP_INT_TYPE_EXTINT, 0_u8), (MP_INT_TYPE_NMI, 1_u8)] {
        config.push(MPC_ENTRY_LOCAL_INTERRUPT);
        config.push(interrupt_type);
        config.extend_from_slice(&0_u16.to_le_bytes()); //    flags (conforming)
        config.push(0); //                                    source bus id
        config.push(0); //                                    source bus irq
        config.push(LOCAL_INT_DEST_ALL); //                   destination local-APIC id
        config.push(lapic_lintin); //                         destination LINTIN#
        entry_count += 1;
    }

    // The built length must match the formula the Kani harness reasons about.
    debug_assert_eq!(config.len(), config_table_length(num_cpus) as usize);

    // Patch the header now the entries are known: base table length, entry
    // count, then the checksum that drives the whole table's sum to zero.
    let base_table_length = config.len() as u16;
    config[4..6].copy_from_slice(&base_table_length.to_le_bytes());
    config[34..36].copy_from_slice(&entry_count.to_le_bytes());
    let table_checksum = checksum(&config);
    config[7] = table_checksum;

    // --- Floating pointer: 16 bytes pointing at the configuration table ---
    let mut floating_pointer = [0_u8; MP_FLOATING_POINTER_LEN];
    floating_pointer[0..4].copy_from_slice(MP_FP_SIGNATURE);
    floating_pointer[4..8].copy_from_slice(&(MPTABLE_CONFIG_ADDR as u32).to_le_bytes());
    floating_pointer[8] = 1; //   length, in 16-byte paragraphs
    floating_pointer[9] = MP_SPEC_REVISION;
    // Byte 10 is the checksum; bytes 11..16 are MP feature bytes, all left
    // zero — feature byte 1 == 0 means "the configuration table is present".
    let fp_checksum = checksum(&floating_pointer);
    floating_pointer[10] = fp_checksum;

    // --- Write both structures into the BIOS hole ---
    guest_memory
        .write_slice(
            &floating_pointer,
            GuestAddress(MPTABLE_FLOATING_POINTER_ADDR),
        )
        .map_err(|e| format!("failed to write the MP floating pointer: {e}"))?;
    guest_memory
        .write_slice(&config, GuestAddress(MPTABLE_CONFIG_ADDR))
        .map_err(|e| format!("failed to write the MP configuration table: {e}"))?;

    Ok(())
}

/// Formal proofs checked by the Kani model checker (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::{checksum, config_table_length, MP_FLOATING_POINTER_LEN};
    use crate::vmm::MAX_VCPUS;

    /// Proof: the checksum byte always drives the structure's 8-bit total to
    /// zero — the exact condition the guest kernel checks before it will trust
    /// an MP table. A wrong checksum makes the kernel discard the table and
    /// silently fall back to a single CPU.
    #[kani::proof]
    fn proof_checksum_zeroes_the_total() {
        let bytes: [u8; MP_FLOATING_POINTER_LEN] = kani::any();
        let check = checksum(&bytes);
        let total = bytes.iter().fold(check, |acc, &b| acc.wrapping_add(b));
        assert!(total == 0);
    }

    /// Proof: the configuration-table length fits the 16-bit `base_table_length`
    /// header field for every vCPU count the hypervisor allows — so that field
    /// can never silently truncate and desynchronise the kernel's table walk.
    #[kani::proof]
    fn proof_config_table_length_fits_u16() {
        let num_cpus: u32 = kani::any();
        kani::assume(num_cpus >= 1 && num_cpus <= MAX_VCPUS);
        assert!(config_table_length(num_cpus) <= u16::MAX as u32);
    }
}

/// Tests: the pure checksum/flags logic and the full table builder (hermetic —
/// no KVM, no guest kernel, just host-side guest memory).
#[cfg(test)]
mod tests {
    use super::{
        checksum, config_table_length, processor_flags, setup_mptable, MPC_SIGNATURE,
        MPTABLE_CONFIG_ADDR, MPTABLE_FLOATING_POINTER_ADDR, MP_FLOATING_POINTER_LEN,
        MP_FP_SIGNATURE, PROCESSOR_FLAG_BSP, PROCESSOR_FLAG_ENABLED,
    };
    use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

    #[test]
    fn checksum_drives_the_total_to_zero() {
        for bytes in [
            &[0_u8][..],
            &[1, 2, 3, 4, 250, 99][..],
            &[255, 255, 255][..],
        ] {
            let total = bytes
                .iter()
                .fold(checksum(bytes), |acc, &b| acc.wrapping_add(b));
            assert_eq!(total, 0, "checksum must zero the 8-bit sum of {bytes:?}");
        }
    }

    #[test]
    fn processor_flags_mark_enabled_and_the_bsp() {
        assert_eq!(
            processor_flags(true),
            PROCESSOR_FLAG_ENABLED | PROCESSOR_FLAG_BSP
        );
        assert_eq!(processor_flags(false), PROCESSOR_FLAG_ENABLED);
        // Every processor entry is enabled, bootstrap processor or not.
        assert!(processor_flags(true) & PROCESSOR_FLAG_ENABLED != 0);
        assert!(processor_flags(false) & PROCESSOR_FLAG_ENABLED != 0);
    }

    #[test]
    fn config_table_length_matches_the_built_table() {
        // 44-byte header + 4 * 20-byte processors + 8-byte bus + 8-byte I/O
        // APIC + 16 * 8-byte I/O interrupts + 2 * 8-byte local interrupts.
        assert_eq!(config_table_length(4), 44 + 80 + 8 + 8 + 128 + 16);
    }

    #[test]
    fn setup_mptable_writes_a_valid_findable_table() {
        let memory = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x20_0000)])
            .expect("guest memory");
        setup_mptable(&memory, 4).expect("setup_mptable");

        // Floating pointer: the "_MP_" signature, and all 16 bytes sum to zero.
        let mut floating_pointer = [0_u8; MP_FLOATING_POINTER_LEN];
        memory
            .read_slice(
                &mut floating_pointer,
                GuestAddress(MPTABLE_FLOATING_POINTER_ADDR),
            )
            .expect("read floating pointer");
        assert_eq!(&floating_pointer[0..4], MP_FP_SIGNATURE);
        assert_eq!(
            floating_pointer
                .iter()
                .fold(0_u8, |a, &b| a.wrapping_add(b)),
            0,
            "the floating pointer must checksum to zero"
        );

        // Configuration table: the "PCMP" signature, a zero checksum over the
        // whole base table, and 4 processors + 1 bus + 2 local interrupts.
        let base_len: u16 = memory
            .read_obj(GuestAddress(MPTABLE_CONFIG_ADDR + 4))
            .expect("read base table length");
        assert_eq!(u32::from(base_len), config_table_length(4));
        let mut config = vec![0_u8; base_len as usize];
        memory
            .read_slice(&mut config, GuestAddress(MPTABLE_CONFIG_ADDR))
            .expect("read configuration table");
        assert_eq!(&config[0..4], MPC_SIGNATURE);
        assert_eq!(
            config.iter().fold(0_u8, |a, &b| a.wrapping_add(b)),
            0,
            "the configuration table must checksum to zero"
        );
        // 4 processors + 1 bus + 1 I/O APIC + 16 I/O interrupts + 2 local
        // interrupts.
        let entry_count = u16::from_le_bytes([config[34], config[35]]);
        assert_eq!(entry_count, 4 + 1 + 1 + 16 + 2);
    }
}
