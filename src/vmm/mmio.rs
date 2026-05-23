//! Minimal MMIO device bus.
//!
//! KVM reports a guest access to an address that is not backed by a memory
//! region as a `VcpuExit::MmioRead` / `VcpuExit::MmioWrite`. The VMM run loop
//! forwards those here, and this bus routes each access to the device
//! registered for the matching guest-physical range.
//!
//! Devices implement the local [`MmioDevice`] trait. A local trait — rather
//! than `vm-device`'s `MutDeviceMmio` — keeps the bus free of the rust-vmm
//! `virtio` crate dependency soup, fully under test, and lets the routing
//! arithmetic be proven by the Kani harnesses at the bottom of this file.
//!
//! `mmio_write` receives a reference to guest memory: a device such as
//! virtio-blk needs it to walk the virtqueue structures the guest set up when
//! it is notified through a register write.

use vm_memory::GuestMemoryMmap;

/// A device reachable over the MMIO bus.
///
/// `offset` is relative to the device's base address: the bus has already
/// subtracted the base, and `proof_contained_addr_has_valid_offset` guarantees
/// the offset is strictly less than the device's registered length.
///
/// `MmioDevice` requires `Send`: once a VM runs more than one vCPU the bus is
/// shared across the vCPU threads as an `Arc<Mutex<MmioBus>>`, so every device
/// on it must be safe to reach from whichever thread holds the lock.
pub trait MmioDevice: Send {
    /// Handles a guest read of `data.len()` bytes at `offset` within the device.
    fn mmio_read(&mut self, offset: u64, data: &mut [u8]);
    /// Handles a guest write of `data` at `offset` within the device.
    ///
    /// `guest_memory` lets a device follow guest-physical pointers (e.g. a
    /// virtqueue) when a register write asks it to.
    fn mmio_write(&mut self, offset: u64, data: &[u8], guest_memory: &GuestMemoryMmap);
}

/// One device occupying the guest-physical range `[base, base + len)`.
struct MmioRegion {
    base: u64,
    len: u64,
    device: Box<dyn MmioDevice>,
}

/// Routes guest MMIO accesses to the device registered for each address range.
#[derive(Default)]
pub struct MmioBus {
    regions: Vec<MmioRegion>,
}

/// Returns `true` iff the range `[base, base + len)` contains `addr`.
///
/// `addr - base` is only evaluated once `addr >= base` (short-circuit), so it
/// can never underflow — a fact the Kani harnesses verify.
const fn region_contains(base: u64, len: u64, addr: u64) -> bool {
    addr >= base && addr - base < len
}

impl MmioBus {
    /// Creates an empty bus.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `device` for the guest-physical range `[base, base + len)`.
    pub fn register(&mut self, base: u64, len: u64, device: Box<dyn MmioDevice>) {
        self.regions.push(MmioRegion { base, len, device });
    }

    /// Routes a guest MMIO read to its device. Returns `false` if no device
    /// owns `addr` (the caller then supplies zeros to the guest).
    pub fn read(&mut self, addr: u64, data: &mut [u8]) -> bool {
        for region in &mut self.regions {
            if region_contains(region.base, region.len, addr) {
                region.device.mmio_read(addr - region.base, data);
                return true;
            }
        }
        false
    }

    /// Routes a guest MMIO write to its device. Returns `false` if no device
    /// owns `addr`.
    pub fn write(&mut self, addr: u64, data: &[u8], guest_memory: &GuestMemoryMmap) -> bool {
        for region in &mut self.regions {
            if region_contains(region.base, region.len, addr) {
                region
                    .device
                    .mmio_write(addr - region.base, data, guest_memory);
                return true;
            }
        }
        false
    }
}

/// Formal proofs checked by the Kani model checker (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::region_contains;

    /// Proof: a contained address yields a valid in-range offset — `addr - base`
    /// never underflows, and the resulting offset is strictly below `len`, so a
    /// device can never receive an out-of-range register offset.
    #[kani::proof]
    fn proof_contained_addr_has_valid_offset() {
        let base: u64 = kani::any();
        let len: u64 = kani::any();
        let addr: u64 = kani::any();
        if region_contains(base, len, addr) {
            assert!(addr >= base);
            let offset = addr - base;
            assert!(offset < len);
        }
    }

    /// Proof: an address below the region base is never contained.
    #[kani::proof]
    fn proof_addr_below_base_is_not_contained() {
        let base: u64 = kani::any();
        let len: u64 = kani::any();
        let addr: u64 = kani::any();
        kani::assume(addr < base);
        assert!(!region_contains(base, len, addr));
    }

    /// Proof: a zero-length region contains no address at all.
    #[kani::proof]
    fn proof_empty_region_contains_nothing() {
        let base: u64 = kani::any();
        let addr: u64 = kani::any();
        assert!(!region_contains(base, 0, addr));
    }
}

/// Tests for the address routing.
#[cfg(test)]
mod tests {
    use super::{MmioBus, MmioDevice};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use vm_memory::{GuestAddress, GuestMemoryMmap};

    /// Test device: a read echoes the low byte of the offset it was given; a
    /// write records that offset into a shared cell the test can inspect.
    struct OffsetRecorder {
        seen_offset: Arc<AtomicU64>,
    }
    impl MmioDevice for OffsetRecorder {
        fn mmio_read(&mut self, offset: u64, data: &mut [u8]) {
            if let Some(first) = data.first_mut() {
                *first = offset as u8;
            }
        }
        fn mmio_write(&mut self, offset: u64, _data: &[u8], _guest_memory: &GuestMemoryMmap) {
            self.seen_offset.store(offset, Ordering::SeqCst);
        }
    }

    /// A small host-side guest memory, enough for the routing tests.
    fn test_memory() -> GuestMemoryMmap {
        GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x1000)]).expect("test memory")
    }

    #[test]
    fn read_routes_to_device_with_base_relative_offset() {
        let mut bus = MmioBus::new();
        bus.register(
            0x1000,
            0x100,
            Box::new(OffsetRecorder {
                seen_offset: Arc::new(AtomicU64::new(0)),
            }),
        );
        let mut data = [0_u8; 1];
        assert!(bus.read(0x1042, &mut data));
        assert_eq!(data[0], 0x42);
    }

    #[test]
    fn write_routes_to_device_with_base_relative_offset() {
        let seen = Arc::new(AtomicU64::new(0));
        let mut bus = MmioBus::new();
        bus.register(
            0x2000,
            0x100,
            Box::new(OffsetRecorder {
                seen_offset: Arc::clone(&seen),
            }),
        );
        assert!(bus.write(0x2055, &[0xff], &test_memory()));
        assert_eq!(seen.load(Ordering::SeqCst), 0x55);
    }

    #[test]
    fn unknown_address_is_not_routed() {
        let mut bus = MmioBus::new();
        bus.register(
            0x1000,
            0x100,
            Box::new(OffsetRecorder {
                seen_offset: Arc::new(AtomicU64::new(0)),
            }),
        );
        let mut data = [0_u8; 4];
        assert!(!bus.read(0x9999, &mut data));
        assert!(!bus.write(0x9999, &[0_u8; 4], &test_memory()));
    }
}
