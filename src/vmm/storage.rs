//! virtio-mmio block device — transport, identity, and the virtqueue datapath.
//!
//! [`create_block_device`] opens a disk image and returns a [`VirtioBlockMmio`]
//! that sits on the MMIO bus. It presents a valid virtio-mmio block-device
//! identity, exposes the disk capacity, and — once the guest driver has set up
//! a split virtqueue — services block I/O requests through [`process_virtqueue`].
//!
//! # Host memory safety
//!
//! Every access to guest memory goes through `GuestMemoryMmap`'s `read_obj`,
//! `read_slice`, `write_obj` and `write_slice`. Each of those bounds-checks the
//! address against the registered guest memory regions and returns `Err` for an
//! out-of-range address. A buggy or malicious guest driver can therefore make a
//! request *fail*, but it can never make this code read or write host memory
//! outside the guest mapping. All such calls below are `?`-propagated, so a bad
//! address aborts the request instead of corrupting the host.
//!
//! # Interrupt routing
//!
//! `provision_kvm_machine_native` registers the device's [`EventFd`] with KVM as
//! an irqfd on GSI 5. When a request completes, [`process_virtqueue`] writes `1`
//! to that `EventFd`; KVM then injects IRQ 5 through the in-kernel IOAPIC, and
//! the guest's virtio-blk driver runs its completion handler.

use crate::vmm::mmio::MmioDevice;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};
use vmm_sys_util::eventfd::EventFd;

/// virtio block sector size, in bytes.
const SECTOR_SIZE: u64 = 512;
/// Largest single request this MVP will service, as a guard against a buggy or
/// malicious descriptor length.
const MAX_REQUEST_BYTES: usize = 1 << 20;
/// `EFD_NONBLOCK` — the device only ever writes the irqfd, and a non-blocking
/// fd lets a test read it back without risking a hang.
const EFD_NONBLOCK: i32 = 0o4000;

// --- virtio-mmio transport register offsets --------------------------------

const REG_MAGIC: u64 = 0x000;
const REG_VERSION: u64 = 0x004;
const REG_DEVICE_ID: u64 = 0x008;
const REG_VENDOR_ID: u64 = 0x00c;
const REG_DEVICE_FEATURES: u64 = 0x010;
const REG_DEVICE_FEATURES_SEL: u64 = 0x014;
const REG_QUEUE_NUM_MAX: u64 = 0x034;
const REG_QUEUE_NUM: u64 = 0x038;
const REG_QUEUE_READY: u64 = 0x044;
const REG_QUEUE_NOTIFY: u64 = 0x050;
const REG_INTERRUPT_STATUS: u64 = 0x060;
const REG_INTERRUPT_ACK: u64 = 0x064;
const REG_STATUS: u64 = 0x070;
const REG_QUEUE_DESC_LOW: u64 = 0x080;
const REG_QUEUE_DESC_HIGH: u64 = 0x084;
const REG_QUEUE_AVAIL_LOW: u64 = 0x090;
const REG_QUEUE_AVAIL_HIGH: u64 = 0x094;
const REG_QUEUE_USED_LOW: u64 = 0x0a0;
const REG_QUEUE_USED_HIGH: u64 = 0x0a4;
/// Device configuration space: virtio-blk capacity (sectors), low 32-bit word.
const REG_CONFIG_CAPACITY_LOW: u64 = 0x100;
/// High 32-bit word of the virtio-blk capacity.
const REG_CONFIG_CAPACITY_HIGH: u64 = 0x104;

// --- virtio-mmio constant register values ----------------------------------

/// MagicValue: the little-endian ASCII "virt".
const VIRTIO_MMIO_MAGIC: u32 = 0x7472_6976;
/// Modern virtio-mmio transport version.
const VIRTIO_MMIO_VERSION: u32 = 2;
/// virtio device ID for a block device.
const VIRTIO_ID_BLOCK: u32 = 2;
/// Vendor ID reported to the guest (the conventional virtio vendor).
const VIRTIO_VENDOR_ID: u32 = 0x1af4;
/// `VIRTIO_F_VERSION_1` is feature bit 32 — i.e. bit 0 of feature word 1.
const VIRTIO_F_VERSION_1_WORD1: u32 = 1;
/// Largest virtqueue this device supports (a power of two, as virtio requires).
const QUEUE_SIZE_MAX: u16 = 8;

// --- split-virtqueue layout constants --------------------------------------

/// Size of one `virtq_desc` entry, in bytes (`addr` u64, `len` u32, `flags`
/// u16, `next` u16).
const VIRTQ_DESC_SIZE: u64 = 16;
/// Descriptor flag: the chain continues at the `next` index.
const VIRTQ_DESC_F_NEXT: u16 = 1;
/// Descriptor flag: the buffer is written by the device (read requests).
const VIRTQ_DESC_F_WRITE: u16 = 2;

// --- virtio-blk request constants ------------------------------------------

/// virtio-blk request type: read from disk into guest memory.
const VIRTIO_BLK_T_IN: u32 = 0;
/// virtio-blk request type: write guest memory out to disk.
const VIRTIO_BLK_T_OUT: u32 = 1;
/// virtio-blk request type: flush the backing image to stable storage.
const VIRTIO_BLK_T_FLUSH: u32 = 4;
/// virtio-blk request type: read the device's identity string.
const VIRTIO_BLK_T_GET_ID: u32 = 8;
/// virtio-blk request status: success.
const VIRTIO_BLK_S_OK: u8 = 0;
/// virtio-blk request status: a host-side I/O error.
const VIRTIO_BLK_S_IOERR: u8 = 1;
/// virtio-blk request status: the request type is not supported.
const VIRTIO_BLK_S_UNSUPP: u8 = 2;
/// Length of the virtio-blk identity buffer, in bytes.
const VIRTIO_BLK_ID_BYTES: usize = 20;
/// Identity string reported for a `VIRTIO_BLK_T_GET_ID` request.
const DEVICE_ID: &[u8] = b"ai-vmm-virtio-blk";

/// One decoded split-virtqueue descriptor (a host-side copy of the 16-byte
/// `virtq_desc` read out of guest memory).
#[derive(Clone, Copy)]
struct VirtqDesc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

/// A virtio-mmio block device backed by a host disk image.
pub struct VirtioBlockMmio {
    /// Backing disk image, opened read/write — the block device's backend.
    disk: File,
    /// EventFd registered with KVM as this device's irqfd.
    interrupt_evt: EventFd,
    /// Selects which 32-bit word the `DeviceFeatures` register reads back.
    device_features_sel: u32,
    /// virtio device status register.
    status: u32,
    /// virtio interrupt-status register.
    interrupt_status: u32,
    /// Negotiated virtqueue size (entry count).
    queue_size: u16,
    /// `1` once the driver has marked the virtqueue ready.
    queue_ready: u32,
    /// Guest-physical address of the descriptor table.
    queue_desc: u64,
    /// Guest-physical address of the available ring.
    queue_avail: u64,
    /// Guest-physical address of the used ring.
    queue_used: u64,
    /// Next available-ring index this device has yet to consume.
    last_avail_idx: u16,
    /// Next used-ring index this device will write.
    used_idx: u16,
}

/// Returns the slot a ring index maps to, wrapping within `queue_size`.
///
/// Guards `queue_size == 0` so it can never divide by zero; the Kani harness
/// proves the result is always a valid slot.
fn ring_slot(index: u16, queue_size: u16) -> u16 {
    if queue_size == 0 {
        0
    } else {
        index % queue_size
    }
}

/// Returns the guest-physical address of descriptor `index` in a table based at
/// `table`, or `None` if the address arithmetic would overflow.
fn descriptor_addr(table: u64, index: u16) -> Option<u64> {
    table.checked_add(u64::from(index) * VIRTQ_DESC_SIZE)
}

/// Adds `offset` to a guest-controlled virtqueue ring base, returning `None`
/// on overflow instead of wrapping to a bogus low address.
///
/// `queue_avail` and `queue_used` are set verbatim by the guest, so this is
/// the guard that stops a malicious base near `u64::MAX` from panicking the
/// VMM on the ring address arithmetic. `proof_ring_addr_never_wraps` proves
/// the result, when present, never wraps below the base.
fn ring_addr(base: u64, offset: u64) -> Option<u64> {
    base.checked_add(offset)
}

impl VirtioBlockMmio {
    /// Returns the device's interrupt `EventFd`, for `VmFd::register_irqfd`.
    pub fn irq_eventfd(&self) -> &EventFd {
        &self.interrupt_evt
    }

    /// Disk capacity in 512-byte sectors, read live from the backing image.
    fn capacity_sectors(&self) -> u64 {
        self.disk
            .metadata()
            .map(|meta| meta.len() / SECTOR_SIZE)
            .unwrap_or(0)
    }

    /// The `DeviceFeatures` value for the currently selected feature word.
    fn device_features(&self) -> u32 {
        if self.device_features_sel == 1 {
            VIRTIO_F_VERSION_1_WORD1
        } else {
            0
        }
    }

    /// Reads descriptor `index` out of the guest's descriptor table.
    ///
    /// Each field is read with a primitive `read_obj`, which bounds-checks the
    /// address — no `unsafe`, and no possibility of a host out-of-bounds access.
    fn read_descriptor(
        &self,
        guest_memory: &GuestMemoryMmap,
        index: u16,
    ) -> Result<VirtqDesc, Box<dyn std::error::Error>> {
        let base = descriptor_addr(self.queue_desc, index)
            .ok_or("virtqueue descriptor address overflow")?;
        let addr: u64 = guest_memory.read_obj(GuestAddress(base))?;
        let len: u32 = guest_memory.read_obj(GuestAddress(base + 8))?;
        let flags: u16 = guest_memory.read_obj(GuestAddress(base + 12))?;
        let next: u16 = guest_memory.read_obj(GuestAddress(base + 14))?;
        Ok(VirtqDesc {
            addr,
            len,
            flags,
            next,
        })
    }

    /// Services every request the driver has published since the last call.
    ///
    /// Triggered by a write to the `QueueNotify` register. It walks the
    /// available ring from `last_avail_idx` to the driver's current `idx`,
    /// processes each request, and (per request) raises the completion
    /// interrupt.
    fn process_virtqueue(
        &mut self,
        guest_memory: &GuestMemoryMmap,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.queue_ready != 1 || self.queue_size == 0 || self.queue_desc == 0 {
            return Ok(());
        }

        // Available ring: { flags: u16, idx: u16, ring: [u16; queue_size] }.
        // `queue_avail` is guest-controlled, so every ring address goes through
        // the overflow-checked `ring_addr`.
        let avail_idx_addr =
            ring_addr(self.queue_avail, 2).ok_or("virtqueue avail-ring address overflow")?;
        let avail_idx: u16 = guest_memory.read_obj(GuestAddress(avail_idx_addr))?;

        while self.last_avail_idx != avail_idx {
            let slot = ring_slot(self.last_avail_idx, self.queue_size);
            // avail.ring[slot] is at queue_avail + 4 + 2*slot.
            let entry_addr = ring_addr(self.queue_avail, 4 + u64::from(slot) * 2)
                .ok_or("virtqueue avail-ring address overflow")?;
            let head: u16 = guest_memory.read_obj(GuestAddress(entry_addr))?;

            self.process_request(guest_memory, head)?;

            self.last_avail_idx = self.last_avail_idx.wrapping_add(1);
        }
        Ok(())
    }

    /// Processes the single request whose descriptor chain starts at `head`.
    ///
    /// The chain is `[header] [data?] [status]`: the 16-byte header and the
    /// one-byte status descriptor are always present, with a data descriptor
    /// between them for every request type except `FLUSH`. Whatever the
    /// outcome — success, an unknown request type, or a host I/O error — the
    /// request is always completed in the used ring, so the guest's
    /// virtio-blk driver can never block waiting for a reply.
    fn process_request(
        &mut self,
        guest_memory: &GuestMemoryMmap,
        head: u16,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Descriptor 0 — the virtio_blk request header.
        let header = self.read_descriptor(guest_memory, head)?;
        // virtio_blk_outhdr: { type: u32, reserved: u32, sector: u64 }.
        let request_type: u32 = guest_memory.read_obj(GuestAddress(header.addr))?;
        let sector: u64 = guest_memory.read_obj(GuestAddress(header.addr + 8))?;

        // Walk the rest of the chain: descriptors between the header and the
        // final (no-`NEXT`) one are data buffers; the final one is the
        // one-byte status. This MVP services the common layout of at most one
        // data descriptor.
        if header.flags & VIRTQ_DESC_F_NEXT == 0 {
            return Err("virtio-blk request chain has no status descriptor".into());
        }
        let mut data: Option<VirtqDesc> = None;
        let mut current = self.read_descriptor(guest_memory, header.next)?;
        // A well-formed chain visits each descriptor at most once, so it can
        // never be longer than the queue. A longer walk means a cyclic chain:
        // bail out instead of looping forever and hanging the VMM.
        let mut steps = 0_u16;
        while current.flags & VIRTQ_DESC_F_NEXT != 0 {
            steps += 1;
            if steps >= self.queue_size {
                return Err("virtio-blk descriptor chain is cyclic or too long".into());
            }
            let next = current.next;
            data = Some(current);
            current = self.read_descriptor(guest_memory, next)?;
        }
        let status_desc = current;

        // Service the request; every outcome resolves to a status byte.
        let (status, used_len) = self.service_request(guest_memory, request_type, sector, data);

        // Write the status byte into the guest's status descriptor.
        guest_memory.write_obj(status, GuestAddress(status_desc.addr))?;

        // Publish the completion in the used ring:
        // used = { flags: u16, idx: u16, ring: [{ id: u32, len: u32 }; size] }.
        // `queue_used` is guest-controlled — every address goes through
        // the overflow-checked `ring_addr`.
        let used_slot = ring_slot(self.used_idx, self.queue_size);
        let id_offset = 4 + u64::from(used_slot) * 8;
        let id_addr =
            ring_addr(self.queue_used, id_offset).ok_or("virtqueue used-ring address overflow")?;
        let len_addr = ring_addr(self.queue_used, id_offset + 4)
            .ok_or("virtqueue used-ring address overflow")?;
        let used_idx_addr =
            ring_addr(self.queue_used, 2).ok_or("virtqueue used-ring address overflow")?;
        guest_memory.write_obj(u32::from(head), GuestAddress(id_addr))?;
        guest_memory.write_obj(used_len as u32, GuestAddress(len_addr))?;
        self.used_idx = self.used_idx.wrapping_add(1);
        guest_memory.write_obj(self.used_idx, GuestAddress(used_idx_addr))?;

        // Raise the completion interrupt: set the "used buffer" status bit, then
        // signal the irqfd so KVM injects the guest IRQ.
        self.interrupt_status |= 1;
        self.interrupt_evt.write(1)?;

        Ok(())
    }

    /// Services one parsed request, returning `(status_byte, bytes_written)`.
    ///
    /// A read, write, flush, identity query, an unknown request type, or a
    /// host I/O failure all resolve to a virtio-blk status byte here; the
    /// caller then completes the request unconditionally.
    fn service_request(
        &mut self,
        guest_memory: &GuestMemoryMmap,
        request_type: u32,
        sector: u64,
        data: Option<VirtqDesc>,
    ) -> (u8, usize) {
        let file_offset = sector * SECTOR_SIZE;
        match request_type {
            VIRTIO_BLK_T_IN => {
                // Disk -> host buffer -> guest. A read needs a device-writable
                // data buffer; `write_slice` bounds-checks the guest range.
                let Some(data) = data else {
                    return (VIRTIO_BLK_S_IOERR, 0);
                };
                let len = data.len as usize;
                if data.flags & VIRTQ_DESC_F_WRITE == 0 || len > MAX_REQUEST_BYTES {
                    return (VIRTIO_BLK_S_IOERR, 0);
                }
                let mut buffer = vec![0_u8; len];
                if self.disk.read_exact_at(&mut buffer, file_offset).is_err() {
                    return (VIRTIO_BLK_S_IOERR, 0);
                }
                match guest_memory.write_slice(&buffer, GuestAddress(data.addr)) {
                    Ok(()) => (VIRTIO_BLK_S_OK, len),
                    Err(_) => (VIRTIO_BLK_S_IOERR, 0),
                }
            }
            VIRTIO_BLK_T_OUT => {
                // Guest -> host buffer (bounds-checked read) -> disk.
                let Some(data) = data else {
                    return (VIRTIO_BLK_S_IOERR, 0);
                };
                let len = data.len as usize;
                if len > MAX_REQUEST_BYTES {
                    return (VIRTIO_BLK_S_IOERR, 0);
                }
                let mut buffer = vec![0_u8; len];
                if guest_memory
                    .read_slice(&mut buffer, GuestAddress(data.addr))
                    .is_err()
                {
                    return (VIRTIO_BLK_S_IOERR, 0);
                }
                match self.disk.write_all_at(&buffer, file_offset) {
                    Ok(()) => (VIRTIO_BLK_S_OK, 0),
                    Err(_) => (VIRTIO_BLK_S_IOERR, 0),
                }
            }
            VIRTIO_BLK_T_FLUSH => match self.disk.sync_all() {
                Ok(()) => (VIRTIO_BLK_S_OK, 0),
                Err(_) => (VIRTIO_BLK_S_IOERR, 0),
            },
            VIRTIO_BLK_T_GET_ID => {
                // Write the device identity string into the guest buffer.
                let Some(data) = data else {
                    return (VIRTIO_BLK_S_IOERR, 0);
                };
                let len = DEVICE_ID
                    .len()
                    .min(data.len as usize)
                    .min(VIRTIO_BLK_ID_BYTES);
                match guest_memory.write_slice(&DEVICE_ID[..len], GuestAddress(data.addr)) {
                    Ok(()) => (VIRTIO_BLK_S_OK, len),
                    Err(_) => (VIRTIO_BLK_S_IOERR, 0),
                }
            }
            // An unknown request type is reported, never left dangling.
            _ => (VIRTIO_BLK_S_UNSUPP, 0),
        }
    }
}

impl MmioDevice for VirtioBlockMmio {
    fn mmio_read(&mut self, offset: u64, data: &mut [u8]) {
        let value: u32 = match offset {
            REG_MAGIC => VIRTIO_MMIO_MAGIC,
            REG_VERSION => VIRTIO_MMIO_VERSION,
            REG_DEVICE_ID => VIRTIO_ID_BLOCK,
            REG_VENDOR_ID => VIRTIO_VENDOR_ID,
            REG_DEVICE_FEATURES => self.device_features(),
            REG_QUEUE_NUM_MAX => u32::from(QUEUE_SIZE_MAX),
            REG_QUEUE_READY => self.queue_ready,
            REG_INTERRUPT_STATUS => self.interrupt_status,
            REG_STATUS => self.status,
            REG_CONFIG_CAPACITY_LOW => self.capacity_sectors() as u32,
            REG_CONFIG_CAPACITY_HIGH => (self.capacity_sectors() >> 32) as u32,
            _ => 0,
        };
        write_u32_le(data, value);
    }

    fn mmio_write(&mut self, offset: u64, data: &[u8], guest_memory: &GuestMemoryMmap) {
        let value = read_u32_le(data);
        match offset {
            REG_DEVICE_FEATURES_SEL => self.device_features_sel = value,
            REG_STATUS => self.status = value,
            REG_INTERRUPT_ACK => self.interrupt_status &= !value,
            REG_QUEUE_NUM => self.queue_size = value as u16,
            REG_QUEUE_READY => self.queue_ready = value,
            REG_QUEUE_DESC_LOW => write_low(&mut self.queue_desc, value),
            REG_QUEUE_DESC_HIGH => write_high(&mut self.queue_desc, value),
            REG_QUEUE_AVAIL_LOW => write_low(&mut self.queue_avail, value),
            REG_QUEUE_AVAIL_HIGH => write_high(&mut self.queue_avail, value),
            REG_QUEUE_USED_LOW => write_low(&mut self.queue_used, value),
            REG_QUEUE_USED_HIGH => write_high(&mut self.queue_used, value),
            REG_QUEUE_NOTIFY => {
                if let Err(e) = self.process_virtqueue(guest_memory) {
                    eprintln!("[vmm] virtio-blk virtqueue processing error: {e}");
                }
            }
            _ => {}
        }
    }
}

/// Opens `disk_image_path` read/write and builds a virtio-mmio block device
/// backed by it.
pub fn create_block_device(
    disk_image_path: &str,
) -> Result<VirtioBlockMmio, Box<dyn std::error::Error>> {
    let disk = OpenOptions::new()
        .read(true)
        .write(true)
        .open(disk_image_path)
        .map_err(|e| format!("cannot open disk image '{disk_image_path}': {e}"))?;
    let interrupt_evt = EventFd::new(EFD_NONBLOCK)
        .map_err(|e| format!("cannot create the virtio-blk interrupt EventFd: {e}"))?;
    Ok(VirtioBlockMmio {
        disk,
        interrupt_evt,
        device_features_sel: 0,
        status: 0,
        interrupt_status: 0,
        queue_size: 0,
        queue_ready: 0,
        queue_desc: 0,
        queue_avail: 0,
        queue_used: 0,
        last_avail_idx: 0,
        used_idx: 0,
    })
}

/// Replaces the low 32 bits of `target` with `value`.
fn write_low(target: &mut u64, value: u32) {
    *target = (*target & 0xffff_ffff_0000_0000) | u64::from(value);
}

/// Replaces the high 32 bits of `target` with `value`.
fn write_high(target: &mut u64, value: u32) {
    *target = (*target & 0x0000_0000_ffff_ffff) | (u64::from(value) << 32);
}

/// Writes the low `min(4, data.len())` bytes of `value` into `data`,
/// little-endian — the byte order of every virtio-mmio register.
fn write_u32_le(data: &mut [u8], value: u32) {
    let bytes = value.to_le_bytes();
    for (slot, byte) in data.iter_mut().zip(bytes.iter()) {
        *slot = *byte;
    }
}

/// Reads a little-endian `u32` from the first `min(4, data.len())` bytes of
/// `data`; missing bytes are treated as zero.
fn read_u32_le(data: &[u8]) -> u32 {
    let mut buf = [0_u8; 4];
    for (slot, byte) in buf.iter_mut().zip(data.iter()) {
        *slot = *byte;
    }
    u32::from_le_bytes(buf)
}

/// Formal proofs checked by the Kani model checker (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::{descriptor_addr, read_u32_le, ring_addr, ring_slot, write_u32_le};

    /// Proof: the register codec round-trips — any `u32` written into a 4-byte
    /// MMIO buffer reads back unchanged.
    #[kani::proof]
    fn proof_u32_le_round_trips() {
        let value: u32 = kani::any();
        let mut buf = [0_u8; 4];
        write_u32_le(&mut buf, value);
        assert!(read_u32_le(&buf) == value);
    }

    /// Proof: `ring_slot` never divides by zero and always yields a slot inside
    /// the ring — a wrong wrap would index outside the available/used ring.
    #[kani::proof]
    fn proof_ring_slot_stays_in_bounds() {
        let index: u16 = kani::any();
        let queue_size: u16 = kani::any();
        let slot = ring_slot(index, queue_size);
        if queue_size == 0 {
            assert!(slot == 0);
        } else {
            assert!(slot < queue_size);
        }
    }

    /// Proof: a descriptor address never wraps below its table base — the
    /// checked arithmetic returns `None` instead of aliasing a low address.
    #[kani::proof]
    fn proof_descriptor_addr_never_wraps_below_table() {
        let table: u64 = kani::any();
        let index: u16 = kani::any();
        if let Some(addr) = descriptor_addr(table, index) {
            assert!(addr >= table);
        }
    }

    /// Proof: a virtqueue ring address never wraps below its base — a
    /// guest-controlled `queue_avail` / `queue_used` can never be turned into a
    /// bogus low address that would slip past the guest-memory bounds check.
    #[kani::proof]
    fn proof_ring_addr_never_wraps() {
        let base: u64 = kani::any();
        let offset: u64 = kani::any();
        if let Some(addr) = ring_addr(base, offset) {
            assert!(addr >= base);
            assert!(addr >= offset);
        }
    }
}

/// Tests for the register codec, the device identity, and the full virtqueue
/// datapath (exercised entirely host-side, without KVM or a guest kernel).
#[cfg(test)]
mod tests {
    use super::{
        create_block_device, read_u32_le, write_u32_le, VIRTIO_ID_BLOCK, VIRTIO_MMIO_MAGIC,
        VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_WRITE,
    };
    use crate::vmm::mmio::MmioDevice;
    use std::io::Write;
    use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

    #[test]
    fn u32_le_codec_round_trips() {
        let mut buf = [0_u8; 4];
        write_u32_le(&mut buf, 0x1234_5678);
        assert_eq!(read_u32_le(&buf), 0x1234_5678);
    }

    #[test]
    fn block_device_presents_virtio_identity_and_capacity() {
        let path = std::env::temp_dir().join("ai-vmm-storage-id-test.img");
        {
            let mut file = std::fs::File::create(&path).expect("create temp disk");
            file.write_all(&[0_u8; 4096]).expect("size temp disk");
        }
        let mut device =
            create_block_device(path.to_str().expect("utf-8 path")).expect("create_block_device");

        let mut reg = [0_u8; 4];
        device.mmio_read(0x000, &mut reg);
        assert_eq!(read_u32_le(&reg), VIRTIO_MMIO_MAGIC);
        device.mmio_read(0x008, &mut reg);
        assert_eq!(read_u32_le(&reg), VIRTIO_ID_BLOCK);
        device.mmio_read(0x100, &mut reg);
        assert_eq!(read_u32_le(&reg), 8); // 4096 bytes / 512 = 8 sectors

        let _ = std::fs::remove_file(&path);
    }

    /// Writes one split-virtqueue descriptor into guest memory.
    fn write_descriptor(
        memory: &GuestMemoryMmap,
        table: u64,
        index: u64,
        addr: u64,
        len: u32,
        flags: u16,
        next: u16,
    ) {
        let base = table + index * 16;
        memory
            .write_obj(addr, GuestAddress(base))
            .expect("desc addr");
        memory
            .write_obj(len, GuestAddress(base + 8))
            .expect("desc len");
        memory
            .write_obj(flags, GuestAddress(base + 12))
            .expect("desc flags");
        memory
            .write_obj(next, GuestAddress(base + 14))
            .expect("desc next");
    }

    // Fixed guest-physical layout shared by the datapath tests.
    const DESC_TABLE: u64 = 0x1000;
    const AVAIL_RING: u64 = 0x2000;
    const USED_RING: u64 = 0x3000;
    const HEADER_ADDR: u64 = 0x4000;
    const DATA_ADDR: u64 = 0x5000;
    const STATUS_ADDR: u64 = 0x6000;

    /// Builds a `VirtioBlockMmio` over `disk_path` with the virtqueue registers
    /// pointed at the fixed layout above.
    fn device_with_queue(disk_path: &str) -> super::VirtioBlockMmio {
        let mut device = create_block_device(disk_path).expect("create_block_device");
        device.queue_desc = DESC_TABLE;
        device.queue_avail = AVAIL_RING;
        device.queue_used = USED_RING;
        device.queue_size = 8;
        device.queue_ready = 1;
        device
    }

    /// Lays out a one-request, three-descriptor chain (header, data, status)
    /// and publishes it in the available ring.
    fn publish_request(
        memory: &GuestMemoryMmap,
        request_type: u32,
        sector: u64,
        data_len: u32,
        data_writable: bool,
    ) {
        // virtio_blk_outhdr: { type: u32, reserved: u32, sector: u64 }.
        memory
            .write_obj(request_type, GuestAddress(HEADER_ADDR))
            .expect("header type");
        memory
            .write_obj(sector, GuestAddress(HEADER_ADDR + 8))
            .expect("header sector");

        let data_flags = if data_writable {
            VIRTQ_DESC_F_NEXT | VIRTQ_DESC_F_WRITE
        } else {
            VIRTQ_DESC_F_NEXT
        };
        write_descriptor(memory, DESC_TABLE, 0, HEADER_ADDR, 16, VIRTQ_DESC_F_NEXT, 1);
        write_descriptor(memory, DESC_TABLE, 1, DATA_ADDR, data_len, data_flags, 2);
        write_descriptor(memory, DESC_TABLE, 2, STATUS_ADDR, 1, VIRTQ_DESC_F_WRITE, 0);

        // Available ring: { flags: u16, idx: u16, ring: [u16; size] }.
        memory
            .write_obj(1_u16, GuestAddress(AVAIL_RING + 2))
            .expect("avail idx");
        memory
            .write_obj(0_u16, GuestAddress(AVAIL_RING + 4))
            .expect("avail ring[0]");
    }

    #[test]
    fn cyclic_descriptor_chain_is_rejected_not_hung() {
        let memory =
            GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x10_0000)]).expect("memory");
        let path = std::env::temp_dir().join("ai-vmm-storage-cyclic.img");
        std::fs::write(&path, [0_u8; 512]).expect("write temp disk");
        let mut device = device_with_queue(path.to_str().expect("utf-8 path"));

        // Header -> descriptor 1, whose `next` points back at itself: a cycle.
        write_descriptor(
            &memory,
            DESC_TABLE,
            0,
            HEADER_ADDR,
            16,
            VIRTQ_DESC_F_NEXT,
            1,
        );
        write_descriptor(&memory, DESC_TABLE, 1, DATA_ADDR, 512, VIRTQ_DESC_F_NEXT, 1);
        memory
            .write_obj(1_u16, GuestAddress(AVAIL_RING + 2))
            .expect("avail idx");
        memory
            .write_obj(0_u16, GuestAddress(AVAIL_RING + 4))
            .expect("avail ring[0]");

        // The bounded chain walk must return an error promptly, never loop.
        assert!(device.process_virtqueue(&memory).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn overflowing_queue_address_is_rejected_not_panicking() {
        let memory =
            GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x10_0000)]).expect("memory");
        let path = std::env::temp_dir().join("ai-vmm-storage-overflow.img");
        std::fs::write(&path, [0_u8; 512]).expect("write temp disk");
        let mut device = device_with_queue(path.to_str().expect("utf-8 path"));

        // A guest-controlled avail-ring base near u64::MAX must resolve to a
        // clean error, not overflow the VMM's address arithmetic into a panic.
        device.queue_avail = u64::MAX;
        assert!(device.process_virtqueue(&memory).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn datapath_reads_a_block_into_guest_memory() {
        let memory =
            GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x10_0000)]).expect("memory");
        let path = std::env::temp_dir().join("ai-vmm-datapath-read.img");
        std::fs::write(&path, [0xab_u8; 512]).expect("write temp disk");

        let mut device = device_with_queue(path.to_str().expect("utf-8 path"));
        publish_request(&memory, 0 /* VIRTIO_BLK_T_IN */, 0, 512, true);

        device
            .process_virtqueue(&memory)
            .expect("process_virtqueue must succeed");

        // The disk block was copied into the guest data buffer.
        let mut buffer = [0_u8; 512];
        memory
            .read_slice(&mut buffer, GuestAddress(DATA_ADDR))
            .expect("read guest data");
        assert!(buffer.iter().all(|&byte| byte == 0xab));
        // Status is VIRTIO_BLK_S_OK, the used ring advanced, the IRQ fired.
        let status: u8 = memory.read_obj(GuestAddress(STATUS_ADDR)).expect("status");
        assert_eq!(status, 0);
        let used_idx: u16 = memory
            .read_obj(GuestAddress(USED_RING + 2))
            .expect("used idx");
        assert_eq!(used_idx, 1);
        let used_id: u32 = memory
            .read_obj(GuestAddress(USED_RING + 4))
            .expect("used id");
        assert_eq!(used_id, 0);
        assert_eq!(device.irq_eventfd().read().expect("irqfd"), 1);

        let _ = std::fs::remove_file(&path);
        eprintln!("[datapath] virtio-blk read request serviced end to end");
    }

    #[test]
    fn datapath_writes_a_block_out_to_disk() {
        let memory =
            GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x10_0000)]).expect("memory");
        let path = std::env::temp_dir().join("ai-vmm-datapath-write.img");
        std::fs::write(&path, [0_u8; 512]).expect("write temp disk");

        let mut device = device_with_queue(path.to_str().expect("utf-8 path"));
        // Place the bytes the guest wants written into its data buffer.
        memory
            .write_slice(&[0xcd_u8; 512], GuestAddress(DATA_ADDR))
            .expect("fill guest data");
        publish_request(&memory, 1 /* VIRTIO_BLK_T_OUT */, 0, 512, false);

        device
            .process_virtqueue(&memory)
            .expect("process_virtqueue must succeed");

        // The host disk image now holds the bytes from the guest buffer.
        let on_disk = std::fs::read(&path).expect("read back temp disk");
        assert_eq!(on_disk.len(), 512);
        assert!(on_disk.iter().all(|&byte| byte == 0xcd));
        let status: u8 = memory.read_obj(GuestAddress(STATUS_ADDR)).expect("status");
        assert_eq!(status, 0);
        assert_eq!(device.irq_eventfd().read().expect("irqfd"), 1);

        let _ = std::fs::remove_file(&path);
        eprintln!("[datapath] virtio-blk write request serviced end to end");
    }
}
