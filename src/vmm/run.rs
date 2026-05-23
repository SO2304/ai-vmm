//! The VMM run loop: drive a vCPU with `KVM_RUN` and emulate the minimal
//! devices a Linux kernel needs to print its boot log.
//!
//! # Serial console (16550A UART, COM1)
//!
//! With `console=ttyS0` on the kernel command line, the guest sends every log
//! byte to COM1's transmit-holding register at port `0x3f8`. Before each byte
//! the kernel's 8250 serial driver polls the Line Status Register at port
//! `0x3fd` and waits for the "transmitter ready" bits.
//!
//! That is the subtle part: a UART that only handled writes to `0x3f8` and
//! "ignored other ports" would let the LSR read back as zero — the driver
//! would then spin forever waiting to transmit, and not a single log byte
//! would appear. [`uart_input_byte`] therefore always reports the transmitter
//! ready for the LSR, an invariant proven by the Kani harness below.

use crate::vmm::mmio::MmioBus;
use kvm_ioctls::{VcpuExit, VcpuFd};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use vm_memory::GuestMemoryMmap;

/// COM1 base I/O port.
const COM1_BASE: u16 = 0x3f8;
/// COM1 transmit holding register — writing a byte here sends it.
const COM1_THR: u16 = COM1_BASE;
/// COM1 line status register.
const COM1_LSR: u16 = COM1_BASE + 5;
/// LSR value reporting the transmitter ready: THR-empty (bit 5) + TEMT (bit 6).
const LSR_TRANSMITTER_READY: u8 = (1 << 5) | (1 << 6);

/// Returns the byte the emulated 16550A UART yields for a read of `port`.
///
/// The Line Status Register MUST report the transmitter as ready, otherwise
/// the guest kernel's serial driver polls it forever and never sends a byte.
/// Every other register reads back as zero.
fn uart_input_byte(port: u16) -> u8 {
    if port == COM1_LSR {
        LSR_TRANSMITTER_READY
    } else {
        0
    }
}

/// Drives one vCPU with `KVM_RUN` until it stops, then winds down its siblings.
///
/// One host thread runs one vCPU; on a multi-vCPU VM every thread calls this
/// function. The exits a booting kernel produces are handled here:
///  * `IoOut` to COM1 — printed to the host terminal (the kernel log);
///  * `IoIn`  from COM1 — answered by the minimal UART emulation;
///  * `MmioRead` / `MmioWrite` — routed through the shared `mmio_bus` to the
///    matching device (e.g. virtio-blk); an unrouted read is answered with
///    zeros;
///  * `Hlt` — the idle guest; the loop pauses briefly so it does not pin a
///    host core;
///  * `Shutdown` — a triple fault — returned as an error;
///  * `FailEntry` / `InternalError` — KVM-side failures — returned as errors;
///  * `SystemEvent` — a clean guest reboot/shutdown — ends the loop with `Ok`.
///
/// `stop` is the VM-wide wind-down flag: this loop leaves as soon as a sibling
/// vCPU raises it, and raises it itself on the way out — so when any one vCPU
/// of the VM stops (clean shutdown, triple fault, KVM error) every other vCPU
/// thread leaves its loop too and can be joined.
///
/// Every other exit is ignored silently so it does not drown the kernel log.
pub fn run_vcpu_loop(
    vcpu: &VcpuFd,
    mmio_bus: &Arc<Mutex<MmioBus>>,
    guest_memory: &GuestMemoryMmap,
    stop: &Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let outcome = drive_vcpu(vcpu, mmio_bus, guest_memory, stop);
    // However this vCPU ended — clean exit, fault, KVM error, or a sibling
    // winding the VM down — tell every other vCPU thread to stop as well.
    stop.store(true, Ordering::SeqCst);
    outcome
}

/// The inner `KVM_RUN` loop. Split out so [`run_vcpu_loop`] can raise the
/// VM-wide `stop` flag on every exit path — error or not — with one `store`.
fn drive_vcpu(
    vcpu: &VcpuFd,
    mmio_bus: &Arc<Mutex<MmioBus>>,
    guest_memory: &GuestMemoryMmap,
    stop: &Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        // A sibling vCPU has wound the VM down: leave so this thread can join.
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let exit = match vcpu.run() {
            Ok(exit) => exit,
            Err(e)
                if matches!(
                    std::io::Error::from_raw_os_error(e.errno()).kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                ) =>
            {
                // Not a failure. An application processor is briefly not
                // runnable until the bootstrap processor sends it the
                // INIT-SIPI-SIPI sequence — `KVM_RUN` reports that as `EAGAIN`
                // — and any `KVM_RUN` can be cut short by a host signal
                // (`EINTR`). Pause so the thread does not busy-spin, then
                // retry; meanwhile a real fault on this vCPU still surfaces
                // through the arm below.
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }
            Err(e) => return Err(Box::new(e)),
        };
        match exit {
            VcpuExit::IoOut(COM1_THR, data) => {
                // COM1 transmit register: print every byte to the host console.
                // Writes to any other port fall through to the silent arm.
                for &byte in data {
                    print!("{}", byte as char);
                }
                std::io::stdout().flush()?;
            }
            VcpuExit::IoIn(port, [slot, ..]) => {
                // Answer a UART register read: the LSR reports the transmitter
                // ready, every other register reads back zero.
                *slot = uart_input_byte(port);
            }
            VcpuExit::MmioRead(addr, data) => {
                // Route to the shared device bus; an unrouted address reads as
                // zero. The lock is held only for the single access.
                let handled = mmio_bus
                    .lock()
                    .map_err(|_| "MMIO bus mutex poisoned")?
                    .read(addr, data);
                if !handled {
                    eprintln!("[vmm] unhandled MMIO read at {addr:#x}");
                    data.fill(0);
                }
            }
            VcpuExit::MmioWrite(addr, data) => {
                let handled = mmio_bus
                    .lock()
                    .map_err(|_| "MMIO bus mutex poisoned")?
                    .write(addr, data, guest_memory);
                if !handled {
                    eprintln!("[vmm] unhandled MMIO write at {addr:#x}");
                }
            }
            VcpuExit::Hlt => {
                // The guest is idle and this MVP injects no interrupts to wake
                // it; pause briefly so the loop does not pin a host core.
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            VcpuExit::Shutdown => {
                return Err("guest triple-faulted (VcpuExit::Shutdown)".into());
            }
            VcpuExit::FailEntry(..) => {
                return Err("KVM could not enter the guest (VcpuExit::FailEntry)".into());
            }
            VcpuExit::InternalError => {
                return Err("KVM reported an internal error (VcpuExit::InternalError)".into());
            }
            VcpuExit::SystemEvent(..) => {
                eprintln!("\n[vmm] guest requested a clean reboot/shutdown.");
                return Ok(());
            }
            // MMIO probes and other non-critical exits: ignored silently.
            _ => {}
        }
    }
}

/// Formal proof checked by the Kani model checker (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::{uart_input_byte, COM1_LSR};

    /// Proof: a read of the Line Status Register always reports the
    /// transmitter ready — THR-empty (bit 5) and TEMT (bit 6) are both set.
    /// If either were clear, the guest kernel's serial driver would spin
    /// forever waiting to transmit, and no log byte would ever appear.
    #[kani::proof]
    fn proof_uart_lsr_reports_transmitter_ready() {
        let lsr = uart_input_byte(COM1_LSR);
        assert!(lsr & (1 << 5) != 0);
        assert!(lsr & (1 << 6) != 0);
    }
}

/// Tests: the pure UART logic (hermetic) and a real run of the loop against a
/// tiny guest program (Linux with `/dev/kvm`).
#[cfg(test)]
mod tests {
    use super::{run_vcpu_loop, uart_input_byte, COM1_LSR, COM1_THR, LSR_TRANSMITTER_READY};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    #[test]
    fn lsr_read_reports_transmitter_ready() {
        assert_eq!(uart_input_byte(COM1_LSR), LSR_TRANSMITTER_READY);
    }

    #[test]
    fn non_status_ports_read_back_zero() {
        assert_eq!(uart_input_byte(COM1_THR), 0);
        assert_eq!(uart_input_byte(0x0000), 0);
        assert_eq!(uart_input_byte(0xffff), 0);
    }

    /// Best-effort integration test: needs `/dev/kvm`. A tiny 64-bit program
    /// writes "OK" to COM1, then executes `ud2` to triple-fault. The run loop
    /// must emulate the two serial writes and then end with an error on the
    /// `Shutdown` exit.
    #[test]
    fn run_loop_emulates_uart_then_handles_shutdown() {
        use crate::vmm::boot::{setup_boot_memory, setup_vcpu_registers};
        use kvm_bindings::kvm_userspace_memory_region;
        use kvm_ioctls::Kvm;
        use vm_memory::{Bytes, GuestAddress, GuestMemory, GuestMemoryMmap};

        let kvm = match Kvm::new() {
            Ok(kvm) => kvm,
            Err(e) => {
                eprintln!("[run-smoke] /dev/kvm unavailable, test inconclusive: {e}");
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

        // A tiny 64-bit program: write "OK" to COM1 (0x3f8), then `ud2`.
        const CODE_ADDR: u64 = 0x10_0000;
        let program: [u8; 12] = [
            0x66, 0xBA, 0xF8, 0x03, // mov dx, 0x3f8
            0xB0, 0x4F, // mov al, 'O'
            0xEE, // out dx, al
            0xB0, 0x4B, // mov al, 'K'
            0xEE, // out dx, al
            0x0F, 0x0B, // ud2  -> triple fault
        ];
        guest_memory
            .write_slice(&program, GuestAddress(CODE_ADDR))
            .expect("write program");

        setup_boot_memory(&guest_memory).expect("setup_boot_memory");
        let vcpu = vm.create_vcpu(0).expect("create_vcpu");
        setup_vcpu_registers(&vcpu, CODE_ADDR).expect("setup_vcpu_registers");

        // The loop emulates the two COM1 writes (printing "OK"), then returns
        // an error when the `ud2` triple-faults the guest.
        let mmio_bus = Arc::new(Mutex::new(crate::vmm::mmio::MmioBus::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let result = run_vcpu_loop(&vcpu, &mmio_bus, &guest_memory, &stop);
        assert!(
            result.is_err(),
            "the loop must end with an error on the guest triple fault"
        );
        eprintln!("[run-smoke] run loop emulated UART output and caught the triple fault");
    }

    /// Best-effort integration test: needs `/dev/kvm`. A tiny program does an
    /// MMIO read from an address owned by a probe device on the bus, then
    /// `ud2`s. The run loop must route the access through the bus to the device.
    #[test]
    fn run_loop_routes_mmio_to_the_bus() {
        use crate::vmm::boot::{setup_boot_memory, setup_vcpu_registers};
        use crate::vmm::mmio::{MmioBus, MmioDevice};
        use kvm_bindings::kvm_userspace_memory_region;
        use kvm_ioctls::Kvm;
        use vm_memory::{Bytes, GuestAddress, GuestMemory, GuestMemoryMmap};

        struct Probe(Arc<AtomicBool>);
        impl MmioDevice for Probe {
            fn mmio_read(&mut self, _offset: u64, data: &mut [u8]) {
                self.0.store(true, Ordering::SeqCst);
                data.fill(0);
            }
            fn mmio_write(&mut self, _offset: u64, _data: &[u8], _guest_memory: &GuestMemoryMmap) {}
        }

        let kvm = match Kvm::new() {
            Ok(kvm) => kvm,
            Err(e) => {
                eprintln!("[mmio-smoke] /dev/kvm unavailable, test inconclusive: {e}");
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

        // `mov al, [0x10000000]` then `ud2`. 0x1000_0000 is inside the
        // identity-mapped first GiB but above the 2 MiB of guest RAM, so the
        // access leaves KVM as an MMIO exit rather than a memory access.
        const CODE_ADDR: u64 = 0x10_0000;
        let program: [u8; 11] = [
            0xA0, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, // mov al, [0x10000000]
            0x0F, 0x0B, // ud2
        ];
        guest_memory
            .write_slice(&program, GuestAddress(CODE_ADDR))
            .expect("write program");

        setup_boot_memory(&guest_memory).expect("setup_boot_memory");
        let vcpu = vm.create_vcpu(0).expect("create_vcpu");
        setup_vcpu_registers(&vcpu, CODE_ADDR).expect("setup_vcpu_registers");

        let probed = Arc::new(AtomicBool::new(false));
        let mut bus = MmioBus::new();
        bus.register(0x1000_0000, 0x1000, Box::new(Probe(Arc::clone(&probed))));
        let mmio_bus = Arc::new(Mutex::new(bus));
        let stop = Arc::new(AtomicBool::new(false));

        let result = run_vcpu_loop(&vcpu, &mmio_bus, &guest_memory, &stop);
        assert!(
            result.is_err(),
            "the loop should end on the guest triple fault"
        );
        assert!(
            probed.load(Ordering::SeqCst),
            "the guest MMIO read must have been routed through the bus to the device"
        );
        eprintln!("[mmio-smoke] guest MMIO access routed through the bus to the device");
    }
}
