//! Live integration test: boots a real Linux kernel and root filesystem
//! through the full `ai-vmm` provisioning pipeline.
//!
//! The test needs a `./vmlinux` (ELF kernel) and a `./rootfs.ext4` (a disk
//! image whose root filesystem carries an `init` executable) in the working
//! directory. When they are absent — as on CI, which does not carry those
//! large binaries — the test skips cleanly so it never blocks a build.
//!
//! It never blocks indefinitely: the boot runs on a separate thread and the
//! test observes it for a bounded window only.

use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

/// How long to let the guest run before declaring the boot a success.
const BOOT_OBSERVE_SECS: u64 = 20;

/// Boots a real OS and confirms the VMM drives it without faulting.
///
/// This is a smoke test of the whole provisioning pipeline: a genuine boot
/// failure (a triple fault, or any pipeline error) makes it fail, while a
/// guest that boots and keeps running passes. Run with `--nocapture` to watch
/// the real kernel and systemd boot log on the emulated serial console.
#[test]
fn test_live_ubuntu_boot() {
    // KVM provisioning only works on Linux; elsewhere there is nothing to do.
    if !cfg!(target_os = "linux") {
        eprintln!("[live] not a Linux host — live boot test skipped.");
        return;
    }

    // The kernel and root-filesystem images are large, uncommitted binaries.
    // Without them there is nothing to boot, so skip cleanly.
    if !Path::new("./vmlinux").exists() || !Path::new("./rootfs.ext4").exists() {
        eprintln!("[live] ./vmlinux and/or ./rootfs.ext4 not present — live boot test skipped.");
        return;
    }

    eprintln!("[live] booting a real OS: 1 vCPU, 256 MiB RAM, rootfs ./rootfs.ext4 ...");

    // `provision_kvm_machine_native` runs the whole pipeline: KVM creation, the
    // host vCPU-capability check, RAM allocation, Direct Kernel Boot
    // (`setup_vcpu_registers`, `setup_zero_page`), the virtio-blk MMIO device,
    // and finally the vCPU run loop. The run loop blocks for as long as the
    // guest runs, so it is driven on a separate thread. The error is converted
    // to a `String` before crossing the channel because `Box<dyn Error>` is
    // not `Send`.
    let (tx, rx) = mpsc::channel::<Result<String, String>>();
    std::thread::spawn(move || {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let outcome = ai_vmm::vmm::provision_kvm_machine_native(
            1,
            256,
            "live-ubuntu",
            None,
            Some("./rootfs.ext4"),
            "./vmlinux",
            &stop,
        )
        .map_err(|e| e.to_string());
        // The receiver may already be gone if the test timed out — ignore that.
        let _ = tx.send(outcome);
    });

    // Wait only a bounded time, so the test can never hang. The boot thread is
    // detached: if it is still running, it is torn down when this test binary
    // exits.
    match rx.recv_timeout(Duration::from_secs(BOOT_OBSERVE_SECS)) {
        Ok(Ok(report)) => {
            // The guest reached a clean shutdown within the observation window.
            eprintln!("[live] guest exited cleanly: {report}");
        }
        Ok(Err(error)) => {
            // The pipeline failed — e.g. a triple fault makes the run loop
            // return early. That is a genuine boot failure.
            panic!("live OS boot failed: {error}");
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            // Still running after the observation window: the guest booted and
            // did not triple-fault or abort the pipeline. With the Firecracker
            // kernel and Ubuntu rootfs this window covers the kernel coming up
            // and handing off to `/sbin/init` (systemd) — visible on the
            // serial console when the test is run with `--nocapture`.
            eprintln!("[live] guest booted and ran for {BOOT_OBSERVE_SECS}s without crashing.");
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("the boot thread terminated unexpectedly");
        }
    }
}
