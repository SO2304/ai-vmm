//! VM ledger — a lightweight, declarative record of provisioned VMs.
//!
//! `ai-vmm` is a declarative control plane: every VM is fully described by its
//! [`ExecutionPlan`]. This module persists those plans to a per-user
//! `vms.toml`, which makes the plan itself the snapshot — re-applying a stored
//! plan re-materialises the VM, with no guest-memory image to ship between
//! hosts. `ai-vmm list` / `inspect` read the ledger back; `forget` prunes it.
//!
//! The ledger holds plain declarative state, not secrets, and is not KVM-gated:
//! it behaves identically on every platform.

use crate::agent::ExecutionPlan;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Upper bound on the number of records the ledger keeps. Older records roll
/// off past this, so the on-disk file can never grow without bound.
const MAX_RECORDS: usize = 64;

/// The lifecycle state of a VM, as observed by the control plane.
///
/// The lifecycle is strictly forward-only — `Booting → Running → {Stopped,
/// Failed}` — and a terminal state is final: a finished VM is never resurrected.
/// [`VmLifecycle::can_transition_to`] is the state machine, and the Kani
/// harnesses prove it admits only valid transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VmLifecycle {
    /// The VM has been provisioned and is being booted.
    Booting,
    /// The guest is up and running.
    Running,
    /// The guest has exited; the VM is finished. Also the state assumed for a
    /// record written before this field existed.
    #[default]
    Stopped,
    /// Provisioning or the boot failed; the VM never reached a usable state.
    Failed,
}

impl VmLifecycle {
    /// Whether a transition from `self` to `next` is valid in the VM lifecycle.
    ///
    /// Valid edges: `Booting → {Running, Stopped, Failed}` and
    /// `Running → {Stopped, Failed}`. A terminal state has no outgoing edge,
    /// nothing ever re-enters `Booting`, and no state transitions to itself.
    pub const fn can_transition_to(self, next: VmLifecycle) -> bool {
        use VmLifecycle::{Booting, Failed, Running, Stopped};
        matches!(
            (self, next),
            (Booting, Running)
                | (Booting, Stopped)
                | (Booting, Failed)
                | (Running, Stopped)
                | (Running, Failed)
        )
    }
}

impl std::fmt::Display for VmLifecycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            VmLifecycle::Booting => "booting",
            VmLifecycle::Running => "running",
            VmLifecycle::Stopped => "stopped",
            VmLifecycle::Failed => "failed",
        })
    }
}

/// One ledger entry: a VM's declared resources, its lifecycle state, and when
/// the plan was applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmRecord {
    /// Name of the virtual machine.
    pub vm_name: String,
    /// Virtual cores declared.
    pub vcpus: u32,
    /// RAM declared, in mebibytes.
    pub memory_mb: u32,
    /// Host bridge attached, or `None` for an isolated VM.
    pub network_bridge: Option<String>,
    /// Root-filesystem image, or `None` for a diskless VM.
    pub disk_image_path: Option<String>,
    /// Unix time, in seconds, when the plan was applied.
    pub applied_at: u64,
    /// Current lifecycle state of the VM.
    #[serde(default)]
    pub state: VmLifecycle,
}

impl VmRecord {
    /// Builds a record from an approved plan, stamped with the current time.
    fn from_plan(plan: &ExecutionPlan) -> Self {
        VmRecord {
            vm_name: plan.vm_name.clone(),
            vcpus: plan.vcpus,
            memory_mb: plan.memory_mb,
            network_bridge: plan.network_bridge.clone(),
            disk_image_path: plan.disk_image_path.clone(),
            applied_at: unix_now(),
            state: VmLifecycle::Booting,
        }
    }
}

/// The declarative VM ledger: an ordered list of records, oldest first.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Ledger {
    #[serde(default)]
    records: Vec<VmRecord>,
}

impl Ledger {
    /// Loads the ledger from `path`. A missing file yields an empty ledger;
    /// only an unreadable or malformed file is an error.
    pub fn load_from(path: &Path) -> Result<Ledger, Box<dyn std::error::Error>> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(toml::from_str(&text)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Ledger::default()),
            Err(e) => Err(Box::new(e)),
        }
    }

    /// Writes the ledger to `path`, creating the parent directory if needed.
    pub fn save_to(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }

    /// The records, oldest first.
    pub fn records(&self) -> &[VmRecord] {
        &self.records
    }

    /// Returns the record for `vm_name`, if the ledger holds one.
    pub fn get(&self, vm_name: &str) -> Option<&VmRecord> {
        self.records.iter().find(|r| r.vm_name == vm_name)
    }

    /// Records an applied plan.
    ///
    /// Declaring is an upsert: an existing record for the same VM name is
    /// replaced, so the ledger never holds duplicate names. The ledger is then
    /// capped at [`MAX_RECORDS`] by dropping its oldest entries.
    pub fn record(&mut self, plan: &ExecutionPlan) {
        self.records.retain(|r| r.vm_name != plan.vm_name);
        let start = first_kept_index(self.records.len());
        self.records.drain(..start);
        self.records.push(VmRecord::from_plan(plan));
    }

    /// Removes the record for `vm_name`, returning whether one was found.
    pub fn forget(&mut self, vm_name: &str) -> bool {
        let before = self.records.len();
        self.records.retain(|r| r.vm_name != vm_name);
        self.records.len() != before
    }

    /// Advances the lifecycle state of a recorded VM.
    ///
    /// The transition must be valid (see [`VmLifecycle::can_transition_to`]):
    /// a terminal record is never reopened. Returns whether a record was found
    /// and updated.
    pub fn set_state(&mut self, vm_name: &str, state: VmLifecycle) -> bool {
        match self.records.iter_mut().find(|r| r.vm_name == vm_name) {
            Some(record) if record.state.can_transition_to(state) => {
                record.state = state;
                true
            }
            _ => false,
        }
    }
}

/// Index of the first record to keep when one more is about to be appended, so
/// that no more than [`MAX_RECORDS`] entries remain. Records before this index
/// are the oldest and are dropped.
///
/// `proof_first_kept_index_is_a_valid_split` proves the result never exceeds
/// `old_len` (so [`Ledger::record`]'s drain range is always valid), and
/// `proof_ledger_stays_bounded` proves the post-append length is capped.
fn first_kept_index(old_len: usize) -> usize {
    old_len.saturating_sub(MAX_RECORDS - 1)
}

/// Returns the current Unix time in seconds, saturating to 0 before the epoch.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Path of the per-user VM ledger (`~/.local/share/ai-vmm/vms.toml` on Linux).
pub fn ledger_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dirs = directories::ProjectDirs::from("com", "ai-vmm", "ai-vmm")
        .ok_or("could not determine a data directory for this platform")?;
    Ok(dirs.data_dir().join("vms.toml"))
}

/// Lets the Kani harnesses draw a symbolic [`VmLifecycle`] value.
#[cfg(kani)]
impl kani::Arbitrary for VmLifecycle {
    fn any() -> Self {
        match kani::any::<u8>() % 4 {
            0 => VmLifecycle::Booting,
            1 => VmLifecycle::Running,
            2 => VmLifecycle::Stopped,
            _ => VmLifecycle::Failed,
        }
    }
}

/// Formal proofs checked by the Kani model checker (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::{first_kept_index, VmLifecycle, MAX_RECORDS};

    /// Proof: the kept-from index never points past the ledger's end, so the
    /// `drain(..start)` range in `Ledger::record` is always a valid slice.
    #[kani::proof]
    fn proof_first_kept_index_is_a_valid_split() {
        let old_len: usize = kani::any();
        assert!(first_kept_index(old_len) <= old_len);
    }

    /// Proof: after dropping the oldest records and appending one, the ledger
    /// never exceeds `MAX_RECORDS` — the on-disk file stays bounded.
    #[kani::proof]
    fn proof_ledger_stays_bounded() {
        let old_len: usize = kani::any();
        let kept = old_len - first_kept_index(old_len);
        assert!(kept + 1 <= MAX_RECORDS);
    }

    /// Proof: a terminal VM state is final — `Stopped` and `Failed` have no
    /// outgoing transition, so a finished VM can never be marked alive again.
    #[kani::proof]
    fn proof_terminal_states_are_final() {
        let next: VmLifecycle = kani::any();
        assert!(!VmLifecycle::Stopped.can_transition_to(next));
        assert!(!VmLifecycle::Failed.can_transition_to(next));
    }

    /// Proof: the lifecycle is strictly forward-only — no transition re-enters
    /// `Booting`, and no state ever transitions to itself.
    #[kani::proof]
    fn proof_lifecycle_is_forward_only() {
        let from: VmLifecycle = kani::any();
        assert!(!from.can_transition_to(VmLifecycle::Booting));
        assert!(!from.can_transition_to(from));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal plan with the given VM name.
    fn plan(name: &str) -> ExecutionPlan {
        ExecutionPlan {
            vcpus: 2,
            memory_mb: 2048,
            vm_name: name.to_string(),
            network_bridge: None,
            disk_image_path: None,
        }
    }

    #[test]
    fn record_then_get_round_trips() {
        let mut ledger = Ledger::default();
        ledger.record(&plan("db-prod"));
        let record = ledger.get("db-prod").expect("record present");
        assert_eq!(record.vcpus, 2);
        assert_eq!(record.memory_mb, 2048);
    }

    #[test]
    fn recording_the_same_name_replaces_not_duplicates() {
        let mut ledger = Ledger::default();
        ledger.record(&plan("web"));
        ledger.record(&plan("web"));
        assert_eq!(ledger.records().len(), 1);
    }

    #[test]
    fn ledger_is_capped_at_max_records() {
        let mut ledger = Ledger::default();
        for index in 0..(MAX_RECORDS + 20) {
            ledger.record(&plan(&format!("vm-{index}")));
        }
        assert_eq!(ledger.records().len(), MAX_RECORDS);
        // The oldest record rolled off; the newest survived.
        assert!(ledger.get("vm-0").is_none());
        assert!(ledger.get(&format!("vm-{}", MAX_RECORDS + 19)).is_some());
    }

    #[test]
    fn forget_removes_a_record() {
        let mut ledger = Ledger::default();
        ledger.record(&plan("temp"));
        assert!(ledger.forget("temp"));
        assert!(!ledger.forget("temp"));
        assert!(ledger.get("temp").is_none());
    }

    #[test]
    fn lifecycle_transitions_are_forward_only() {
        use VmLifecycle::{Booting, Failed, Running, Stopped};
        // Valid forward edges.
        assert!(Booting.can_transition_to(Running));
        assert!(Booting.can_transition_to(Stopped));
        assert!(Booting.can_transition_to(Failed));
        assert!(Running.can_transition_to(Stopped));
        assert!(Running.can_transition_to(Failed));
        // Terminal states are final; nothing returns to Booting; no self-loop.
        assert!(!Stopped.can_transition_to(Running));
        assert!(!Failed.can_transition_to(Running));
        assert!(!Running.can_transition_to(Booting));
        assert!(!Running.can_transition_to(Running));
    }

    #[test]
    fn set_state_advances_a_record_and_rejects_invalid_transitions() {
        let mut ledger = Ledger::default();
        ledger.record(&plan("api"));
        // A freshly recorded VM starts in `Booting`.
        assert_eq!(
            ledger.get("api").expect("record").state,
            VmLifecycle::Booting
        );
        // A valid forward transition is applied.
        assert!(ledger.set_state("api", VmLifecycle::Stopped));
        assert_eq!(
            ledger.get("api").expect("record").state,
            VmLifecycle::Stopped
        );
        // An invalid transition out of a terminal state is refused.
        assert!(!ledger.set_state("api", VmLifecycle::Running));
        assert_eq!(
            ledger.get("api").expect("record").state,
            VmLifecycle::Stopped
        );
        // An unknown VM is not updated.
        assert!(!ledger.set_state("ghost", VmLifecycle::Failed));
    }

    #[test]
    fn save_and_load_round_trip() {
        let mut ledger = Ledger::default();
        ledger.record(&plan("persisted"));
        let path = std::env::temp_dir().join("ai-vmm-ledger-roundtrip.toml");
        ledger.save_to(&path).expect("save");
        let loaded = Ledger::load_from(&path).expect("load");
        assert_eq!(loaded.records().len(), 1);
        assert_eq!(loaded.get("persisted").expect("record").memory_mb, 2048);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_from_missing_file_is_empty() {
        let path = std::env::temp_dir().join("ai-vmm-ledger-absent.toml");
        let _ = std::fs::remove_file(&path);
        let ledger = Ledger::load_from(&path).expect("a missing ledger is not an error");
        assert!(ledger.records().is_empty());
    }
}
