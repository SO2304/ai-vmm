//! Prompts and tool schemas for the agentic control plane.
//!
//! This module isolates the "contract" between the hypervisor and the
//! reasoning model: the system prompt that defines the model's role and the
//! JSON schema of every tool exposed to it. The schemas are kept in a
//! provider-neutral form ([`ToolSpec`]); `agent` wraps them into either the
//! Anthropic or the OpenAI-compatible tool format.

use serde_json::{json, Value};

/// System prompt: positions the model as the infrastructure planner of an
/// enterprise KVM hypervisor. The model produces an execution plan; it never
/// has direct hardware access, and nothing is applied without explicit human
/// approval (a Terraform-style plan / approve / apply workflow).
pub static SYSTEM_PROMPT: &str = "\
You are the control plane of an enterprise KVM hypervisor named `ai-vmm`. \
You act as an INFRASTRUCTURE PLANNER: you translate infrastructure operators' \
natural-language intents into a precise execution plan.

Crucial: calling the `provision_kvm_machine` tool does NOT immediately create \
anything. It produces a PLAN that a human operator reviews and must explicitly \
approve before it is applied to real hardware. Nothing runs without approval.

Strict rules:
- You have NO direct hardware access. The `provision_kvm_machine` tool is the \
only way to express a provisioning plan.
- As soon as a virtual-machine creation request is expressed, you MUST call \
`provision_kvm_machine` with the exact parameters derived from the intent.
- The hypervisor keeps a registry of VMs already provisioned. If the operator \
asks to CLONE, COPY, RESIZE, MODIFY, COMPARE or VERIFY an existing VM, you MUST \
FIRST call `list_vms` to see what exists and/or `inspect_vm` to read the exact \
recorded specification of the named VM. Build your `provision_kvm_machine` plan \
from the specification you actually read back — never from assumptions.
- `vcpus`: integer number of virtual cores (1 to 255).
- `memory_mb`: amount of RAM in mebibytes (8 MiB minimum).
- `vm_name`: a readable, non-empty identifier for the virtual machine.
- `network_bridge` (optional): the host bridge interface to attach the VM to \
(e.g. br0, virbr0). Omit it, or use \"none\", for an isolated VM with no network.
- `disk_image_path` (optional): the path to a disk image to attach as the VM's \
root filesystem over virtio-blk (e.g. ./rootfs.ext4). Set it to the disk the \
user names. If the user names none, omit it — the hypervisor then supplies its \
own configured default root filesystem. Every VM boots from a root filesystem.
- Convert units faithfully: 1 GiB = 1024 MiB, 1 TiB = 1048576 MiB.
- Never fabricate arbitrary values: derive them exactly from the request. When \
in doubt about a value, pick the most conservative option.
- If the intent is ambiguous or unrelated to provisioning a VM, do not call \
any tool and answer in plain text.

Final bound checking and memory safety are enforced by the hypervisor: your \
role is planning, not enforcing the limits.";

/// A tool exposed to the model, in a provider-neutral form.
///
/// `agent` wraps this into the Anthropic format (`input_schema`) or the
/// OpenAI-compatible format (`function.parameters`); the `parameters` field is
/// a standard JSON Schema object reused verbatim by both.
pub struct ToolSpec {
    /// Tool name, exactly as the model must call it.
    pub name: &'static str,
    /// Human-readable description shown to the model.
    pub description: &'static str,
    /// JSON Schema of the tool's arguments.
    pub parameters: Value,
}

/// Returns every tool exposed to the model, provider-neutral.
pub fn tool_specs() -> Vec<ToolSpec> {
    vec![provision_spec(), list_vms_spec(), inspect_vm_spec()]
}

/// Schema of `provision_kvm_machine` — the terminal tool that yields the plan.
fn provision_spec() -> ToolSpec {
    ToolSpec {
        name: "provision_kvm_machine",
        description: "Describes a virtual machine to provision on the host's KVM \
hypervisor. Calling this tool produces an execution plan (vCPUs, memory, name, \
optional network bridge, optional root-filesystem disk) that a human operator \
reviews and approves before anything is applied. Use it as soon as an operator \
asks to create a VM.",
        parameters: json!({
            "type": "object",
            "properties": {
                "vcpus": {
                    "type": "integer",
                    "description": "Number of virtual cores (vCPUs) to allocate.",
                    "minimum": 1,
                    "maximum": 255
                },
                "memory_mb": {
                    "type": "integer",
                    "description": "VM RAM, in mebibytes (MiB).",
                    "minimum": 8,
                    "maximum": 1048576
                },
                "vm_name": {
                    "type": "string",
                    "description": "Readable, non-empty identifier for the VM.",
                    "minLength": 1
                },
                "network_bridge": {
                    "type": "string",
                    "description": "Optional host bridge interface to attach the VM to (e.g. br0, virbr0). Omit, or use \"none\", for an isolated VM with no network."
                },
                "disk_image_path": {
                    "type": "string",
                    "description": "Optional path to a disk image to attach as the VM's root filesystem via virtio-blk (e.g. './rootfs.ext4'). Set it to the disk the user names; omit it if the user names none, and the hypervisor supplies its configured default root filesystem."
                }
            },
            "required": ["vcpus", "memory_mb", "vm_name"]
        }),
    }
}

/// Schema of `list_vms` — reads the names of every VM in the local registry.
fn list_vms_spec() -> ToolSpec {
    ToolSpec {
        name: "list_vms",
        description: "Lists the names of every virtual machine already recorded \
in the local ai-vmm registry. Takes no arguments. Call it to discover what \
exists before cloning, comparing or modifying a VM.",
        parameters: json!({
            "type": "object",
            "properties": {}
        }),
    }
}

/// Schema of `inspect_vm` — reads one VM's recorded specification.
fn inspect_vm_spec() -> ToolSpec {
    ToolSpec {
        name: "inspect_vm",
        description: "Returns the recorded specification (vCPUs, memory, network \
bridge, disk image) of one virtual machine in the local ai-vmm registry, \
selected by name. Call it to read a VM's exact configuration before cloning or \
modifying it.",
        parameters: json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the VM to inspect, as returned by list_vms.",
                    "minLength": 1
                }
            },
            "required": ["name"]
        }),
    }
}
