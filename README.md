# ai-vmm — agentic hypervisor

`ai-vmm` turns a natural-language intent into a real virtual machine. You
describe what you want; a reasoning model produces an execution plan; you
review it; and on approval the control plane provisions and boots the VM
natively through KVM.

```
ai-vmm run "create a VM with 2 vCPUs and 2 GiB RAM named db"
```

The model does the **reasoning**. Every resource bound is enforced in Rust and
**formally proven** by [Kani](https://github.com/model-checking/kani) — the
model never sets the limits.

---

## Requirements

- **A Linux host with `/dev/kvm`.** The user must be in the `kvm` group (or
  root). The crate builds on Windows and macOS, but actual provisioning is
  Linux-only — on Windows, run it under WSL2.
- **A reasoning back end**, either:
  - the hosted Anthropic API (bring your own key), or
  - a local OpenAI-compatible server — Ollama, vLLM, llama.cpp — for a fully
    air-gapped deployment with no network egress.
- **A guest kernel and a root filesystem** (see below).

## Install

```sh
cargo build --release
# the binary is target/release/ai-vmm
```

## Quick start

```sh
# 1. Configure the reasoning back end (once).
ai-vmm auth login                      # hosted Anthropic — paste your API key
#   ...or, for an air-gapped local model:
#   export AI_VMM_PROVIDER=openai
#   export AI_VMM_BASE_URL=http://localhost:11434/v1
#   export AI_VMM_MODEL=llama3

# 2. Point ai-vmm at a kernel and a default root filesystem (once).
export AI_VMM_KERNEL=/path/to/vmlinux
export AI_VMM_DISK=/path/to/rootfs.ext4

# 3. Provision a VM, from any directory.
ai-vmm run "create a VM with 2 vCPUs and 2 GiB RAM named db"
```

`ai-vmm run` plans, shows the plan for review, and applies it only after you
type `y`. Add `--yes` to skip the prompt in scripts. To provision without the
AI round-trip:

```sh
ai-vmm provision --name db --vcpus 2 --memory 2G --disk /path/to/rootfs.ext4
```

## The guest kernel and root filesystem

`ai-vmm` performs a **Direct Kernel Boot**, so it needs two artefacts:

- **Kernel** — an *uncompressed* `vmlinux` ELF image (not a `bzImage`).
- **Root filesystem** — an `ext4` disk image carrying an `/sbin/init`.

The
[Firecracker getting-started guide](https://github.com/firecracker-microvm/firecracker/blob/main/docs/getting-started.md)
publishes ready-made kernel and rootfs downloads that work directly; or build
your own rootfs with `debootstrap` into an `ext4` image. Place them wherever
you like and point `ai-vmm` at them with the configuration below — a kernel
and a root image are large binaries and are intentionally not committed to the
repository.

## Configuration

`ai-vmm` resolves the kernel and disk in this order — first hit wins:

| Setting | 1. Environment | 2. `credentials.toml` | 3. Default |
|---|---|---|---|
| Kernel  | `AI_VMM_KERNEL` | `kernel_path` | `./vmlinux` |
| Disk    | `AI_VMM_DISK`   | `disk_path`   | — (a disk is required) |

The configuration file lives at `~/.config/ai-vmm/credentials.toml` (created by
`ai-vmm auth login`). A complete example:

```toml
anthropic_api_key = "sk-ant-..."
kernel_path = "/home/me/images/vmlinux"
disk_path   = "/home/me/images/rootfs.ext4"
```

Provider overrides — `AI_VMM_PROVIDER`, `AI_VMM_BASE_URL`, `AI_VMM_MODEL`,
`ANTHROPIC_API_KEY` — are intended for CI and air-gapped hosts.

## Commands

| Command | Purpose |
|---|---|
| `ai-vmm auth login` | Store the Anthropic API key locally (BYOK). |
| `ai-vmm run "<intent>"` | Plan a VM from natural language, review, apply. |
| `ai-vmm provision --name N --vcpus C --memory M --disk D` | Provision directly — no AI, no prompt. |
| `ai-vmm list` | List the VMs recorded in the local ledger. |
| `ai-vmm inspect <name>` | Show one VM's full recorded plan. |
| `ai-vmm forget <name>` | Remove a VM from the ledger. |
| `ai-vmm serve [--addr host:port]` | Run the HTTP control-plane API server. |

## HTTP control-plane API

`ai-vmm serve` exposes the agentic planner and the VM ledger over HTTP, so
automation and AI agents can drive the control plane programmatically:

```sh
export AI_VMM_API_TOKEN=$(openssl rand -hex 16)   # required — the bearer token
ai-vmm serve --addr 127.0.0.1:8080
```

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /healthz` | none | Liveness probe. |
| `POST /v1/plan` | bearer | Natural-language intent → a reviewable plan (JSON). |
| `POST /v1/vms` | bearer | Provision and boot a VM as a supervised worker (returns `202` at once). |
| `GET /v1/vms` | bearer | List every VM and its lifecycle state. |
| `GET /v1/vms/{name}` | bearer | One VM's recorded plan. |
| `POST /v1/vms/{name}/stop` | bearer | Gracefully shut a running VM down. |
| `DELETE /v1/vms/{name}` | bearer | Remove a VM from the ledger. |

Every `/v1` route requires `Authorization: Bearer $AI_VMM_API_TOKEN`, and the
server refuses to start without a token; the token comparison is constant-time.
`POST /v1/vms` spawns each VM as its own worker process — so the call returns
immediately while the guest boots in the background, its console captured to a
per-VM log file under the data directory.

## How it works

A Terraform-style **plan → review → apply** workflow. The reasoning model emits
a plan; `ai-vmm` validates every value (`validate_spec`), reconciles it against
the host's real KVM and memory capacity, then provisions natively with
`rust-vmm` (`kvm-ioctls`, `vm-memory`). Multi-vCPU VMs are brought up through an
Intel MP table; the guest mounts an `ext4` root over a virtio-blk MMIO device.

## Security

- The control plane trusts the model for *reasoning*, never for *bounds*: the
  limits are enforced in Rust and proven by Kani.
- The API key follows a BYOK model — stored once in a `0600` `credentials.toml`
  (directory `0700` on Unix), never logged, never committed.
- Guest memory is an anonymous private `mmap`, isolated from the host.
- Every external input — API responses, tool arguments — is treated as
  untrusted and explicitly validated before any hardware contact.

## Known limitations

- KVM provisioning is **Linux-only**.
- `--network` / `network_bridge` provisions host-side TAP plumbing; attaching
  the guest NIC to it (in-guest networking) is not yet implemented.
- A VM runs in the foreground until it exits; one VM at a time.
- Under **nested virtualization** (a KVM guest inside WSL2) the guest kernel's
  APIC/timer calibration is timing-sensitive and can occasionally stall before
  root mount. On real hardware, or a single-level KVM host, the boot is
  deterministic.

## Building and verifying from source

```sh
cargo build --release
cargo clippy --all-targets -- -D warnings   # strict lint, zero warnings
cargo test                                  # unit + live boot tests
cargo kani                                   # formal proofs
```

A GitHub Actions workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
runs this same baseline — formatting, Clippy, and tests on Linux and Windows,
and the Kani proofs on Linux — on every push and pull request.

## License

MIT — see [LICENSE](LICENSE).
