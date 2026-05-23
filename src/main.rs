//! `ai-vmm` — Agentic Hypervisor.
//!
//! Professional CLI with a Terraform-style workflow. `ai-vmm auth login`
//! stores your Anthropic API key once (BYOK). `ai-vmm run "<intent>"` asks
//! Claude to produce an execution plan, shows it for review, and applies it to
//! real KVM hardware only after explicit operator approval.

use ai_vmm::agent::{self, ExecutionPlan};
use ai_vmm::{config, registry, server, vmm};
use clap::{Parser, Subcommand};

/// Top-level command-line interface.
#[derive(Parser)]
#[command(
    name = "ai-vmm",
    version,
    about = "Agentic Hypervisor — KVM control plane driven by Claude"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Top-level subcommands.
#[derive(Subcommand)]
enum Commands {
    /// Manage the locally stored Anthropic API credentials.
    Auth {
        #[command(subcommand)]
        command: AuthCommands,
    },
    /// Plan a provisioning request, then apply it after approval.
    Run {
        /// Natural-language intent, e.g. "Create a VM with 2 vcpus and 2048 MiB named db-prod".
        prompt: String,
        /// Apply the plan without the interactive confirmation (for scripts).
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Provision a VM directly from explicit specs — no AI, no prompt (headless).
    Provision {
        /// Name of the virtual machine.
        #[arg(long)]
        name: String,
        /// Number of virtual cores.
        #[arg(long)]
        vcpus: u32,
        /// RAM: a plain MiB count or a unit suffix (e.g. 2048, 512M, 2G).
        #[arg(long)]
        memory: String,
        /// Host bridge to attach to (omitted: an isolated VM).
        #[arg(long)]
        network: Option<String>,
        /// Root-filesystem disk image (omitted: a diskless VM).
        #[arg(long)]
        disk: Option<String>,
    },
    /// List the VMs recorded in the local ledger.
    List,
    /// Show the full recorded plan for one VM.
    Inspect {
        /// Name of the VM to inspect.
        name: String,
    },
    /// Remove a VM from the local ledger.
    Forget {
        /// Name of the VM to remove.
        name: String,
    },
    /// Run the HTTP control-plane API server (planning + ledger endpoints).
    Serve {
        /// Address to bind, as host:port.
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
    },
}

/// Subcommands of `ai-vmm auth`.
#[derive(Subcommand)]
enum AuthCommands {
    /// Store your Anthropic API key locally (Bring Your Own Key).
    Login,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(e) = dispatch(cli).await {
        eprintln!("[error] {e}");
        std::process::exit(1);
    }
}

/// Routes a parsed command to its handler.
async fn dispatch(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Auth { command } => match command {
            AuthCommands::Login => auth_login(),
        },
        Commands::Run { prompt, yes } => run_intent(&prompt, yes).await,
        Commands::Provision {
            name,
            vcpus,
            memory,
            network,
            disk,
        } => provision_direct(name, vcpus, &memory, network, disk),
        Commands::List => list_vms(),
        Commands::Inspect { name } => inspect_vm(&name),
        Commands::Forget { name } => forget_vm(&name),
        Commands::Serve { addr } => server::serve(&addr).await,
    }
}

/// Plans a provisioning request, shows it, and applies it after approval.
///
/// This is the Terraform-style flow: plan -> review -> approve -> apply.
/// `auto_approve` (`--yes`) skips the interactive confirmation, so the
/// natural-language path is usable from a non-interactive script.
async fn run_intent(prompt: &str, auto_approve: bool) -> Result<(), Box<dyn std::error::Error>> {
    let settings = config::load_provider_settings()?;

    let mut plan = match agent::ask_agent(prompt, &settings).await? {
        Some(plan) => plan,
        // The model only replied with text; `ask_agent` already printed it.
        None => return Ok(()),
    };

    // Resolve the root-filesystem disk into the plan, and the kernel image
    // path, before the plan is shown — so the operator reviews what will
    // really boot. A plan with no disk is refused here, with guidance.
    let kernel_path = finalize_plan(&mut plan)?;

    display_plan(&plan);

    if auto_approve || confirm_apply()? {
        apply_plan(&plan, &kernel_path)
    } else {
        println!("Plan cancelled — no changes were made.");
        Ok(())
    }
}

/// Provisions a VM directly from explicit specs — the headless path.
///
/// There is no AI round-trip and no interactive confirmation: the explicit
/// flags ARE the operator's approval. The hypervisor still enforces every
/// bound (`validate_spec`, the host capability checks), so a bad value is
/// rejected just as firmly as on the natural-language path.
fn provision_direct(
    name: String,
    vcpus: u32,
    memory: &str,
    network: Option<String>,
    disk: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let memory_mb = vmm::parse_memory_mb(memory)?;
    let mut plan = ExecutionPlan {
        vcpus,
        memory_mb,
        vm_name: name,
        network_bridge: network,
        disk_image_path: disk,
    };
    let kernel_path = finalize_plan(&mut plan)?;
    display_plan(&plan);
    apply_plan(&plan, &kernel_path)
}

/// Resolves a plan's root-filesystem disk before it is shown and applied.
///
/// The disk is filled in from the plan itself, the `AI_VMM_DISK` environment
/// variable, or the `disk_path` configuration entry. A VM with no root
/// filesystem cannot boot to a usable system, so a plan still missing a disk
/// after resolution is refused here — with guidance — rather than booting a
/// kernel that would panic. Returns the resolved kernel image path.
fn finalize_plan(plan: &mut ExecutionPlan) -> Result<String, Box<dyn std::error::Error>> {
    plan.disk_image_path = config::resolve_disk_path(plan.disk_image_path.as_deref());
    if plan.disk_image_path.is_none() {
        return Err(
            "this VM has no root-filesystem disk image — it could not boot to a \
                    usable system. Name a disk in your request, pass `--disk <path>` to \
                    `provision`, set the AI_VMM_DISK environment variable, or add a \
                    `disk_path` entry to credentials.toml. The README explains how to \
                    obtain a root filesystem."
                .into(),
        );
    }
    Ok(config::resolve_kernel_path())
}

/// Prints the execution plan in a Terraform-style review block.
fn display_plan(plan: &ExecutionPlan) {
    let network = plan
        .network_bridge
        .as_deref()
        .unwrap_or("none (isolated VM)");
    let storage = plan.disk_image_path.as_deref().unwrap_or("none");

    println!();
    println!("────────────────────────── Execution plan ──────────────────────────");
    println!("  🟢 Action      CREATE");
    println!("  🖥️  VM          {}", plan.vm_name);
    println!(
        "  ⚙️  Resources   {} vCPU, {} MiB RAM",
        plan.vcpus, plan.memory_mb
    );
    println!("  🌐 Network     {network}");
    println!("  💾 Storage     {storage}");
    println!("─────────────────────────────────────────────────────────────────────");
}

/// Asks the operator to approve the plan (human-in-the-loop).
///
/// Only an explicit "y"/"yes" (case-insensitive) approves; every other answer
/// — including an empty line — safely cancels. Carriage returns and surrounding
/// spaces are trimmed.
fn confirm_apply() -> Result<bool, Box<dyn std::error::Error>> {
    use std::io::Write;

    print!("\nApply this plan? [y/N] ");
    std::io::stdout().flush()?;

    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    let answer = answer.trim().to_ascii_lowercase();

    Ok(matches!(answer.as_str(), "y" | "yes"))
}

/// Applies an approved plan by calling the native KVM provisioning layer.
fn apply_plan(plan: &ExecutionPlan, kernel_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    println!("Applying plan...");
    // Record the declared VM before the (blocking) boot, so the ledger
    // reflects the plan even while the guest is still running.
    record_plan(plan);

    // A caller-owned stop flag, raised by Ctrl+C, asks the hypervisor for a
    // graceful shutdown: the VM winds down, the function returns, and the
    // lifecycle outcome is recorded — instead of the process being killed
    // mid-flight and the ledger left stale at `Booting`.
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || {
            eprintln!("\n[vmm] Ctrl+C — requesting a graceful shutdown...");
            stop.store(true, Ordering::SeqCst);
        })?;
    }

    let report = vmm::provision_kvm_machine_native(
        plan.vcpus,
        plan.memory_mb,
        &plan.vm_name,
        plan.network_bridge.as_deref(),
        plan.disk_image_path.as_deref(),
        kernel_path,
        &stop,
    );
    // Record the lifecycle outcome before surfacing any error: a guest that
    // exited — cleanly or on a Ctrl+C wind-down — is `Stopped`, a pipeline or
    // boot failure is `Failed`.
    let outcome = match &report {
        Ok(_) => registry::VmLifecycle::Stopped,
        Err(_) => registry::VmLifecycle::Failed,
    };
    mark_vm_state(&plan.vm_name, outcome);
    let report = report?;
    println!("[vmm] {report}");
    println!("✅ Plan applied successfully.");
    Ok(())
}

/// Records an applied plan in the VM ledger.
///
/// Best-effort: a ledger write failure is reported but never blocks
/// provisioning — the ledger is a convenience, not part of the boot path.
fn record_plan(plan: &ExecutionPlan) {
    match persist_to_ledger(plan) {
        Ok(()) => println!("[ledger] VM '{}' recorded.", plan.vm_name),
        Err(e) => eprintln!("[ledger] note: VM not recorded ({e})."),
    }
}

/// Loads the ledger, upserts `plan`, and writes it back.
fn persist_to_ledger(plan: &ExecutionPlan) -> Result<(), Box<dyn std::error::Error>> {
    let path = registry::ledger_path()?;
    let mut ledger = registry::Ledger::load_from(&path)?;
    ledger.record(plan);
    ledger.save_to(&path)
}

/// Updates a VM's lifecycle state in the ledger.
///
/// Best-effort, like [`record_plan`]: a ledger write failure is reported but
/// never blocks — the lifecycle state is a convenience, not part of the boot
/// path.
fn mark_vm_state(vm_name: &str, state: registry::VmLifecycle) {
    let result = (|| -> Result<bool, Box<dyn std::error::Error>> {
        let path = registry::ledger_path()?;
        let mut ledger = registry::Ledger::load_from(&path)?;
        let updated = ledger.set_state(vm_name, state);
        if updated {
            ledger.save_to(&path)?;
        }
        Ok(updated)
    })();
    match result {
        Ok(true) => println!("[ledger] VM '{vm_name}' is now {state}."),
        Ok(false) => {}
        Err(e) => eprintln!("[ledger] note: VM state not updated ({e})."),
    }
}

/// Prints the VMs recorded in the local ledger (Terraform-style `state list`).
fn list_vms() -> Result<(), Box<dyn std::error::Error>> {
    let ledger = registry::Ledger::load_from(&registry::ledger_path()?)?;
    let records = ledger.records();
    if records.is_empty() {
        println!("No VMs recorded yet — run: ai-vmm run \"<intent>\"");
        return Ok(());
    }
    println!("{} VM(s) in the ledger:", records.len());
    for record in records {
        let storage = record.disk_image_path.as_deref().unwrap_or("none");
        println!(
            "  • {:<24} [{:<7}] {} vCPU, {} MiB RAM, storage: {}",
            record.vm_name, record.state, record.vcpus, record.memory_mb, storage
        );
    }
    Ok(())
}

/// Prints one VM's full recorded plan (Terraform-style `state show`).
fn inspect_vm(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let ledger = registry::Ledger::load_from(&registry::ledger_path()?)?;
    let record = ledger
        .get(name)
        .ok_or_else(|| format!("no VM named '{name}' in the ledger"))?;
    let network = record
        .network_bridge
        .as_deref()
        .unwrap_or("none (isolated VM)");
    let storage = record.disk_image_path.as_deref().unwrap_or("none");
    println!("VM '{}'", record.vm_name);
    println!("  🔄 State       {}", record.state);
    println!(
        "  ⚙️  Resources   {} vCPU, {} MiB RAM",
        record.vcpus, record.memory_mb
    );
    println!("  🌐 Network     {network}");
    println!("  💾 Storage     {storage}");
    println!("  🕐 Applied at  {} (Unix time)", record.applied_at);
    Ok(())
}

/// Removes a VM from the local ledger (Terraform-style `state rm`).
fn forget_vm(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = registry::ledger_path()?;
    let mut ledger = registry::Ledger::load_from(&path)?;
    if !ledger.forget(name) {
        return Err(format!("no VM named '{name}' in the ledger").into());
    }
    ledger.save_to(&path)?;
    println!("VM '{name}' removed from the ledger.");
    Ok(())
}

/// Reads an Anthropic API key and stores it locally.
///
/// On an interactive terminal the key is read with echo disabled (via
/// `rpassword`), so the secret never appears on screen or in the scrollback.
/// When stdin is not a terminal — piped input or CI, e.g.
/// `echo $KEY | ai-vmm auth login` — the key is read directly from stdin
/// (there is nothing to echo, and `rpassword` would otherwise block on the
/// controlling terminal).
fn auth_login() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::IsTerminal;

    let key = if std::io::stdin().is_terminal() {
        println!("Paste your Anthropic API key, then press Enter.");
        println!("The input is hidden — it is not echoed to the terminal.");
        rpassword::prompt_password("API key> ")?
    } else {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        line
    };

    config::save_api_key(&key)?;
    let path = config::credentials_file_path()?;
    println!("API key stored at {}", path.display());
    println!("You can now run: ai-vmm run \"<your request>\"");
    Ok(())
}
