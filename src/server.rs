//! HTTP control-plane API — `ai-vmm serve`.
//!
//! Exposes the agentic planner and the VM ledger over HTTP so that automation
//! — and AI agents — can drive the control plane programmatically, not only
//! from the CLI.
//!
//! Every endpoint returns promptly. Planning is a single model round-trip and
//! the ledger endpoints are small file reads. Provisioning is supervised: the
//! daemon spawns each VM as its own `ai-vmm` worker process, so `POST /v1/vms`
//! returns `202 Accepted` at once while the guest boots in the background, its
//! console captured to a per-VM log file. `POST /v1/vms/{name}/stop` sends that
//! worker a graceful interrupt — it winds the VM down and records the outcome
//! in the ledger. A worker is a separate OS process, so one VM can never
//! destabilise the daemon or another VM.
//!
//! # Authentication
//!
//! Every `/v1/*` route requires `Authorization: Bearer <token>`, where the
//! token is the value of the `AI_VMM_API_TOKEN` environment variable. The
//! server refuses to start without it — the control plane is never served
//! unauthenticated. The token comparison is constant-time and Kani-proven to
//! compute exact equality, so it cannot be tricked into accepting a wrong
//! token. `/healthz` is the only unauthenticated route.

use crate::agent::{self, AgentReply, ExecutionPlan};
use crate::registry::{self, VmRecord};
use axum::{
    extract::{Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

/// Environment variable carrying the bearer token the server requires.
const API_TOKEN_ENV_VAR: &str = "AI_VMM_API_TOKEN";

/// Shared server state, reached through an `Arc` by every request handler.
struct ApiState {
    /// AI-provider settings used to answer planning requests.
    settings: crate::config::ProviderSettings,
    /// The bearer token every `/v1` request must present.
    api_token: String,
    /// PID of every VM worker the daemon has spawned and not yet reaped, keyed
    /// by VM name. A reaper thread removes an entry once its worker exits.
    running: Mutex<HashMap<String, u32>>,
    /// This daemon's own executable path — used to spawn VM worker processes.
    self_exe: PathBuf,
}

// --- Authentication primitives (pure, Kani-verified) -----------------------

/// Extracts the token from an HTTP `Authorization: Bearer <token>` value.
///
/// Returns `None` for any header that does not begin with the exact,
/// case-sensitive prefix `Bearer ` — `Basic`, lower-case `bearer`, or a bare
/// token are all rejected.
fn extract_bearer_token(header_value: &str) -> Option<&str> {
    header_value.strip_prefix("Bearer ")
}

/// Constant-time byte-string equality.
///
/// Every byte of equal-length inputs is compared regardless of where a
/// mismatch first occurs, so the comparison time does not leak how many leading
/// bytes of a guessed token were correct. `proof_constant_time_eq_is_exact`
/// proves it computes exactly `a == b`.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0_u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= *x ^ *y;
    }
    diff == 0
}

/// Returns whether a request carrying `presented` is authorized against the
/// server's `expected` token.
///
/// An empty `expected` authorizes nobody (defence in depth — the server already
/// refuses to start without a token), and an absent header is rejected.
fn request_is_authorized(presented: Option<&str>, expected: &str) -> bool {
    if expected.is_empty() {
        return false;
    }
    match presented {
        Some(token) => constant_time_eq(token.as_bytes(), expected.as_bytes()),
        None => false,
    }
}

// --- VM-name validation (pure, Kani-verified) ------------------------------

/// Largest VM name the supervisor accepts.
const MAX_VM_NAME_LEN: usize = 64;

/// Whether `byte` is allowed anywhere in a VM name.
///
/// ASCII alphanumerics plus `_`, `-`, `.`. Every other byte — a path separator,
/// NUL, whitespace, a shell metacharacter — is rejected, so the name is safe
/// both as a spawned-process argument and as a per-VM log filename.
const fn is_allowed_vm_name_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' || byte == b'.'
}

/// Whether `byte` is allowed as the FIRST byte of a VM name — an ASCII
/// alphanumeric. This guarantees the name can never be mistaken for a
/// `-`-prefixed command-line flag by the spawned worker's argument parser.
const fn is_safe_vm_name_first_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
}

/// Whether `name` is a safe VM name: non-empty, at most [`MAX_VM_NAME_LEN`]
/// bytes, starting with an ASCII alphanumeric, and containing only allowlisted
/// bytes. The supervisor passes the name to a spawned worker process and uses
/// it as a log filename, so it is validated before either use.
fn is_valid_vm_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_VM_NAME_LEN {
        return false;
    }
    is_safe_vm_name_first_byte(bytes[0]) && bytes.iter().all(|&b| is_allowed_vm_name_byte(b))
}

// --- Request / response bodies ---------------------------------------------

/// The request body of `POST /v1/plan`.
#[derive(Deserialize)]
struct PlanRequest {
    /// The natural-language provisioning intent.
    intent: String,
}

/// The request body of `POST /v1/vms`.
#[derive(Deserialize)]
struct ProvisionRequest {
    /// Name for the new VM.
    name: String,
    /// Number of virtual cores.
    vcpus: u32,
    /// RAM, in mebibytes.
    memory_mb: u32,
    /// Root-filesystem disk image; falls back to the configured default.
    #[serde(default)]
    disk_image_path: Option<String>,
}

/// JSON error body.
#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

/// The body of a successful `POST /v1/plan` response.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PlanOutcome {
    /// The model produced a provisioning plan.
    Plan { plan: ExecutionPlan },
    /// The model replied with text only — no provisioning was requested.
    Message { message: String },
}

/// The body of a successful `GET /v1/vms` response.
#[derive(Serialize)]
struct VmListBody {
    vms: Vec<VmRecord>,
}

/// Builds a JSON error response with the given status.
fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorBody {
            error: message.into(),
        }),
    )
        .into_response()
}

/// Checks the `Authorization` header against the server token. Returns the
/// `401` response to send back when the request is not authorized, or `None`
/// when the request may proceed.
fn check_auth(headers: &HeaderMap, expected: &str) -> Option<Response> {
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(extract_bearer_token);
    if request_is_authorized(presented, expected) {
        None
    } else {
        Some(error_response(
            StatusCode::UNAUTHORIZED,
            "missing or invalid bearer token",
        ))
    }
}

/// Loads the VM ledger from its per-user path.
fn load_ledger() -> Result<registry::Ledger, Box<dyn std::error::Error>> {
    let path = registry::ledger_path()?;
    registry::Ledger::load_from(&path)
}

// --- Handlers --------------------------------------------------------------

/// `GET /healthz` — unauthenticated liveness probe.
async fn healthz() -> Response {
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" }))).into_response()
}

/// `POST /v1/plan` — turn a natural-language intent into a reviewable plan.
async fn plan(
    State(state): State<Arc<ApiState>>,
    headers: HeaderMap,
    Json(request): Json<PlanRequest>,
) -> Response {
    if let Some(rejection) = check_auth(&headers, &state.api_token) {
        return rejection;
    }
    if request.intent.trim().is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "the 'intent' field must not be empty",
        );
    }
    match agent::ask_agent_reply(&request.intent, &state.settings).await {
        Ok(AgentReply::Plan(plan)) => {
            (StatusCode::OK, Json(PlanOutcome::Plan { plan })).into_response()
        }
        Ok(AgentReply::Text(text)) => (
            StatusCode::OK,
            Json(PlanOutcome::Message {
                message: text.trim().to_string(),
            }),
        )
            .into_response(),
        Err(e) => error_response(StatusCode::BAD_GATEWAY, e.to_string()),
    }
}

/// `GET /v1/vms` — list every VM recorded in the ledger.
async fn list_vms(State(state): State<Arc<ApiState>>, headers: HeaderMap) -> Response {
    if let Some(rejection) = check_auth(&headers, &state.api_token) {
        return rejection;
    }
    match load_ledger() {
        Ok(ledger) => (
            StatusCode::OK,
            Json(VmListBody {
                vms: ledger.records().to_vec(),
            }),
        )
            .into_response(),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `GET /v1/vms/:name` — the full recorded plan of one VM.
async fn get_vm(
    State(state): State<Arc<ApiState>>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Response {
    if let Some(rejection) = check_auth(&headers, &state.api_token) {
        return rejection;
    }
    match load_ledger() {
        Ok(ledger) => match ledger.get(&name) {
            Some(record) => (StatusCode::OK, Json(record.clone())).into_response(),
            None => error_response(StatusCode::NOT_FOUND, format!("no VM named '{name}'")),
        },
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `DELETE /v1/vms/:name` — remove a VM from the ledger.
async fn delete_vm(
    State(state): State<Arc<ApiState>>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Response {
    if let Some(rejection) = check_auth(&headers, &state.api_token) {
        return rejection;
    }
    let path = match registry::ledger_path() {
        Ok(path) => path,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let mut ledger = match registry::Ledger::load_from(&path) {
        Ok(ledger) => ledger,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    if !ledger.forget(&name) {
        return error_response(StatusCode::NOT_FOUND, format!("no VM named '{name}'"));
    }
    match ledger.save_to(&path) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// Path of a VM worker's console log file, under the per-user data directory.
fn vm_log_path(name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dirs = directories::ProjectDirs::from("com", "ai-vmm", "ai-vmm")
        .ok_or("could not determine a data directory for this platform")?;
    Ok(dirs.data_dir().join("logs").join(format!("{name}.log")))
}

/// Spawns the worker `ai-vmm provision` process for a VM, its console captured
/// to `log_path`.
///
/// The worker is a headless, separate process: it validates, boots, supervises
/// and records the lifecycle of exactly one VM, and a graceful interrupt makes
/// it wind that VM down. The VM name was already checked by [`is_valid_vm_name`]
/// before this is called, so it is safe both as an argument and as a filename.
fn spawn_worker(
    self_exe: &std::path::Path,
    request: &ProvisionRequest,
    disk: &str,
    log_path: &std::path::Path,
) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    if let Some(dir) = log_path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let log = std::fs::File::create(log_path)?;
    let log_err = log.try_clone()?;
    let child = Command::new(self_exe)
        .arg("provision")
        .arg("--name")
        .arg(&request.name)
        .arg("--vcpus")
        .arg(request.vcpus.to_string())
        .arg("--memory")
        .arg(request.memory_mb.to_string())
        .arg("--disk")
        .arg(disk)
        .stdin(Stdio::null())
        .stdout(log)
        .stderr(log_err)
        .spawn()?;
    Ok(child)
}

/// `POST /v1/vms` — provision and boot a VM as a supervised worker process.
async fn provision_vm(
    State(state): State<Arc<ApiState>>,
    headers: HeaderMap,
    Json(request): Json<ProvisionRequest>,
) -> Response {
    if let Some(rejection) = check_auth(&headers, &state.api_token) {
        return rejection;
    }
    // The name becomes a worker argument and a log filename — validate it.
    if !is_valid_vm_name(&request.name) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid VM name: 1-64 bytes, starting with a letter or digit, \
             containing only letters, digits, '_', '-' or '.'",
        );
    }
    // The vCPU / memory bounds are enforced and Kani-proven by `validate_spec`.
    if let Err(e) = crate::vmm::validate_spec(request.vcpus, request.memory_mb, None, None) {
        return error_response(StatusCode::BAD_REQUEST, e.to_string());
    }
    // Resolve the root-filesystem disk; a VM cannot boot without one.
    let disk = match crate::config::resolve_disk_path(request.disk_image_path.as_deref()) {
        Some(disk) => disk,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "no root-filesystem disk: set disk_image_path, or configure a default",
            )
        }
    };
    let log_path = match vm_log_path(&request.name) {
        Ok(path) => path,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };

    // Spawn under the registry lock: the duplicate check and the PID insert are
    // atomic, so two concurrent requests can never both start the same VM.
    let child = {
        let mut running = match state.running.lock() {
            Ok(running) => running,
            Err(_) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "supervisor registry mutex poisoned",
                )
            }
        };
        if running.contains_key(&request.name) {
            return error_response(
                StatusCode::CONFLICT,
                format!("a VM named '{}' is already running", request.name),
            );
        }
        match spawn_worker(&state.self_exe, &request, &disk, &log_path) {
            Ok(child) => {
                running.insert(request.name.clone(), child.id());
                child
            }
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        }
    };

    // Reaper: own the `Child`, wait for the worker to exit (reaping it so it
    // never lingers as a zombie), then drop the registry entry. Holding the
    // `Child` until `wait` also pins the PID, so a concurrent `stop` can never
    // signal a recycled PID while the entry exists.
    let reaper_state = Arc::clone(&state);
    let reaped_name = request.name.clone();
    std::thread::spawn(move || {
        let mut child = child;
        let _ = child.wait();
        if let Ok(mut running) = reaper_state.running.lock() {
            running.remove(&reaped_name);
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "name": request.name,
            "status": "provisioning",
            "log": log_path.display().to_string(),
        })),
    )
        .into_response()
}

/// `POST /v1/vms/:name/stop` — ask a running VM worker to shut down gracefully.
async fn stop_vm(
    State(state): State<Arc<ApiState>>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Response {
    if let Some(rejection) = check_auth(&headers, &state.api_token) {
        return rejection;
    }
    let pid = {
        let running = match state.running.lock() {
            Ok(running) => running,
            Err(_) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "supervisor registry mutex poisoned",
                )
            }
        };
        match running.get(&name) {
            Some(&pid) => pid,
            None => {
                return error_response(
                    StatusCode::NOT_FOUND,
                    format!("no running VM named '{name}'"),
                )
            }
        }
    };
    // A graceful interrupt: the worker's Ctrl+C handler winds the VM down and
    // records the outcome. The PID is pinned by the reaper's `Child` handle
    // until the worker exits, so this is never a recycled PID.
    match Command::new("kill")
        .arg("-INT")
        .arg(pid.to_string())
        .output()
    {
        Ok(output) if output.status.success() => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({ "name": name, "status": "stopping" })),
        )
            .into_response(),
        Ok(output) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "could not signal the VM worker: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("could not run 'kill': {e}"),
        ),
    }
}

// --- Server ----------------------------------------------------------------

/// Runs the HTTP control-plane API server, binding `addr` (e.g.
/// `127.0.0.1:8080`).
///
/// Refuses to start unless `AI_VMM_API_TOKEN` is set to a non-empty secret —
/// the control plane is never served unauthenticated.
pub async fn serve(addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let api_token = std::env::var(API_TOKEN_ENV_VAR)
        .ok()
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
        .ok_or("set AI_VMM_API_TOKEN to a non-empty secret before starting the server")?;

    let settings = crate::config::load_provider_settings()?;
    let state = Arc::new(ApiState {
        settings,
        api_token,
        running: Mutex::new(HashMap::new()),
        self_exe: std::env::current_exe()?,
    });

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/plan", post(plan))
        .route("/v1/vms", get(list_vms).post(provision_vm))
        .route("/v1/vms/:name", get(get_vm).delete(delete_vm))
        .route("/v1/vms/:name/stop", post(stop_vm))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("[server] ai-vmm control plane listening on http://{addr}");
    println!("[server] /v1 routes require: Authorization: Bearer <AI_VMM_API_TOKEN>");
    axum::serve(listener, app).await?;
    Ok(())
}

/// Formal proofs checked by the Kani model checker (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::{constant_time_eq, is_allowed_vm_name_byte, is_safe_vm_name_first_byte};

    /// Proof: the constant-time comparison computes *exactly* byte-string
    /// equality — it accepts the matching token and rejects every other. A
    /// short-circuit or masking bug here would be a silent authentication
    /// bypass.
    #[kani::proof]
    fn proof_constant_time_eq_is_exact() {
        let a: [u8; 8] = kani::any();
        let b: [u8; 8] = kani::any();
        assert!(constant_time_eq(&a, &b) == (a == b));
    }

    /// Proof: inputs of different lengths never compare equal, in either order.
    #[kani::proof]
    fn proof_constant_time_eq_rejects_length_mismatch() {
        let short: [u8; 4] = kani::any();
        let long: [u8; 8] = kani::any();
        assert!(!constant_time_eq(&short, &long));
        assert!(!constant_time_eq(&long, &short));
    }

    /// Proof: a validated VM name's first byte is alphanumeric — never `-` — so
    /// the name passed to a spawned worker can never be mistaken for a
    /// command-line flag (argument injection).
    #[kani::proof]
    fn proof_vm_name_first_byte_blocks_flag_injection() {
        let byte: u8 = kani::any();
        if is_safe_vm_name_first_byte(byte) {
            assert!(byte != b'-');
            assert!(byte.is_ascii_alphanumeric());
        }
    }

    /// Proof: no byte allowed in a VM name is a path separator or NUL — so the
    /// name is always safe to use as a per-VM log filename, with no path
    /// traversal.
    #[kani::proof]
    fn proof_vm_name_byte_excludes_path_separators() {
        let byte: u8 = kani::any();
        if is_allowed_vm_name_byte(byte) {
            assert!(byte != b'/');
            assert!(byte != 0);
        }
    }
}

/// Tests for the pure authentication primitives.
#[cfg(test)]
mod tests {
    use super::{constant_time_eq, extract_bearer_token, request_is_authorized};

    #[test]
    fn constant_time_eq_matches_byte_equality() {
        assert!(constant_time_eq(b"secret-token", b"secret-token"));
        assert!(!constant_time_eq(b"secret-token", b"secret-toxen"));
        assert!(!constant_time_eq(b"short", b"longer-value"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn extract_bearer_token_requires_the_exact_prefix() {
        assert_eq!(extract_bearer_token("Bearer abc123"), Some("abc123"));
        assert_eq!(extract_bearer_token("Bearer "), Some(""));
        // The prefix is exact and case-sensitive.
        assert_eq!(extract_bearer_token("bearer abc123"), None);
        assert_eq!(extract_bearer_token("Basic abc123"), None);
        assert_eq!(extract_bearer_token("abc123"), None);
        assert_eq!(extract_bearer_token(""), None);
    }

    #[test]
    fn authorization_requires_an_exact_token_match() {
        assert!(request_is_authorized(Some("right-token"), "right-token"));
        assert!(!request_is_authorized(Some("wrong-token"), "right-token"));
        assert!(!request_is_authorized(None, "right-token"));
        // An unset (empty) server token authorizes nobody.
        assert!(!request_is_authorized(Some(""), ""));
        assert!(!request_is_authorized(None, ""));
    }

    #[test]
    fn vm_name_validation_accepts_safe_names_and_rejects_unsafe() {
        use super::is_valid_vm_name;
        assert!(is_valid_vm_name("db-prod"));
        assert!(is_valid_vm_name("web01"));
        assert!(is_valid_vm_name("vm.test_3"));
        // Empty, over-long, or a non-alphanumeric first byte are rejected.
        assert!(!is_valid_vm_name(""));
        assert!(!is_valid_vm_name(&"a".repeat(65)));
        assert!(!is_valid_vm_name("-rf")); // would be parsed as a flag
        assert!(!is_valid_vm_name(".hidden"));
        // Path separators and metacharacters are rejected.
        assert!(!is_valid_vm_name("a/b"));
        assert!(!is_valid_vm_name("a b"));
        assert!(!is_valid_vm_name("a;b"));
    }
}
