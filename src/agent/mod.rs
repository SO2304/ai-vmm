//! Agentic layer: a provider-agnostic client that turns a natural-language
//! intent into a reviewable execution plan.
//!
//! The same logic drives two back ends without any vendor SDK — only `reqwest`
//! and `serde_json`:
//!  * `Anthropic` — the hosted Claude Messages API;
//!  * `OpenAiCompatible` — any local server speaking `/v1/chat/completions`
//!    (Ollama, vLLM, llama.cpp), which makes a fully air-gapped deployment
//!    possible with no hosted dependency.
//!
//! The agent runs a bounded tool-use loop: the model may call `list_vms` and
//! `inspect_vm` to read the local VM registry before it commits to a
//! `provision_kvm_machine` plan, so a clone/modify request is grounded in the
//! real recorded specification rather than in assumptions.

pub mod prompts;

use crate::config::{AiProvider, ProviderSettings};
use crate::registry;
use serde::Serialize;
use serde_json::{json, Value};

/// Endpoint of the hosted Anthropic Messages API.
const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";

/// Targeted Anthropic API version.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default Anthropic reasoning model.
///
/// Note: the historical `claude-3-opus-20240229` model has been retired. We
/// target a current Claude 4.x model here. A configured `model` overrides it.
const ANTHROPIC_MODEL: &str = "claude-opus-4-7";

/// Default base URL for an OpenAI-compatible server — Ollama's local default.
const DEFAULT_OPENAI_BASE_URL: &str = "http://localhost:11434/v1";

/// Token budget for the model's response.
const MAX_TOKENS: u32 = 1024;

/// Hard cap on tool-use round-trips, so a misbehaving model cannot loop
/// forever between the registry tools without ever producing a plan.
const MAX_TURNS: usize = 6;

/// Name of the terminal tool: calling it yields the execution plan itself.
const PROVISION_TOOL: &str = "provision_kvm_machine";

/// A reviewable provisioning plan produced by the model.
///
/// [`ask_agent`] returns this instead of acting on the hardware: the operator
/// reviews and explicitly approves it before it is applied (a Terraform-style
/// plan / approve / apply workflow).
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionPlan {
    /// Number of virtual cores requested.
    pub vcpus: u32,
    /// Amount of RAM requested, in mebibytes.
    pub memory_mb: u32,
    /// Name of the virtual machine to create.
    pub vm_name: String,
    /// Host bridge interface to attach to, or `None` for an isolated VM.
    pub network_bridge: Option<String>,
    /// Disk image to attach as the root filesystem, or `None` for a diskless VM.
    pub disk_image_path: Option<String>,
}

/// One tool call requested by the model, in a provider-neutral form.
struct ToolCall {
    /// Provider-assigned call id, echoed back with the result.
    id: String,
    /// Tool name.
    name: String,
    /// Decoded arguments object (always a JSON object).
    arguments: Value,
}

/// The outcome of one model turn.
enum ModelTurn {
    /// The model replied with text only — no provisioning was requested.
    Text(String),
    /// The model requested one or more tool calls. `assistant_message` is the
    /// turn to replay into the conversation, already in the provider's shape.
    Calls {
        assistant_message: Value,
        calls: Vec<ToolCall>,
    },
}

/// The control plane's answer to a natural-language intent.
///
/// Either a reviewable provisioning plan, or a plain-text reply when the model
/// requested no provisioning. The CLI flattens this to `Option<ExecutionPlan>`
/// (see [`ask_agent`]); the HTTP control plane returns the text to the caller.
pub enum AgentReply {
    /// The model produced a provisioning plan.
    Plan(ExecutionPlan),
    /// The model replied with text only — no provisioning was requested.
    Text(String),
}

/// Submits a natural-language intent to the control plane and returns its
/// answer — a plan, or a plain-text reply — without printing anything.
///
/// The agent runs a bounded tool-use loop against the configured provider: the
/// model may call `list_vms` / `inspect_vm`, whose results are executed locally
/// against the VM registry and fed back, before it calls `provision_kvm_machine`
/// — the terminal tool whose arguments become the [`ExecutionPlan`]. The plan
/// is returned for human review; it is NOT applied here.
///
/// This is the form the HTTP control plane consumes: it surfaces the model's
/// text reply to the caller rather than writing it to the console.
pub async fn ask_agent_reply(
    user_intent: &str,
    settings: &ProviderSettings,
) -> Result<AgentReply, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let mut messages: Vec<Value> = vec![json!({ "role": "user", "content": user_intent })];

    for _ in 0..MAX_TURNS {
        let response = send_turn(&client, settings, &messages).await?;
        match parse_turn(settings.provider, &response)? {
            ModelTurn::Text(text) => return Ok(AgentReply::Text(text)),
            ModelTurn::Calls {
                assistant_message,
                calls,
            } => {
                // `provision_kvm_machine` is terminal: its arguments are the plan.
                if let Some(plan_call) = calls.iter().find(|c| c.name == PROVISION_TOOL) {
                    return Ok(AgentReply::Plan(build_plan(&plan_call.arguments)?));
                }
                // Registry tools: execute them and feed the results back so the
                // model can ground its plan in the real fleet state.
                println!(
                    "[agent] the model is consulting the VM registry ({} call(s))...",
                    calls.len()
                );
                messages.push(assistant_message);
                append_tool_results(settings.provider, &mut messages, &calls);
            }
        }
    }

    Err("the agent exhausted its tool-call budget without producing a plan".into())
}

/// Console form of [`ask_agent_reply`] used by the CLI: returns the plan, or
/// `Ok(None)` after printing the model's text reply to the operator.
///
/// The plan is returned for human review; it is NOT applied here.
pub async fn ask_agent(
    user_intent: &str,
    settings: &ProviderSettings,
) -> Result<Option<ExecutionPlan>, Box<dyn std::error::Error>> {
    match ask_agent_reply(user_intent, settings).await? {
        AgentReply::Plan(plan) => Ok(Some(plan)),
        AgentReply::Text(text) => {
            println!("[agent] the model did not request any provisioning.");
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                println!("[agent] reply: {trimmed}");
            }
            Ok(None)
        }
    }
}

/// Sends one conversation turn to the configured provider and returns the
/// parsed JSON response body.
async fn send_turn(
    client: &reqwest::Client,
    settings: &ProviderSettings,
    messages: &[Value],
) -> Result<Value, Box<dyn std::error::Error>> {
    let (url, request) = match settings.provider {
        AiProvider::Anthropic => {
            let key = settings
                .api_key
                .as_deref()
                .ok_or("the Anthropic provider requires an API key")?;
            let body = anthropic_request(settings, messages);
            let request = client
                .post(ANTHROPIC_API_URL)
                .header("x-api-key", key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .json(&body);
            (ANTHROPIC_API_URL.to_string(), request)
        }
        AiProvider::OpenAiCompatible => {
            let base = settings
                .base_url
                .as_deref()
                .unwrap_or(DEFAULT_OPENAI_BASE_URL);
            let url = format!("{}/chat/completions", base.trim_end_matches('/'));
            let body = openai_request(settings, messages)?;
            let mut request = client.post(&url);
            // A local air-gapped server (Ollama) needs no key; a hosted
            // OpenAI-compatible endpoint takes a bearer token when present.
            if let Some(key) = settings.api_key.as_deref() {
                request = request.header("authorization", format!("Bearer {key}"));
            }
            (url, request.json(&body))
        }
    };

    let response = request.send().await?;
    let status = response.status();
    let raw_body = response.text().await?;
    if !status.is_success() {
        return Err(
            format!("the AI provider returned HTTP {status} from {url}: {raw_body}").into(),
        );
    }
    serde_json::from_str(&raw_body)
        .map_err(|e| format!("unreadable AI response from {url}: {e} — body: {raw_body}").into())
}

/// Builds the request body for the Anthropic Messages API.
fn anthropic_request(settings: &ProviderSettings, messages: &[Value]) -> Value {
    let model = settings.model.as_deref().unwrap_or(ANTHROPIC_MODEL);
    json!({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": prompts::SYSTEM_PROMPT,
        "tools": anthropic_tools(),
        "messages": messages,
    })
}

/// Builds the request body for an OpenAI-compatible `/v1/chat/completions`
/// endpoint. The model identifier is mandatory here — a local server has no
/// universal default.
fn openai_request(
    settings: &ProviderSettings,
    messages: &[Value],
) -> Result<Value, Box<dyn std::error::Error>> {
    let model = settings.model.as_deref().ok_or(
        "the OpenAI-compatible provider needs a model — set AI_VMM_MODEL or the \
         `model` field in credentials.toml (e.g. \"llama3.1\")",
    )?;
    // OpenAI-compatible APIs carry the system prompt as the first message.
    let mut full_messages = Vec::with_capacity(messages.len() + 1);
    full_messages.push(json!({ "role": "system", "content": prompts::SYSTEM_PROMPT }));
    full_messages.extend_from_slice(messages);
    Ok(json!({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "tools": openai_tools(),
        "tool_choice": "auto",
        "messages": full_messages,
    }))
}

/// Wraps the neutral tool specs into the Anthropic tool format.
fn anthropic_tools() -> Value {
    let tools: Vec<Value> = prompts::tool_specs()
        .into_iter()
        .map(|spec| {
            json!({
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.parameters,
            })
        })
        .collect();
    Value::Array(tools)
}

/// Wraps the neutral tool specs into the OpenAI `function` tool format.
fn openai_tools() -> Value {
    let tools: Vec<Value> = prompts::tool_specs()
        .into_iter()
        .map(|spec| {
            json!({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            })
        })
        .collect();
    Value::Array(tools)
}

/// Parses one model turn from a provider's raw JSON response.
fn parse_turn(
    provider: AiProvider,
    response: &Value,
) -> Result<ModelTurn, Box<dyn std::error::Error>> {
    match provider {
        AiProvider::Anthropic => parse_anthropic_turn(response),
        AiProvider::OpenAiCompatible => parse_openai_turn(response),
    }
}

/// Parses an Anthropic Messages API response into a [`ModelTurn`].
fn parse_anthropic_turn(response: &Value) -> Result<ModelTurn, Box<dyn std::error::Error>> {
    let content = response
        .get("content")
        .and_then(Value::as_array)
        .ok_or("invalid Anthropic response: missing or malformed 'content' field")?;
    let stop_reason = response
        .get("stop_reason")
        .and_then(Value::as_str)
        .unwrap_or_default();

    if stop_reason != "tool_use" {
        return Ok(ModelTurn::Text(text_of_anthropic(content)));
    }

    let calls: Vec<ToolCall> = content
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
        .map(|block| ToolCall {
            id: block
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            name: block
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            arguments: block.get("input").cloned().unwrap_or_else(|| json!({})),
        })
        .collect();
    if calls.is_empty() {
        return Err("Anthropic stop_reason is 'tool_use' but no tool_use block was found".into());
    }
    Ok(ModelTurn::Calls {
        assistant_message: json!({ "role": "assistant", "content": content }),
        calls,
    })
}

/// Parses an OpenAI-compatible `/v1/chat/completions` response into a
/// [`ModelTurn`].
fn parse_openai_turn(response: &Value) -> Result<ModelTurn, Box<dyn std::error::Error>> {
    let message = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .ok_or("invalid OpenAI response: missing 'choices[0].message'")?;

    match message.get("tool_calls").and_then(Value::as_array) {
        Some(raw_calls) if !raw_calls.is_empty() => {
            let mut calls = Vec::with_capacity(raw_calls.len());
            for raw in raw_calls {
                let function = raw
                    .get("function")
                    .ok_or("an OpenAI tool_call is missing its 'function' field")?;
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                // OpenAI delivers the arguments as a JSON-encoded string.
                let raw_arguments = function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let arguments: Value = serde_json::from_str(raw_arguments)
                    .map_err(|e| format!("OpenAI tool '{name}' sent malformed arguments: {e}"))?;
                calls.push(ToolCall {
                    id: raw
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    name: name.to_string(),
                    arguments,
                });
            }
            Ok(ModelTurn::Calls {
                assistant_message: message.clone(),
                calls,
            })
        }
        _ => Ok(ModelTurn::Text(
            message
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        )),
    }
}

/// Executes the registry tool calls and appends the results to `messages` in
/// the provider's own conversation format.
fn append_tool_results(provider: AiProvider, messages: &mut Vec<Value>, calls: &[ToolCall]) {
    match provider {
        AiProvider::Anthropic => {
            // Anthropic carries every result as a block of one user message.
            let blocks: Vec<Value> = calls
                .iter()
                .map(|call| {
                    json!({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": execute_local_tool(&call.name, &call.arguments),
                    })
                })
                .collect();
            messages.push(json!({ "role": "user", "content": blocks }));
        }
        AiProvider::OpenAiCompatible => {
            // OpenAI carries each result as its own `tool` message.
            for call in calls {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": execute_local_tool(&call.name, &call.arguments),
                }));
            }
        }
    }
}

/// Executes one local registry tool, returning a string for the model.
///
/// Any failure is reported back to the model as a JSON `{"error": ...}` string
/// rather than aborting the agent, so the model can react and retry.
fn execute_local_tool(name: &str, arguments: &Value) -> String {
    match run_local_tool(name, arguments) {
        Ok(result) => result,
        Err(e) => json!({ "error": e.to_string() }).to_string(),
    }
}

/// Runs a registry tool against the on-disk VM ledger.
fn run_local_tool(name: &str, arguments: &Value) -> Result<String, Box<dyn std::error::Error>> {
    let ledger = registry::Ledger::load_from(&registry::ledger_path()?)?;
    match name {
        "list_vms" => {
            let names: Vec<&str> = ledger
                .records()
                .iter()
                .map(|record| record.vm_name.as_str())
                .collect();
            Ok(json!({ "vms": names }).to_string())
        }
        "inspect_vm" => {
            let vm_name = arguments
                .get("name")
                .and_then(Value::as_str)
                .ok_or("inspect_vm requires a string 'name' argument")?;
            match ledger.get(vm_name) {
                Some(record) => Ok(serde_json::to_string(record)?),
                None => Ok(
                    json!({ "error": format!("no VM named '{vm_name}' in the registry") })
                        .to_string(),
                ),
            }
        }
        other => Err(format!("the model called an unknown tool: '{other}'").into()),
    }
}

/// Type-converts the `provision_kvm_machine` arguments into an execution plan.
///
/// The plan is returned for human review; it is not executed here.
fn build_plan(input: &Value) -> Result<ExecutionPlan, Box<dyn std::error::Error>> {
    let vcpus: u32 = input
        .get("vcpus")
        .and_then(Value::as_u64)
        .ok_or("missing or non-integer 'vcpus' argument")?
        .try_into()
        .map_err(|_| "'vcpus' argument out of 32-bit integer range")?;

    let memory_mb: u32 = input
        .get("memory_mb")
        .and_then(Value::as_u64)
        .ok_or("missing or non-integer 'memory_mb' argument")?
        .try_into()
        .map_err(|_| "'memory_mb' argument out of 32-bit integer range")?;

    let vm_name = input
        .get("vm_name")
        .and_then(Value::as_str)
        .ok_or("missing or non-string 'vm_name' argument")?;

    // Optional arguments: a missing key, JSON null, an empty string or the
    // literal "none" all normalize to `None`.
    let network_bridge = normalize_optional(input.get("network_bridge").and_then(Value::as_str));
    let disk_image_path = normalize_optional(input.get("disk_image_path").and_then(Value::as_str));

    Ok(ExecutionPlan {
        vcpus,
        memory_mb,
        vm_name: vm_name.to_string(),
        network_bridge,
        disk_image_path,
    })
}

/// Normalizes an optional string argument coming from the model.
///
/// A missing value, an empty or whitespace-only string, or the literal
/// `"none"` (any case) all become `None`; otherwise the trimmed value is kept.
fn normalize_optional(value: Option<&str>) -> Option<String> {
    let trimmed = value?.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Concatenates all text blocks (`type == "text"`) of an Anthropic response.
fn text_of_anthropic(content: &[Value]) -> String {
    content
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Tests for the provider-neutral parsing, plan building and tool wrapping.
#[cfg(test)]
mod tests {
    use super::{
        anthropic_tools, build_plan, normalize_optional, openai_tools, parse_anthropic_turn,
        parse_openai_turn, ModelTurn,
    };
    use serde_json::json;

    #[test]
    fn normalizes_model_supplied_optionals() {
        assert_eq!(normalize_optional(None), None);
        assert_eq!(normalize_optional(Some("")), None);
        assert_eq!(normalize_optional(Some("   ")), None);
        assert_eq!(normalize_optional(Some("none")), None);
        assert_eq!(normalize_optional(Some("NONE")), None);
        assert_eq!(
            normalize_optional(Some("  ./rootfs.ext4  ")),
            Some("./rootfs.ext4".to_string())
        );
    }

    #[test]
    fn build_plan_extracts_every_field() {
        let plan = build_plan(&json!({
            "vcpus": 4,
            "memory_mb": 8192,
            "vm_name": "db-prod",
            "network_bridge": "br0",
            "disk_image_path": "./rootfs.ext4"
        }))
        .expect("a complete tool call yields a plan");
        assert_eq!(plan.vcpus, 4);
        assert_eq!(plan.memory_mb, 8192);
        assert_eq!(plan.vm_name, "db-prod");
        assert_eq!(plan.network_bridge.as_deref(), Some("br0"));
        assert_eq!(plan.disk_image_path.as_deref(), Some("./rootfs.ext4"));
    }

    #[test]
    fn build_plan_rejects_missing_required_field() {
        assert!(build_plan(&json!({ "memory_mb": 2048, "vm_name": "x" })).is_err());
    }

    #[test]
    fn parses_an_anthropic_tool_call() {
        let response = json!({
            "stop_reason": "tool_use",
            "content": [
                { "type": "text", "text": "Checking the registry." },
                { "type": "tool_use", "id": "tu_1", "name": "list_vms", "input": {} }
            ]
        });
        match parse_anthropic_turn(&response).expect("parse") {
            ModelTurn::Calls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "list_vms");
                assert_eq!(calls[0].id, "tu_1");
            }
            ModelTurn::Text(_) => panic!("expected a tool call"),
        }
    }

    #[test]
    fn parses_an_anthropic_text_reply() {
        let response = json!({
            "stop_reason": "end_turn",
            "content": [{ "type": "text", "text": "Hello operator." }]
        });
        match parse_anthropic_turn(&response).expect("parse") {
            ModelTurn::Text(text) => assert_eq!(text, "Hello operator."),
            ModelTurn::Calls { .. } => panic!("expected a text reply"),
        }
    }

    #[test]
    fn parses_an_openai_tool_call_with_string_arguments() {
        let response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "inspect_vm",
                            "arguments": "{\"name\":\"db-prod\"}"
                        }
                    }]
                }
            }]
        });
        match parse_openai_turn(&response).expect("parse") {
            ModelTurn::Calls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "inspect_vm");
                // OpenAI ships arguments as a JSON string; it must be decoded.
                assert_eq!(
                    calls[0].arguments.get("name").and_then(|v| v.as_str()),
                    Some("db-prod")
                );
            }
            ModelTurn::Text(_) => panic!("expected a tool call"),
        }
    }

    #[test]
    fn parses_an_openai_text_reply() {
        let response = json!({
            "choices": [{ "message": { "role": "assistant", "content": "Nothing to do." } }]
        });
        match parse_openai_turn(&response).expect("parse") {
            ModelTurn::Text(text) => assert_eq!(text, "Nothing to do."),
            ModelTurn::Calls { .. } => panic!("expected a text reply"),
        }
    }

    #[test]
    fn both_tool_formats_expose_the_three_tools() {
        let anthropic = anthropic_tools();
        let openai = openai_tools();
        assert_eq!(anthropic.as_array().map(Vec::len), Some(3));
        assert_eq!(openai.as_array().map(Vec::len), Some(3));
        // Anthropic uses `input_schema`; OpenAI nests under `function`.
        assert!(anthropic[0].get("input_schema").is_some());
        assert_eq!(
            openai[0].get("type").and_then(|v| v.as_str()),
            Some("function")
        );
        assert!(openai[0].get("function").is_some());
    }
}
