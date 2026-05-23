//! Local configuration store: BYOK credentials and AI-provider settings.
//!
//! The user configures their credentials once with `ai-vmm auth login`; they
//! are persisted to a per-user `credentials.toml` and reused afterwards. The
//! same file also carries the AI-provider settings — which back end to talk to
//! and, for an OpenAI-compatible server, its base URL and model.
//!
//! `ai-vmm` speaks to two kinds of back end:
//!  * `Anthropic` — the hosted Claude Messages API (needs an API key);
//!  * `OpenAiCompatible` — a local server (Ollama, vLLM, llama.cpp), which
//!    makes a fully air-gapped, on-premise deployment possible with no key.
//!
//! `ANTHROPIC_API_KEY`, `AI_VMM_PROVIDER`, `AI_VMM_BASE_URL` and `AI_VMM_MODEL`
//! override the stored file when set (intended for CI and air-gapped hosts).
//!
//! The pure key-length validation (`validate_key_len`) is allocation-free and
//! `format!`-free, so it is verified by the Kani harnesses at the bottom of
//! this file — mirroring the `vmm` module's approach.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Minimum accepted length of a trimmed API key, in bytes.
///
/// This is a sanity envelope (it rejects empty or obviously truncated input),
/// not a format check: it stays deliberately permissive about the exact shape
/// of the key so that future Anthropic key formats keep working.
const MIN_KEY_LEN: usize = 8;

/// Maximum accepted length of a trimmed API key, in bytes.
const MAX_KEY_LEN: usize = 8192;

/// Environment variable that, when set, overrides the stored API key.
const API_KEY_ENV_VAR: &str = "ANTHROPIC_API_KEY";

/// Environment variable that, when set, overrides the stored AI provider.
const PROVIDER_ENV_VAR: &str = "AI_VMM_PROVIDER";

/// Environment variable that, when set, overrides the stored provider URL.
const BASE_URL_ENV_VAR: &str = "AI_VMM_BASE_URL";

/// Environment variable that, when set, overrides the stored model identifier.
const MODEL_ENV_VAR: &str = "AI_VMM_MODEL";

/// Environment variable that, when set, overrides the kernel image path.
const KERNEL_PATH_ENV_VAR: &str = "AI_VMM_KERNEL";

/// Environment variable that, when set, overrides the default disk image path.
const DISK_PATH_ENV_VAR: &str = "AI_VMM_DISK";

/// Kernel image path used when neither the environment nor the configuration
/// file names one — preserves the original "drop a `vmlinux` in the working
/// directory" behaviour as a last resort.
const DEFAULT_KERNEL_PATH: &str = "./vmlinux";

/// The reasoning back end the control plane talks to.
///
/// `Anthropic` is the hosted Claude Messages API; `OpenAiCompatible` is any
/// server speaking the OpenAI `/v1/chat/completions` protocol — Ollama, vLLM
/// or llama.cpp — which needs no hosted dependency and so enables a fully
/// air-gapped deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AiProvider {
    /// The hosted Anthropic Claude Messages API.
    #[default]
    #[serde(rename = "anthropic")]
    Anthropic,
    /// A local OpenAI-compatible server (Ollama, vLLM, llama.cpp).
    #[serde(rename = "openai")]
    OpenAiCompatible,
}

/// On-disk configuration schema.
///
/// Intentionally does not derive `Debug`: the struct holds a secret, and a
/// derived `Debug` would risk leaking the key into logs or panic messages.
#[derive(Default, Serialize, Deserialize)]
struct Config {
    /// The Anthropic API key, if one has been stored.
    anthropic_api_key: Option<String>,
    /// Which AI back end to talk to.
    #[serde(default)]
    provider: AiProvider,
    /// Base URL of an OpenAI-compatible server (e.g. `http://localhost:11434/v1`).
    #[serde(default)]
    base_url: Option<String>,
    /// Model identifier to request from the provider.
    #[serde(default)]
    model: Option<String>,
    /// Path to the `vmlinux` ELF kernel image booted for every VM.
    #[serde(default)]
    kernel_path: Option<String>,
    /// Default root-filesystem disk image, used when a request names none.
    #[serde(default)]
    disk_path: Option<String>,
}

/// Resolved AI-provider settings consumed by the agent.
///
/// Like [`Config`], this intentionally does not derive `Debug`: it may hold
/// the API key, and a derived `Debug` could leak it into logs.
pub struct ProviderSettings {
    /// Which back end to talk to.
    pub provider: AiProvider,
    /// API key, if one is configured. An air-gapped local server needs none.
    pub api_key: Option<String>,
    /// Base URL of an OpenAI-compatible server; ignored for `Anthropic`.
    pub base_url: Option<String>,
    /// Model identifier; the agent applies a provider default when `None`.
    pub model: Option<String>,
}

/// Why a candidate API key was rejected by [`validate_key_len`].
///
/// A "flat" error type: no allocation, no dynamic formatting, so it stays
/// trivial to verify with Kani.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyError {
    /// The key is empty (or contained only whitespace).
    Empty,
    /// The key is shorter than [`MIN_KEY_LEN`].
    TooShort,
    /// The key is longer than [`MAX_KEY_LEN`].
    TooLong,
}

impl KeyError {
    /// Readable, static message describing the rejection cause.
    const fn as_str(self) -> &'static str {
        match self {
            KeyError::Empty => "the API key is empty",
            KeyError::TooShort => "the API key is too short to be valid",
            KeyError::TooLong => "the API key is unreasonably long",
        }
    }
}

impl std::fmt::Display for KeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for KeyError {}

/// Validates the length of a trimmed API key.
///
/// Pure, deterministic and allocation-free — verified by the Kani harnesses.
fn validate_key_len(len: usize) -> Result<(), KeyError> {
    if len == 0 {
        return Err(KeyError::Empty);
    }
    if len < MIN_KEY_LEN {
        return Err(KeyError::TooShort);
    }
    if len > MAX_KEY_LEN {
        return Err(KeyError::TooLong);
    }
    Ok(())
}

/// Trims surrounding whitespace from a candidate key and validates its length.
///
/// Returns the cleaned key slice on success. Trimming matters because keys
/// typed or pasted on a terminal carry a trailing newline.
fn check_key(key: &str) -> Result<&str, KeyError> {
    let trimmed = key.trim();
    validate_key_len(trimmed.len())?;
    Ok(trimmed)
}

/// Selects a usable, trimmed value from a raw environment-variable value.
///
/// A missing variable, or one that is empty or whitespace-only, yields `None`,
/// so the caller falls back to the stored configuration file.
fn nonempty_env_value(raw: Option<String>) -> Option<String> {
    let trimmed = raw?.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// Parses an [`AiProvider`] from a user-supplied string, accepting friendly
/// aliases (`claude`, `ollama`, `vllm`, ...). Returns `None` for an
/// unrecognised value.
fn parse_provider(raw: &str) -> Option<AiProvider> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "anthropic" | "claude" => Some(AiProvider::Anthropic),
        "openai" | "openai_compatible" | "ollama" | "vllm" | "llamacpp" => {
            Some(AiProvider::OpenAiCompatible)
        }
        _ => None,
    }
}

/// Friendly, actionable message shown when no usable key is configured.
const fn not_configured_message() -> &'static str {
    "no Anthropic API key configured — run `ai-vmm auth login` to store one, or \
     set AI_VMM_PROVIDER=openai to use a local air-gapped model instead"
}

/// Returns the absolute path of the `credentials.toml` file for this user.
///
/// Uses the platform's standard per-user configuration directory
/// (`~/.config/ai-vmm/` on Linux).
pub fn credentials_file_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dirs = directories::ProjectDirs::from("com", "ai-vmm", "ai-vmm")
        .ok_or("could not determine a configuration directory for this platform")?;
    Ok(dirs.config_dir().join("credentials.toml"))
}

/// Reads the stored configuration file, or a default config if none exists.
fn load_config() -> Result<Config, Box<dyn std::error::Error>> {
    let path = credentials_file_path()?;
    if !path.exists() {
        return Ok(Config::default());
    }
    let contents = std::fs::read_to_string(&path)?;
    Ok(toml::from_str(&contents)?)
}

/// Persists the Anthropic API key to the local credentials file.
///
/// Any AI-provider settings already in the file are preserved — only the key
/// is updated. Creates the configuration directory if needed, trims and
/// validates the key, then writes `credentials.toml`. On Unix the directory is
/// set to `0700` and the file to `0600`, so the secret is readable by its
/// owner only.
pub fn save_api_key(key: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let validated = check_key(key)?;
    let path = credentials_file_path()?;

    // Create the configuration directory, owner-only on Unix.
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
        }
    }

    // Preserve any provider settings already stored; only the key changes. A
    // corrupt or unreadable file falls back to defaults, so `auth login` can
    // always recover rather than failing to overwrite it.
    let mut config = load_config().unwrap_or_default();
    config.anthropic_api_key = Some(validated.to_string());
    let serialized = toml::to_string(&config)?;

    // Create the file owner-only (0600) from the start on Unix, so the secret
    // is never even briefly readable by other users.
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&path)?;
    file.write_all(serialized.as_bytes())?;

    // Re-assert 0600 in case the file already existed with looser permissions.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }

    Ok(())
}

/// Loads the resolved AI-provider settings.
///
/// The stored `credentials.toml` is overlaid with any environment overrides
/// (`ANTHROPIC_API_KEY`, `AI_VMM_PROVIDER`, `AI_VMM_BASE_URL`, `AI_VMM_MODEL`)
/// — intended for CI and air-gapped deployments. A key is mandatory for the
/// `Anthropic` provider; an `OpenAiCompatible` server may need none.
pub fn load_provider_settings() -> Result<ProviderSettings, Box<dyn std::error::Error>> {
    // 1. Start from the stored configuration file.
    let mut config = load_config()?;

    // 2. Apply environment overrides.
    if let Some(key) = nonempty_env_value(std::env::var(API_KEY_ENV_VAR).ok()) {
        config.anthropic_api_key = Some(key);
    }
    if let Some(raw) = nonempty_env_value(std::env::var(PROVIDER_ENV_VAR).ok()) {
        config.provider = parse_provider(&raw)
            .ok_or_else(|| format!("{PROVIDER_ENV_VAR} names an unknown provider: '{raw}'"))?;
    }
    if let Some(url) = nonempty_env_value(std::env::var(BASE_URL_ENV_VAR).ok()) {
        config.base_url = Some(url);
    }
    if let Some(model) = nonempty_env_value(std::env::var(MODEL_ENV_VAR).ok()) {
        config.model = Some(model);
    }

    // 3. Validate the key when present; require one for the Anthropic provider.
    let api_key = match config.anthropic_api_key.as_deref() {
        Some(raw) => Some(check_key(raw)?.to_string()),
        None => None,
    };
    if config.provider == AiProvider::Anthropic && api_key.is_none() {
        return Err(not_configured_message().into());
    }

    Ok(ProviderSettings {
        provider: config.provider,
        api_key,
        base_url: config.base_url,
        model: config.model,
    })
}

/// Cleans an explicit disk value: trims it, and treats an empty string or the
/// literal `none` (any case) as "no disk given" — `none` is what the agent
/// emits, and the headless flag, for a request that names no disk.
fn clean_explicit_disk(explicit: Option<&str>) -> Option<String> {
    let trimmed = explicit?.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Resolves the path to the `vmlinux` kernel image to boot.
///
/// Resolution order, first hit wins: the `AI_VMM_KERNEL` environment variable,
/// the `kernel_path` entry of `credentials.toml`, then `./vmlinux` in the
/// working directory. The path is not checked for existence here — the
/// provisioning layer reports a precise, actionable error if it is missing —
/// so this never fails, and `ai-vmm` works from any directory once the kernel
/// is configured once.
pub fn resolve_kernel_path() -> String {
    if let Some(env) = nonempty_env_value(std::env::var(KERNEL_PATH_ENV_VAR).ok()) {
        return env;
    }
    if let Some(path) = load_config().ok().and_then(|config| config.kernel_path) {
        return path;
    }
    DEFAULT_KERNEL_PATH.to_string()
}

/// Resolves the root-filesystem disk image for a VM.
///
/// Resolution order, first hit wins: an explicit path (a `--disk` flag, or the
/// disk a natural-language request named), the `AI_VMM_DISK` environment
/// variable, then the `disk_path` entry of `credentials.toml`. Returns `None`
/// when none is configured; the caller then refuses, with guidance, to
/// provision a VM that would have no root filesystem.
pub fn resolve_disk_path(explicit: Option<&str>) -> Option<String> {
    if let Some(path) = clean_explicit_disk(explicit) {
        return Some(path);
    }
    if let Some(env) = nonempty_env_value(std::env::var(DISK_PATH_ENV_VAR).ok()) {
        return Some(env);
    }
    load_config().ok().and_then(|config| config.disk_path)
}

/// Formal proofs checked by the Kani model checker (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::{validate_key_len, KeyError, MAX_KEY_LEN, MIN_KEY_LEN};

    /// Proof: an empty key is always rejected as [`KeyError::Empty`].
    #[kani::proof]
    fn proof_empty_key_is_rejected() {
        assert!(validate_key_len(0) == Err(KeyError::Empty));
    }

    /// Proof: any length within `[MIN_KEY_LEN, MAX_KEY_LEN]` is accepted.
    #[kani::proof]
    fn proof_valid_length_is_accepted() {
        let len: usize = kani::any();
        kani::assume(len >= MIN_KEY_LEN && len <= MAX_KEY_LEN);
        assert!(validate_key_len(len).is_ok());
    }

    /// Proof: when a length is accepted, it is necessarily within bounds.
    #[kani::proof]
    fn proof_accepted_length_is_within_bounds() {
        let len: usize = kani::any();
        if validate_key_len(len).is_ok() {
            assert!(len >= MIN_KEY_LEN && len <= MAX_KEY_LEN);
        }
    }
}

/// Tests for the pure validation, environment-selection and provider logic.
#[cfg(test)]
mod tests {
    use super::{
        check_key, clean_explicit_disk, nonempty_env_value, parse_provider, validate_key_len,
        AiProvider, KeyError, MAX_KEY_LEN, MIN_KEY_LEN,
    };

    #[test]
    fn rejects_empty_length() {
        assert_eq!(validate_key_len(0), Err(KeyError::Empty));
    }

    #[test]
    fn rejects_short_length() {
        assert_eq!(validate_key_len(MIN_KEY_LEN - 1), Err(KeyError::TooShort));
    }

    #[test]
    fn rejects_long_length() {
        assert_eq!(validate_key_len(MAX_KEY_LEN + 1), Err(KeyError::TooLong));
    }

    #[test]
    fn accepts_in_range_length() {
        assert!(validate_key_len(MIN_KEY_LEN).is_ok());
        assert!(validate_key_len(MAX_KEY_LEN).is_ok());
    }

    #[test]
    fn check_key_trims_surrounding_whitespace() {
        let cleaned = check_key("  sk-ant-example-key  \n").expect("valid padded key");
        assert_eq!(cleaned, "sk-ant-example-key");
    }

    #[test]
    fn check_key_rejects_blank_input() {
        assert!(check_key("   \n\t ").is_err());
    }

    #[test]
    fn selects_usable_value_from_env() {
        assert_eq!(nonempty_env_value(None), None);
        assert_eq!(nonempty_env_value(Some(String::new())), None);
        assert_eq!(nonempty_env_value(Some("   \n".to_string())), None);
        assert_eq!(
            nonempty_env_value(Some("  sk-ant-example  ".to_string())),
            Some("sk-ant-example".to_string())
        );
    }

    #[test]
    fn parses_provider_names_and_aliases() {
        assert_eq!(parse_provider("anthropic"), Some(AiProvider::Anthropic));
        assert_eq!(parse_provider("  Claude "), Some(AiProvider::Anthropic));
        assert_eq!(parse_provider("openai"), Some(AiProvider::OpenAiCompatible));
        assert_eq!(parse_provider("OLLAMA"), Some(AiProvider::OpenAiCompatible));
        assert_eq!(parse_provider("vllm"), Some(AiProvider::OpenAiCompatible));
        assert_eq!(parse_provider("nonsense"), None);
    }

    #[test]
    fn ai_provider_defaults_to_anthropic() {
        assert_eq!(AiProvider::default(), AiProvider::Anthropic);
    }

    #[test]
    fn clean_explicit_disk_trims_and_drops_placeholders() {
        // A real path is trimmed and kept.
        assert_eq!(
            clean_explicit_disk(Some("  ./rootfs.ext4 ")),
            Some("./rootfs.ext4".to_string())
        );
        // Absent, blank or the literal "none" all mean "no disk given".
        assert_eq!(clean_explicit_disk(None), None);
        assert_eq!(clean_explicit_disk(Some("   ")), None);
        assert_eq!(clean_explicit_disk(Some("none")), None);
        assert_eq!(clean_explicit_disk(Some("NONE")), None);
    }
}
