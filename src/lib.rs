//! `ai-vmm` — Agentic Hypervisor, library crate.
//!
//! This crate holds every module of `ai-vmm`. The `ai-vmm` binary
//! (`src/main.rs`) is a thin CLI layered on top of it, and the integration
//! tests in `tests/` drive these same modules directly — which is only
//! possible because the code lives in a library target rather than inside the
//! binary.

pub mod agent;
pub mod config;
pub mod registry;
pub mod server;
pub mod vmm;
