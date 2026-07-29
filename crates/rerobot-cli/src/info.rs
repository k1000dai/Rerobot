//! `lerobot-info`: a real, dependency-free port of
//! `lerobot.scripts.lerobot_info`.
//!
//! The report carries upstream's key set, in upstream's order, and nothing else
//! — the whole point of this command is that its output is comparable with what
//! a Python user pastes into a bug report. Rerobot's own compatibility status is
//! deliberately *not* here; it lives in `<command> --help` and in
//! `docs/compatibility.md`.
//!
//! Upstream's report is mostly a list of installed Python package versions. A
//! Rust build has no such packages, so those keys keep their names but report
//! [`NOT_PORTED`] instead of a fabricated version. The keys that do carry
//! meaning — platform, `ffmpeg`, the console entry point inventory, and the
//! `Using GPU in script?` placeholder — are real.

use crate::which;
use rerobot_compat::{ENTRY_POINTS, UPSTREAM_VERSION};
use rerobot_core::sysinfo::{
    format_dict_for_markdown, parse_ffmpeg_version, FFMPEG_PARSE_FAILED, NOT_AVAILABLE,
};
use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;

/// Value reported for keys that only make sense for the Python distribution.
///
/// Distinct from upstream's `N/A`, which means "looked, and it is not
/// installed": Rerobot cannot look at all, and says so.
pub const NOT_PORTED: &str = "N/A (not ported)";

/// Outcome of probing `ffmpeg`, matching the three paths through upstream's
/// `get_ffmpeg_version`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfmpegProbe {
    /// `shutil.which("ffmpeg")` found nothing. Reported as `N/A`.
    NotFound,
    /// The binary ran to completion; this is its stdout.
    Ran(String),
    /// The binary resolved on `PATH` but could not be run to completion -- a
    /// spawn failure, a non-zero exit under `check=True`, or output that is not
    /// text. Upstream catches this as `subprocess.SubprocessError` and reports
    /// the parse-failed sentinel, not `N/A`; `N/A` is reachable only through the
    /// `which` check.
    Failed,
}

/// Everything `lerobot-info` needs from the outside world, so the report itself
/// stays a pure function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Environment {
    /// Rerobot workspace version.
    pub rerobot_version: String,
    /// Upstream version this port targets.
    pub upstream_version: String,
    /// Human-readable OS/architecture string.
    pub platform: String,
    /// Result of probing `ffmpeg -version`.
    pub ffmpeg: FfmpegProbe,
}

impl Environment {
    /// Probe the real machine.
    pub fn detect() -> Self {
        Self {
            rerobot_version: env!("CARGO_PKG_VERSION").to_string(),
            upstream_version: UPSTREAM_VERSION.to_string(),
            platform: detect_platform(),
            ffmpeg: detect_ffmpeg(),
        }
    }
}

/// Value for the `Platform` key.
///
/// Upstream calls `platform.platform()`, which composes an OS *release* string
/// (`macOS-15.0-arm64-arm-64bit`, `Linux-6.5.0-x86_64-with-glibc2.35`). Rust's
/// standard library exposes no OS release or libc version, so this reports the
/// two components it can know for certain instead of shelling out to `uname` or
/// guessing. Recorded as a deliberate divergence in `docs/compatibility.md`.
fn detect_platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Probe `ffmpeg -version` the way upstream does: resolve the name on `PATH`
/// first, then run what resolution produced.
fn detect_ffmpeg() -> FfmpegProbe {
    detect_ffmpeg_in(std::env::var_os("PATH").as_deref())
}

/// [`detect_ffmpeg`] against an explicit search path, so the resolve-then-run
/// split is testable without mutating the environment.
pub fn detect_ffmpeg_in(search_path: Option<&OsStr>) -> FfmpegProbe {
    match which::which_in("ffmpeg", search_path) {
        // Only `shutil.which` returning `None` reaches `N/A`.
        None => FfmpegProbe::NotFound,
        Some(path) => probe_ffmpeg_at(&path),
    }
}

/// Run an already-resolved `ffmpeg` and classify the result.
///
/// Every failure after resolution is [`FfmpegProbe::Failed`], never
/// [`FfmpegProbe::NotFound`]: upstream has a path by this point, so its report
/// says the tool is installed and the version could not be read.
pub fn probe_ffmpeg_at(path: &Path) -> FfmpegProbe {
    let Ok(output) = Command::new(path).arg("-version").output() else {
        // Spawn failure (missing, not executable, not a runnable image). Upstream
        // would raise `OSError` here, which its `except` clause does not cover;
        // reporting "installed, unreadable version" beats aborting the report.
        return FfmpegProbe::Failed;
    };
    if !output.status.success() {
        // `check=True` -> `CalledProcessError`, a `SubprocessError`.
        return FfmpegProbe::Failed;
    }
    match String::from_utf8(output.stdout) {
        Ok(stdout) => FfmpegProbe::Ran(stdout),
        // `text=True` would raise `UnicodeDecodeError`, also uncaught upstream.
        Err(_) => FfmpegProbe::Failed,
    }
}

/// The `lerobot scripts` value: upstream's `str([ep.name for ep in ...])`, i.e.
/// a Python list repr of the console entry point names in declaration order.
fn scripts_list() -> String {
    format!(
        "[{}]",
        ENTRY_POINTS
            .iter()
            .map(|e| format!("'{}'", e.name))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Ordered key/value report: exactly `get_sys_info`'s 15 keys, in its order.
pub fn sys_info(env: &Environment) -> Vec<(String, String)> {
    let ffmpeg = match &env.ffmpeg {
        FfmpegProbe::NotFound => NOT_AVAILABLE.to_string(),
        FfmpegProbe::Ran(stdout) => parse_ffmpeg_version(stdout),
        FfmpegProbe::Failed => FFMPEG_PARSE_FAILED.to_string(),
    };

    let pairs: Vec<(&str, String)> = vec![
        // Upstream reports the installed `lerobot` distribution version. There
        // is no such distribution here, so this names the version being targeted
        // and the port doing the targeting rather than inventing one.
        (
            "LeRobot version",
            format!(
                "{} (upstream target; Rerobot {}, a partial Rust port)",
                env.upstream_version, env.rerobot_version
            ),
        ),
        ("Platform", env.platform.clone()),
        ("Python version", NOT_PORTED.to_string()),
        ("Huggingface Hub version", NOT_PORTED.to_string()),
        ("Transformers version", NOT_PORTED.to_string()),
        ("Datasets version", NOT_PORTED.to_string()),
        ("Numpy version", NOT_PORTED.to_string()),
        ("FFmpeg version", ffmpeg),
        ("PyTorch version", NOT_PORTED.to_string()),
        ("Torchcodec version", NOT_PORTED.to_string()),
        (
            "Is PyTorch built with CUDA support?",
            NOT_PORTED.to_string(),
        ),
        ("Cuda version", NOT_PORTED.to_string()),
        ("GPU model", NOT_PORTED.to_string()),
        // A prompt for whoever pastes the report, not a probe, so it ports as-is.
        ("Using GPU in script?", "<fill in>".to_string()),
        ("lerobot scripts", scripts_list()),
    ];

    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// The full markdown report printed by `lerobot-info`.
pub fn report(env: &Environment) -> String {
    let pairs = sys_info(env);
    format_dict_for_markdown(pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())))
}
