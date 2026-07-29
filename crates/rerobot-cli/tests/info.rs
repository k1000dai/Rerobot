//! Tests for the pure report builder behind `lerobot-info`.
//!
//! The key set and key order are compared against the pinned upstream source
//! `src/lerobot/scripts/lerobot_info.py` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1 — `get_sys_info` builds one dict in
//! a fixed order, and a bug report from a Rerobot user has to look familiar to
//! someone reading upstream issues.

use rerobot_cli::info::{report, sys_info, Environment, FfmpegProbe, NOT_PORTED};
use rerobot_compat::ENTRY_POINTS;

/// `get_sys_info`'s keys, in insertion order, transcribed from upstream.
const UPSTREAM_KEYS: &[&str] = &[
    "LeRobot version",
    "Platform",
    "Python version",
    "Huggingface Hub version",
    "Transformers version",
    "Datasets version",
    "Numpy version",
    "FFmpeg version",
    "PyTorch version",
    "Torchcodec version",
    "Is PyTorch built with CUDA support?",
    "Cuda version",
    "GPU model",
    "Using GPU in script?",
    "lerobot scripts",
];

fn env(ffmpeg: FfmpegProbe) -> Environment {
    Environment {
        rerobot_version: "0.1.0".to_string(),
        upstream_version: "0.6.1".to_string(),
        platform: "macos-aarch64".to_string(),
        ffmpeg,
    }
}

fn ran(stdout: &str) -> FfmpegProbe {
    FfmpegProbe::Ran(stdout.to_string())
}

fn value(e: &Environment, key: &str) -> String {
    sys_info(e)
        .into_iter()
        .find(|(k, _)| k == key)
        .unwrap_or_else(|| panic!("report has no {key:?} key"))
        .1
}

fn ffmpeg_version(e: &Environment) -> String {
    value(e, "FFmpeg version")
}

#[test]
fn the_report_has_exactly_the_upstream_keys_in_the_upstream_order() {
    let keys: Vec<String> = sys_info(&env(FfmpegProbe::NotFound))
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(keys, UPSTREAM_KEYS.to_vec());
}

#[test]
fn the_report_adds_no_keys_of_its_own() {
    // Rerobot-specific metadata belongs in `--help` and `docs/compatibility.md`,
    // not in a report whose whole purpose is to be comparable with upstream's.
    let keys: Vec<String> = sys_info(&env(FfmpegProbe::NotFound))
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(keys.len(), 15);
    for key in &keys {
        assert!(
            UPSTREAM_KEYS.contains(&key.as_str()),
            "{key:?} is not an upstream key"
        );
    }
}

#[test]
fn report_keys_are_unique() {
    let mut keys: Vec<String> = sys_info(&env(FfmpegProbe::NotFound))
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    let before = keys.len();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), before);
}

#[test]
fn using_gpu_in_script_is_upstreams_fill_in_placeholder() {
    // Upstream writes the literal `"<fill in>"`: it is a prompt to the person
    // pasting the report, not a probe, so it ports across unchanged.
    assert_eq!(
        value(&env(FfmpegProbe::NotFound), "Using GPU in script?"),
        "<fill in>"
    );
}

#[test]
fn ffmpeg_absent_from_path_reports_not_available() {
    // Upstream: `shutil.which("ffmpeg") is None` -> "N/A".
    assert_eq!(ffmpeg_version(&env(FfmpegProbe::NotFound)), "N/A");
}

#[test]
fn present_ffmpeg_reports_its_parsed_version() {
    let e = env(ran("ffmpeg version 7.1 Copyright (c) 2000-2024"));
    assert_eq!(ffmpeg_version(&e), "7.1");
}

#[test]
fn unparseable_ffmpeg_reports_the_upstream_sentinel() {
    assert_eq!(
        ffmpeg_version(&env(ran("garbage"))),
        "Installed (version parsing failed)"
    );
}

#[test]
fn ffmpeg_that_cannot_be_run_reports_the_parse_failed_sentinel_not_not_available() {
    // Upstream catches `subprocess.SubprocessError` -- which includes the
    // `CalledProcessError` raised by `check=True` on a non-zero exit -- and
    // returns the parse-failed sentinel. It reaches "N/A" only via `which`.
    assert_eq!(
        ffmpeg_version(&env(FfmpegProbe::Failed)),
        "Installed (version parsing failed)"
    );
}

#[test]
fn ffmpeg_that_runs_but_prints_nothing_reports_the_parse_failed_sentinel() {
    // Upstream: `"".splitlines()[0]` raises IndexError, same except branch.
    assert_eq!(
        ffmpeg_version(&env(ran(""))),
        "Installed (version parsing failed)"
    );
}

#[test]
fn python_only_keys_report_not_ported_rather_than_a_fabricated_version() {
    let e = env(FfmpegProbe::NotFound);
    for key in [
        "Python version",
        "Huggingface Hub version",
        "Transformers version",
        "Datasets version",
        "Numpy version",
        "PyTorch version",
        "Torchcodec version",
        "Is PyTorch built with CUDA support?",
        "Cuda version",
        "GPU model",
    ] {
        assert_eq!(value(&e, key), NOT_PORTED, "{key} must not be fabricated");
    }
}

#[test]
fn the_not_ported_sentinel_is_distinguishable_from_upstreams_n_a() {
    // Upstream's `N/A` means "checked, and it is not installed". Rerobot cannot
    // check at all, and says so rather than borrowing the same word.
    assert_eq!(NOT_PORTED, "N/A (not ported)");
    assert_ne!(NOT_PORTED, "N/A");
}

#[test]
fn the_lerobot_version_key_names_the_upstream_target_and_the_port_version() {
    // Upstream reports the installed `lerobot` distribution version. There is
    // no such distribution here, so the value states both the version this port
    // targets and the port's own version instead of inventing one.
    let e = env(FfmpegProbe::NotFound);
    assert_eq!(
        value(&e, "LeRobot version"),
        "0.6.1 (upstream target; Rerobot 0.1.0, a partial Rust port)"
    );
}

#[test]
fn the_platform_key_passes_the_probed_platform_through_unchanged() {
    let e = env(FfmpegProbe::NotFound);
    assert_eq!(value(&e, "Platform"), "macos-aarch64");
}

#[test]
fn the_scripts_key_is_a_python_style_list_of_executable_names() {
    // Upstream's value is `str([ep.name for ep in ...])`, i.e. a Python list
    // repr with single-quoted names.
    let e = env(FfmpegProbe::NotFound);
    let expected = format!(
        "[{}]",
        ENTRY_POINTS
            .iter()
            .map(|entry| format!("'{}'", entry.name))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert_eq!(value(&e, "lerobot scripts"), expected);
    assert!(expected.starts_with("['lerobot-calibrate', 'lerobot-find-cameras', "));
    assert!(expected.ends_with("'lerobot-rollout']"));
}

#[test]
fn the_scripts_key_carries_no_compatibility_status() {
    // Finding 5: upstream output must not be replaced with status metadata.
    // Status lives in `--help` and `docs/compatibility.md`.
    let scripts = value(&env(FfmpegProbe::NotFound), "lerobot scripts");
    for status in ["partial", "unimplemented", "hardware-gated", "implemented"] {
        assert!(
            !scripts.contains(status),
            "the scripts list must not carry {status:?}: {scripts}"
        );
    }
    assert!(!scripts.contains('='));
}

#[test]
fn the_scripts_key_lists_every_entry_point() {
    let scripts = value(&env(FfmpegProbe::NotFound), "lerobot scripts");
    for entry in ENTRY_POINTS {
        assert!(scripts.contains(entry.name), "missing {}", entry.name);
    }
    assert_eq!(scripts.matches(", ").count(), ENTRY_POINTS.len() - 1);
}

#[test]
fn report_renders_every_pair_as_a_markdown_bullet() {
    let text = report(&env(ran("ffmpeg version 7.1 x")));
    let pairs = sys_info(&env(ran("ffmpeg version 7.1 x")));
    assert_eq!(text.lines().count(), pairs.len());
    for ((k, v), line) in pairs.iter().zip(text.lines()) {
        assert_eq!(line, format!("- {k}: {v}"));
    }
}

#[test]
fn detect_produces_a_usable_environment() {
    let e = Environment::detect();
    assert_eq!(e.rerobot_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(e.upstream_version, "0.6.1");
    assert!(!e.platform.is_empty());
    assert!(!e.platform.contains("unknown-unknown"));
}
