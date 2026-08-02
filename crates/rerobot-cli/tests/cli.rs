//! End-to-end tests for the `lerobot-*` executables.

use rerobot_cli::{
    dispatch, help_text, unsupported_message, version_line, COMPATIBILITY_URL, EXIT_UNSUPPORTED,
    REPOSITORY,
};
use rerobot_compat::ENTRY_POINTS;
use std::process::{Command, Output};

fn arg(s: &str) -> Vec<String> {
    vec![s.to_string()]
}

fn run(bin: &str, args: &[&str]) -> Output {
    Command::new(bin)
        .args(args)
        .output()
        .expect("executable runs")
}

/// Path of a built `lerobot-*` executable.
fn bin_path(name: &str) -> &'static str {
    match name {
        "lerobot-info" => env!("CARGO_BIN_EXE_lerobot-info"),
        "lerobot-calibrate" => env!("CARGO_BIN_EXE_lerobot-calibrate"),
        "lerobot-find-cameras" => env!("CARGO_BIN_EXE_lerobot-find-cameras"),
        "lerobot-find-port" => env!("CARGO_BIN_EXE_lerobot-find-port"),
        "lerobot-record" => env!("CARGO_BIN_EXE_lerobot-record"),
        "lerobot-replay" => env!("CARGO_BIN_EXE_lerobot-replay"),
        "lerobot-setup-motors" => env!("CARGO_BIN_EXE_lerobot-setup-motors"),
        "lerobot-teleoperate" => env!("CARGO_BIN_EXE_lerobot-teleoperate"),
        "lerobot-eval" => env!("CARGO_BIN_EXE_lerobot-eval"),
        "lerobot-train" => env!("CARGO_BIN_EXE_lerobot-train"),
        "lerobot-train-tokenizer" => env!("CARGO_BIN_EXE_lerobot-train-tokenizer"),
        "lerobot-dataset-viz" => env!("CARGO_BIN_EXE_lerobot-dataset-viz"),
        "lerobot-find-joint-limits" => env!("CARGO_BIN_EXE_lerobot-find-joint-limits"),
        "lerobot-imgtransform-viz" => env!("CARGO_BIN_EXE_lerobot-imgtransform-viz"),
        "lerobot-edit-dataset" => env!("CARGO_BIN_EXE_lerobot-edit-dataset"),
        "lerobot-setup-can" => env!("CARGO_BIN_EXE_lerobot-setup-can"),
        "lerobot-annotate" => env!("CARGO_BIN_EXE_lerobot-annotate"),
        "lerobot-rollout" => env!("CARGO_BIN_EXE_lerobot-rollout"),
        other => panic!("no executable built for {other}"),
    }
}

#[test]
fn every_upstream_entry_point_is_built_under_its_upstream_name() {
    for e in ENTRY_POINTS {
        let path = bin_path(e.name);
        assert!(
            std::path::Path::new(path).exists(),
            "{} was not built at {path}",
            e.name
        );
        let file = std::path::Path::new(path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            file.starts_with(e.name),
            "{path} does not deploy as {}",
            e.name
        );
    }
}

#[test]
fn help_works_for_every_entry_point_and_states_its_status() {
    for e in ENTRY_POINTS {
        let out = run(bin_path(e.name), &["--help"]);
        assert!(out.status.success(), "{} --help must exit 0", e.name);
        let text = String::from_utf8(out.stdout).unwrap();
        assert!(
            text.contains(e.name),
            "{} --help omits its own name",
            e.name
        );
        assert!(
            text.contains(e.status.as_str()),
            "{} --help omits its compatibility status",
            e.name
        );
        assert!(
            text.contains("lerobot 0.6.1"),
            "{} --help omits the upstream version it targets",
            e.name
        );
    }
}

#[test]
fn short_help_flag_is_accepted() {
    let out = run(bin_path("lerobot-train"), &["-h"]);
    assert!(out.status.success());
    assert!(String::from_utf8(out.stdout)
        .unwrap()
        .contains("lerobot-train"));
}

#[test]
fn version_works_for_every_entry_point() {
    for e in ENTRY_POINTS {
        let out = run(bin_path(e.name), &["--version"]);
        assert!(out.status.success(), "{} --version must exit 0", e.name);
        let text = String::from_utf8(out.stdout).unwrap();
        assert!(text.starts_with(e.name));
        assert!(text.contains(env!("CARGO_PKG_VERSION")));
    }
}

#[test]
fn unported_commands_exit_nonzero_with_a_stable_error() {
    for e in ENTRY_POINTS.iter().filter(|e| e.status.is_unsupported()) {
        let out = run(bin_path(e.name), &[]);
        assert_eq!(
            out.status.code(),
            Some(EXIT_UNSUPPORTED),
            "{} must exit {EXIT_UNSUPPORTED}",
            e.name
        );
        assert!(out.stdout.is_empty(), "{} must not print to stdout", e.name);
        let err = String::from_utf8(out.stderr).unwrap();
        assert!(
            err.starts_with(&format!("{}: unsupported", e.name)),
            "{} stderr was {err:?}",
            e.name
        );
    }
}

#[test]
fn unported_commands_stay_unsupported_even_with_arguments() {
    let out = run(
        bin_path("lerobot-record"),
        &["--robot.type=so101_follower", "-x"],
    );
    assert_eq!(out.status.code(), Some(EXIT_UNSUPPORTED));
    assert!(String::from_utf8(out.stderr)
        .unwrap()
        .contains("not implemented"));
}

#[test]
fn sixteen_of_eighteen_commands_are_unsupported() {
    let unsupported = ENTRY_POINTS
        .iter()
        .filter(|e| e.status.is_unsupported())
        .count();
    assert_eq!(unsupported, 16);
    // The two that are not: `lerobot-info` in full, and `lerobot-train` for the
    // ACT local-dataset slice, including embedded PNG/JPEG camera columns.
    let runnable: Vec<&str> = ENTRY_POINTS
        .iter()
        .filter(|e| !e.status.is_unsupported())
        .map(|e| e.name)
        .collect();
    assert_eq!(runnable, vec!["lerobot-train", "lerobot-info"]);
}

#[test]
fn lerobot_info_runs_end_to_end() {
    let out = run(bin_path("lerobot-info"), &[]);
    assert!(out.status.success(), "lerobot-info must exit 0");
    let text = String::from_utf8(out.stdout).unwrap();
    for key in [
        "- LeRobot version:",
        "- Platform:",
        "- FFmpeg version:",
        "- Using GPU in script?: <fill in>",
        "- lerobot scripts:",
    ] {
        assert!(
            text.contains(key),
            "lerobot-info output missing {key:?}:\n{text}"
        );
    }
    for line in text.lines() {
        assert!(line.starts_with("- "), "unexpected line {line:?}");
    }
}

#[test]
fn lerobot_info_lists_all_eighteen_scripts() {
    let out = run(bin_path("lerobot-info"), &[]);
    let text = String::from_utf8(out.stdout).unwrap();
    for e in ENTRY_POINTS {
        assert!(
            text.contains(e.name),
            "lerobot-info does not list {}",
            e.name
        );
    }
}

#[test]
fn lerobot_info_rejects_unknown_flags() {
    let out = run(bin_path("lerobot-info"), &["--nope"]);
    assert_eq!(out.status.code(), Some(rerobot_cli::EXIT_USAGE));
    assert!(String::from_utf8(out.stderr).unwrap().contains("--nope"));
}

#[test]
fn dispatch_help_is_pure_and_matches_help_text() {
    let outcome = dispatch("lerobot-train", &arg("--help"));
    assert_eq!(outcome.code, 0);
    assert_eq!(outcome.stdout, help_text("lerobot-train"));
    assert!(outcome.stderr.is_empty());
}

#[test]
fn dispatch_version_matches_version_line() {
    let outcome = dispatch("lerobot-eval", &arg("--version"));
    assert_eq!(outcome.code, 0);
    assert_eq!(outcome.stdout, version_line("lerobot-eval"));
}

#[test]
fn dispatch_of_an_unsupported_command_matches_unsupported_message() {
    let outcome = dispatch("lerobot-eval", &[]);
    assert_eq!(outcome.code, EXIT_UNSUPPORTED);
    assert_eq!(outcome.stderr, unsupported_message("lerobot-eval"));
    assert!(outcome.stdout.is_empty());
}

#[test]
fn help_takes_precedence_over_other_arguments() {
    let args = vec!["--dataset.repo_id=x".to_string(), "--help".to_string()];
    assert_eq!(dispatch("lerobot-record", &args).code, 0);
}

#[test]
fn dispatch_rejects_an_unknown_command_name() {
    let outcome = dispatch("lerobot-nope", &[]);
    assert_eq!(outcome.code, rerobot_cli::EXIT_USAGE);
    assert!(outcome.stderr.contains("lerobot-nope"));
}

#[test]
fn unsupported_message_names_the_upstream_target() {
    let msg = unsupported_message("lerobot-train");
    assert!(msg.contains("lerobot.scripts.lerobot_train:main"));
    assert!(msg.contains("not implemented"));
    assert!(
        !msg.contains('\n'),
        "the error must stay a single greppable line"
    );
}

#[test]
fn hardware_gated_commands_say_so() {
    let msg = unsupported_message("lerobot-calibrate");
    assert!(msg.contains("hardware-gated"));
}

#[test]
fn the_unsupported_message_points_at_a_resolvable_repository_url() {
    // A local `docs/compatibility.md` path is useless to someone who installed
    // the executables with `cargo install` and has no checkout.
    for e in ENTRY_POINTS.iter().filter(|e| e.status.is_unsupported()) {
        let msg = unsupported_message(e.name);
        assert!(
            msg.contains(REPOSITORY),
            "{} does not name {REPOSITORY}: {msg}",
            e.name
        );
        assert!(
            !msg.contains('\n'),
            "{} must stay a single greppable line",
            e.name
        );
    }
}

#[test]
fn help_points_at_a_resolvable_repository_url_not_only_a_local_path() {
    for e in ENTRY_POINTS {
        let text = help_text(e.name);
        assert!(
            text.contains(COMPATIBILITY_URL),
            "{} --help does not link {COMPATIBILITY_URL}",
            e.name
        );
        assert!(
            text.contains(REPOSITORY),
            "{} --help does not name the repository",
            e.name
        );
    }
    assert!(
        COMPATIBILITY_URL.starts_with(REPOSITORY),
        "the docs link must live under the repository URL"
    );
    assert!(REPOSITORY.starts_with("https://"));
}

#[test]
fn the_repository_url_matches_the_published_package_metadata() {
    // `cargo install`ed users reach the project through this URL, so it has to
    // be the same one crates.io shows.
    assert_eq!(REPOSITORY, env!("CARGO_PKG_REPOSITORY"));
}

#[test]
fn every_executable_prints_a_repository_url_in_its_help() {
    for e in ENTRY_POINTS {
        let out = run(bin_path(e.name), &["--help"]);
        assert!(out.status.success());
        let text = String::from_utf8(out.stdout).unwrap();
        assert!(
            text.contains(COMPATIBILITY_URL),
            "{} --help omits the compatibility URL",
            e.name
        );
    }
}
