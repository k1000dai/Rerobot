# rerobot-cli

The `lerobot-*` executables of [Hugging Face LeRobot][upstream], ported to Rust
as part of [Rerobot].

[upstream]: https://github.com/huggingface/lerobot
[Rerobot]: https://github.com/k1000dai/Rerobot

All 18 upstream console entry points are installed under byte-identical names,
so deployment tooling and documentation that reference them keep working:

```shell
cargo install --path crates/rerobot-cli --locked
```

Exactly one of them runs at this milestone.

```shell
lerobot-info                 # a real port of lerobot.scripts.lerobot_info
lerobot-train --help         # works, and states that it is unimplemented
lerobot-train; echo $?       # -> 2
```

## Contract for not-yet-ported commands

A command that has not been ported never silently succeeds:

* stdout is empty;
* stderr carries exactly one line beginning `<name>: unsupported in Rerobot`,
  naming the upstream `module:function` it corresponds to;
* the process exits with status `2`.

`--help` and `--version` take precedence over every other argument, so help
stays reachable even for commands that cannot run. `--help` states the command's
compatibility status and the upstream version it targets.

```rust
use rerobot_cli::{dispatch, help_text, unsupported_message, EXIT_UNSUPPORTED};

let outcome = dispatch("lerobot-train", &[]);
assert_eq!(outcome.code, EXIT_UNSUPPORTED);
assert!(outcome.stdout.is_empty());
assert!(outcome.stderr.starts_with("lerobot-train: unsupported"));
assert!(outcome.stderr.contains("lerobot.scripts.lerobot_train:main"));
assert!(!outcome.stderr.contains('\n')); // one greppable line

// --help wins over anything else on the command line.
let args = vec!["--dataset.repo_id=x".to_string(), "--help".to_string()];
let outcome = dispatch("lerobot-record", &args);
assert_eq!(outcome.code, 0);
assert_eq!(outcome.stdout, help_text("lerobot-record"));

// The same message the executable prints is available programmatically.
assert_eq!(unsupported_message("lerobot-train"), dispatch("lerobot-train", &[]).stderr);
```

## `lerobot-info`

The report builder is a pure function of a probed [`info::Environment`], so it
is testable without touching the machine. It emits upstream's 15 `get_sys_info`
keys, in upstream's order, and no others — the point of this command is that its
output is comparable with what a Python user pastes into a bug report, so port
status is deliberately absent from it and lives in `--help` instead.

Keys that report Python package versions cannot apply to a Rust build; they are
reported as `N/A (not ported)` — distinct from upstream's `N/A`, which means
"looked, and it is not installed" — rather than invented.

```rust
use rerobot_cli::info::{report, sys_info, Environment, FfmpegProbe, NOT_PORTED};

let env = Environment {
    rerobot_version: "0.1.0".to_string(),
    upstream_version: "0.6.1".to_string(),
    platform: "macos-aarch64".to_string(),
    ffmpeg: FfmpegProbe::Ran("ffmpeg version 7.1 Copyright (c) 2000-2024".to_string()),
};

let pairs = sys_info(&env);
let ffmpeg = pairs.iter().find(|(k, _)| k == "FFmpeg version").unwrap();
assert_eq!(ffmpeg.1, "7.1");

let torch = pairs.iter().find(|(k, _)| k == "PyTorch version").unwrap();
assert_eq!(torch.1, NOT_PORTED);

// Markdown bullets, exactly like upstream's format_dict_for_markdown.
assert!(report(&env).starts_with("- LeRobot version: 0.6.1"));

// `ffmpeg` present but failing is not the same as `ffmpeg` absent: upstream
// reaches "N/A" only through its `shutil.which` check.
let broken = Environment { ffmpeg: FfmpegProbe::Failed, ..env };
let pairs = sys_info(&broken);
let ffmpeg = pairs.iter().find(|(k, _)| k == "FFmpeg version").unwrap();
assert_eq!(ffmpeg.1, "Installed (version parsing failed)");
```

See [`docs/compatibility.md`][compat] for the per-command status table. Every
`--help` output and every unsupported-command error prints that URL, so it stays
reachable for anyone who installed the executables without a checkout.

[compat]: https://github.com/k1000dai/Rerobot/blob/main/docs/compatibility.md

## License

Apache-2.0, matching upstream. See `LICENSE` and `NOTICE` in the repository
root.
