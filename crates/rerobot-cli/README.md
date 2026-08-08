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

Three of them run at this milestone. The other fifteen exist and fail explicitly.

```shell
# `lerobot-info` is a full port of `lerobot.scripts.lerobot_info`.
lerobot-info

# `lerobot-train` trains for real, for one vertical slice: the ACT policy on a
# local LeRobot v3.0 dataset on disk, including embedded PNG/JPEG camera columns,
# on CPU. It writes upstream's
# checkpoint layout, which upstream can load back.
lerobot-train --help         # lists exactly which arguments it accepts
lerobot-train --dataset.repo_id=ID --dataset.root=DIR --output_dir=DIR \
              --policy.type=act --steps=1

# Anything outside that slice is refused with a reason, never ignored:
lerobot-train --policy.type=diffusion; echo $?   # -> 2
lerobot-train --wandb.project=demo;    echo $?   # -> 2

# A hardware-independent local deployment path loads a checkpoint and emits
# actions from dataset observations:
lerobot-rollout --policy.path=outputs/train/demo/checkpoints/000001/pretrained_model \\
                --dataset.root=path/to/dataset --steps=10

# The fifteen unported commands say so and exit 2:
lerobot-eval; echo $?        # -> 2
```

`lerobot-train` invoked with no arguments is a *usage* error (exit 64) naming the
first requirement it is missing, not an unsupported-command error: it is a
command that runs, so a missing argument is the user's, not the port's.

## Contract for not-yet-ported commands

Sixteen of the eighteen are not ported. None of them ever silently succeeds:

* stdout is empty;
* stderr carries exactly one line beginning `<name>: unsupported in Rerobot`,
  naming the upstream `module:function` it corresponds to;
* the process exits with status `2`.

`--help` and `--version` take precedence over every other argument, so help
stays reachable even for commands that cannot run. `--help` states the command's
compatibility status and the upstream version it targets.

```rust
use rerobot_cli::{dispatch, help_text, unsupported_message, EXIT_UNSUPPORTED};

let outcome = dispatch("lerobot-eval", &[]);
assert_eq!(outcome.code, EXIT_UNSUPPORTED);
assert!(outcome.stdout.is_empty());
assert!(outcome.stderr.starts_with("lerobot-eval: unsupported"));
assert!(outcome.stderr.contains("lerobot.scripts.lerobot_eval:main"));
assert!(!outcome.stderr.contains('\n')); // one greppable line

// `lerobot-train` is runnable for one slice, so a bare invocation is a *usage*
// error naming what is missing rather than an unsupported-command error.
let outcome = dispatch("lerobot-train", &[]);
assert_eq!(outcome.code, rerobot_cli::EXIT_USAGE);
assert!(outcome.stderr.contains("--policy.type=act is required"));

// --help wins over anything else on the command line.
let args = vec!["--dataset.repo_id=x".to_string(), "--help".to_string()];
let outcome = dispatch("lerobot-record", &args);
assert_eq!(outcome.code, 0);
assert_eq!(outcome.stdout, help_text("lerobot-record"));

// The same message the executable prints is available programmatically.
assert_eq!(unsupported_message("lerobot-eval"), dispatch("lerobot-eval", &[]).stderr);
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
