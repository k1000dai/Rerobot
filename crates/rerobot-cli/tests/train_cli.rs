//! End-to-end tests for the `lerobot-train` executable: the real binary, run as a
//! subprocess against the committed dataset fixture.
//!
//! This is where "the command trains" is asserted about the *command* rather than
//! about the library: a user typing the documented invocation gets a checkpoint on
//! disk and exit status zero, and a user typing anything outside the slice gets a
//! non-zero status and a message naming what is missing.
//!
//! The dataset fixture lives in `rerobot-train`, which is why this file is excluded
//! from `rerobot-cli`'s published archive — the same arrangement
//! `rerobot-compat/tests/docs_consistency.rs` uses for the root compatibility
//! ledger.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU32, Ordering};

/// The executable under test, as cargo built it for this integration test.
fn bin_path(name: &str) -> PathBuf {
    let mut path = std::env::current_exe().expect("the test binary has a path");
    path.pop(); // the deps/ directory
    path.pop(); // the target profile directory
    path.push(name);
    if cfg!(windows) {
        path.set_extension("exe");
    }
    assert!(
        path.is_file(),
        "{} was not built; run `cargo test` rather than the binary directly",
        path.display()
    );
    path
}

fn fixture_dataset() -> PathBuf {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../rerobot-train/tests/fixtures/state_only");
    assert!(
        path.join("meta/info.json").is_file(),
        "the dataset fixture is missing at {}",
        path.display()
    );
    path
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rerobot-train-cli-{}-{label}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("cannot create the test directory");
        Self(path)
    }

    fn child(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(args: &[String]) -> Output {
    Command::new(bin_path("lerobot-train"))
        .args(args)
        .output()
        .expect("lerobot-train can be spawned")
}

/// The documented one-step invocation, plus whatever the caller adds.
fn slice_args(output_dir: &Path, extra: &[&str]) -> Vec<String> {
    let mut args = vec![
        "--dataset.repo_id=rerobot/state_only_slice".to_owned(),
        format!("--dataset.root={}", fixture_dataset().display()),
        format!("--output_dir={}", output_dir.display()),
        "--policy.type=act".to_owned(),
        "--steps=1".to_owned(),
        "--batch_size=2".to_owned(),
        "--policy.chunk_size=2".to_owned(),
        "--policy.n_action_steps=2".to_owned(),
        "--policy.dim_model=32".to_owned(),
        "--policy.n_heads=4".to_owned(),
        "--policy.dim_feedforward=64".to_owned(),
        "--policy.n_encoder_layers=1".to_owned(),
        "--policy.n_decoder_layers=1".to_owned(),
        "--policy.n_vae_encoder_layers=1".to_owned(),
        "--policy.latent_dim=8".to_owned(),
    ];
    args.extend(extra.iter().map(|argument| (*argument).to_owned()));
    args
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr is UTF-8")
}

// ---------------------------------------------------------------------------
// The command trains
// ---------------------------------------------------------------------------

#[test]
fn the_documented_invocation_trains_for_one_step_and_writes_a_checkpoint() {
    let dir = TempDir::new("e2e");
    let output_dir = dir.child("out");
    let result = run(&slice_args(&output_dir, &[]));
    let stdout = stdout_of(&result);
    let stderr = stderr_of(&result);
    assert!(
        result.status.success(),
        "lerobot-train exited {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        result.status.code()
    );

    // The loop reported what it did, in upstream's shape.
    assert!(stdout.contains("Creating dataset"), "stdout:\n{stdout}");
    assert!(stdout.contains("dataset.num_frames=4"), "stdout:\n{stdout}");
    assert!(
        stdout.contains("dataset.num_episodes=1"),
        "stdout:\n{stdout}"
    );
    assert!(stdout.contains("step:1 loss:"), "stdout:\n{stdout}");
    assert!(
        stdout.contains("Checkpoint policy after step 1"),
        "stdout:\n{stdout}"
    );
    assert!(stdout.contains("End of training"), "stdout:\n{stdout}");

    // The checkpoint is on disk, in upstream's layout.
    let checkpoint = output_dir.join("checkpoints/000001");
    for relative in [
        "pretrained_model/config.json",
        "pretrained_model/model.safetensors",
        "pretrained_model/train_config.json",
        "pretrained_model/policy_preprocessor.json",
        "pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors",
        "pretrained_model/policy_postprocessor.json",
        "pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "training_state/training_step.json",
        "training_state/rng_state.safetensors",
        "training_state/optimizer_state.safetensors",
        "training_state/optimizer_param_groups.json",
    ] {
        assert!(
            checkpoint.join(relative).is_file(),
            "the checkpoint has no {relative}"
        );
    }
    assert!(
        output_dir.join("checkpoints/last").exists(),
        "the last-checkpoint marker was not written"
    );
    // And the run named it on stdout so a script can find it.
    assert!(
        stdout.contains("Checkpoint: ") && stdout.contains("000001"),
        "stdout does not name the checkpoint:\n{stdout}"
    );
}

#[test]
fn the_written_weights_are_a_real_safetensors_file_of_the_expected_size() {
    let dir = TempDir::new("weights");
    let output_dir = dir.child("out");
    assert!(run(&slice_args(&output_dir, &[])).status.success());
    let weights = output_dir.join("checkpoints/000001/pretrained_model/model.safetensors");
    let bytes = std::fs::read(&weights).expect("the weights file reads");
    // safetensors starts with a little-endian u64 header length followed by JSON.
    assert!(bytes.len() > 8, "the weights file is empty");
    let header_length = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    assert!(
        8 + header_length <= bytes.len(),
        "the safetensors header length is out of range"
    );
    let header = std::str::from_utf8(&bytes[8..8 + header_length]).expect("the header is UTF-8");
    assert!(
        header.contains("model.action_head.weight"),
        "the header does not name upstream's tensors: {header}"
    );
    assert!(
        header.contains("model.vae_encoder_cls_embed.weight"),
        "the header does not name the VAE encoder's tensors"
    );
}

#[test]
fn two_steps_write_one_checkpoint_at_the_configured_frequency_and_one_at_the_end() {
    let dir = TempDir::new("save-freq");
    let output_dir = dir.child("out");
    let result = run(&slice_args(
        &output_dir,
        &["--steps=2", "--save_freq=1", "--log_freq=1"],
    ));
    assert!(result.status.success(), "stderr:\n{}", stderr_of(&result));
    assert!(output_dir.join("checkpoints/000001").is_dir());
    assert!(output_dir.join("checkpoints/000002").is_dir());
    let stdout = stdout_of(&result);
    assert!(stdout.contains("step:1 loss:"));
    assert!(stdout.contains("step:2 loss:"));
}

#[test]
fn zero_save_frequency_disables_periodic_saves_but_keeps_the_final_checkpoint() {
    let dir = TempDir::new("zero-save-freq");
    let output_dir = dir.child("out");
    let result = run(&slice_args(
        &output_dir,
        &["--steps=2", "--save_freq=0", "--log_freq=1"],
    ));
    assert!(result.status.success(), "stderr:\n{}", stderr_of(&result));
    assert!(
        !output_dir.join("checkpoints/000001").exists(),
        "save_freq=0 must disable periodic checkpoints"
    );
    assert!(
        output_dir.join("checkpoints/000002").is_dir(),
        "the final checkpoint is unconditional"
    );
}

#[test]
fn negative_save_frequency_disables_periodic_saves_but_keeps_the_final_checkpoint() {
    let dir = TempDir::new("negative-save-freq");
    let output_dir = dir.child("out");
    let result = run(&slice_args(
        &output_dir,
        &["--steps=2", "--save_freq=-1", "--log_freq=1"],
    ));
    assert!(result.status.success(), "stderr:\n{}", stderr_of(&result));
    assert!(
        !output_dir.join("checkpoints/000001").exists(),
        "save_freq<0 must disable periodic checkpoints"
    );
    assert!(
        output_dir.join("checkpoints/000002").is_dir(),
        "the final checkpoint is unconditional"
    );
}

#[test]
fn save_checkpoint_false_trains_without_writing_one() {
    let dir = TempDir::new("no-save");
    let output_dir = dir.child("out");
    let result = run(&slice_args(&output_dir, &["--save_checkpoint=false"]));
    assert!(result.status.success(), "stderr:\n{}", stderr_of(&result));
    assert!(stdout_of(&result).contains("step:1 loss:"));
    assert!(
        !output_dir.join("checkpoints").exists(),
        "no checkpoint should have been written"
    );
}

#[test]
fn the_same_seed_produces_the_same_loss_from_the_command_line() {
    let loss_of = |label: &str, seed: &str| {
        let dir = TempDir::new(label);
        let output_dir = dir.child("out");
        let result = run(&slice_args(&output_dir, &[&format!("--seed={seed}")]));
        assert!(result.status.success(), "stderr:\n{}", stderr_of(&result));
        let stdout = stdout_of(&result);
        stdout
            .lines()
            .find(|line| line.starts_with("step:1 loss:"))
            .expect("the step line is present")
            .to_owned()
    };
    assert_eq!(loss_of("seed-a", "1000"), loss_of("seed-b", "1000"));
    assert_ne!(loss_of("seed-c", "1000"), loss_of("seed-d", "77"));
}

#[test]
fn a_flag_can_be_given_as_two_arguments_like_draccus_accepts() {
    let dir = TempDir::new("space-separated");
    let output_dir = dir.child("out");
    let mut args = slice_args(&output_dir, &[]);
    args.push("--log_freq".to_owned());
    args.push("1".to_owned());
    let result = run(&args);
    assert!(result.status.success(), "stderr:\n{}", stderr_of(&result));
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

#[test]
fn help_states_the_partial_status_and_lists_what_is_accepted_and_refused() {
    let result = run(&["--help".to_owned()]);
    assert!(result.status.success());
    let stdout = stdout_of(&result);
    assert!(
        stdout.contains("Compatibility status: partial"),
        "stdout:\n{stdout}"
    );
    assert!(stdout.contains("--dataset.root=DIR"), "stdout:\n{stdout}");
    assert!(stdout.contains("--policy.type=act"), "stdout:\n{stdout}");
    assert!(stdout.contains("--policy.chunk_size"), "stdout:\n{stdout}");
    assert!(
        stdout.contains("Refused, with a reason naming what is missing:"),
        "stdout:\n{stdout}"
    );
    // Every refused flag is listed, so the boundary is discoverable without
    // reading the source.
    for flag in ["--config_path", "--resume", "--wandb", "--policy.path"] {
        assert!(
            stdout.contains(flag),
            "help does not list {flag}:\n{stdout}"
        );
    }
    assert!(
        stdout.contains("state-only"),
        "help does not state the dataset restriction:\n{stdout}"
    );
}

#[test]
fn help_wins_over_a_full_command_line() {
    let dir = TempDir::new("help-wins");
    let output_dir = dir.child("out");
    let mut args = slice_args(&output_dir, &[]);
    args.push("--help".to_owned());
    let result = run(&args);
    assert!(result.status.success());
    assert!(stdout_of(&result).contains("Compatibility status: partial"));
    assert!(!output_dir.exists(), "--help must not start a training run");
}

// ---------------------------------------------------------------------------
// Refusals: exit status and message
// ---------------------------------------------------------------------------

#[test]
fn a_bare_invocation_names_the_first_missing_requirement() {
    let result = run(&[]);
    assert_eq!(result.status.code(), Some(64), "usage errors exit 64");
    let stderr = stderr_of(&result);
    assert!(
        stderr.contains("--policy.type=act is required"),
        "stderr:\n{stderr}"
    );
    assert!(result.stdout.is_empty());
}

#[test]
fn each_missing_requirement_is_named_in_turn() {
    let dir = TempDir::new("missing");
    for (omit, expected) in [
        ("--dataset.repo_id", "--dataset.repo_id is required"),
        ("--dataset.root", "--dataset.root is required"),
        ("--output_dir", "--output_dir is required"),
    ] {
        let full = slice_args(&dir.child("out"), &[]);
        let args: Vec<String> = full
            .into_iter()
            .filter(|argument| !argument.starts_with(omit))
            .collect();
        let result = run(&args);
        assert_eq!(result.status.code(), Some(64), "omitting {omit}");
        let stderr = stderr_of(&result);
        assert!(
            stderr.contains(expected),
            "omitting {omit} did not report it:\n{stderr}"
        );
    }
}

#[test]
fn a_flag_naming_an_unported_feature_is_refused_by_the_parser() {
    // These never reach the run: the flag itself names something absent, so the
    // parser refuses it by name and the message says what is missing.
    let dir = TempDir::new("unsupported-parse");
    for (flag, fragment) in [
        ("--resume=true", "resume needs the optimizer"),
        ("--wandb.project=demo", "Weights & Biases"),
        ("--policy.path=lerobot/act_aloha", "Hub client"),
        ("--env.type=aloha", "Gymnasium"),
        ("--config_path=x.json", "Draccus config loader"),
        ("--dataset.streaming=true", "Hub client"),
        ("--optimizer.lr=1e-4", "Draccus optimizer registry"),
        ("--peft=x", "PEFT"),
        ("--sample_weighting=x", "per-sample loss weighting"),
        ("--cudnn_deterministic=true", "no cuDNN here"),
        ("--prefetch_factor=2", "nothing to prefetch"),
        ("--dataset.image_transforms=x", "state-only"),
    ] {
        let result = run(&slice_args(&dir.child("out"), &[flag]));
        assert_eq!(
            result.status.code(),
            Some(2),
            "{flag} should exit 2, stderr:\n{}",
            stderr_of(&result)
        );
        let stderr = stderr_of(&result);
        assert!(
            stderr.contains(fragment),
            "{flag} did not explain itself:\n{stderr}"
        );
        assert!(
            stderr.contains("is not supported in this slice"),
            "{flag} did not say it is unsupported:\n{stderr}"
        );
    }
}

#[test]
fn a_flag_whose_value_leaves_the_slice_is_refused_when_the_config_is_validated() {
    // These flags *are* accepted -- the run uses them -- but only within the slice.
    // A value outside it is refused a moment later, when the configuration is
    // validated, and the message names the value rather than the flag. Both layers
    // exit 2 and both name what is missing; which layer catches a flag depends on
    // whether the flag or only some of its values are out of scope.
    let dir = TempDir::new("unsupported-validate");
    for (flag, fragment) in [
        ("--num_workers=4", "calling thread"),
        ("--policy.device=mps", "only \"cpu\" is accepted"),
        ("--policy.use_amp=true", "mixed precision"),
        ("--policy.use_peft=true", "PEFT"),
        ("--use_policy_training_preset=false", "optimizer registry"),
    ] {
        let result = run(&slice_args(&dir.child("out"), &[flag]));
        assert_eq!(
            result.status.code(),
            Some(2),
            "{flag} should exit 2, stderr:\n{}",
            stderr_of(&result)
        );
        let stderr = stderr_of(&result);
        assert!(
            stderr.contains(fragment),
            "{flag} did not explain itself:\n{stderr}"
        );
        assert!(
            stderr.contains("unsupported in this slice"),
            "{flag} did not say it is unsupported:\n{stderr}"
        );
    }
}

#[test]
fn the_accepted_form_of_a_partly_supported_flag_still_works() {
    // The other half of the rule above: `--num_workers=0` and
    // `--wandb.enable=false` describe what this slice already does, so a command
    // copied from upstream that happens to spell them out must run rather than fail.
    let dir = TempDir::new("accepted-forms");
    let output_dir = dir.child("out");
    let result = run(&slice_args(
        &output_dir,
        &[
            "--num_workers=0",
            "--wandb.enable=false",
            "--policy.device=cpu",
        ],
    ));
    assert!(result.status.success(), "stderr:\n{}", stderr_of(&result));
    assert!(output_dir.join("checkpoints/000001").is_dir());
}

#[test]
fn a_non_finite_float_flag_is_refused_before_anything_is_trained() {
    // `--policy.dropout=nan` produced a *successful* run whose every reported number
    // was NaN: `step:1 loss:NaN grdn:NaN lr:NaN`, exit code 0, and a checkpoint full
    // of NaN weights on disk. A run that cannot produce a usable model must not
    // report success.
    let dir = TempDir::new("non-finite");
    for flag in [
        "--policy.dropout=nan",
        "--policy.dropout=inf",
        "--policy.dropout=-inf",
        "--policy.kl_weight=nan",
        "--policy.kl_weight=inf",
        "--policy.optimizer_lr=nan",
        "--policy.optimizer_lr=inf",
        "--policy.optimizer_weight_decay=nan",
        "--policy.optimizer_lr_backbone=nan",
        "--tolerance_s=nan",
        "--tolerance_s=inf",
    ] {
        let result = run(&slice_args(&dir.child("out"), &[flag]));
        let stdout = stdout_of(&result);
        let stderr = stderr_of(&result);
        assert_ne!(
            result.status.code(),
            Some(0),
            "{flag} produced a successful run:\n{stdout}"
        );
        assert!(
            !stdout.contains("step:1"),
            "{flag} trained a step instead of being refused:\n{stdout}"
        );
        assert!(
            stderr.contains("finite"),
            "{flag} was not refused as non-finite:\n{stderr}"
        );
    }
}

#[test]
fn a_float_flag_outside_its_meaningful_range_is_refused() {
    // Finiteness alone is not enough: a negative dropout or learning rate is finite
    // and still cannot train. Upstream would produce nonsense; this refuses.
    let dir = TempDir::new("out-of-range");
    for (flag, fragment) in [
        ("--policy.dropout=-0.5", "dropout"),
        ("--policy.dropout=1.0", "dropout"),
        ("--policy.dropout=2.0", "dropout"),
        ("--policy.kl_weight=-1.0", "kl_weight"),
        ("--policy.optimizer_lr=0.0", "optimizer_lr"),
        ("--policy.optimizer_lr=-1e-5", "optimizer_lr"),
        (
            "--policy.optimizer_weight_decay=-0.1",
            "optimizer_weight_decay",
        ),
        (
            "--policy.optimizer_lr_backbone=-1.0",
            "optimizer_lr_backbone",
        ),
        ("--tolerance_s=-1.0", "tolerance_s"),
    ] {
        let result = run(&slice_args(&dir.child("out"), &[flag]));
        let stdout = stdout_of(&result);
        let stderr = stderr_of(&result);
        assert_ne!(
            result.status.code(),
            Some(0),
            "{flag} produced a successful run:\n{stdout}"
        );
        assert!(
            !stdout.contains("step:1"),
            "{flag} trained a step instead of being refused:\n{stdout}"
        );
        assert!(
            stderr.contains(fragment),
            "{flag} was not refused by name:\n{stderr}"
        );
    }
}

#[test]
fn a_dropout_at_the_edges_of_its_range_is_still_accepted() {
    // The range check must not have narrowed what works: zero dropout is the oracle
    // configuration, and a small positive one is the default.
    let dir = TempDir::new("dropout-edges");
    for (index, flag) in ["--policy.dropout=0.0", "--policy.dropout=0.9"]
        .iter()
        .enumerate()
    {
        let result = run(&slice_args(&dir.child(&format!("out{index}")), &[flag]));
        assert!(
            result.status.success(),
            "{flag} should train, stderr:\n{}",
            stderr_of(&result)
        );
    }
}

#[test]
fn a_worker_count_that_does_not_fit_a_u32_is_refused_rather_than_truncated() {
    // `--num_workers=4294967296` is 2^32, which a `u64 as u32` cast turns into 0 --
    // the one value `TrainConfig::validate` accepts. The run then proceeded to build
    // the dataset, silently honouring a worker count it does not implement. Every
    // multiple of 2^32 has the same effect, so this is a hole in the parser's
    // applied-or-refused contract rather than a cosmetic overflow.
    let dir = TempDir::new("worker-overflow");
    for value in ["4294967296", "8589934592", "18446744073709551615"] {
        let result = run(&slice_args(
            &dir.child("out"),
            &[&format!("--num_workers={value}")],
        ));
        let stdout = stdout_of(&result);
        let stderr = stderr_of(&result);
        assert_ne!(
            result.status.code(),
            Some(0),
            "--num_workers={value} was accepted:\n{stdout}"
        );
        assert!(
            !stdout.contains("Creating dataset"),
            "--num_workers={value} reached dataset creation instead of being refused:\n{stdout}"
        );
        assert!(
            stderr.contains("num_workers"),
            "--num_workers={value} was not refused by name:\n{stderr}"
        );
    }
}

#[test]
fn an_out_of_range_integer_flag_is_refused_by_name_rather_than_wrapped() {
    // The same hole, for the other integer flags: none of them may reach the run
    // holding a value the parser silently narrowed.
    let dir = TempDir::new("integer-overflow");
    for flag in [
        "--batch_size=18446744073709551616",
        "--steps=18446744073709551616",
        "--log_freq=99999999999999999999999",
        "--seed=18446744073709551616",
    ] {
        let result = run(&slice_args(&dir.child("out"), &[flag]));
        assert_eq!(
            result.status.code(),
            Some(64),
            "{flag} should be a usage error, stderr:\n{}",
            stderr_of(&result)
        );
        let stderr = stderr_of(&result);
        let name = flag.split('=').next().unwrap().trim_start_matches("--");
        assert!(
            stderr.contains(name),
            "{flag} was not refused by name:\n{stderr}"
        );
    }
}

#[test]
fn an_unknown_flag_exits_sixty_four_and_points_at_help() {
    let dir = TempDir::new("unknown");
    let result = run(&slice_args(&dir.child("out"), &["--nonsense=1"]));
    assert_eq!(result.status.code(), Some(64));
    let stderr = stderr_of(&result);
    assert!(stderr.contains("--nonsense is not a lerobot-train argument"));
    assert!(stderr.contains("lerobot-train --help"));
}

#[test]
fn an_unknown_policy_field_is_rejected_rather_than_ignored() {
    let dir = TempDir::new("unknown-policy");
    let result = run(&slice_args(&dir.child("out"), &["--policy.n_obs_stepz=1"]));
    assert_eq!(result.status.code(), Some(64));
    assert!(stderr_of(&result).contains("policy.n_obs_stepz"));
}

#[test]
fn a_positional_argument_is_rejected() {
    let dir = TempDir::new("positional");
    let mut args = slice_args(&dir.child("out"), &[]);
    args.push("extra".to_owned());
    let result = run(&args);
    assert_eq!(result.status.code(), Some(64));
    assert!(stderr_of(&result).contains("unexpected argument \"extra\""));
}

#[test]
fn a_flag_with_no_value_is_rejected_rather_than_defaulted() {
    let result = run(&["--steps".to_owned()]);
    assert_eq!(result.status.code(), Some(64));
    assert!(stderr_of(&result).contains("expected a value"));
}

#[test]
fn a_malformed_value_names_the_flag_and_what_was_expected() {
    let dir = TempDir::new("bad-value");
    let result = run(&slice_args(&dir.child("out"), &["--steps=lots"]));
    assert_eq!(result.status.code(), Some(64));
    let stderr = stderr_of(&result);
    assert!(stderr.contains("--steps"), "stderr:\n{stderr}");
    assert!(stderr.contains("expected an integer"), "stderr:\n{stderr}");
}

#[test]
fn another_policy_type_is_refused_by_name() {
    let dir = TempDir::new("other-policy");
    let full = slice_args(&dir.child("out"), &[]);
    let args: Vec<String> = full
        .into_iter()
        .map(|argument| {
            if argument == "--policy.type=act" {
                "--policy.type=diffusion".to_owned()
            } else {
                argument
            }
        })
        .collect();
    let result = run(&args);
    assert_eq!(result.status.code(), Some(2));
    let stderr = stderr_of(&result);
    assert!(stderr.contains("diffusion"), "stderr:\n{stderr}");
    assert!(stderr.contains("only policy"), "stderr:\n{stderr}");
}

#[test]
fn a_non_cpu_device_is_refused_at_the_command_line() {
    let dir = TempDir::new("cli-cuda");
    let result = run(&slice_args(&dir.child("out"), &["--policy.device=cuda"]));
    assert_eq!(result.status.code(), Some(2));
    assert!(stderr_of(&result).contains("only \"cpu\" is accepted"));
}

#[test]
fn an_absent_dataset_root_fails_without_reaching_the_network() {
    let dir = TempDir::new("absent-root");
    let output_dir = dir.child("out");
    let args = vec![
        "--dataset.repo_id=lerobot/pusht".to_owned(),
        format!("--dataset.root={}", dir.child("nope").display()),
        format!("--output_dir={}", output_dir.display()),
        "--policy.type=act".to_owned(),
        "--steps=1".to_owned(),
    ];
    let result = run(&args);
    assert_eq!(result.status.code(), Some(2));
    let stderr = stderr_of(&result);
    assert!(
        stderr.contains("never downloads from the Hub"),
        "stderr:\n{stderr}"
    );
    assert!(
        !output_dir.exists(),
        "a failed run must not leave an output directory"
    );
}

#[test]
fn rerunning_into_the_same_output_directory_is_refused() {
    let dir = TempDir::new("rerun");
    let output_dir = dir.child("out");
    assert!(run(&slice_args(&output_dir, &[])).status.success());
    let second = run(&slice_args(&output_dir, &[]));
    assert_eq!(second.status.code(), Some(2));
    assert!(stderr_of(&second).contains("resume is not supported"));
}

#[test]
fn a_chunk_size_off_the_frame_grid_is_impossible_but_a_huge_one_is_refused_cleanly() {
    // chunk_size is always a whole number of frames, so the tolerance check cannot
    // fail through the CLI. What can fail is asking for more actions than the
    // chunk holds, which is upstream's own ValueError.
    let dir = TempDir::new("bad-chunk");
    let full = slice_args(&dir.child("out"), &[]);
    let args: Vec<String> = full
        .into_iter()
        .map(|argument| {
            if argument == "--policy.n_action_steps=2" {
                "--policy.n_action_steps=5".to_owned()
            } else {
                argument
            }
        })
        .collect();
    let result = run(&args);
    assert_eq!(result.status.code(), Some(2));
    let stderr = stderr_of(&result);
    assert!(
        stderr.contains("chunk size is the upper bound"),
        "the message is not upstream's:\n{stderr}"
    );
}

// ---------------------------------------------------------------------------
// The checkpoint the executable wrote can be loaded back
// ---------------------------------------------------------------------------

#[test]
fn the_checkpoint_the_executable_wrote_reloads_and_predicts() {
    // The round trip that matters for a real user: the files `lerobot-train` left
    // on disk are enough, on their own, to rebuild the policy and run it. Nothing
    // from the training process is reused here -- the config is re-read from
    // `config.json` and the weights from `model.safetensors`, exactly as a resume or
    // an evaluation would.
    let dir = TempDir::new("reload-from-binary");
    let output_dir = dir.child("out");
    let result = run(&slice_args(&output_dir, &[]));
    assert!(result.status.success(), "stderr:\n{}", stderr_of(&result));

    let pretrained = output_dir.join("checkpoints/000001/pretrained_model");
    let config_text = std::fs::read_to_string(pretrained.join("config.json")).unwrap();
    let policy = rerobot_core::policy::act::ActConfig::from_checkpoint_json(&config_text)
        .expect("the config the executable wrote is a valid ACT checkpoint config");
    // The features were resolved from the dataset, not left at their empty default.
    assert_eq!(
        policy
            .input_features
            .as_ref()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        vec!["observation.state", "observation.environment_state"]
    );
    assert_eq!(
        policy
            .output_features
            .as_ref()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        vec!["action"]
    );

    let mut rng = rerobot_core::random::SplitMix64::new(0);
    let mut model = rerobot_train::model::act::ActModel::new(
        &policy,
        &rerobot_train::candle_core::Device::Cpu,
        &mut rng,
    )
    .expect("a model builds from the checkpoint's own config");
    model
        .load(&pretrained.join("model.safetensors"))
        .expect("the weights the executable wrote load into it");

    // And it runs: a batch straight off the same dataset produces finite actions of
    // the shape the config promises.
    let windows = rerobot_train::indexmap::IndexMap::from([(
        "action".to_owned(),
        rerobot_core::dataset::delta::action_delta_timestamps(2, 10),
    )]);
    let dataset =
        rerobot_train::data::dataset::StateOnlyDataset::load(&fixture_dataset(), &windows, 1e-4)
            .unwrap();
    let frames: Vec<_> = (0..2).map(|index| dataset.get(index).unwrap()).collect();
    let batch =
        rerobot_train::data::batch::collate(&frames, &rerobot_train::candle_core::Device::Cpu)
            .unwrap();
    let actions = model.predict_action_steps(&batch).unwrap();
    assert_eq!(actions.dims(), &[2, 2, 2]);
    let values = actions.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "the reloaded policy predicted a non-finite action: {values:?}"
    );

    // Loading twice is idempotent, so the reload is a function of the file rather
    // than of the model's prior state.
    let first = values.clone();
    model.load(&pretrained.join("model.safetensors")).unwrap();
    let second = model
        .predict_action_steps(&batch)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(first, second);
}

#[test]
fn the_saved_weights_are_the_trained_ones_and_train_config_json_records_enough_to_prove_it() {
    // Two claims at once, and they support each other.
    //
    // First, that `model.safetensors` holds the weights *after* the step rather than
    // the initial draw: a fresh model seeded the way the run was seeded differs from
    // the checkpoint.
    //
    // Second, that `train_config.json` records enough of the run to reproduce it.
    // The batch size and seed are read out of that file rather than restated from
    // the command line, so the reconstruction below cannot silently drift away from
    // the flags -- and if the file omitted either, this test would fail rather than
    // quietly compare two different runs. That is the whole purpose of the file.
    let dir = TempDir::new("trained-not-initial");
    let output_dir = dir.child("out");
    assert!(run(&slice_args(&output_dir, &[])).status.success());
    let pretrained = output_dir.join("checkpoints/000001/pretrained_model");

    let policy = rerobot_core::policy::act::ActConfig::from_checkpoint_json(
        &std::fs::read_to_string(pretrained.join("config.json")).unwrap(),
    )
    .unwrap();

    let train_config = std::fs::read_to_string(pretrained.join("train_config.json")).unwrap();
    let rerobot_core::dataset::json::JsonLike::Object(recorded) =
        rerobot_core::dataset::json::loads(&train_config).unwrap()
    else {
        panic!("train_config.json is not an object");
    };
    let integer = |name: &str| -> u64 {
        match &recorded[name] {
            rerobot_core::dataset::json::JsonLike::Int(value) => {
                value.to_string().parse().expect("a machine integer")
            }
            other => panic!("{name} is a {}", other.type_name()),
        }
    };
    let batch_size = integer("batch_size") as usize;
    let seed = integer("seed");
    assert_eq!(
        batch_size, 2,
        "train_config.json did not record the batch size"
    );
    assert_eq!(seed, 1000, "train_config.json did not record the seed");

    // The initial draw the run started from, reproduced from what the file records.
    let mut initial = {
        let mut config = rerobot_train::config::TrainConfig::new(
            "rerobot/state_only_slice".to_owned(),
            fixture_dataset(),
            dir.child("unused"),
        );
        config.policy = policy.clone();
        config.seed = Some(seed);
        config.batch_size = batch_size;
        config.validate().unwrap();
        rerobot_train::run::TrainSession::new(&config).unwrap()
    };

    let mut trained = rerobot_train::model::act::ActModel::new(
        &policy,
        &rerobot_train::candle_core::Device::Cpu,
        &mut rerobot_core::random::SplitMix64::new(7),
    )
    .unwrap();
    trained.load(&pretrained.join("model.safetensors")).unwrap();

    let before = rerobot_train::optim::state_dict_distance(
        &initial.model.state_dict().unwrap(),
        &trained.state_dict().unwrap(),
    )
    .unwrap();
    assert!(
        before > 0.0,
        "the checkpoint holds the initial weights, so the step was not saved"
    );

    // ... and taking the same step from that draw lands on the checkpoint exactly.
    // This is the strongest available statement about the saved file: it is the
    // state this configuration and this seed deterministically produce after one
    // step, reproduced in a separate process from the one that wrote it.
    initial.step(1).unwrap();
    let after = rerobot_train::optim::state_dict_distance(
        &initial.model.state_dict().unwrap(),
        &trained.state_dict().unwrap(),
    )
    .unwrap();
    assert!(
        after < before,
        "taking the step did not move the model towards the checkpoint \
         ({after} is not below {before})"
    );
    assert!(
        after < 1e-6,
        "the checkpoint is not the state one step from the recorded seed produces \
         (distance {after}, having started {before} away)"
    );
}
