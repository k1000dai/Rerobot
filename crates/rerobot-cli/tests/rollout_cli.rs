//! End-to-end tests for the hardware-independent `lerobot-rollout` slice.

use rerobot_cli::rollout::parse;
use rerobot_train::config::TrainConfig;
use rerobot_train::run::train;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU32, Ordering};

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "rerobot-rollout-cli-{}-{label}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
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

fn fixture_dataset() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/state_only")
}

fn trained_policy() -> (TempDir, PathBuf) {
    let dir = TempDir::new("fixture");
    let output = dir.child("train");
    let mut config = TrainConfig::new(
        "rerobot/state_only_slice".to_owned(),
        fixture_dataset(),
        output.clone(),
    );
    config.steps = 1;
    config.batch_size = 2;
    config.log_freq = 1;
    config.save_freq = 1.into();
    config.policy.chunk_size = 2.into();
    config.policy.n_action_steps = 2.into();
    config.policy.dim_model = 32.into();
    config.policy.n_heads = 4.into();
    config.policy.dim_feedforward = 64.into();
    config.policy.n_encoder_layers = 1.into();
    config.policy.n_decoder_layers = 1.into();
    config.policy.n_vae_encoder_layers = 1.into();
    config.policy.latent_dim = 8.into();
    config.policy.pretrained_backbone_weights = None;
    train(&config, &mut |_| {}).unwrap();
    (dir, output.join("checkpoints/000001/pretrained_model"))
}

fn text(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).unwrap()
}

#[test]
fn rollout_help_exposes_the_offline_policy_and_dataset_boundary() {
    let output = Command::new(env!("CARGO_BIN_EXE_lerobot-rollout"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = text(&output);
    assert!(stdout.contains("--policy.path=DIR"), "{stdout}");
    assert!(stdout.contains("--dataset.root=DIR"), "{stdout}");
    assert!(stdout.contains("hardware-independent"), "{stdout}");
}

#[test]
fn rollout_loads_a_trained_checkpoint_and_emits_actions() {
    let (_dir, checkpoint) = trained_policy();
    let output = Command::new(env!("CARGO_BIN_EXE_lerobot-rollout"))
        .args([
            format!("--policy.path={}", checkpoint.display()),
            format!("--dataset.root={}", fixture_dataset().display()),
            "--steps=3".to_owned(),
        ])
        .output()
        .unwrap();
    let stdout = text(&output);
    let stderr = String::from_utf8(output.stderr.clone()).unwrap();
    assert!(
        output.status.success(),
        "rollout exited {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status.code()
    );
    assert!(stdout.contains("frame:0 action:"), "{stdout}");
    assert!(stdout.contains("frame:1 action:"), "{stdout}");
    assert!(stdout.contains("frame:2 action:"), "{stdout}");
    assert!(stdout.contains("queried:true"), "{stdout}");
    assert!(stdout.contains("queried:false"), "{stdout}");
}

#[test]
fn rollout_refuses_robot_flags_instead_of_claiming_hardware_deployment() {
    let output = Command::new(env!("CARGO_BIN_EXE_lerobot-rollout"))
        .args(["--robot.type=so101_follower"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("robot drivers"), "{stderr}");
}

#[test]
fn rollout_does_not_silently_narrow_arbitrary_precision_integer_flags() {
    let args = vec![
        "--policy.path=/tmp/policy".to_owned(),
        "--dataset.root=/tmp/dataset".to_owned(),
        "--steps=184467440737095516160000000000000000000".to_owned(),
    ];
    let error = parse(&args).expect_err("the oversized Python integer must be refused explicitly");
    assert!(error.to_string().contains("outside the supported range"));
}

#[test]
fn rollout_rejects_trace_counts_that_would_exceed_the_memory_bound() {
    let args = vec![
        "--policy.path=/tmp/policy".to_owned(),
        "--dataset.root=/tmp/dataset".to_owned(),
        format!(
            "--steps={}",
            rerobot_train::limits::MAX_ROLLOUT_TRACE_STEPS + 1
        ),
    ];
    let error = parse(&args).expect_err("the in-memory rollout trace must be bounded");
    assert!(error.to_string().contains("rollout trace"), "{error}");
}
