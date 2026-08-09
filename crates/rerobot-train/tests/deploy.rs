//! Differentially meaningful smoke tests for the offline ACT deployment boundary.

mod common;

use common::{copy_fixture_dataset, fixture_dataset, reduced_config, TempDir};
use rerobot_train::deploy::{InferenceSession, InferenceStep, TemporalEnsembler};
use rerobot_train::run::train;
use std::path::PathBuf;

fn trained_checkpoint() -> (TempDir, PathBuf) {
    let dir = TempDir::new("deploy");
    let output = dir.child("train");
    let config = reduced_config(fixture_dataset(), output.clone());
    train(&config, &mut |_| {}).expect("the reduced ACT run trains");
    (dir, output.join("checkpoints/000001/pretrained_model"))
}

#[test]
fn an_oversized_checkpoint_config_is_refused_before_json_parsing() {
    let (_dir, checkpoint) = trained_checkpoint();
    let config_path = checkpoint.join("config.json");
    let oversized = vec![b' '; rerobot_train::limits::MAX_CHECKPOINT_JSON_BYTES as usize + 1];
    std::fs::write(&config_path, oversized).expect("the fixture config is replaceable");

    let error = match InferenceSession::load(&checkpoint, &fixture_dataset(), None) {
        Ok(_) => panic!("an oversized config must not be parsed or loaded"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("config.json"));
    assert!(error.to_string().contains("exceeds"));
}

#[test]
fn a_saved_act_checkpoint_loads_and_selects_finite_actions_from_a_dataset_frame() {
    let (_dir, checkpoint) = trained_checkpoint();
    let mut session = InferenceSession::load(&checkpoint, &fixture_dataset(), None)
        .expect("the checkpoint is a deployable local ACT policy");

    let action = session
        .select_action(0)
        .expect("the first dataset observation produces an action");

    assert_eq!(action.frame_index, 0);
    assert_eq!(action.action.len(), 2);
    assert!(action.action.iter().all(|value| value.is_finite()));
    assert!(action.queried_policy);
}

#[test]
fn action_queue_reuses_a_chunk_before_querying_the_policy_again() {
    let (_dir, checkpoint) = trained_checkpoint();
    let mut session = InferenceSession::load(&checkpoint, &fixture_dataset(), None)
        .expect("the checkpoint loads");

    let first = session.select_action(0).unwrap();
    let second = session.select_action(1).unwrap();

    assert!(first.queried_policy);
    assert!(!second.queried_policy);
    assert_eq!(second.frame_index, 1);
    assert_eq!(second.action.len(), first.action.len());
}

#[test]
fn offline_rollout_reports_each_requested_frame_in_order() {
    let (_dir, checkpoint) = trained_checkpoint();
    let mut session = InferenceSession::load(&checkpoint, &fixture_dataset(), None)
        .expect("the checkpoint loads");

    let trace: Vec<InferenceStep> = session.rollout(0, 3).expect("rollout completes");

    assert_eq!(
        trace
            .iter()
            .map(|step| step.frame_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(trace.iter().filter(|step| step.queried_policy).count(), 2);
}

#[test]
fn temporal_checkpoint_queries_the_policy_on_each_dataset_frame() {
    let (_dir, checkpoint) = trained_checkpoint();
    let config_path = checkpoint.join("config.json");
    let config = std::fs::read_to_string(&config_path).unwrap();
    let config = config
        .replace("\"n_action_steps\": 2", "\"n_action_steps\": 1")
        .replace(
            "\"temporal_ensemble_coeff\": null",
            "\"temporal_ensemble_coeff\": 0.0",
        );
    assert!(config.contains("\"n_action_steps\": 1"));
    assert!(config.contains("\"temporal_ensemble_coeff\": 0.0"));
    std::fs::write(config_path, config).unwrap();

    let mut session = InferenceSession::load(&checkpoint, &fixture_dataset(), None)
        .expect("a temporal-ensemble ACT checkpoint is deployable");
    let first = session.select_action(0).expect("the first frame runs");
    let second = session.select_action(1).expect("the second frame runs");

    assert!(first.queried_policy);
    assert!(second.queried_policy);
}

#[test]
fn deployment_uses_checkpoint_processor_statistics_not_observation_dataset_statistics() {
    let (_dir, checkpoint) = trained_checkpoint();
    let mut reference = InferenceSession::load(&checkpoint, &fixture_dataset(), None)
        .expect("the original checkpoint and dataset load");
    let expected = reference
        .select_action(0)
        .expect("the reference observation produces an action")
        .action;

    let shifted = TempDir::new("deploy-shifted-stats");
    let shifted_dataset = shifted.child("dataset");
    copy_fixture_dataset(&shifted_dataset);
    let stats_path = shifted_dataset.join("meta/stats.json");
    let stats = std::fs::read_to_string(&stats_path).expect("the copied stats file reads");
    let shifted_stats = stats
        .replace("0.4375", "100.0")
        .replace("0.5625", "200.0")
        .replace("11.5", "300.0")
        .replace("-2.5", "400.0")
        .replace("0.0625", "50.0")
        .replace("-0.0625", "-50.0")
        .replace("0.36975499987602234", "1.0")
        .replace("1.1180340051651", "1.0");
    std::fs::write(stats_path, shifted_stats).expect("the shifted stats file writes");

    let mut deployed = InferenceSession::load(&checkpoint, &shifted_dataset, None)
        .expect("the checkpoint remains deployable with different observation statistics");
    let actual = deployed
        .select_action(0)
        .expect("the shifted-stat observation produces an action")
        .action;

    assert_eq!(
        actual, expected,
        "deployment must use the checkpoint's saved processor state for both input normalization and action unnormalization"
    );
}

#[test]
fn a_checkpoint_without_config_or_weights_is_refused_before_dataset_use() {
    let dir = TempDir::new("missing-policy");
    let error = match InferenceSession::load(dir.path(), &fixture_dataset(), None) {
        Ok(_) => panic!("a policy directory without config and weights is not deployable"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(message.contains("config.json"), "{message}");
}

#[test]
fn temporal_ensembler_returns_the_first_action_then_blends_overlapping_chunks() {
    let mut ensembler =
        TemporalEnsembler::new(0.0, 2).expect("the coefficient and chunk are valid");

    assert_eq!(
        ensembler.update(vec![vec![1.0], vec![3.0]]).unwrap(),
        vec![1.0]
    );
    assert_eq!(
        ensembler.update(vec![vec![5.0], vec![7.0]]).unwrap(),
        vec![4.0]
    );
    assert_eq!(
        ensembler.update(vec![vec![9.0], vec![11.0]]).unwrap(),
        vec![8.0]
    );
}

#[test]
fn temporal_ensembler_uses_float32_exponential_weights() {
    let mut ensembler = TemporalEnsembler::new(std::f64::consts::LN_2, 2)
        .expect("the coefficient and chunk are valid");

    assert_eq!(
        ensembler.update(vec![vec![1.0], vec![3.0]]).unwrap(),
        vec![1.0]
    );
    let blended = ensembler
        .update(vec![vec![5.0], vec![7.0]])
        .expect("the second chunk has the same shape");
    assert!((blended[0] - 3.6666667).abs() < 1e-6, "{blended:?}");
}

#[test]
fn resetting_temporal_ensembler_discards_previous_chunks() {
    let mut ensembler = TemporalEnsembler::new(0.0, 2).unwrap();
    ensembler.update(vec![vec![1.0], vec![3.0]]).unwrap();
    ensembler.reset();

    assert_eq!(
        ensembler.update(vec![vec![9.0], vec![11.0]]).unwrap(),
        vec![9.0]
    );
}
