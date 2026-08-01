//! Behaviour parity tests for reading the committed state-only dataset fixture,
//! derived from `lerobot/datasets/{lerobot_dataset,dataset_reader,io_utils}.py` at
//! commit f37be3edbee60f3a09a5183788b91eb19f0c07d1.
//!
//! The fixture was written by upstream's own `LeRobotDataset.create` /
//! `add_frame` / `save_episode` / `finalize` (see
//! `tools/goldens/make_dataset_fixture.py`), so these tests check the reader
//! against upstream's writer rather than against a hand-rolled parquet file.

mod common;

use common::{fixture_dataset, TempDir};
use indexmap::IndexMap;
use rerobot_core::dataset::delta::action_delta_timestamps;
use rerobot_core::types::FeatureType;
use rerobot_train::data::dataset::StateOnlyDataset;
use rerobot_train::data::meta::DatasetMetadata;
use rerobot_train::error::TrainError;

fn action_window(chunk_size: i64) -> IndexMap<String, Vec<f64>> {
    IndexMap::from([("action".to_owned(), action_delta_timestamps(chunk_size, 10))])
}

fn load(chunk_size: i64) -> StateOnlyDataset {
    StateOnlyDataset::load(&fixture_dataset(), &action_window(chunk_size), 1e-4)
        .expect("the fixture loads")
}

// ---------------------------------------------------------------------------
// `meta/`
// ---------------------------------------------------------------------------

#[test]
fn the_fixtures_metadata_is_read_in_full() {
    let metadata = DatasetMetadata::load(&fixture_dataset()).expect("the fixture loads");
    assert_eq!(metadata.fps().unwrap(), 10);
    assert_eq!(metadata.total_frames().unwrap(), 4);
    assert_eq!(metadata.total_episodes().unwrap(), 1);
    assert_eq!(metadata.info.codebase_version, "v3.0");
}

#[test]
fn info_json_declares_the_three_state_features_and_the_five_bookkeeping_ones() {
    let metadata = DatasetMetadata::load(&fixture_dataset()).unwrap();
    assert_eq!(
        metadata.feature_keys().collect::<Vec<_>>(),
        vec![
            "observation.state",
            "observation.environment_state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]
    );
    let state = metadata.feature("observation.state").unwrap();
    assert_eq!(state.dtype, "float32");
    assert_eq!(state.shape, vec![2]);
    assert_eq!(state.width(), Ok(2));
    assert_eq!(
        state.names.as_deref(),
        Some(&["x".to_owned(), "y".to_owned()][..])
    );
}

#[test]
fn the_stats_file_upstream_wrote_is_read_with_its_statistics_intact() {
    let metadata = DatasetMetadata::load(&fixture_dataset()).unwrap();
    let state = metadata.stats.get("observation.state").expect("present");
    assert_eq!(state.mean(), Some(&[0.4375, 0.5625][..]));
    assert_eq!(state.min(), Some(&[0.0, 0.0][..]));
    assert_eq!(state.max(), Some(&[1.0, 1.0][..]));
    let action = metadata.stats.get("action").expect("present");
    assert_eq!(action.mean(), Some(&[0.0625, -0.0625][..]));
}

#[test]
fn the_episode_table_carries_the_dataset_index_boundaries() {
    let metadata = DatasetMetadata::load(&fixture_dataset()).unwrap();
    assert_eq!(metadata.episodes.len(), 1);
    let episode = &metadata.episodes[0];
    assert_eq!(episode.episode_index, 0);
    assert_eq!(episode.length, 4);
    assert_eq!(episode.dataset_from_index, 0);
    assert_eq!(episode.dataset_to_index, 4);
    assert_eq!(episode.data_chunk_index, 0);
    assert_eq!(episode.data_file_index, 0);
    assert_eq!(episode.tasks, vec!["reach the target".to_owned()]);
    assert_eq!(metadata.episode_from_indices(), vec![0]);
    assert_eq!(metadata.episode_to_indices(), vec![4]);
}

#[test]
fn the_task_table_maps_the_index_the_frames_carry() {
    let metadata = DatasetMetadata::load(&fixture_dataset()).unwrap();
    assert_eq!(metadata.task(0), Some("reach the target"));
    assert_eq!(metadata.task(1), None);
}

#[test]
fn policy_features_classify_by_key_and_drop_the_bookkeeping_columns() {
    let metadata = DatasetMetadata::load(&fixture_dataset()).unwrap();
    let features = metadata.policy_features();
    assert_eq!(
        features.keys().collect::<Vec<_>>(),
        vec![
            "observation.state",
            "observation.environment_state",
            "action"
        ]
    );
    assert_eq!(features["observation.state"].r#type, FeatureType::State);
    assert_eq!(
        features["observation.environment_state"].r#type,
        FeatureType::Env
    );
    assert_eq!(features["action"].r#type, FeatureType::Action);
}

#[test]
fn the_input_output_split_puts_actions_on_the_output_side() {
    let metadata = DatasetMetadata::load(&fixture_dataset()).unwrap();
    let (inputs, outputs) = metadata.policy_feature_split();
    assert_eq!(
        inputs.keys().collect::<Vec<_>>(),
        vec!["observation.state", "observation.environment_state"]
    );
    assert_eq!(outputs.keys().collect::<Vec<_>>(), vec!["action"]);
}

#[test]
fn an_absent_dataset_root_says_it_will_not_download() {
    let dir = TempDir::new("absent");
    let error = DatasetMetadata::load(&dir.child("nope")).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("never \n                 downloads")
            || error.to_string().contains("never downloads"),
        "message does not disclaim the Hub: {error}"
    );
}

#[test]
fn a_dataset_declaring_a_camera_is_refused_rather_than_half_read() {
    let dir = TempDir::new("visual");
    let root = dir.child("ds");
    std::fs::create_dir_all(root.join("meta")).unwrap();
    std::fs::write(
        root.join("meta/info.json"),
        r#"{
            "codebase_version": "v3.0",
            "fps": 10,
            "features": {
                "observation.images.top": {"dtype": "video", "shape": [3, 96, 96], "names": null},
                "action": {"dtype": "float32", "shape": [2], "names": null}
            },
            "total_episodes": 1,
            "total_frames": 4,
            "total_tasks": 1
        }"#,
    )
    .unwrap();
    let error = DatasetMetadata::load(&root).unwrap_err();
    assert!(
        matches!(error, TrainError::Unsupported(_)),
        "expected an explicit refusal, got {error}"
    );
    assert!(
        error.to_string().contains("observation.images.top"),
        "the refusal does not name the feature: {error}"
    );
    assert!(
        error.to_string().contains("state-only"),
        "the refusal does not say why: {error}"
    );
    // And it must say what the reader cannot decode and what the policy path *can*
    // consume, so that a user with a camera dataset is not left guessing whether ACT
    // supports cameras at all.
    for expected in ["MP4", "PNG or JPEG", "Batch::with_images"] {
        assert!(
            error.to_string().contains(expected),
            "the refusal does not mention {expected}: {error}"
        );
    }
}

// ---------------------------------------------------------------------------
// Frames and delta windows
// ---------------------------------------------------------------------------

#[test]
fn the_dataset_has_one_item_per_frame() {
    let dataset = load(2);
    assert_eq!(dataset.len(), 4);
    assert_eq!(dataset.num_frames(), 4);
    assert_eq!(dataset.num_episodes(), 1);
    assert!(!dataset.is_empty());
    assert!(dataset.has_action_window());
}

#[test]
fn the_action_window_is_the_chunk_range_in_frames() {
    let dataset = load(2);
    assert_eq!(dataset.delta_indices()["action"], vec![0, 1]);
    let dataset = load(4);
    assert_eq!(dataset.delta_indices()["action"], vec![0, 1, 2, 3]);
}

#[test]
fn every_frame_carries_the_values_upstream_wrote() {
    let dataset = load(2);
    let expected_state = [[0.0f32, 1.0], [0.25, 0.75], [0.5, 0.5], [1.0, 0.0]];
    let expected_env = [[10.0f32, -1.0], [11.0, -2.0], [12.0, -3.0], [13.0, -4.0]];
    for (index, (state, env)) in expected_state.iter().zip(&expected_env).enumerate() {
        let frame = dataset.get(index).unwrap();
        assert_eq!(frame.index, index as i64);
        assert_eq!(frame.frame_index, index as i64);
        assert_eq!(frame.episode_index, 0);
        assert_eq!(frame.task, "reach the target");
        assert_eq!(frame.value("observation.state"), Some(&state[..]));
        assert_eq!(frame.value("observation.environment_state"), Some(&env[..]));
    }
}

#[test]
fn timestamps_are_the_frame_index_over_fps_as_float32() {
    let dataset = load(2);
    for index in 0..4 {
        let frame = dataset.get(index).unwrap();
        assert_eq!(frame.timestamp, (index as f32) / 10.0);
    }
}

#[test]
fn the_action_chunk_of_an_interior_frame_is_the_next_two_actions() {
    let dataset = load(2);
    let frame = dataset.get(1).unwrap();
    assert_eq!(
        frame.window("action"),
        Some(&[vec![0.25f32, -0.25], vec![0.0, 0.0]][..])
    );
    assert_eq!(frame.is_pad("action"), Some(&[false, false][..]));
}

#[test]
fn the_action_chunk_of_the_last_frame_repeats_it_and_flags_the_repeat_as_padding() {
    let dataset = load(2);
    let frame = dataset.get(3).unwrap();
    assert_eq!(
        frame.window("action"),
        Some(&[vec![-0.5f32, 0.5], vec![-0.5, 0.5]][..])
    );
    assert_eq!(frame.is_pad("action"), Some(&[false, true][..]));
}

#[test]
fn a_chunk_longer_than_the_episode_pads_the_tail() {
    let dataset = load(6);
    let frame = dataset.get(0).unwrap();
    assert_eq!(
        frame.is_pad("action"),
        Some(&[false, false, false, false, true, true][..])
    );
    let window = frame.window("action").unwrap();
    // The clamped entries all repeat the episode's last action.
    assert_eq!(window[4], vec![-0.5, 0.5]);
    assert_eq!(window[5], vec![-0.5, 0.5]);
}

#[test]
fn a_frame_past_the_end_is_an_error_rather_than_a_wrap() {
    let dataset = load(2);
    let error = dataset.get(4).unwrap_err();
    assert!(
        error.to_string().contains("out of range"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_delta_window_naming_an_absent_feature_is_refused() {
    let mut windows = action_window(2);
    windows.insert("observation.images.top".to_owned(), vec![0.0]);
    let error = StateOnlyDataset::load(&fixture_dataset(), &windows, 1e-4).unwrap_err();
    assert!(
        error.to_string().contains("observation.images.top"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_window_off_the_frame_grid_is_refused_by_the_tolerance_check() {
    let windows = IndexMap::from([("action".to_owned(), vec![0.0, 0.05])]);
    let error = StateOnlyDataset::load(&fixture_dataset(), &windows, 1e-4).unwrap_err();
    assert!(
        matches!(error, TrainError::DeltaTimestamps(_)),
        "unexpected error: {error}"
    );
    assert!(
        error.to_string().contains("multiples of 1/10"),
        "the message does not name the frame rate: {error}"
    );
}

// ---------------------------------------------------------------------------
// The sampler over this dataset
// ---------------------------------------------------------------------------

#[test]
fn the_sampler_covers_every_frame_of_the_only_episode() {
    let dataset = load(2);
    let sampler = dataset.sampler(None, 0, false, 1000).unwrap();
    assert_eq!(sampler.len(), 4);
    assert_eq!(sampler.indices(), vec![0, 1, 2, 3]);
}

#[test]
fn dropping_the_last_frames_shortens_the_sampler_not_the_dataset() {
    let dataset = load(2);
    let sampler = dataset.sampler(None, 1, false, 1000).unwrap();
    assert_eq!(sampler.indices(), vec![0, 1, 2]);
    assert_eq!(dataset.len(), 4, "the dataset itself is unchanged");
}

#[test]
fn a_shuffled_sampler_is_reproducible_for_a_seed() {
    let dataset = load(2);
    let mut left = dataset.sampler(None, 0, true, 7).unwrap();
    let mut right = dataset.sampler(None, 0, true, 7).unwrap();
    assert_eq!(left.next_epoch(), right.next_epoch());
}
