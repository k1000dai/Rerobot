//! The pre/postprocessor artifacts a checkpoint carries, pinned byte for byte
//! against upstream's own `save_pretrained` output.
//!
//! `lerobot/common/train_utils.py:145-155` passes both processors to
//! `save_checkpoint`, and `lerobot/scripts/lerobot_train.py:675-683` builds them
//! from the dataset's statistics. Four files land in `pretrained_model/`:
//!
//! ```text
//! policy_preprocessor.json
//! policy_preprocessor_step_3_normalizer_processor.safetensors
//! policy_postprocessor.json
//! policy_postprocessor_step_0_unnormalizer_processor.safetensors
//! ```
//!
//! A checkpoint without them has lost its normalization state: the weights were
//! trained on normalized inputs, so anything loading the policy has to know the
//! mean and standard deviation the training data was divided by. That is not a
//! cosmetic omission, and it is why these are compared against goldens
//! `tools/goldens/make_act_goldens.py` produced by calling upstream's own writer
//! rather than against Rerobot's own output.

mod common;

use candle_core::{Device, Tensor};
use common::{embedded_image_fixture, fixture_dataset, reduced_config, TempDir};
use indexmap::IndexMap;
use rerobot_core::dataset::json::{loads, JsonLike};
use rerobot_core::types::NormalizationMode;
use rerobot_train::data::batch::Batch;
use rerobot_train::data::image::CameraNormalization;
use rerobot_train::processor::{
    rename_observation_batch, write_processor_artifacts, write_processor_artifacts_with_cameras,
    write_processor_artifacts_with_cameras_and_rename, LoadedPolicyProcessors,
    POLICY_POSTPROCESSOR_NAME, POLICY_PREPROCESSOR_NAME,
};
use std::collections::HashMap;
use std::path::PathBuf;

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/goldens/processors")
}

/// The four names upstream writes, in the order this test checks them.
const ARTIFACTS: [&str; 4] = [
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
];

/// Write the artifacts into a temporary directory from the fixture's metadata.
fn written(label: &str) -> (TempDir, PathBuf) {
    let dir = TempDir::new(label);
    let target = dir.child("pretrained_model");
    std::fs::create_dir_all(&target).unwrap();

    let metadata =
        rerobot_train::data::meta::DatasetMetadata::load(&fixture_dataset()).expect("fixture");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.policy.dropout = 0.0;
    let (inputs, outputs) = metadata.policy_feature_split();
    config.policy.input_features = Some(inputs);
    config.policy.output_features = Some(outputs);

    write_processor_artifacts(&target, &config.policy, &metadata.stats)
        .expect("the processor artifacts are written");
    (dir, target)
}

fn safetensors(path: &PathBuf) -> HashMap<String, candle_core::Tensor> {
    candle_core::safetensors::load(path, &candle_core::Device::Cpu)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

fn values(tensor: &candle_core::Tensor) -> Vec<f32> {
    tensor
        .flatten_all()
        .unwrap()
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

// ---------------------------------------------------------------------------
// The names
// ---------------------------------------------------------------------------

#[test]
fn all_four_upstream_artifacts_are_written_under_their_upstream_names() {
    let (_dir, target) = written("names");
    for name in ARTIFACTS {
        assert!(
            target.join(name).is_file(),
            "the checkpoint has no {name}; upstream's save_checkpoint writes it"
        );
    }
    // And nothing else, so the step index in each filename cannot drift.
    let mut found: Vec<String> = std::fs::read_dir(&target)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    found.sort_unstable();
    let mut expected: Vec<String> = ARTIFACTS.iter().map(|name| (*name).to_owned()).collect();
    expected.sort_unstable();
    assert_eq!(found, expected);
    assert_eq!(POLICY_PREPROCESSOR_NAME, "policy_preprocessor");
    assert_eq!(POLICY_POSTPROCESSOR_NAME, "policy_postprocessor");
}

#[test]
fn saved_rename_map_can_be_read_before_feature_resolution() {
    let (_dir, target) = written("rename-map-read");
    let path = target.join("policy_preprocessor.json");
    let original = std::fs::read_to_string(&path).unwrap();
    let changed = original.replace(
        "\"rename_map\": {}",
        "\"rename_map\": {\"observation.images.left\": \"observation.images.top\"}",
    );
    assert_ne!(
        changed, original,
        "the fixture must contain an empty rename map"
    );
    std::fs::write(&path, changed).unwrap();

    let rename_map = LoadedPolicyProcessors::load_rename_map(&target).unwrap();

    assert_eq!(
        rename_map
            .get("observation.images.left")
            .map(String::as_str),
        Some("observation.images.top")
    );
}

#[test]
fn written_processor_artifacts_retain_a_saved_rename_map() {
    let dir = TempDir::new("rename-map-write");
    let target = dir.child("pretrained_model");
    let metadata = rerobot_train::data::meta::DatasetMetadata::load(&fixture_dataset()).unwrap();
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    let (inputs, outputs) = metadata.policy_feature_split();
    config.policy.input_features = Some(inputs);
    config.policy.output_features = Some(outputs);
    let rename_map = IndexMap::from([(
        "observation.images.left".to_owned(),
        "observation.images.top".to_owned(),
    )]);

    write_processor_artifacts_with_cameras_and_rename(
        &target,
        &config.policy,
        &metadata.stats,
        &IndexMap::new(),
        &rename_map,
    )
    .unwrap();

    let loaded =
        rerobot_train::processor::LoadedPolicyProcessors::load_rename_map(&target).unwrap();
    assert_eq!(
        loaded.get("observation.images.left").map(String::as_str),
        Some("observation.images.top")
    );
}

#[test]
fn loading_renamed_processor_state_uses_the_destination_feature_key() {
    let dir = TempDir::new("rename-state-load");
    let target = dir.child("pretrained_model");
    let metadata = rerobot_train::data::meta::DatasetMetadata::load(&fixture_dataset()).unwrap();
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    let (mut inputs, outputs) = metadata.policy_feature_split();
    let state = inputs.shift_remove("observation.state").unwrap();
    inputs.insert("observation.robot_state".to_owned(), state);
    config.policy.input_features = Some(inputs);
    config.policy.output_features = Some(outputs);
    config
        .policy
        .normalization_mapping
        .insert("STATE".to_owned(), NormalizationMode::MeanStd);
    let rename_map = IndexMap::from([(
        "observation.state".to_owned(),
        "observation.robot_state".to_owned(),
    )]);

    write_processor_artifacts_with_cameras_and_rename(
        &target,
        &config.policy,
        &metadata.stats,
        &IndexMap::new(),
        &rename_map,
    )
    .unwrap();

    let loaded = LoadedPolicyProcessors::load(&target, &config.policy).unwrap();
    assert_eq!(
        loaded.normalizer().mode("observation.robot_state"),
        Some(NormalizationMode::MeanStd)
    );
}

#[test]
fn loading_renamed_camera_state_uses_the_destination_feature_key() {
    let dir = TempDir::new("rename-camera-load");
    let target = dir.child("pretrained_model");
    let metadata =
        rerobot_train::data::meta::DatasetMetadata::load(&embedded_image_fixture()).unwrap();
    let mut config = reduced_config(embedded_image_fixture(), dir.child("out"));
    let (mut inputs, outputs) = metadata.policy_feature_split();
    let camera = inputs.shift_remove("observation.images.top").unwrap();
    inputs.insert("observation.images.left".to_owned(), camera);
    config.policy.input_features = Some(inputs);
    config.policy.output_features = Some(outputs);
    let rename_map = IndexMap::from([(
        "observation.images.top".to_owned(),
        "observation.images.left".to_owned(),
    )]);
    let mut cameras = IndexMap::new();
    cameras.insert(
        "observation.images.top".to_owned(),
        CameraNormalization::new(vec![0.25, 0.5, 0.75], vec![0.1, 0.2, 0.3]).unwrap(),
    );

    write_processor_artifacts_with_cameras_and_rename(
        &target,
        &config.policy,
        &metadata.stats,
        &cameras,
        &rename_map,
    )
    .unwrap();

    let loaded = LoadedPolicyProcessors::load(&target, &config.policy).unwrap();
    assert_eq!(
        loaded
            .camera_normalizations()
            .get("observation.images.left")
            .expect("the renamed camera receives its saved statistics")
            .mean(),
        &[0.25, 0.5, 0.75]
    );
}

#[test]
fn camera_normalization_is_selected_after_one_observation_rename() {
    let images = IndexMap::from([(
        "observation.images.left".to_owned(),
        Tensor::zeros((1, 3, 2, 2), candle_core::DType::F32, &Device::Cpu).unwrap(),
    )]);
    let normalizations = IndexMap::from([(
        "observation.images.top".to_owned(),
        CameraNormalization::new(vec![0.25, 0.5, 0.75], vec![0.1, 0.2, 0.3]).unwrap(),
    )]);
    let rename_map = IndexMap::from([
        (
            "observation.images.left".to_owned(),
            "observation.images.top".to_owned(),
        ),
        (
            "observation.images.top".to_owned(),
            "observation.images.wrist".to_owned(),
        ),
    ]);

    let selected = rerobot_train::processor::camera_normalizations_for_input_images(
        &images,
        &normalizations,
        &rename_map,
    );

    assert_eq!(
        selected
            .get("observation.images.left")
            .expect("the raw camera receives the target camera statistics")
            .mean(),
        &[0.25, 0.5, 0.75]
    );
}

#[test]
fn visual_processor_state_round_trips_per_camera_statistics() {
    let dir = TempDir::new("processor-camera-stats");
    let target = dir.child("pretrained_model");
    std::fs::create_dir_all(&target).unwrap();
    let metadata = rerobot_train::data::meta::DatasetMetadata::load(&embedded_image_fixture())
        .expect("embedded image fixture");
    let mut config = reduced_config(embedded_image_fixture(), dir.child("out"));
    let (inputs, outputs) = metadata.policy_feature_split();
    config.policy.input_features = Some(inputs);
    config.policy.output_features = Some(outputs);
    let mut cameras = IndexMap::new();
    cameras.insert(
        "observation.images.top".to_owned(),
        CameraNormalization::new(vec![0.25, 0.5, 0.75], vec![0.1, 0.2, 0.3]).unwrap(),
    );

    write_processor_artifacts_with_cameras(&target, &config.policy, &metadata.stats, &cameras)
        .expect("processor artifacts are written");
    let loaded =
        LoadedPolicyProcessors::load(&target, &config.policy).expect("processor artifacts reload");
    let restored = loaded
        .camera_normalizations()
        .get("observation.images.top")
        .expect("camera statistics are restored");
    assert_eq!(restored.mean(), &[0.25, 0.5, 0.75]);
    assert_eq!(restored.std(), &[0.1, 0.2, 0.3]);
}

#[test]
fn visual_processor_state_rejects_partial_camera_statistics() {
    let dir = TempDir::new("processor-partial-camera-stats");
    let target = dir.child("pretrained_model");
    std::fs::create_dir_all(&target).unwrap();
    let metadata = rerobot_train::data::meta::DatasetMetadata::load(&embedded_image_fixture())
        .expect("embedded image fixture");
    let mut config = reduced_config(embedded_image_fixture(), dir.child("out"));
    let (inputs, outputs) = metadata.policy_feature_split();
    config.policy.input_features = Some(inputs);
    config.policy.output_features = Some(outputs);
    let mut cameras = IndexMap::new();
    cameras.insert(
        "observation.images.top".to_owned(),
        CameraNormalization::new(vec![0.25, 0.5, 0.75], vec![0.1, 0.2, 0.3]).unwrap(),
    );
    write_processor_artifacts_with_cameras(&target, &config.policy, &metadata.stats, &cameras)
        .expect("processor artifacts are written");

    for filename in [
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ] {
        let path = target.join(filename);
        let mut tensors = candle_core::safetensors::load(&path, &candle_core::Device::Cpu)
            .expect("the state loads");
        tensors.remove("observation.images.top.std");
        candle_core::safetensors::save(&tensors, &path).expect("the damaged state is writable");
    }

    let error = LoadedPolicyProcessors::load(&target, &config.policy)
        .expect_err("a visual feature with only one statistic must be refused");
    assert!(
        error
            .to_string()
            .contains("requires both mean and std statistics"),
        "partial camera state was not named: {error}"
    );
}

#[test]
fn loaded_pipeline_renames_observation_keys_before_normalizing_a_batch() {
    let (_dir, target) = written("processor-rename-runtime");
    let config_path = target.join("policy_preprocessor.json");
    let config = std::fs::read_to_string(&config_path).expect("the preprocessor config reads");
    let config = config.replace(
        "\"rename_map\": {}",
        "\"rename_map\": {\"state\": \"observation.state\"}",
    );
    std::fs::write(&config_path, config).expect("the preprocessor rename map writes");

    let metadata =
        rerobot_train::data::meta::DatasetMetadata::load(&fixture_dataset()).expect("fixture");
    let mut policy = reduced_config(fixture_dataset(), target.join("out")).policy;
    let (inputs, outputs) = metadata.policy_feature_split();
    policy.input_features = Some(inputs);
    policy.output_features = Some(outputs);
    let processors =
        LoadedPolicyProcessors::load(&target, &policy).expect("the saved processor pipeline loads");

    let mut features = IndexMap::new();
    features.insert(
        "state".to_owned(),
        Tensor::new(vec![0.4375_f32, 0.5625], &Device::Cpu).expect("the raw state builds"),
    );
    let raw = Batch {
        features,
        images: IndexMap::new(),
        padding: IndexMap::from([(
            "state".to_owned(),
            Tensor::new(vec![1_u8], &Device::Cpu).expect("the padding builds"),
        )]),
        tasks: vec!["".to_owned()],
        indices: vec![0],
    };

    let processed = processors
        .process_observation_batch(&raw)
        .expect("the pipeline renames and normalizes the batch");
    assert!(processed.features.contains_key("observation.state"));
    assert!(!processed.features.contains_key("state"));
    assert!(processed.padding.contains_key("state"));
    assert!(!processed.padding.contains_key("observation.state"));
    let values = processed
        .features
        .get("observation.state")
        .expect("the renamed feature exists")
        .to_vec1::<f32>()
        .expect("the normalized tensor reads");
    assert!(values.iter().all(|value| value.abs() < 1e-6), "{values:?}");
}

#[test]
fn observation_rename_leaves_the_action_feature_untouched() {
    let mut features = IndexMap::new();
    features.insert(
        "observation.state".to_owned(),
        Tensor::new(vec![1.0_f32], &Device::Cpu).expect("the state builds"),
    );
    features.insert(
        "action".to_owned(),
        Tensor::new(vec![2.0_f32], &Device::Cpu).expect("the action builds"),
    );
    let raw = Batch {
        features,
        images: IndexMap::new(),
        padding: IndexMap::new(),
        tasks: vec![String::new()],
        indices: vec![0],
    };
    let rename_map = IndexMap::from([
        ("observation.state".to_owned(), "state".to_owned()),
        ("action".to_owned(), "renamed.action".to_owned()),
    ]);

    let renamed = rename_observation_batch(&raw, &rename_map);

    assert!(renamed.features.contains_key("state"));
    assert!(renamed.features.contains_key("action"));
    assert!(!renamed.features.contains_key("renamed.action"));
}

#[test]
fn malformed_saved_rename_map_is_rejected_before_deployment() {
    let (_dir, target) = written("processor-malformed-rename");
    let config_path = target.join("policy_preprocessor.json");
    let config = std::fs::read_to_string(&config_path).expect("the preprocessor config reads");
    std::fs::write(
        &config_path,
        config.replace("\"rename_map\": {}", "\"rename_map\": {\"state\": 7}"),
    )
    .expect("the malformed preprocessor config writes");

    let mut policy = reduced_config(fixture_dataset(), target.join("out")).policy;
    let metadata =
        rerobot_train::data::meta::DatasetMetadata::load(&fixture_dataset()).expect("fixture");
    let (inputs, outputs) = metadata.policy_feature_split();
    policy.input_features = Some(inputs);
    policy.output_features = Some(outputs);
    let error = LoadedPolicyProcessors::load(&target, &policy)
        .expect_err("a non-string rename target must be rejected");
    assert!(error
        .to_string()
        .contains("rename_map entry \"state\" must be a string"));
}

// ---------------------------------------------------------------------------
// The JSON, byte for byte
// ---------------------------------------------------------------------------

#[test]
fn the_preprocessor_json_is_byte_identical_to_upstreams() {
    let (_dir, target) = written("pre-json");
    let ours = std::fs::read_to_string(target.join("policy_preprocessor.json")).unwrap();
    let theirs = std::fs::read_to_string(goldens_dir().join("policy_preprocessor.json")).unwrap();
    assert_eq!(
        ours, theirs,
        "policy_preprocessor.json differs from the one upstream's save_pretrained wrote"
    );
}

#[test]
fn the_postprocessor_json_is_byte_identical_to_upstreams() {
    let (_dir, target) = written("post-json");
    let ours = std::fs::read_to_string(target.join("policy_postprocessor.json")).unwrap();
    let theirs = std::fs::read_to_string(goldens_dir().join("policy_postprocessor.json")).unwrap();
    assert_eq!(
        ours, theirs,
        "policy_postprocessor.json differs from the one upstream's save_pretrained wrote"
    );
}

#[test]
fn the_preprocessor_declares_upstreams_four_steps_in_order() {
    // Structural, on top of the byte comparison, so a failure says *what* drifted.
    let (_dir, target) = written("pre-steps");
    let text = std::fs::read_to_string(target.join("policy_preprocessor.json")).unwrap();
    let JsonLike::Object(root) = loads(&text).unwrap() else {
        panic!("not an object")
    };
    assert_eq!(root["name"], JsonLike::Str("policy_preprocessor".into()));
    let JsonLike::Array(steps) = &root["steps"] else {
        panic!("`steps` is not a list")
    };
    let names: Vec<&str> = steps
        .iter()
        .map(|step| match step {
            JsonLike::Object(object) => match &object["registry_name"] {
                JsonLike::Str(name) => name.as_str(),
                _ => panic!("registry_name is not a string"),
            },
            _ => panic!("a step is not an object"),
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "rename_observations_processor",
            "to_batch_processor",
            "device_processor",
            "normalizer_processor",
        ]
    );
    // The normalizer is step 3, which is where its filename's `step_3` comes from.
    let JsonLike::Object(normalizer) = &steps[3] else {
        panic!()
    };
    assert_eq!(
        normalizer["state_file"],
        JsonLike::Str("policy_preprocessor_step_3_normalizer_processor.safetensors".into())
    );
}

#[test]
fn the_postprocessor_declares_upstreams_two_steps_in_order() {
    let (_dir, target) = written("post-steps");
    let text = std::fs::read_to_string(target.join("policy_postprocessor.json")).unwrap();
    let JsonLike::Object(root) = loads(&text).unwrap() else {
        panic!("not an object")
    };
    assert_eq!(root["name"], JsonLike::Str("policy_postprocessor".into()));
    let JsonLike::Array(steps) = &root["steps"] else {
        panic!("`steps` is not a list")
    };
    assert_eq!(steps.len(), 2);
    let JsonLike::Object(unnormalizer) = &steps[0] else {
        panic!()
    };
    assert_eq!(
        unnormalizer["registry_name"],
        JsonLike::Str("unnormalizer_processor".into())
    );
    assert_eq!(
        unnormalizer["state_file"],
        JsonLike::Str("policy_postprocessor_step_0_unnormalizer_processor.safetensors".into())
    );
    // The unnormalizer's declared features are the *output* features only: it turns
    // the policy's prediction back into action units and touches nothing else.
    let JsonLike::Object(config) = &unnormalizer["config"] else {
        panic!()
    };
    let JsonLike::Object(features) = &config["features"] else {
        panic!()
    };
    assert_eq!(features.keys().collect::<Vec<_>>(), vec!["action"]);
}

// ---------------------------------------------------------------------------
// The safetensors state
// ---------------------------------------------------------------------------

#[test]
fn the_normalizer_state_matches_upstreams_tensor_for_tensor() {
    let (_dir, target) = written("pre-state");
    for name in [
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ] {
        let ours = safetensors(&target.join(name));
        let theirs = safetensors(&goldens_dir().join(name));

        let mut our_keys: Vec<&str> = ours.keys().map(String::as_str).collect();
        let mut their_keys: Vec<&str> = theirs.keys().map(String::as_str).collect();
        our_keys.sort_unstable();
        their_keys.sort_unstable();
        assert_eq!(our_keys, their_keys, "{name}: the tensor names differ");

        for key in &their_keys {
            let ours = &ours[*key];
            let theirs = &theirs[*key];
            assert_eq!(
                ours.dims(),
                theirs.dims(),
                "{name}: {key} has shape {:?} but upstream wrote {:?}",
                ours.dims(),
                theirs.dims()
            );
            assert_eq!(
                ours.dtype(),
                candle_core::DType::F32,
                "{name}: {key} must be f32, as upstream's cast produces"
            );
            assert_eq!(
                values(ours),
                values(theirs),
                "{name}: {key} differs from upstream's value"
            );
        }
    }
}

#[test]
fn the_normalizer_state_covers_every_dataset_feature_not_only_the_policys() {
    // Upstream hands `NormalizerProcessorStep` the dataset's whole `stats` dict, so
    // the saved state carries `timestamp`, `frame_index` and the rest too. Writing
    // only the policy's three features would look reasonable and be wrong.
    let (_dir, target) = written("coverage");
    let state =
        safetensors(&target.join("policy_preprocessor_step_3_normalizer_processor.safetensors"));
    for key in [
        "observation.state.mean",
        "observation.state.std",
        "observation.environment_state.mean",
        "action.mean",
        "action.std",
        "timestamp.mean",
        "frame_index.mean",
        "episode_index.mean",
        "index.mean",
        "task_index.mean",
    ] {
        assert!(state.contains_key(key), "the normalizer state has no {key}");
    }
    // `count` is a single element even where the feature is wider, as upstream's
    // `np.atleast_1d` leaves it.
    assert_eq!(state["action.count"].dims(), &[1]);
    assert_eq!(state["action.mean"].dims(), &[2]);
}

#[test]
fn the_statistics_are_the_ones_the_run_actually_normalized_with() {
    // The point of saving them: they have to be the numbers the weights were trained
    // against, not a default or a re-derivation.
    let (_dir, target) = written("values");
    let state =
        safetensors(&target.join("policy_preprocessor_step_3_normalizer_processor.safetensors"));
    assert_eq!(
        values(&state["observation.state.mean"]),
        vec![0.4375, 0.5625]
    );
    assert_eq!(values(&state["action.mean"]), vec![0.0625, -0.0625]);
    assert_eq!(
        values(&state["observation.environment_state.mean"]),
        vec![11.5, -2.5]
    );
}
