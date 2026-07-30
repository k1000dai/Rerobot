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

use common::{fixture_dataset, reduced_config, TempDir};
use rerobot_core::dataset::json::{loads, JsonLike};
use rerobot_train::processor::{
    write_processor_artifacts, POLICY_POSTPROCESSOR_NAME, POLICY_PREPROCESSOR_NAME,
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
