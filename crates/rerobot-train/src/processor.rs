//! The pre/postprocessor artifacts every upstream checkpoint carries.
//!
//! `lerobot/scripts/lerobot_train.py:675-683` hands both processors to
//! `save_checkpoint`, and `lerobot/common/train_utils.py:145-155` calls
//! `save_pretrained` on each, writing four files into `pretrained_model/`:
//!
//! | File | What it is |
//! | --- | --- |
//! | `policy_preprocessor.json` | the four steps `make_pre_post_processors` builds for ACT, in order |
//! | `policy_preprocessor_step_3_normalizer_processor.safetensors` | the dataset statistics the normalizer step divides by |
//! | `policy_postprocessor.json` | the two steps that turn a prediction back into action units |
//! | `policy_postprocessor_step_0_unnormalizer_processor.safetensors` | the same statistics, for the inverse |
//!
//! Without them a checkpoint has lost its normalization state. The weights were
//! trained on inputs divided by a particular mean and standard deviation, so
//! anything that loads the policy has to know which — a `config.json` and a
//! `model.safetensors` alone are not a deployable pretrained artifact.
//!
//! # What is and is not ported here
//!
//! The *artifacts* are ported: their names, their JSON structure byte for byte, and
//! their safetensors contents. The processor *runtime* is not — Rerobot normalizes
//! with [`rerobot_core::policy::normalize::Normalizer`] rather than by running a
//! pipeline of registry-named steps. So the step list this module writes is a
//! faithful description of what upstream would have built for this configuration,
//! and three of the four preprocessor steps
//! (`rename_observations_processor`, `to_batch_processor`, `device_processor`) name
//! behaviour Rerobot performs structurally rather than as a step. That is recorded
//! in `docs/compatibility.md`; the alternative — omitting them — would produce a
//! file upstream cannot load.

use crate::error::{Result, TrainError};
use rerobot_core::dataset::json::{dumps_indent_ascii, JsonLike, JsonObject};
use rerobot_core::dataset::stats::DatasetStats;
use rerobot_core::policy::act::ActConfig;
use rerobot_core::policy::normalize::NORMALIZATION_EPS;
use rerobot_core::types::PolicyFeature;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// `POLICY_PREPROCESSOR_DEFAULT_NAME`.
pub const POLICY_PREPROCESSOR_NAME: &str = "policy_preprocessor";
/// `POLICY_POSTPROCESSOR_DEFAULT_NAME`.
pub const POLICY_POSTPROCESSOR_NAME: &str = "policy_postprocessor";

/// The index of the normalizer step in upstream's ACT preprocessor pipeline.
///
/// It appears in the state file's name, so it is a wire detail rather than an
/// implementation one: `policy_preprocessor_step_3_normalizer_processor.safetensors`.
const NORMALIZER_STEP_INDEX: usize = 3;

/// The index of the unnormalizer step in upstream's ACT postprocessor pipeline.
const UNNORMALIZER_STEP_INDEX: usize = 0;

/// `ProcessorPipeline._save_pretrained` writes `json.dump(..., indent=2)`, not the
/// four-space indent a policy `config.json` uses.
const PROCESSOR_JSON_INDENT: usize = 2;

/// Write all four artifacts into `directory`, returning their paths in write order.
pub fn write_processor_artifacts(
    directory: &Path,
    policy: &ActConfig,
    stats: &DatasetStats,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(directory).map_err(|error| TrainError::io(directory, &error))?;

    let inputs = policy.input_features.clone().unwrap_or_default();
    let outputs = policy.output_features.clone().unwrap_or_default();
    let device = policy.device.clone().unwrap_or_else(|| "cpu".to_owned());

    // The normalizer sees inputs *and* outputs: it normalizes the action target too,
    // because the loss is computed in normalized space.
    let mut normalizer_features = inputs.clone();
    for (key, feature) in &outputs {
        normalizer_features.insert(key.clone(), feature.clone());
    }

    let normalizer_state = format!(
        "{POLICY_PREPROCESSOR_NAME}_step_{NORMALIZER_STEP_INDEX}_normalizer_processor.safetensors"
    );
    let unnormalizer_state = format!(
        "{POLICY_POSTPROCESSOR_NAME}_step_{UNNORMALIZER_STEP_INDEX}_unnormalizer_processor.safetensors"
    );

    let mut written = Vec::with_capacity(4);

    let preprocessor = preprocessor_json(policy, &normalizer_features, &device, &normalizer_state);
    written.push(write_json(
        directory,
        &format!("{POLICY_PREPROCESSOR_NAME}.json"),
        &preprocessor,
    )?);
    written.push(write_state(directory, &normalizer_state, stats)?);

    let postprocessor = postprocessor_json(policy, &outputs, &device, &unnormalizer_state);
    written.push(write_json(
        directory,
        &format!("{POLICY_POSTPROCESSOR_NAME}.json"),
        &postprocessor,
    )?);
    // Upstream hands the unnormalizer the same `stats` dict, so its state file is
    // the same tensors even though its declared features are only the outputs.
    written.push(write_state(directory, &unnormalizer_state, stats)?);

    Ok(written)
}

fn write_json(directory: &Path, name: &str, value: &JsonLike) -> Result<PathBuf> {
    let path = directory.join(name);
    // No trailing newline: `json.dump` does not write one.
    std::fs::write(&path, dumps_indent_ascii(value, PROCESSOR_JSON_INDENT))
        .map_err(|error| TrainError::io(&path, &error))?;
    Ok(path)
}

/// `NormalizerProcessorStep.state_dict()`: every statistic of every feature the
/// dataset carries, flattened to `<feature>.<statistic>` and cast to `f32`.
///
/// Every feature, not only the policy's: upstream passes `dataset.meta.stats`
/// wholesale, so `timestamp`, `frame_index`, `episode_index`, `index` and
/// `task_index` are in there too.
fn write_state(directory: &Path, name: &str, stats: &DatasetStats) -> Result<PathBuf> {
    let path = directory.join(name);
    let mut tensors: HashMap<String, candle_core::Tensor> = HashMap::new();
    for feature in stats.keys() {
        let entry = stats
            .get(feature)
            .expect("the key came from `keys()` a line above");
        for statistic in entry.keys() {
            let values = entry
                .get(statistic)
                .expect("the name came from `keys()` a line above");
            let floats: Vec<f32> = values.iter().map(|value| *value as f32).collect();
            let width = floats.len();
            tensors.insert(
                format!("{feature}.{statistic}"),
                candle_core::Tensor::from_vec(floats, width, &candle_core::Device::Cpu)?,
            );
        }
    }
    candle_core::safetensors::save(&tensors, &path)?;
    Ok(path)
}

fn preprocessor_json(
    policy: &ActConfig,
    features: &indexmap::IndexMap<String, PolicyFeature>,
    device: &str,
    state_file: &str,
) -> JsonLike {
    let mut steps = Vec::with_capacity(4);

    // `rename_observations_processor`, with the empty map a fresh run produces:
    // `--rename_map` requires a pretrained checkpoint, which this slice refuses.
    let mut rename = JsonObject::new();
    rename.insert("rename_map".into(), JsonLike::Object(JsonObject::new()));
    steps.push(step("rename_observations_processor", rename, None));

    steps.push(step("to_batch_processor", JsonObject::new(), None));
    steps.push(step("device_processor", device_config(device), None));
    steps.push(step(
        "normalizer_processor",
        normalization_config(policy, features),
        Some(state_file),
    ));

    pipeline(POLICY_PREPROCESSOR_NAME, steps)
}

fn postprocessor_json(
    policy: &ActConfig,
    outputs: &indexmap::IndexMap<String, PolicyFeature>,
    device: &str,
    state_file: &str,
) -> JsonLike {
    let steps = vec![
        step(
            "unnormalizer_processor",
            normalization_config(policy, outputs),
            Some(state_file),
        ),
        step("device_processor", device_config(device), None),
    ];
    pipeline(POLICY_POSTPROCESSOR_NAME, steps)
}

fn pipeline(name: &str, steps: Vec<JsonLike>) -> JsonLike {
    let mut root = JsonObject::new();
    root.insert("name".into(), JsonLike::Str(name.to_owned()));
    root.insert("steps".into(), JsonLike::Array(steps));
    JsonLike::Object(root)
}

fn step(registry_name: &str, config: JsonObject, state_file: Option<&str>) -> JsonLike {
    let mut object = JsonObject::new();
    object.insert(
        "registry_name".into(),
        JsonLike::Str(registry_name.to_owned()),
    );
    object.insert("config".into(), JsonLike::Object(config));
    if let Some(state_file) = state_file {
        object.insert("state_file".into(), JsonLike::Str(state_file.to_owned()));
    }
    JsonLike::Object(object)
}

fn device_config(device: &str) -> JsonObject {
    let mut config = JsonObject::new();
    config.insert("device".into(), JsonLike::Str(device.to_owned()));
    // `float_dtype` is null unless the run asked for a non-default dtype, which this
    // slice does not support.
    config.insert("float_dtype".into(), JsonLike::Null);
    config
}

/// The `eps` / `features` / `norm_map` block both normalization steps carry.
fn normalization_config(
    policy: &ActConfig,
    features: &indexmap::IndexMap<String, PolicyFeature>,
) -> JsonObject {
    let mut config = JsonObject::new();
    config.insert("eps".into(), JsonLike::Float(NORMALIZATION_EPS));

    let mut encoded = JsonObject::new();
    for (key, feature) in features {
        encoded.insert(key.clone(), encode_feature(feature));
    }
    config.insert("features".into(), JsonLike::Object(encoded));

    let mut norm_map = JsonObject::new();
    for (feature_type, mode) in &policy.normalization_mapping {
        norm_map.insert(
            feature_type.clone(),
            JsonLike::Str(mode.as_str().to_owned()),
        );
    }
    config.insert("norm_map".into(), JsonLike::Object(norm_map));
    config
}

/// A `PolicyFeature` the way `draccus.encode` writes it: the enum's *name*, and the
/// shape as a list.
fn encode_feature(feature: &PolicyFeature) -> JsonLike {
    let mut object = JsonObject::new();
    object.insert(
        "type".into(),
        JsonLike::Str(feature.r#type.as_str().to_owned()),
    );
    object.insert(
        "shape".into(),
        JsonLike::Array(
            feature
                .shape
                .iter()
                .map(|dimension| JsonLike::Int(dimension.clone()))
                .collect(),
        ),
    );
    JsonLike::Object(object)
}
