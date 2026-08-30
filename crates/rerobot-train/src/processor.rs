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
//! their safetensors contents. The native ACT deployment boundary also validates
//! and consumes the saved rename map and numeric normalizer state for scalar
//! observations and action unnormalization. It does not yet execute an arbitrary
//! registry-named pipeline: the batch, device and full multi-step processor
//! lifecycles remain represented structurally rather than exposed as a general
//! runtime. That boundary is recorded in `docs/compatibility.md`; the alternative
//! — omitting the steps — would produce a file upstream cannot load.

use crate::data::batch::Batch;
use crate::data::image::CameraNormalization;
use crate::data::meta::ACTION;
use crate::error::{Result, TrainError};
use candle_core::Tensor;
use indexmap::IndexMap;
use rerobot_core::dataset::json::{dumps_indent_ascii, JsonLike, JsonObject};
use rerobot_core::dataset::stats::{DatasetStats, FeatureStats};
use rerobot_core::policy::act::ActConfig;
use rerobot_core::policy::normalize::{Normalizer, NORMALIZATION_EPS};
use rerobot_core::types::PolicyFeature;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The loaded processor state used by hardware-independent policy inference.
///
/// The checkpoint's processor artifacts are not documentation: upstream loads
/// them and applies the saved statistics to observations before inference and to
/// actions after inference. This type is the native ACT subset of that runtime.
#[derive(Debug, Clone)]
pub struct LoadedPolicyProcessors {
    normalizer: Normalizer,
    camera_normalizations: IndexMap<String, CameraNormalization>,
    rename_map: IndexMap<String, String>,
    stats: DatasetStats,
}

impl LoadedPolicyProcessors {
    /// Read only the saved observation-key mapping.
    ///
    /// Training uses this small first pass to resolve dataset feature names before
    /// it constructs the model. The complete [`Self::load`] call still validates
    /// both pipelines and all statistics before the session starts.
    pub fn load_rename_map(checkpoint_dir: &Path) -> Result<IndexMap<String, String>> {
        let path = checkpoint_dir.join(POLICY_PREPROCESSOR_NAME.to_owned() + ".json");
        let preprocessor = load_json(&path, "policy_preprocessor.json")?;
        rename_map_from_pipeline(&preprocessor)
    }

    /// Load and validate the pre/postprocessor artifacts from one checkpoint.
    ///
    /// The state is taken from the checkpoint, not from the observation dataset.
    /// This matters when a policy is deployed against another dataset whose
    /// statistics describe a different collection run.
    pub fn load(checkpoint_dir: &Path, policy: &ActConfig) -> Result<Self> {
        Self::load_internal(checkpoint_dir, policy, None)
    }

    /// Load processor artifacts while replacing their saved observation rename map.
    ///
    /// Training passes the user-facing `rename_map` here when a pretrained policy
    /// is configured with an explicit mapping. The saved JSON is still validated;
    /// only the mapping applied to statistics and runtime batches is overridden.
    pub fn load_with_rename_map(
        checkpoint_dir: &Path,
        policy: &ActConfig,
        rename_map: &IndexMap<String, String>,
    ) -> Result<Self> {
        Self::load_internal(checkpoint_dir, policy, Some(rename_map))
    }

    fn load_internal(
        checkpoint_dir: &Path,
        policy: &ActConfig,
        rename_map_override: Option<&IndexMap<String, String>>,
    ) -> Result<Self> {
        let preprocessor_path = checkpoint_dir.join(POLICY_PREPROCESSOR_NAME.to_owned() + ".json");
        let postprocessor_path =
            checkpoint_dir.join(POLICY_POSTPROCESSOR_NAME.to_owned() + ".json");
        let preprocessor = load_json(&preprocessor_path, "policy_preprocessor.json")?;
        let postprocessor = load_json(&postprocessor_path, "policy_postprocessor.json")?;
        validate_pipeline(
            &preprocessor,
            POLICY_PREPROCESSOR_NAME,
            &[
                "rename_observations_processor",
                "to_batch_processor",
                "device_processor",
                "normalizer_processor",
            ],
            Some("policy_preprocessor_step_3_normalizer_processor.safetensors"),
        )?;
        validate_pipeline(
            &postprocessor,
            POLICY_POSTPROCESSOR_NAME,
            &["unnormalizer_processor", "device_processor"],
            Some("policy_postprocessor_step_0_unnormalizer_processor.safetensors"),
        )?;
        let saved_rename_map = rename_map_from_pipeline(&preprocessor)?;
        let rename_map = rename_map_override.unwrap_or(&saved_rename_map);

        let preprocessor_state_path =
            checkpoint_dir.join("policy_preprocessor_step_3_normalizer_processor.safetensors");
        let postprocessor_state_path =
            checkpoint_dir.join("policy_postprocessor_step_0_unnormalizer_processor.safetensors");
        let (stats, preprocessor_state) = load_state(&preprocessor_state_path)?;
        let (_, postprocessor_state) = load_state(&postprocessor_state_path)?;
        if preprocessor_state
            .keys()
            .collect::<std::collections::BTreeSet<_>>()
            != postprocessor_state
                .keys()
                .collect::<std::collections::BTreeSet<_>>()
        {
            return Err(TrainError::checkpoint(
                &postprocessor_state_path,
                "the postprocessor state keys differ from the preprocessor state",
            ));
        }

        // Processor state is saved in the source namespace, then the rename step
        // exposes the destination namespace to the normalizer. Apply the same
        // one-pass mapping to the grouped statistics before resolving either
        // scalar or camera features.
        let stats = rename_stats(&stats, rename_map);

        let mut features = policy.input_features.clone().unwrap_or_default();
        features.extend(policy.output_features.clone().unwrap_or_default());
        let camera_normalizations = camera_normalizations_from_stats(&features, &stats)?;
        let scalar_features = features
            .into_iter()
            .filter(|(_, feature)| feature.r#type != rerobot_core::types::FeatureType::Visual)
            .collect();
        let normalizer = Normalizer::new(&scalar_features, &policy.normalization_mapping, &stats)
            .map_err(|error| TrainError::Metadata(error.to_string()))?;
        Ok(Self {
            normalizer,
            camera_normalizations,
            rename_map: (*rename_map).clone(),
            stats,
        })
    }

    /// The normalizer used by the preprocessor and postprocessor.
    pub fn normalizer(&self) -> &Normalizer {
        &self.normalizer
    }

    /// Per-camera normalization restored from visual feature statistics.
    pub fn camera_normalizations(&self) -> &IndexMap<String, CameraNormalization> {
        &self.camera_normalizations
    }

    /// The saved statistics after the observation rename step has been applied.
    pub fn stats(&self) -> &DatasetStats {
        &self.stats
    }

    /// Apply the saved observation pipeline's rename and normalization steps.
    ///
    /// This is the native subset of upstream's preprocessor runtime used by ACT.
    /// Renaming happens before normalization, and the input batch is not mutated.
    pub fn process_observation_batch(&self, batch: &Batch) -> Result<Batch> {
        let renamed = rename_observation_batch(batch, &self.rename_map);
        renamed.normalized(&self.normalizer)
    }

    /// The saved observation-key mapping, in upstream insertion order.
    pub fn rename_map(&self) -> &IndexMap<String, String> {
        &self.rename_map
    }
}

/// Rename observation feature and camera keys using upstream's one-pass mapping.
///
/// `IndexMap::insert` preserves the first insertion position while replacing a
/// colliding value, matching Python dict assignment. Only exact observation keys
/// in the mapping are renamed; derived padding keys are not implicitly rewritten.
fn renamed_observation_key(key: &str, rename_map: &IndexMap<String, String>) -> String {
    if key == ACTION {
        return key.to_owned();
    }
    if let Some(mapped) = rename_map.get(key) {
        return mapped.clone();
    }
    key.to_owned()
}

/// Apply a saved observation rename to grouped processor statistics.
fn rename_stats(stats: &DatasetStats, rename_map: &IndexMap<String, String>) -> DatasetStats {
    let mut renamed = IndexMap::with_capacity(stats.keys().count());
    for feature in stats.keys() {
        let target = renamed_observation_key(feature, rename_map);
        let source = stats
            .get(feature)
            .expect("a key yielded by DatasetStats::keys must be present");
        let mut values = IndexMap::new();
        for statistic in source.keys() {
            values.insert(
                statistic.to_owned(),
                source
                    .get(statistic)
                    .expect("a key yielded by FeatureStats::keys must be present")
                    .to_vec(),
            );
        }
        renamed.insert(target, FeatureStats::from_entries(values));
    }
    DatasetStats::from_entries(renamed)
}

/// Rename a collection of raw camera tensors using the saved observation mapping.
pub fn rename_observation_images(
    images: &IndexMap<String, Tensor>,
    rename_map: &IndexMap<String, String>,
) -> IndexMap<String, Tensor> {
    let mut renamed = IndexMap::with_capacity(images.len());
    for (key, tensor) in images {
        renamed.insert(renamed_observation_key(key, rename_map), tensor.clone());
    }
    renamed
}

/// Select camera statistics for raw input keys before the observation mapping runs.
///
/// The returned map deliberately keeps the raw keys: [`Batch::with_image_normalizations`]
/// attaches and normalizes those tensors before [`rename_observation_batch`] performs
/// the one-pass key rewrite. Applying the rewrite while constructing the map and then
/// again while processing a batch would make a mapping such as `left -> top` followed
/// by `top -> wrist` depend on the second entry, which is not upstream's behavior.
pub fn camera_normalizations_for_input_images(
    images: &IndexMap<String, Tensor>,
    normalizations: &IndexMap<String, CameraNormalization>,
    rename_map: &IndexMap<String, String>,
) -> IndexMap<String, CameraNormalization> {
    let mut selected = IndexMap::with_capacity(images.len());
    for key in images.keys() {
        let renamed = renamed_observation_key(key, rename_map);
        if let Some(normalization) = normalizations
            .get(&renamed)
            .or_else(|| normalizations.get(key))
        {
            selected.insert(key.clone(), normalization.clone());
        }
    }
    selected
}

/// Rename a batch's observation features and camera keys using the saved one-pass
/// mapping. Padding masks are not observation keys and therefore pass through
/// untouched. The input tensors are shared by clone, but the returned maps and
/// task/index vectors are independent.
pub fn rename_observation_batch(batch: &Batch, rename_map: &IndexMap<String, String>) -> Batch {
    let mut features = IndexMap::with_capacity(batch.features.len());
    for (key, tensor) in &batch.features {
        features.insert(renamed_observation_key(key, rename_map), tensor.clone());
    }
    let images = rename_observation_images(&batch.images, rename_map);
    // The upstream observation processor sees one ordered dictionary. Batch keeps
    // scalar and camera tensors in separate maps, so resolve a cross-map rename
    // collision as the later camera entry would overwrite the scalar entry.
    for key in images.keys() {
        features.shift_remove(key);
    }
    Batch {
        features,
        images,
        padding: batch.padding.clone(),
        tasks: batch.tasks.clone(),
        indices: batch.indices.clone(),
    }
}

fn rename_map_from_pipeline(document: &JsonLike) -> Result<IndexMap<String, String>> {
    let JsonLike::Object(root) = document else {
        return Err(TrainError::Metadata(
            "policy_preprocessor.json root must be an object".to_owned(),
        ));
    };
    let Some(JsonLike::Array(steps)) = root.get("steps") else {
        return Err(TrainError::Metadata(
            "policy_preprocessor.json steps must be an array".to_owned(),
        ));
    };
    let Some(JsonLike::Object(step)) = steps.first() else {
        return Err(TrainError::Metadata(
            "policy_preprocessor.json has no rename step".to_owned(),
        ));
    };
    let Some(JsonLike::Object(config)) = step.get("config") else {
        return Err(TrainError::Metadata(
            "policy_preprocessor.json rename step config must be an object".to_owned(),
        ));
    };
    let Some(rename_map) = config.get("rename_map") else {
        return Err(TrainError::Metadata(
            "policy_preprocessor.json rename_map is missing".to_owned(),
        ));
    };
    let JsonLike::Object(rename_map) = rename_map else {
        return Err(TrainError::Metadata(
            "policy_preprocessor.json rename_map must be an object".to_owned(),
        ));
    };
    let mut parsed = IndexMap::with_capacity(rename_map.len());
    for (old, new) in rename_map {
        let JsonLike::Str(new) = new else {
            return Err(TrainError::Metadata(format!(
                "policy_preprocessor.json rename_map entry {old:?} must be a string"
            )));
        };
        parsed.insert(old.clone(), new.clone());
    }
    Ok(parsed)
}

const MAX_PROCESSOR_JSON_BYTES: u64 = 16 * 1024 * 1024;
/// Processor statistics are small in every supported LeRobot dataset. Keep the
/// whole-file bound below the general model checkpoint bound because Candle's
/// convenience loader materializes every tensor before this module can inspect
/// the feature names.
const MAX_PROCESSOR_STATE_BYTES: u64 = 64 * 1024 * 1024;

fn load_json(path: &Path, label: &str) -> Result<JsonLike> {
    let file = std::fs::File::open(path).map_err(|error| TrainError::io(path, &error))?;
    let mut bytes = Vec::new();
    file.take(MAX_PROCESSOR_JSON_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| TrainError::io(path, &error))?;
    if bytes.len() as u64 > MAX_PROCESSOR_JSON_BYTES {
        return Err(TrainError::checkpoint(
            path,
            format!("{label} exceeds the {MAX_PROCESSOR_JSON_BYTES}-byte limit"),
        ));
    }
    let text = String::from_utf8(bytes)
        .map_err(|error| TrainError::checkpoint(path, format!("{label} is not UTF-8: {error}")))?;
    rerobot_core::dataset::json::loads(&text).map_err(|error| {
        TrainError::checkpoint(path, format!("{label} is not valid JSON: {error}"))
    })
}

fn validate_pipeline(
    document: &JsonLike,
    expected_name: &str,
    expected_steps: &[&str],
    expected_state_file: Option<&str>,
) -> Result<()> {
    let JsonLike::Object(root) = document else {
        return Err(TrainError::Metadata(format!(
            "{expected_name}.json root must be an object"
        )));
    };
    if root.get("name") != Some(&JsonLike::Str(expected_name.to_owned())) {
        return Err(TrainError::Metadata(format!(
            "{expected_name}.json has the wrong pipeline name"
        )));
    }
    let Some(JsonLike::Array(steps)) = root.get("steps") else {
        return Err(TrainError::Metadata(format!(
            "{expected_name}.json steps must be an array"
        )));
    };
    if steps.len() != expected_steps.len() {
        return Err(TrainError::Metadata(format!(
            "{expected_name}.json has {} steps, expected {}",
            steps.len(),
            expected_steps.len()
        )));
    }
    for (index, (step, expected)) in steps.iter().zip(expected_steps).enumerate() {
        let JsonLike::Object(step) = step else {
            return Err(TrainError::Metadata(format!(
                "{expected_name}.json step {index} must be an object"
            )));
        };
        if step.get("registry_name") != Some(&JsonLike::Str((*expected).to_owned())) {
            return Err(TrainError::Metadata(format!(
                "{expected_name}.json step {index} is not {expected}"
            )));
        }
        let state_file = step.get("state_file").and_then(|value| match value {
            JsonLike::Str(value) => Some(value.as_str()),
            _ => None,
        });
        if index == expected_steps.len() - 1 && expected_name == POLICY_PREPROCESSOR_NAME {
            if state_file != expected_state_file {
                return Err(TrainError::Metadata(
                    "the normalizer processor state file is missing or has the wrong name"
                        .to_owned(),
                ));
            }
        } else if expected_name == POLICY_POSTPROCESSOR_NAME && index == 0 {
            if state_file != expected_state_file {
                return Err(TrainError::Metadata(
                    "the unnormalizer processor state file is missing or has the wrong name"
                        .to_owned(),
                ));
            }
        } else if state_file.is_some() {
            return Err(TrainError::Metadata(format!(
                "{expected_name}.json stateless step {index} unexpectedly has state"
            )));
        }
    }
    Ok(())
}

fn load_state(path: &Path) -> Result<(DatasetStats, HashMap<String, Vec<f32>>)> {
    crate::model::params::validate_safetensors_container(path)?;
    let file_size = std::fs::metadata(path)
        .map_err(|error| TrainError::io(path, &error))?
        .len();
    if file_size > MAX_PROCESSOR_STATE_BYTES {
        return Err(TrainError::checkpoint(
            path,
            format!("processor state exceeds the maximum of {MAX_PROCESSOR_STATE_BYTES} bytes"),
        ));
    }
    let tensors =
        candle_core::safetensors::load(path, &candle_core::Device::Cpu).map_err(|error| {
            TrainError::checkpoint(path, format!("cannot load processor state: {error}"))
        })?;
    if tensors.is_empty() {
        return Err(TrainError::checkpoint(
            path,
            "processor state has no tensors",
        ));
    }
    let mut grouped: IndexMap<String, IndexMap<String, Vec<f64>>> = IndexMap::new();
    let mut flat = HashMap::with_capacity(tensors.len());
    for (key, tensor) in tensors {
        let Some((feature, statistic)) = key.rsplit_once('.') else {
            return Err(TrainError::checkpoint(
                path,
                format!("processor state tensor {key:?} has no feature.statistic name"),
            ));
        };
        if feature.is_empty() || statistic.is_empty() || tensor.dtype() != candle_core::DType::F32 {
            return Err(TrainError::checkpoint(
                path,
                format!("processor state tensor {key:?} has an invalid name or dtype"),
            ));
        }
        let values = tensor
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|error| {
                TrainError::checkpoint(
                    path,
                    format!("cannot read processor state tensor {key:?}: {error}"),
                )
            })?;
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err(TrainError::checkpoint(
                path,
                format!("processor state tensor {key:?} is empty or non-finite"),
            ));
        }
        flat.insert(key.to_owned(), values.clone());
        grouped.entry(feature.to_owned()).or_default().insert(
            statistic.to_owned(),
            values.into_iter().map(f64::from).collect(),
        );
    }
    let stats = DatasetStats::from_entries(
        grouped
            .into_iter()
            .map(|(feature, values)| (feature, FeatureStats::from_entries(values)))
            .collect(),
    );
    Ok((stats, flat))
}

fn camera_normalizations_from_stats(
    features: &IndexMap<String, rerobot_core::types::PolicyFeature>,
    stats: &DatasetStats,
) -> Result<IndexMap<String, CameraNormalization>> {
    let mut normalizations = IndexMap::new();
    for (key, feature) in features {
        if feature.r#type != rerobot_core::types::FeatureType::Visual {
            continue;
        }
        let feature_stats = stats.get(key);
        match (
            feature_stats.and_then(FeatureStats::mean),
            feature_stats.and_then(FeatureStats::std),
        ) {
            (Some(mean), Some(std)) => {
                normalizations.insert(
                    key.clone(),
                    CameraNormalization::new(
                        mean.iter().map(|value| *value as f32).collect(),
                        std.iter().map(|value| *value as f32).collect(),
                    )?,
                );
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(TrainError::Metadata(format!(
                    "camera feature {key:?} requires both mean and std statistics"
                )));
            }
            (None, None) => {}
        }
    }
    Ok(normalizations)
}

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
    write_processor_artifacts_with_cameras(directory, policy, stats, &IndexMap::new())
}

/// Write processor artifacts while retaining the nested per-camera statistics
/// that the upstream normalizer broadcasts across image height and width.
pub fn write_processor_artifacts_with_cameras(
    directory: &Path,
    policy: &ActConfig,
    stats: &DatasetStats,
    camera_stats: &IndexMap<String, CameraNormalization>,
) -> Result<Vec<PathBuf>> {
    write_processor_artifacts_with_cameras_and_rename(
        directory,
        policy,
        stats,
        camera_stats,
        &IndexMap::new(),
    )
}

/// Write processor artifacts while retaining camera statistics and the saved
/// observation-key mapping.
pub fn write_processor_artifacts_with_cameras_and_rename(
    directory: &Path,
    policy: &ActConfig,
    stats: &DatasetStats,
    camera_stats: &IndexMap<String, CameraNormalization>,
    rename_map: &IndexMap<String, String>,
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

    let preprocessor = preprocessor_json(
        policy,
        &normalizer_features,
        &device,
        &normalizer_state,
        rename_map,
    );
    written.push(write_json(
        directory,
        &format!("{POLICY_PREPROCESSOR_NAME}.json"),
        &preprocessor,
    )?);
    written.push(write_state(
        directory,
        &normalizer_state,
        stats,
        camera_stats,
    )?);

    let postprocessor = postprocessor_json(policy, &outputs, &device, &unnormalizer_state);
    written.push(write_json(
        directory,
        &format!("{POLICY_POSTPROCESSOR_NAME}.json"),
        &postprocessor,
    )?);
    // Upstream hands the unnormalizer the same `stats` dict, so its state file is
    // the same tensors even though its declared features are only the outputs.
    written.push(write_state(
        directory,
        &unnormalizer_state,
        stats,
        camera_stats,
    )?);

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
fn write_state(
    directory: &Path,
    name: &str,
    stats: &DatasetStats,
    camera_stats: &IndexMap<String, CameraNormalization>,
) -> Result<PathBuf> {
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
    for (feature, stats) in camera_stats {
        if stats.channels().is_none() {
            continue;
        }
        let mean = stats.mean().to_vec();
        let std = stats.std().to_vec();
        let shape = (mean.len(), 1, 1);
        tensors.insert(
            format!("{feature}.mean"),
            candle_core::Tensor::from_vec(mean, shape, &candle_core::Device::Cpu)?,
        );
        tensors.insert(
            format!("{feature}.std"),
            candle_core::Tensor::from_vec(std, shape, &candle_core::Device::Cpu)?,
        );
    }
    candle_core::safetensors::save(&tensors, &path)?;
    Ok(path)
}

fn preprocessor_json(
    policy: &ActConfig,
    features: &indexmap::IndexMap<String, PolicyFeature>,
    device: &str,
    state_file: &str,
    rename_map: &IndexMap<String, String>,
) -> JsonLike {
    let mut steps = Vec::with_capacity(4);

    // `rename_observations_processor`, preserving the mapping restored from a
    // pretrained pipeline instead of silently resetting it on the next checkpoint.
    let mut mapping = JsonObject::new();
    for (source, target) in rename_map {
        mapping.insert(source.clone(), JsonLike::Str(target.clone()));
    }
    let mut rename = JsonObject::new();
    rename.insert("rename_map".into(), JsonLike::Object(mapping));
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
