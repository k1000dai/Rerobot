//! Offline policy deployment for a locally stored ACT checkpoint.
//!
//! This is deliberately a dataset-backed deployment boundary rather than a fake
//! robot driver. It loads the same local observation representation the training
//! slice reads, applies the checkpoint's feature normalization, and exercises
//! `ACTPolicy.select_action`'s action queue or temporal ensembler. Hardware,
//! Gymnasium environments and video shards remain outside this boundary and are
//! refused explicitly.

use crate::data::batch::{collate, collate_images, Batch};
use crate::data::dataset::StateOnlyDataset;
use crate::data::image::CameraNormalization;
use crate::data::meta::ACTION;
use crate::device;
use crate::error::{Result, TrainError};
use crate::model::act::ActModel;
use crate::processor::{rename_observation_batch, rename_observation_images};

use indexmap::IndexMap;
use num_bigint::BigInt;
use rerobot_core::dataset::delta::action_delta_timestamps;
use rerobot_core::dataset::json::{loads, JsonLike};
use rerobot_core::policy::act::ActConfig;
use rerobot_core::policy::normalize::Normalizer;
use rerobot_core::random::SplitMix64;
use std::collections::VecDeque;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Online temporal action ensembling from upstream `ACTTemporalEnsembler`.
///
/// The model is queried on every control step. Each query supplies a complete
/// action chunk, while the returned action is the oldest overlapping action
/// after the exponential weights have been applied. The vectors are `f32`, as
/// they are in the tensor implementation upstream.
#[derive(Debug, Clone)]
pub struct TemporalEnsembler {
    chunk_size: usize,
    ensemble_weights: Vec<f32>,
    ensemble_weights_cumsum: Vec<f32>,
    ensembled_actions: Option<Vec<Vec<f32>>>,
    ensembled_actions_count: Vec<usize>,
}

impl TemporalEnsembler {
    /// Create an online ensembler for a fixed action-chunk width.
    pub fn new(temporal_ensemble_coeff: f64, chunk_size: usize) -> Result<Self> {
        if !temporal_ensemble_coeff.is_finite() {
            return Err(TrainError::Metadata(format!(
                "temporal_ensemble_coeff must be finite, got {temporal_ensemble_coeff}"
            )));
        }
        if chunk_size == 0 {
            return Err(TrainError::Metadata(
                "temporal ensemble chunk_size must be positive".to_owned(),
            ));
        }
        let coefficient = temporal_ensemble_coeff as f32;
        if !coefficient.is_finite() {
            return Err(TrainError::Metadata(
                "temporal_ensemble_coeff is outside the f32 range".to_owned(),
            ));
        }
        let ensemble_weights = (0..chunk_size)
            .map(|index| (-(coefficient * index as f32)).exp())
            .collect::<Vec<_>>();
        if ensemble_weights.iter().any(|weight| !weight.is_finite()) {
            return Err(TrainError::Metadata(
                "temporal ensemble weights are not finite".to_owned(),
            ));
        }
        let mut total = 0.0f32;
        let mut ensemble_weights_cumsum = Vec::with_capacity(chunk_size);
        for weight in &ensemble_weights {
            total += *weight;
            ensemble_weights_cumsum.push(total);
        }
        Ok(Self {
            chunk_size,
            ensemble_weights,
            ensemble_weights_cumsum,
            ensembled_actions: None,
            ensembled_actions_count: Vec::new(),
        })
    }

    /// Clear all action history at an environment or episode reset.
    pub fn reset(&mut self) {
        self.ensembled_actions = None;
        self.ensembled_actions_count.clear();
    }

    /// Add one full action chunk and consume its oldest ensembled action.
    pub fn update(&mut self, actions: Vec<Vec<f32>>) -> Result<Vec<f32>> {
        if actions.len() != self.chunk_size {
            return Err(TrainError::Metadata(format!(
                "temporal ensemble received {} actions, expected {}",
                actions.len(),
                self.chunk_size
            )));
        }
        let action_dim = actions.first().map_or(0, Vec::len);
        if action_dim == 0 || actions.iter().any(|action| action.len() != action_dim) {
            return Err(TrainError::Metadata(
                "temporal ensemble action chunks must have one non-empty consistent width"
                    .to_owned(),
            ));
        }
        if actions
            .iter()
            .flat_map(|action| action.iter())
            .any(|value| !value.is_finite())
        {
            return Err(TrainError::NonFinite {
                step: 0,
                quantity: "temporal ensemble action".to_owned(),
                value: "non-finite action vector".to_owned(),
            });
        }

        let mut current = if let Some(previous) = self.ensembled_actions.take() {
            if previous.len() != self.ensembled_actions_count.len()
                || previous.iter().any(|action| action.len() != action_dim)
            {
                return Err(TrainError::Metadata(
                    "temporal ensemble history has an inconsistent shape".to_owned(),
                ));
            }
            let mut blended = Vec::with_capacity(self.chunk_size);
            let mut counts = Vec::with_capacity(self.chunk_size);
            for (position, (previous_action, count)) in previous
                .iter()
                .zip(self.ensembled_actions_count.iter())
                .enumerate()
            {
                let new_action = actions.get(position).ok_or_else(|| {
                    TrainError::Metadata(
                        "temporal ensemble history is longer than the incoming chunk".to_owned(),
                    )
                })?;
                let next_weight = *self.ensemble_weights.get(*count).ok_or_else(|| {
                    TrainError::Metadata("temporal ensemble action count overflowed".to_owned())
                })?;
                let previous_weight = *self
                    .ensemble_weights_cumsum
                    .get(count.checked_sub(1).ok_or_else(|| {
                        TrainError::Metadata("temporal ensemble action count was zero".to_owned())
                    })?)
                    .ok_or_else(|| {
                        TrainError::Metadata("temporal ensemble action count overflowed".to_owned())
                    })?;
                let denominator = *self.ensemble_weights_cumsum.get(*count).ok_or_else(|| {
                    TrainError::Metadata("temporal ensemble action count overflowed".to_owned())
                })?;
                if denominator == 0.0 {
                    return Err(TrainError::Metadata(
                        "temporal ensemble weight sum is zero".to_owned(),
                    ));
                }
                let action = previous_action
                    .iter()
                    .zip(new_action.iter())
                    .map(|(old, new)| (*old * previous_weight + *new * next_weight) / denominator)
                    .collect();
                blended.push(action);
                counts.push(count.saturating_add(1).min(self.chunk_size));
            }
            blended.push(actions[self.chunk_size - 1].clone());
            counts.push(1);
            self.ensembled_actions_count = counts;
            blended
        } else {
            self.ensembled_actions_count = vec![1; self.chunk_size];
            actions
        };

        let action = current.remove(0);
        self.ensembled_actions = Some(current);
        self.ensembled_actions_count.remove(0);
        Ok(action)
    }
}

/// One action emitted by the offline deployment loop.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceStep {
    /// The dataset-absolute observation index used for this output.
    pub frame_index: i64,
    /// One action vector, matching upstream `select_action`'s `[batch, action_dim]`
    /// result for its single-observation deployment path.
    pub action: Vec<f32>,
    /// Whether this output caused a fresh model query. A false value means the
    /// action came from ACT's queued chunk, as it does upstream when
    /// `n_action_steps > 1`.
    pub queried_policy: bool,
}

fn should_reset_for_episode_change(previous: Option<i64>, current: i64) -> bool {
    previous.is_some_and(|episode| episode != current)
}

fn checked_deployment_chunk_size(value: &BigInt) -> Result<usize> {
    crate::limits::bounded_usize(value, "chunk_size", crate::limits::MAX_CHUNK_SIZE)
}

/// A local ACT checkpoint connected to a local dataset as an inference source.
///
/// This is a hardware-independent deployment path: the dataset supplies the
/// observations and the returned [`InferenceStep`] values are the actions a robot
/// adapter would consume. The session owns the action queue and must be reset at
/// an environment/episode boundary, matching `ACTPolicy.reset`.
pub struct InferenceSession {
    checkpoint_dir: PathBuf,
    policy_config: ActConfig,
    dataset: Option<StateOnlyDataset>,
    model: ActModel,
    normalizer: Normalizer,
    camera_normalization: CameraNormalization,
    camera_normalizations: IndexMap<String, CameraNormalization>,
    rename_map: IndexMap<String, String>,
    queued_actions: VecDeque<Vec<f32>>,
    temporal_ensembler: Option<TemporalEnsembler>,
}

impl InferenceSession {
    /// Load an ACT checkpoint and connect it to a local LeRobot dataset.
    ///
    /// `device_override` has the same spelling as `--policy.device`; when it is
    /// `None`, the value recorded in `config.json` is used. No device fallback is
    /// performed. Temporal ensembling is restored when the checkpoint declares a
    /// coefficient; otherwise ACT's finite action queue is used.
    pub fn load(
        checkpoint_dir: &Path,
        dataset_root: &Path,
        device_override: Option<&str>,
    ) -> Result<Self> {
        let mut session = Self::load_checkpoint(checkpoint_dir, device_override)?;
        let dataset_root = if crate::hub::is_complete_dataset(dataset_root) {
            dataset_root.to_path_buf()
        } else {
            let train_config = crate::config::TrainConfig::from_checkpoint_dir(checkpoint_dir)?;
            crate::hub::resolve_dataset_root(&train_config.dataset_repo_id, dataset_root, None)?
        };
        let metadata = crate::data::meta::DatasetMetadata::load(&dataset_root)?;
        let fps = metadata.fps()?;
        let chunk_size = checked_deployment_chunk_size(&session.policy_config.chunk_size)?;
        let chunk_size = i64::try_from(chunk_size).map_err(|_| {
            TrainError::unsupported(format!("chunk_size = {chunk_size} does not fit in i64"))
        })?;
        let mut delta_timestamps = IndexMap::new();
        delta_timestamps.insert(ACTION.to_owned(), action_delta_timestamps(chunk_size, fps));
        session.dataset = Some(StateOnlyDataset::load(
            &dataset_root,
            &delta_timestamps,
            1e-4,
        )?);
        Ok(session)
    }

    /// Load a deployable ACT checkpoint without opening or resolving a dataset.
    ///
    /// This is the native counterpart of upstream `ACTPolicy.select_action(batch)`:
    /// the caller supplies already-collated observations through
    /// [`Self::select_action_on_batch`]. It is the boundary for a simulator,
    /// hardware adapter, or another runtime that owns observation acquisition.
    /// `device_override` uses the same spelling as `--policy.device`, and no
    /// fallback is performed.
    pub fn load_checkpoint(checkpoint_dir: &Path, device_override: Option<&str>) -> Result<Self> {
        let config_path = checkpoint_dir.join("config.json");
        let weights_path = checkpoint_dir.join("model.safetensors");
        let config_text = read_checkpoint_json(&config_path, "config.json")?;
        let config = ActConfig::from_checkpoint_json(&config_text).map_err(TrainError::from)?;
        if !weights_path.is_file() {
            return Err(TrainError::checkpoint(
                &weights_path,
                "the ACT policy weights file is missing",
            ));
        }

        let processors = crate::processor::LoadedPolicyProcessors::load(checkpoint_dir, &config)?;
        let rename_map = processors.rename_map().clone();
        let normalizer = processors.normalizer().clone();
        let device_name = device_override.or(config.device.as_deref());
        let device = device::resolve(device_name)?;
        // The initial values are discarded by `load`; construction is still needed
        // because it validates the complete tensor schema before any inference call.
        let mut init_rng = SplitMix64::new(0);
        let mut model = ActModel::new(&config, &device, &mut init_rng)?;
        model.load(&weights_path)?;
        let temporal_ensembler = config
            .temporal_ensemble_coeff
            .map(|coefficient| TemporalEnsembler::new(coefficient, model.shape().chunk_size))
            .transpose()?;
        let camera_normalization = load_camera_normalization(checkpoint_dir)?;
        let mut camera_normalizations = IndexMap::new();
        for (key, feature) in config.input_features.clone().unwrap_or_default() {
            if feature.r#type == rerobot_core::types::FeatureType::Visual {
                camera_normalizations.insert(key, camera_normalization.clone());
            }
        }
        camera_normalizations.extend(processors.camera_normalizations().clone());
        Ok(Self {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            policy_config: config,
            dataset: None,
            model,
            normalizer,
            camera_normalization,
            camera_normalizations,
            rename_map,
            queued_actions: VecDeque::new(),
            temporal_ensembler,
        })
    }

    /// The checkpoint directory this session loaded.
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }

    /// Number of observations available from the connected dataset.
    ///
    /// A checkpoint-only session has no observation source and reports zero.
    pub fn dataset_len(&self) -> usize {
        self.dataset.as_ref().map_or(0, StateOnlyDataset::len)
    }

    /// The Candle device used by this session, for callers constructing batches.
    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }

    /// The checkpoint's configured camera normalization fallback. Deployment uses
    /// the per-camera entries restored from processor statistics when available;
    /// callers normally do not need this accessor because
    /// [`Self::select_action_on_batch`] applies them to raw camera tensors.
    pub fn camera_normalization(&self) -> &CameraNormalization {
        &self.camera_normalization
    }

    /// The checkpoint's per-camera normalization, keyed by image feature name.
    pub fn camera_normalizations(&self) -> &IndexMap<String, CameraNormalization> {
        &self.camera_normalizations
    }

    /// Clear the queued chunk at an environment/episode reset.
    pub fn reset(&mut self) {
        self.queued_actions.clear();
        if let Some(ensembler) = &mut self.temporal_ensembler {
            ensembler.reset();
        }
    }

    /// Select one action for a dataset observation, matching ACT's queue behavior.
    pub fn select_action(&mut self, index: usize) -> Result<InferenceStep> {
        let batch = self.batch(index)?;
        let frame_index = i64::try_from(index).map_err(|_| {
            TrainError::Metadata(format!("dataset frame index {index} does not fit in i64"))
        })?;
        self.select_action_normalized(&batch, frame_index)
    }

    /// Select one action from a caller-owned, single-observation batch.
    ///
    /// The batch must contain exactly one observation. Camera tensors must be raw
    /// `[0, 1]` tensors attached through [`Batch::images`]; this method applies the
    /// checkpoint's saved camera normalization and observation rename map before the
    /// scalar normalizer. The batch is otherwise raw, just as a LeRobot rollout
    /// strategy hands the policy an observation before its preprocessor.
    pub fn select_action_on_batch(&mut self, batch: &Batch) -> Result<InferenceStep> {
        if batch.len() != 1 {
            return Err(TrainError::Metadata(format!(
                "ACT deployment expects one observation, got {}",
                batch.len()
            )));
        }
        let frame_index = *batch.indices.first().ok_or_else(|| {
            TrainError::Metadata("the single-observation batch has no frame index".to_owned())
        })?;
        let mut renamed = rename_observation_batch(batch, &self.rename_map);
        let images = std::mem::take(&mut renamed.images);
        let with_images = if images.is_empty() {
            renamed
        } else {
            renamed.with_image_normalizations(&images, &self.camera_normalizations)?
        };
        let normalized = with_images.normalized(&self.normalizer)?;
        self.select_action_normalized(&normalized, frame_index)
    }

    fn select_action_normalized(
        &mut self,
        batch: &Batch,
        frame_index: i64,
    ) -> Result<InferenceStep> {
        let temporal = self.temporal_ensembler.is_some();
        let queried_policy = temporal || self.queued_actions.is_empty();
        let action = if temporal {
            let actions = self.predict_chunk_from_batch(batch)?;
            self.temporal_ensembler
                .as_mut()
                .expect("temporal flag was checked above")
                .update(actions)?
        } else {
            if queried_policy {
                self.refill_from_batch(batch)?;
            }
            self.queued_actions.pop_front().ok_or_else(|| {
                TrainError::Metadata("ACT returned an empty action chunk".to_owned())
            })?
        };
        let action = self
            .normalizer
            .unnormalize(ACTION, &action)
            .map_err(|error| TrainError::Metadata(error.to_string()))?;
        if !action.iter().all(|value| value.is_finite()) {
            return Err(TrainError::NonFinite {
                step: frame_index.max(0) as u64 + 1,
                quantity: "inference action".to_owned(),
                value: format!("{action:?}"),
            });
        }
        Ok(InferenceStep {
            frame_index,
            action,
            queried_policy,
        })
    }

    /// Run the policy for `steps` observations in increasing dataset order.
    pub fn rollout(&mut self, start_index: usize, steps: usize) -> Result<Vec<InferenceStep>> {
        if steps == 0 {
            return Err(TrainError::unsupported(
                "offline rollout steps must be positive",
            ));
        }
        if steps > crate::limits::MAX_ROLLOUT_TRACE_STEPS {
            return Err(TrainError::unsupported(format!(
                "offline rollout trace has {steps} steps, exceeding the limit {}",
                crate::limits::MAX_ROLLOUT_TRACE_STEPS
            )));
        }
        let end = start_index.checked_add(steps).ok_or_else(|| {
            TrainError::Metadata("offline rollout index range overflowed".to_owned())
        })?;
        let dataset = self.dataset.as_ref().ok_or_else(|| {
            TrainError::unsupported(
                "a checkpoint-only inference session has no dataset; use select_action_on_batch",
            )
        })?;
        let dataset_len = dataset.len();
        if end > dataset_len {
            return Err(TrainError::Metadata(format!(
                "offline rollout ends at frame {end}, but the dataset has {dataset_len} frames"
            )));
        }
        let mut trace = Vec::with_capacity(steps.min(crate::limits::MAX_BATCH_SIZE));
        // Upstream's rollout() calls policy.reset() before every new trace. Do not
        // let an earlier call's queued chunk or temporal ensemble bleed into this one.
        self.reset();
        let mut previous_episode = None;
        for index in start_index..end {
            let episode = self
                .dataset
                .as_ref()
                .expect("the dataset was checked above")
                .episode_index_at(index)?;
            if should_reset_for_episode_change(previous_episode, episode) {
                self.reset();
            }
            previous_episode = Some(episode);
            trace.push(self.select_action(index)?);
        }
        Ok(trace)
    }

    fn refill_from_batch(&mut self, batch: &Batch) -> Result<()> {
        let actions = self.model.predict_action_steps(batch)?;
        self.queued_actions = actions
            .to_vec3::<f32>()?
            .into_iter()
            .next()
            .ok_or_else(|| {
                TrainError::Metadata("ACT returned no action for a non-empty batch".to_owned())
            })?
            .into_iter()
            .collect();
        Ok(())
    }

    fn predict_chunk_from_batch(&self, batch: &Batch) -> Result<Vec<Vec<f32>>> {
        let actions = self.model.predict_action_chunk(batch)?;
        actions.to_vec3::<f32>()?.into_iter().next().ok_or_else(|| {
            TrainError::Metadata("ACT returned no action for a non-empty batch".to_owned())
        })
    }

    fn batch(&self, index: usize) -> Result<Batch> {
        let dataset = self.dataset.as_ref().ok_or_else(|| {
            TrainError::unsupported(
                "a checkpoint-only inference session has no dataset; use select_action_on_batch",
            )
        })?;
        let frame = dataset.get(index)?;
        let raw = collate(std::slice::from_ref(&frame), self.model.device())?;
        let images = collate_images(std::slice::from_ref(&frame), self.model.device())?;
        let renamed = rename_observation_batch(&raw, &self.rename_map);
        let images = rename_observation_images(&images, &self.rename_map);
        let normalized_images = if images.is_empty() {
            renamed
        } else {
            renamed.with_image_normalizations(&images, &self.camera_normalizations)?
        };
        normalized_images.normalized(&self.normalizer)
    }
}

fn read_checkpoint_json(path: &Path, label: &str) -> Result<String> {
    let file = std::fs::File::open(path).map_err(|error| TrainError::io(path, &error))?;
    read_checkpoint_json_file(path, file, label)
}

fn read_checkpoint_json_file(path: &Path, file: std::fs::File, label: &str) -> Result<String> {
    let limit = crate::limits::MAX_CHECKPOINT_JSON_BYTES;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| TrainError::io(path, &error))?;
    if bytes.len() as u64 > limit {
        return Err(TrainError::checkpoint(
            path,
            format!("{label} exceeds the {}-byte limit", limit),
        ));
    }
    String::from_utf8(bytes).map_err(|error| {
        TrainError::checkpoint(path, format!("{label} is not valid UTF-8: {error}"))
    })
}

/// Read the training-time camera-statistics choice from a checkpoint.
///
/// Old or hand-created policy directories do not necessarily carry
/// `train_config.json`; upstream's dataset default is ImageNet statistics, so
/// absence uses that default. A present malformed field is an error, not a silent
/// fallback.
fn load_camera_normalization(checkpoint_dir: &Path) -> Result<CameraNormalization> {
    let path = checkpoint_dir.join("train_config.json");
    let text = match std::fs::File::open(&path) {
        Ok(file) => read_checkpoint_json_file(&path, file, "train_config.json")?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CameraNormalization::imagenet())
        }
        Err(error) => return Err(TrainError::io(&path, &error)),
    };
    let document = loads(&text).map_err(|error| {
        TrainError::checkpoint(
            &path,
            format!("train_config.json is not valid JSON: {error}"),
        )
    })?;
    let use_imagenet = match &document {
        JsonLike::Object(root) => match root.get("dataset") {
            None | Some(JsonLike::Null) => true,
            Some(JsonLike::Object(dataset)) => match dataset.get("use_imagenet_stats") {
                None => true,
                Some(JsonLike::Bool(value)) => *value,
                Some(other) => {
                    return Err(TrainError::checkpoint(
                        &path,
                        format!(
                            "dataset.use_imagenet_stats is {}, not a boolean",
                            other.type_name()
                        ),
                    ))
                }
            },
            Some(other) => {
                return Err(TrainError::checkpoint(
                    &path,
                    format!("dataset is {}, not an object", other.type_name()),
                ))
            }
        },
        other => {
            return Err(TrainError::checkpoint(
                &path,
                format!("the root is {}, not an object", other.type_name()),
            ))
        }
    };
    Ok(if use_imagenet {
        CameraNormalization::imagenet()
    } else {
        CameraNormalization::identity()
    })
}

#[cfg(test)]
mod tests {
    use super::checked_deployment_chunk_size;
    use num_bigint::BigInt;

    #[test]
    fn deployment_chunk_size_is_bounded_before_window_allocation() {
        let oversized = BigInt::from(crate::limits::MAX_CHUNK_SIZE) + 1;
        let error =
            checked_deployment_chunk_size(&oversized).expect_err("oversized chunks must fail");
        assert!(error.to_string().contains("chunk_size"));
        assert!(error
            .to_string()
            .contains(&crate::limits::MAX_CHUNK_SIZE.to_string()));
    }

    #[test]
    fn rollout_resets_only_when_episode_changes() {
        assert!(!super::should_reset_for_episode_change(None, 0));
        assert!(!super::should_reset_for_episode_change(Some(0), 0));
        assert!(super::should_reset_for_episode_change(Some(0), 1));
    }
}
