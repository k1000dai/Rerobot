//! The dataset metadata a training run reads before it touches a single frame:
//! `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet` and
//! `meta/episodes/`.
//!
//! Upstream is `LeRobotDatasetMetadata` (`lerobot/datasets/lerobot_dataset.py`)
//! plus `datasets/io_utils.py`'s four loaders, and it can also reach the Hub. This
//! port is local-directory only. Embedded `image` columns in LeRobot v3.0 parquet
//! are decoded natively; `video` features are refused rather than half-read.

use crate::data::parquet::Table;
use crate::error::{Result, TrainError};
use indexmap::IndexMap;
use num_bigint::BigInt;
use rerobot_core::dataset::info::DatasetInfo;
use rerobot_core::dataset::io::{load_info, load_json};
use rerobot_core::dataset::json::JsonLike;
use rerobot_core::dataset::stats::{stats_from_value, DatasetStats};
use rerobot_core::dataset::{DATA_DIR, DEFAULT_TASKS_PATH, EPISODES_DIR, STATS_PATH};
use rerobot_core::types::{FeatureType, PolicyFeature};
use std::path::{Path, PathBuf};

/// Upstream `OBS_STR`.
pub const OBS_PREFIX: &str = "observation";
/// Upstream `OBS_STATE`.
pub const OBS_STATE: &str = "observation.state";
/// Upstream `OBS_ENV_STATE`.
pub const OBS_ENV_STATE: &str = "observation.environment_state";
/// Upstream `ACTION`.
pub const ACTION: &str = "action";

/// One entry of `info.json`'s `features` map.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSpec {
    /// `dtype`, e.g. `float32`, `int64`, `video`.
    pub dtype: String,
    /// `shape`, as declared.
    pub shape: Vec<i64>,
    /// `names`, when present.
    pub names: Option<Vec<String>>,
}

impl FeatureSpec {
    /// The shape a *policy* sees, which for a camera is channel-first.
    ///
    /// `dataset_to_policy_features` reorders a visual feature declared
    /// `[height, width, channel]` into `[channel, height, width]`, keyed on the last
    /// entry of `names` rather than on the numbers:
    ///
    /// ```python
    /// if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
    ///     shape = (shape[2], shape[0], shape[1])
    /// ```
    ///
    /// Both spellings occur on the Hub — `lerobot/pusht` is channel-first while every
    /// LIBERO conversion is `[256, 256, 3]` with `names` `["height", "width",
    /// "channel"]` — so the declaration alone does not say which, and a 3-channel
    /// frame is not distinguishable from a 3-pixel-wide one by shape.
    ///
    /// One deviation, and it is in the direction of refusing less: upstream indexes
    /// `names[2]` unconditionally, so a visual feature with `"names": null` raises
    /// `TypeError` there. Here a missing or short `names` means no reorder, because
    /// the shape is then already what a policy consumes.
    pub fn policy_shape(&self) -> Vec<i64> {
        if self.dtype != "image" && self.dtype != "video" {
            return self.shape.clone();
        }
        if self.shape.len() != 3 {
            return self.shape.clone();
        }
        let channel_last = self
            .names
            .as_ref()
            .and_then(|names| names.get(2))
            .is_some_and(|name| name == "channel" || name == "channels");
        if channel_last {
            vec![self.shape[2], self.shape[0], self.shape[1]]
        } else {
            self.shape.clone()
        }
    }

    /// How many scalars one frame of this feature carries.
    ///
    /// Checked and bounded, not a plain `product()`. The shape comes from
    /// `meta/info.json`, so both operands are attacker-controlled: an overflowing
    /// product panics in a checked build and *wraps* in release, and a wrapped width
    /// is worse than either — the allocation succeeds at the wrong size and the
    /// reader then walks past its end. The previous version also mapped an
    /// out-of-range dimension to `0`, which silently turned a hostile shape into an
    /// empty feature.
    pub fn width(&self) -> Result<usize> {
        let mut dimensions = Vec::with_capacity(self.shape.len());
        for dimension in &self.shape {
            let value = usize::try_from(*dimension).map_err(|_| {
                TrainError::Metadata(format!(
                    "a feature dimension is {dimension}, which is not a valid extent"
                ))
            })?;
            dimensions.push(crate::limits::within(
                value,
                "a feature dimension",
                crate::limits::MAX_FEATURE_WIDTH,
            )?);
        }
        let product = crate::limits::checked_product(&dimensions, "a feature shape")?;
        // Zero is an extent the upper bounds say nothing about, and it is the one that
        // does not survive the pipeline: the batch collator divides a flat buffer into
        // rows with `slice::chunks`, which *panics* on a chunk size of zero. A panic is
        // not a refusal — no message, no exit code, no partial-work cleanup — so a
        // feature that carries no scalars is refused here, where it is declared, before
        // anything is read or allocated.
        if product == 0 {
            return Err(TrainError::Metadata(format!(
                "a feature declares shape {:?}, which is empty; a feature carrying no \
                 scalars cannot be batched or normalized",
                self.shape
            )));
        }
        crate::limits::within(product, "a feature width", crate::limits::MAX_FEATURE_WIDTH)
    }
}

/// One row of `meta/episodes/`.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeRecord {
    /// `episode_index`.
    pub episode_index: i64,
    /// `tasks`, the natural-language task strings of the episode.
    pub tasks: Vec<String>,
    /// `length`, the frame count.
    pub length: i64,
    /// `data/chunk_index`.
    pub data_chunk_index: i64,
    /// `data/file_index`.
    pub data_file_index: i64,
    /// `dataset_from_index`, inclusive.
    pub dataset_from_index: i64,
    /// `dataset_to_index`, exclusive.
    pub dataset_to_index: i64,
}

/// One row of `meta/tasks.parquet`.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskRecord {
    /// The task string, which is the parquet file's index column.
    pub task: String,
    /// `task_index`, what the frame rows refer to.
    pub task_index: i64,
}

/// What the `meta/episodes/` tree may cost.
///
/// The `data/` files are named by the episode table, so their count is bounded by
/// something already validated. This tree is the other way round: it is *discovered*
/// by walking the directory, and every file is read and materialized before any episode
/// invariant can be checked. That makes it the one place where a dataset's own layout
/// decides how much work happens before any validation, so it needs a cumulative
/// budget of its own.
///
/// Injectable for the same reason the other two budgets are, and [`Self::default`] is
/// the production budget from [`crate::limits`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataBudget {
    /// Parquet files the tree may contain.
    pub max_files: usize,
    /// Episode rows the tree may hold in total.
    pub max_rows: usize,
    /// Decoded scalars the tree may materialize in total.
    pub max_values: usize,
    /// The budget applied to each individual file underneath this one.
    pub read: crate::data::parquet::ReadBudget,
}

impl Default for MetadataBudget {
    fn default() -> Self {
        Self {
            max_files: crate::limits::MAX_EPISODE_FILES,
            max_rows: crate::limits::MAX_EPISODES,
            max_values: crate::limits::MAX_DECODED_VALUES,
            read: crate::data::parquet::ReadBudget::default(),
        }
    }
}

/// Load scalar statistics while keeping the nested image statistics emitted by
/// LeRobot v3.0 available to the camera path. The scalar normalizer still ignores
/// those entries because camera statistics have shape `(channels, 1, 1)` and are
/// broadcast separately by `Batch::with_image_normalizations`.
fn load_stats_document(root: &Path) -> Result<Option<JsonLike>> {
    let path = root.join(STATS_PATH);
    if !path.exists() {
        return Ok(None);
    }
    load_json(&path)
        .map(Some)
        .map_err(|error| TrainError::Metadata(format!("cannot read {}: {error}", path.display())))
}

fn load_stats_for_features(
    root: &Path,
    features: &IndexMap<String, FeatureSpec>,
    document: Option<&JsonLike>,
) -> Result<DatasetStats> {
    let Some(document) = document else {
        return Ok(DatasetStats::default());
    };
    let mut document = document.clone();
    if let JsonLike::Object(values) = &mut document {
        values.retain(|feature, _| {
            features
                .get(feature)
                .is_none_or(|spec| spec.dtype != "image")
        });
    }
    stats_from_value(&document).map_err(|error| {
        TrainError::Metadata(format!(
            "cannot read {}: {error}",
            root.join(STATS_PATH).display()
        ))
    })
}

/// Per-channel statistics for one camera feature.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraStats {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl CameraStats {
    /// Mean values in channel order.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Standard deviations in channel order.
    pub fn std(&self) -> &[f64] {
        &self.std
    }
}

fn camera_stat_vector(
    root: &Path,
    feature: &str,
    statistic: &str,
    value: &JsonLike,
    output: &mut Vec<f64>,
) -> Result<()> {
    match value {
        JsonLike::Array(values) | JsonLike::Tuple(values) => {
            for value in values {
                camera_stat_vector(root, feature, statistic, value, output)?;
            }
        }
        JsonLike::Float(value) => output.push(*value),
        JsonLike::Int(value) => {
            output.push(value.to_string().parse::<f64>().unwrap_or_else(|_| {
                if value.sign() == num_bigint::Sign::Minus {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            }))
        }
        other => {
            return Err(TrainError::Metadata(format!(
                "cannot read {}: camera feature {feature:?} statistic {statistic:?} contains {}, expected numbers",
                root.join(STATS_PATH).display(),
                other.type_name()
            )));
        }
    }
    Ok(())
}

/// Load the nested `(channels, 1, 1)` camera statistics emitted by LeRobot.
fn load_camera_stats(
    root: &Path,
    features: &IndexMap<String, FeatureSpec>,
    document: Option<&JsonLike>,
) -> Result<IndexMap<String, CameraStats>> {
    let Some(JsonLike::Object(values)) = document else {
        return Ok(IndexMap::new());
    };
    let mut cameras = IndexMap::new();
    for (feature, spec) in features {
        if spec.dtype != "image" {
            continue;
        }
        let Some(value) = values.get(feature) else {
            continue;
        };
        let JsonLike::Object(statistics) = value else {
            return Err(TrainError::Metadata(format!(
                "cannot read {}: camera feature {feature:?} statistics must be an object",
                root.join(STATS_PATH).display()
            )));
        };
        let (Some(mean), Some(std)) = (statistics.get("mean"), statistics.get("std")) else {
            return Err(TrainError::Metadata(format!(
                "cannot read {}: camera feature {feature:?} must contain mean and std statistics",
                root.join(STATS_PATH).display()
            )));
        };
        let mut mean_values = Vec::new();
        let mut std_values = Vec::new();
        camera_stat_vector(root, feature, "mean", mean, &mut mean_values)?;
        camera_stat_vector(root, feature, "std", std, &mut std_values)?;
        if mean_values.is_empty() || mean_values.len() != std_values.len() {
            return Err(TrainError::Metadata(format!(
                "cannot read {}: camera feature {feature:?} must have equally sized, non-empty mean and std statistics",
                root.join(STATS_PATH).display()
            )));
        }
        cameras.insert(
            feature.clone(),
            CameraStats {
                mean: mean_values,
                std: std_values,
            },
        );
    }
    Ok(cameras)
}

/// Everything under `meta/`, typed.
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    root: PathBuf,
    /// `meta/info.json`.
    pub info: DatasetInfo,
    /// `meta/stats.json`, empty when the file is absent.
    pub stats: DatasetStats,
    /// Nested per-camera entries from `meta/stats.json`.
    camera_stats: IndexMap<String, CameraStats>,
    /// `meta/tasks.parquet`, in file order.
    pub tasks: Vec<TaskRecord>,
    /// `meta/episodes/`, in file order.
    pub episodes: Vec<EpisodeRecord>,
    features: IndexMap<String, FeatureSpec>,
}

impl DatasetMetadata {
    /// Per-camera statistics, in the feature order of `meta/info.json`.
    pub fn camera_stats(&self) -> &IndexMap<String, CameraStats> {
        &self.camera_stats
    }
}

impl DatasetMetadata {
    /// Read `root/meta/` in full, within the default budget.
    pub fn load(root: &Path) -> Result<Self> {
        Self::load_within(root, &MetadataBudget::default())
    }

    /// [`Self::load`], refusing a `meta/episodes/` tree outside `budget`.
    pub fn load_within(root: &Path, budget: &MetadataBudget) -> Result<Self> {
        if !root.is_dir() {
            return Err(TrainError::io_message(
                root,
                "dataset root does not exist; this slice reads local datasets only and never \
                 downloads from the Hub",
            ));
        }
        let info = load_info(root).map_err(|error| TrainError::Metadata(error.to_string()))?;
        let features = parse_features(&info)?;
        refuse_video_features(&features)?;
        // Every declared width bounded up front, so the name of the offending feature
        // is in the message rather than only the number.
        for (key, spec) in &features {
            spec.width()
                .map_err(|error| TrainError::Metadata(format!("feature {key:?}: {error}")))?;
        }
        let stats_document = load_stats_document(root)?;
        let stats = load_stats_for_features(root, &features, stats_document.as_ref())?;
        let camera_stats = load_camera_stats(root, &features, stats_document.as_ref())?;
        let tasks = load_tasks(root)?;
        let episodes = load_episodes(root, budget)?;
        let metadata = Self {
            root: root.to_path_buf(),
            info,
            stats,
            camera_stats,
            tasks,
            episodes,
            features,
        };
        metadata.validate_episodes()?;
        Ok(metadata)
    }

    /// Refuse an episode table whose ranges cannot describe a real dataset.
    ///
    /// Every field here comes from attacker-controlled parquet, and the reader treats
    /// the ranges as arithmetic: `query_window` computes `ep_end - 1`, and
    /// `EpisodeAwareSampler` subtracts them to get lengths. `i64::MIN` made the first
    /// of those panic in a checked build and wrap in release, and an inverted or
    /// overlapping range silently mis-attributed frames to the wrong episode -- which
    /// would train on the wrong action chunks without any error at all.
    ///
    /// The checks are ordered from cheapest to most structural so that the first
    /// failure is the most specific one.
    fn validate_episodes(&self) -> Result<()> {
        crate::limits::within(
            self.episodes.len(),
            "the number of episodes",
            crate::limits::MAX_EPISODES,
        )?;
        let total_frames = self.total_frames()?;

        // Indices first: everything below resolves an episode *by index*, and
        // `episode_of` / `get` find one by scanning for the first match. Two records
        // sharing an index make one unreachable and silently clamp the other's frames
        // against the wrong range, and a gap makes a frame's `episode_index`
        // unresolvable. Neither produces an error anywhere downstream.
        let mut indices: Vec<i64> = self.episodes.iter().map(|e| e.episode_index).collect();
        indices.sort_unstable();
        for pair in indices.windows(2) {
            if pair[0] == pair[1] {
                return Err(TrainError::Metadata(format!(
                    "two episodes share episode_index {}; a duplicate index makes one of them \
                     unreachable and clamps the other's action windows against the wrong \
                     episode",
                    pair[0]
                )));
            }
        }
        for (position, index) in indices.iter().enumerate() {
            if *index != position as i64 {
                return Err(TrainError::Metadata(format!(
                    "the episode_index values are not the contiguous range 0..{}: position \
                     {position} holds {index}. A frame naming an absent index cannot be \
                     resolved to an episode",
                    self.episodes.len()
                )));
            }
        }

        let mut previous_end: Option<(i64, i64)> = None;
        for episode in &self.episodes {
            let index = episode.episode_index;
            if episode.episode_index < 0 {
                return Err(TrainError::Metadata(format!(
                    "episode_index {index} is negative"
                )));
            }
            if episode.dataset_from_index < 0 {
                return Err(TrainError::Metadata(format!(
                    "episode {index} has a negative dataset_from_index ({})",
                    episode.dataset_from_index
                )));
            }
            if episode.dataset_to_index < 0 {
                return Err(TrainError::Metadata(format!(
                    "episode {index} has a negative dataset_to_index ({})",
                    episode.dataset_to_index
                )));
            }
            if episode.dataset_to_index < episode.dataset_from_index {
                return Err(TrainError::Metadata(format!(
                    "episode {index} has an inverted range: dataset_from_index {} is above \
                     dataset_to_index {}",
                    episode.dataset_from_index, episode.dataset_to_index
                )));
            }
            if episode.length <= 0 {
                return Err(TrainError::Metadata(format!(
                    "episode {index} has length {}; an episode that owns no frame cannot be \
                     sampled from and would leave a hole in the frame domain",
                    episode.length
                )));
            }
            // The range and the recorded length must agree, or the sampler and the
            // reader would disagree about how many frames the episode has.
            let span = episode
                .dataset_to_index
                .checked_sub(episode.dataset_from_index)
                .ok_or_else(|| {
                    TrainError::Metadata(format!(
                        "episode {index} has a range whose width does not fit in i64"
                    ))
                })?;
            if span != episode.length {
                return Err(TrainError::Metadata(format!(
                    "episode {index} declares length {} but its range {}..{} spans {span}",
                    episode.length, episode.dataset_from_index, episode.dataset_to_index
                )));
            }
            if episode.dataset_to_index > total_frames {
                return Err(TrainError::Metadata(format!(
                    "episode {index} ends at dataset_to_index {} but info.json declares \
                     total_frames {total_frames}",
                    episode.dataset_to_index
                )));
            }
            if episode.data_chunk_index < 0 || episode.data_file_index < 0 {
                return Err(TrainError::Metadata(format!(
                    "episode {index} has a negative data file coordinate ({}, {})",
                    episode.data_chunk_index, episode.data_file_index
                )));
            }
            // Sorted by `load_episodes`, so a non-increasing start means two episodes
            // claim the same frames.
            if let Some((previous_index, end)) = previous_end {
                if episode.dataset_from_index < end {
                    return Err(TrainError::Metadata(format!(
                        "episode {index} starts at {} but episode {previous_index} already ends \
                         at {end}; the ranges overlap",
                        episode.dataset_from_index
                    )));
                }
            }
            previous_end = Some((index, episode.dataset_to_index));
        }

        // And the ranges must tile the whole frame domain, exactly. A gap leaves a
        // readable frame that belongs to no episode, so there is nothing to clamp its
        // action window against; a domain that does not start at 0 or stops short of
        // `total_frames` is the same problem at the ends. Sorted by index and already
        // known non-overlapping, so contiguity is a single sweep.
        let mut expected_start = 0i64;
        for episode in &self.episodes {
            if episode.dataset_from_index != expected_start {
                return Err(TrainError::Metadata(format!(
                    "episode {} starts at {} but the episode ranges must cover the frame \
                     domain without a gap; frames {expected_start}..{} belong to no episode",
                    episode.episode_index, episode.dataset_from_index, episode.dataset_from_index
                )));
            }
            expected_start = episode.dataset_to_index;
        }
        if expected_start != total_frames {
            return Err(TrainError::Metadata(format!(
                "the episode ranges cover frames 0..{expected_start} but info.json declares \
                 total_frames {total_frames}; every frame must belong to exactly one episode"
            )));
        }
        Ok(())
    }

    /// The dataset root the metadata was read from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `info.json`'s `fps`.
    ///
    /// # Errors
    ///
    /// When `fps` is not a positive machine integer. Upstream stores a Python
    /// `int`; a frame rate that cannot index a frame grid is refused rather than
    /// truncated.
    pub fn fps(&self) -> Result<i64> {
        i64::try_from(&self.info.fps)
            .ok()
            .filter(|fps| *fps > 0)
            .ok_or_else(|| {
                TrainError::Metadata(format!("fps must be positive, got {}", self.info.fps))
            })
    }

    /// `info.json`'s `total_frames`.
    pub fn total_frames(&self) -> Result<i64> {
        nonnegative(&self.info.total_frames, "total_frames")
    }

    /// `info.json`'s `total_episodes`.
    pub fn total_episodes(&self) -> Result<i64> {
        nonnegative(&self.info.total_episodes, "total_episodes")
    }

    /// One feature's declaration.
    pub fn feature(&self, key: &str) -> Option<&FeatureSpec> {
        self.features.get(key)
    }

    /// Feature keys, in `info.json` order.
    pub fn feature_keys(&self) -> impl Iterator<Item = &str> {
        self.features.keys().map(String::as_str)
    }

    /// `dataset_to_policy_features`: the features a policy sees, typed.
    ///
    /// Keys that are neither an observation nor an action are dropped, exactly as
    /// upstream's `continue` does — that is what removes `timestamp`,
    /// `frame_index`, `episode_index`, `index` and `task_index`.
    pub fn policy_features(&self) -> IndexMap<String, PolicyFeature> {
        let mut out = IndexMap::new();
        for (key, spec) in &self.features {
            let feature_type = if spec.dtype == "image" || spec.dtype == "video" {
                FeatureType::Visual
            } else if key == OBS_ENV_STATE {
                FeatureType::Env
            } else if key.starts_with(OBS_PREFIX) {
                FeatureType::State
            } else if key.starts_with(ACTION) {
                FeatureType::Action
            } else {
                continue;
            };
            out.insert(
                key.clone(),
                PolicyFeature::new(
                    feature_type,
                    spec.policy_shape().into_iter().map(BigInt::from),
                ),
            );
        }
        out
    }

    /// The `(input_features, output_features)` split `make_policy` performs:
    /// action features are outputs, everything else is an input.
    pub fn policy_feature_split(
        &self,
    ) -> (
        IndexMap<String, PolicyFeature>,
        IndexMap<String, PolicyFeature>,
    ) {
        let features = self.policy_features();
        let outputs: IndexMap<String, PolicyFeature> = features
            .iter()
            .filter(|(_, feature)| feature.r#type == FeatureType::Action)
            .map(|(key, feature)| (key.clone(), feature.clone()))
            .collect();
        let inputs = features
            .iter()
            .filter(|(key, _)| !outputs.contains_key(*key))
            .map(|(key, feature)| (key.clone(), feature.clone()))
            .collect();
        (inputs, outputs)
    }

    /// The absolute path of one episode's data file.
    pub fn data_file_path(&self, chunk_index: i64, file_index: i64) -> PathBuf {
        self.root
            .join(DATA_DIR)
            .join(format!("chunk-{chunk_index:03}"))
            .join(format!("file-{file_index:03}.parquet"))
    }

    /// The task string behind a `task_index`.
    pub fn task(&self, task_index: i64) -> Option<&str> {
        self.tasks
            .iter()
            .find(|record| record.task_index == task_index)
            .map(|record| record.task.as_str())
    }

    /// Every episode's `dataset_from_index`, in episode-index order.
    pub fn episode_from_indices(&self) -> Vec<i64> {
        self.episodes
            .iter()
            .map(|episode| episode.dataset_from_index)
            .collect()
    }

    /// Every episode's `dataset_to_index`, in episode-index order.
    pub fn episode_to_indices(&self) -> Vec<i64> {
        self.episodes
            .iter()
            .map(|episode| episode.dataset_to_index)
            .collect()
    }

    /// The episode a dataset-absolute frame index belongs to.
    pub fn episode_of(&self, absolute_index: i64) -> Option<&EpisodeRecord> {
        self.episodes.iter().find(|episode| {
            absolute_index >= episode.dataset_from_index
                && absolute_index < episode.dataset_to_index
        })
    }
}

fn nonnegative(value: &BigInt, field: &str) -> Result<i64> {
    i64::try_from(value)
        .ok()
        .filter(|number| *number >= 0)
        .ok_or_else(|| {
            TrainError::Metadata(format!(
                "{field} must be a non-negative integer, got {value}"
            ))
        })
}

fn parse_features(info: &DatasetInfo) -> Result<IndexMap<String, FeatureSpec>> {
    let mut out = IndexMap::with_capacity(info.features.len());
    for (key, feature) in &info.features {
        let dtype = match feature.get("dtype") {
            Some(JsonLike::Str(dtype)) => dtype.clone(),
            Some(other) => {
                return Err(TrainError::Metadata(format!(
                    "feature {key:?} has a non-string dtype ({})",
                    other.type_name()
                )))
            }
            None => {
                return Err(TrainError::Metadata(format!(
                    "feature {key:?} has no dtype"
                )))
            }
        };
        let shape = match feature.get("shape") {
            Some(JsonLike::Array(items)) | Some(JsonLike::Tuple(items)) => items
                .iter()
                .map(|item| match item {
                    JsonLike::Int(dimension) => i64::try_from(dimension).map_err(|_| {
                        TrainError::Metadata(format!(
                            "feature {key:?} has a shape dimension outside i64 ({dimension})"
                        ))
                    }),
                    other => Err(TrainError::Metadata(format!(
                        "feature {key:?} has a non-integer shape dimension ({})",
                        other.type_name()
                    ))),
                })
                .collect::<Result<Vec<i64>>>()?,
            _ => {
                return Err(TrainError::Metadata(format!(
                    "feature {key:?} has no shape"
                )))
            }
        };
        let names = match feature.get("names") {
            Some(JsonLike::Array(items)) | Some(JsonLike::Tuple(items)) => Some(
                items
                    .iter()
                    .map(|item| match item {
                        JsonLike::Str(name) => name.clone(),
                        other => other.type_name().to_owned(),
                    })
                    .collect(),
            ),
            _ => None,
        };
        out.insert(
            key.clone(),
            FeatureSpec {
                dtype,
                shape,
                names,
            },
        );
    }
    Ok(out)
}

fn refuse_video_features(features: &IndexMap<String, FeatureSpec>) -> Result<()> {
    let visual: Vec<&str> = features
        .iter()
        .filter(|(_, spec)| spec.dtype == "video")
        .map(|(key, _)| key.as_str())
        .collect();
    if visual.is_empty() {
        return Ok(());
    }
    let named: Vec<String> = features
        .iter()
        .filter(|(_, spec)| spec.dtype == "video")
        .map(|(key, spec)| format!("{key} (dtype {:?})", spec.dtype))
        .collect();
    Err(TrainError::unsupported(format!(
        "dataset declares video camera features ({}), and video shards are MP4 files \
         needing an AV1 or H.264 decoder. Video is not supported by this Rust-native \
         reader. LeRobot v3 image features are supported when their parquet column is \
         struct<bytes: binary, path: string> containing PNG or JPEG bytes; ACT consumes \
         their decoded RGB tensors through Batch::with_images and TrainSession::step_on",
        named.join(", ")
    )))
}

fn load_tasks(root: &Path) -> Result<Vec<TaskRecord>> {
    let path = root.join(DEFAULT_TASKS_PATH);
    let table = Table::read(&path)?;
    let indices = table.i64_column(&path, "task_index")?;
    let tasks = table.string_column(&path, "task")?;
    if indices.len() != tasks.len() {
        return Err(TrainError::column(
            &path,
            "task and task_index have different lengths",
        ));
    }
    Ok(tasks
        .into_iter()
        .zip(indices)
        .map(|(task, task_index)| TaskRecord { task, task_index })
        .collect())
}

fn load_episodes(root: &Path, budget: &MetadataBudget) -> Result<Vec<EpisodeRecord>> {
    let directory = root.join(EPISODES_DIR);
    let mut files = collect_parquet_files(&directory, budget.max_files)?;
    files.sort();
    if files.is_empty() {
        return Err(TrainError::io_message(
            &directory,
            "no episode metadata parquet files found",
        ));
    }
    // Checked after the walk and before the first read, which is the only moment at
    // which nothing has been materialized yet.
    crate::limits::within(
        files.len(),
        "the number of episode metadata files",
        budget.max_files,
    )?;

    // Cumulative across the tree: one file inside its own budget says nothing about
    // ten thousand of them.
    let mut total_rows = 0usize;
    let mut total_values = 0usize;
    let mut out = Vec::new();
    for path in files {
        let table = Table::read_within(&path, &budget.read)?;
        total_rows = crate::limits::checked_add(
            total_rows,
            table.rows(),
            "the episode metadata's total row count",
        )?;
        crate::limits::within(
            total_rows,
            "the number of episode metadata rows",
            budget.max_rows,
        )?;
        total_values = crate::limits::checked_add(
            total_values,
            crate::limits::checked_mul(
                table.rows(),
                table.column_names().len(),
                "an episode metadata file's cell count",
            )?,
            "the episode metadata's total decoded size",
        )?;
        crate::limits::within(
            total_values,
            "the episode metadata's total decoded size",
            budget.max_values,
        )?;
        let episode_index = table.i64_column(&path, "episode_index")?;
        let tasks = table.string_list_column(&path, "tasks")?;
        let length = table.i64_column(&path, "length")?;
        let chunk = table.i64_column(&path, "data/chunk_index")?;
        let file = table.i64_column(&path, "data/file_index")?;
        let from = table.i64_column(&path, "dataset_from_index")?;
        let to = table.i64_column(&path, "dataset_to_index")?;
        for row in 0..table.rows() {
            out.push(EpisodeRecord {
                episode_index: episode_index[row],
                tasks: tasks[row].clone(),
                length: length[row],
                data_chunk_index: chunk[row],
                data_file_index: file[row],
                dataset_from_index: from[row],
                dataset_to_index: to[row],
            });
        }
    }
    out.sort_by_key(|episode| episode.episode_index);
    Ok(out)
}

/// Every `*.parquet` under `directory`, recursively.
///
/// Upstream's `load_nested_dataset` globs `**/*.parquet`. The recursion is
/// depth-bounded here because the layout is `chunk-XXX/file-XXX.parquet`: two
/// levels, and a symlink loop in a dataset directory should not hang a run.
fn collect_parquet_files(directory: &Path, max_files: usize) -> Result<Vec<PathBuf>> {
    fn walk(
        directory: &Path,
        depth: usize,
        max_files: usize,
        out: &mut Vec<PathBuf>,
    ) -> Result<()> {
        if depth > 4 {
            return Ok(());
        }
        let entries =
            std::fs::read_dir(directory).map_err(|error| TrainError::io(directory, &error))?;
        for entry in entries {
            let entry = entry.map_err(|error| TrainError::io(directory, &error))?;
            let path = entry.path();
            let kind = entry
                .file_type()
                .map_err(|error| TrainError::io(&path, &error))?;
            if kind.is_dir() {
                walk(&path, depth + 1, max_files, out)?;
            } else if path
                .extension()
                .is_some_and(|extension| extension == "parquet")
            {
                // Bounded *during* the walk, not after it. Collecting the paths of a
                // directory holding millions of files is itself the work being refused,
                // so the count has to stop the walk rather than be checked once it has
                // finished.
                if out.len() >= max_files {
                    return Err(TrainError::io_message(
                        directory,
                        format!(
                            "holds more than the {max_files} episode metadata files the \
                             reader will open"
                        ),
                    ));
                }
                out.push(path);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    if directory.is_dir() {
        walk(directory, 0, max_files, &mut out)?;
    }
    Ok(out)
}
