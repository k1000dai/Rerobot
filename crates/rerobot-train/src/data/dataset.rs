//! Port of the state-only path through `LeRobotDataset.__getitem__` /
//! `DatasetReader.get_item`.
//!
//! What is ported: reading the frame row, expanding every configured
//! delta-timestamp window against the episode's boundaries, producing the
//! `<key>_is_pad` flags, attaching the task string behind `task_index`, and decoding
//! any embedded `dtype: "image"` column into RGB pixels — see [`crate::data::image`]
//! for the codec set and the bounds every decode is subject to.
//!
//! What is not: video decoding, image transforms, depth dequantization, the
//! streaming dataset, episode-filtered index remapping onto a subset of files,
//! and the Hub. A dataset needing any of those is refused by
//! [`crate::data::meta::DatasetMetadata::load`] or here, never partially read.

use crate::data::image::DecodedImage;
use crate::data::meta::{DatasetMetadata, ACTION};
use crate::data::parquet::Table;
use crate::error::{Result, TrainError};
use indexmap::IndexMap;
use rerobot_core::dataset::delta::{check_delta_timestamps, get_delta_indices, query_window};
use rerobot_core::dataset::sampler::EpisodeAwareSampler;

/// One item of the dataset: a frame, plus every delta window configured for it.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// `index`, the dataset-absolute frame index.
    pub index: i64,
    /// `episode_index`.
    pub episode_index: i64,
    /// `frame_index`, the position within the episode.
    pub frame_index: i64,
    /// `timestamp`, in seconds from the episode start.
    pub timestamp: f32,
    /// `task_index`.
    pub task_index: i64,
    /// The task string `task_index` resolves to.
    pub task: String,
    /// Per feature key, the window of rows the delta indices selected.
    ///
    /// A key with no configured delta window holds exactly one row: the frame's
    /// own. A key with one holds `delta_indices.len()` rows, in delta order.
    pub windows: IndexMap<String, Vec<Vec<f32>>>,
    /// Per embedded camera key, the decoded RGB image for this frame.
    pub images: IndexMap<String, DecodedImage>,
    /// Per feature key with a configured window, its `<key>_is_pad` flags.
    pub padding: IndexMap<String, Vec<bool>>,
}

impl Frame {
    /// The single row of a key with no delta window, e.g. `observation.state`.
    pub fn value(&self, key: &str) -> Option<&[f32]> {
        self.windows
            .get(key)
            .and_then(|rows| rows.first())
            .map(Vec::as_slice)
    }

    /// The decoded embedded camera image for one key.
    pub fn image(&self, key: &str) -> Option<&DecodedImage> {
        self.images.get(key)
    }

    /// The full window of a key, e.g. `action`.
    pub fn window(&self, key: &str) -> Option<&[Vec<f32>]> {
        self.windows.get(key).map(Vec::as_slice)
    }

    /// The `<key>_is_pad` flags of a key, when it has a delta window.
    pub fn is_pad(&self, key: &str) -> Option<&[bool]> {
        self.padding.get(key).map(Vec::as_slice)
    }
}

/// What one whole dataset may cost, across every file it names.
///
/// A per-file budget ([`crate::data::parquet::ReadBudget`]) bounds one file and says
/// nothing about ten thousand of them — and the episode table, which names the files,
/// is attacker-controlled too. These three totals are accumulated across the read.
///
/// Injectable for the same reason `ReadBudget` is: proving that an over-budget dataset
/// is refused otherwise needs an over-budget dataset, and a fifty-million-row fixture
/// is not something to commit. [`Self::default`] is the production budget from
/// [`crate::limits`], and `tests/limits.rs` asserts that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetBudget {
    /// Data files the episode table may name.
    pub max_files: usize,
    /// Rows the dataset may hold in total.
    pub max_rows: usize,
    /// Decoded scalars the dataset may materialize in total.
    pub max_values: usize,
    /// The budget applied to each individual file underneath this one.
    pub read: crate::data::parquet::ReadBudget,
}

impl Default for DatasetBudget {
    fn default() -> Self {
        Self {
            max_files: crate::limits::MAX_PARQUET_FILES,
            max_rows: crate::limits::MAX_DATASET_ROWS,
            max_values: crate::limits::MAX_DECODED_VALUES,
            read: crate::data::parquet::ReadBudget::default(),
        }
    }
}

/// A LeRobot v3.0 dataset on local disk, with optional embedded camera images.
///
/// The name is the slice's history rather than its scope: the reader began state-only
/// and now also decodes an embedded `dtype: "image"` column into [`Frame::images`].
/// What it still refuses is a `dtype: "video"` feature, the streaming dataset and the
/// Hub — see [`crate::data::meta::DatasetMetadata::load`].
#[derive(Debug)]
pub struct StateOnlyDataset {
    metadata: DatasetMetadata,
    delta_indices: IndexMap<String, Vec<i64>>,
    /// Per feature key, every frame of the dataset, in absolute index order.
    columns: IndexMap<String, Vec<Vec<f32>>>,
    /// Per embedded camera key, every decoded frame in absolute index order.
    image_columns: IndexMap<String, Vec<DecodedImage>>,
    timestamps: Vec<f32>,
    frame_indices: Vec<i64>,
    episode_indices: Vec<i64>,
    absolute_indices: Vec<i64>,
    task_indices: Vec<i64>,
}

impl StateOnlyDataset {
    /// Read the dataset at `root`, expanding `delta_timestamps` into frame offsets.
    ///
    /// `delta_timestamps` is what `datasets.factory.resolve_delta_timestamps`
    /// produces from the policy config; `tolerance_s` is
    /// `TrainPipelineConfig.tolerance_s`, and a window that does not land on the
    /// frame grid within it is refused exactly as upstream's
    /// `check_delta_timestamps` does.
    pub fn load(
        root: &std::path::Path,
        delta_timestamps: &IndexMap<String, Vec<f64>>,
        tolerance_s: f64,
    ) -> Result<Self> {
        Self::load_within(
            root,
            delta_timestamps,
            tolerance_s,
            &DatasetBudget::default(),
        )
    }

    /// [`Self::load`], refusing anything outside `budget`.
    pub fn load_within(
        root: &std::path::Path,
        delta_timestamps: &IndexMap<String, Vec<f64>>,
        tolerance_s: f64,
        budget: &DatasetBudget,
    ) -> Result<Self> {
        let metadata = DatasetMetadata::load(root)?;
        let fps = metadata.fps()?;
        check_delta_timestamps(delta_timestamps, fps, tolerance_s)?;
        let delta_indices = get_delta_indices(delta_timestamps, fps);

        for key in delta_indices.keys() {
            let Some(feature) = metadata.feature(key) else {
                return Err(TrainError::Metadata(format!(
                    "delta_timestamps names {key:?}, which the dataset does not have"
                )));
            };
            if feature.dtype == "image" {
                return Err(TrainError::unsupported(format!(
                    "delta_timestamps for embedded camera {key:?} are not supported; ACT image inputs \
                     are one RGB frame per dataset item"
                )));
            }
        }

        // Every data file of every episode, read once. Upstream keeps an
        // arrow-backed `datasets.Dataset` and reads rows lazily; the cap in
        // `crate::data::parquet::MAX_ROWS` is what keeps this bounded, and
        // `docs/compatibility.md` records that this slice is eager.
        let mut files: Vec<(i64, i64)> = metadata
            .episodes
            .iter()
            .map(|episode| (episode.data_chunk_index, episode.data_file_index))
            .collect();
        files.sort_unstable();
        files.dedup();
        // Per-file budgets do not bound a dataset of many files. The episode table
        // names them, so the count is attacker-controlled too.
        crate::limits::within(files.len(), "the number of data files", budget.max_files)?;

        let value_keys: Vec<String> = metadata
            .feature_keys()
            .filter(|key| {
                !SCALAR_COLUMNS.contains(key)
                    && metadata
                        .feature(key)
                        .is_some_and(|spec| spec.dtype != "image")
            })
            .map(str::to_owned)
            .collect();
        let image_keys: Vec<String> = metadata
            .feature_keys()
            .filter(|key| {
                metadata
                    .feature(key)
                    .is_some_and(|spec| spec.dtype == "image")
            })
            .map(str::to_owned)
            .collect();

        let mut columns: IndexMap<String, Vec<Vec<f32>>> = value_keys
            .iter()
            .map(|key| (key.clone(), Vec::new()))
            .collect();
        let mut image_columns: IndexMap<String, Vec<DecodedImage>> = image_keys
            .iter()
            .map(|key| (key.clone(), Vec::new()))
            .collect();
        let mut timestamps = Vec::new();
        let mut frame_indices = Vec::new();
        let mut episode_indices = Vec::new();
        let mut absolute_indices = Vec::new();
        let mut task_indices = Vec::new();

        // Cumulative across files: one file inside its own budget says nothing about
        // ten thousand of them.
        let mut total_rows = 0usize;
        let mut total_values = 0usize;

        for (chunk_index, file_index) in files {
            let path = metadata.data_file_path(chunk_index, file_index);
            let table = Table::read_within(&path, &budget.read)?;
            total_rows = crate::limits::checked_add(
                total_rows,
                table.rows(),
                "the dataset's total row count",
            )?;
            crate::limits::within(total_rows, "the dataset's total row count", budget.max_rows)?;
            for key in &value_keys {
                let width = metadata
                    .feature(key)
                    .expect("value_keys came from the feature map")
                    .width()?;
                total_values = crate::limits::checked_add(
                    total_values,
                    crate::limits::checked_mul(table.rows(), width, "a column's decoded size")?,
                    "the dataset's total decoded size",
                )?;
                crate::limits::within(
                    total_values,
                    "the dataset's total decoded size",
                    budget.max_values,
                )?;
                let rows = table.vector_column(&path, key, width)?;
                columns
                    .get_mut(key)
                    .expect("initialized above")
                    .extend(rows);
            }
            for key in &image_keys {
                let spec = metadata
                    .feature(key)
                    .expect("image_keys came from the feature map");
                let shape = image_shape(key, spec)?;
                // The decoded cost, budgeted before a single cell is decoded. One
                // frame is `channels * height * width` f32s, which the declared shape
                // already bounds, and the row count multiplies it.
                total_values = crate::limits::checked_add(
                    total_values,
                    crate::limits::checked_mul(
                        table.rows(),
                        spec.width()?,
                        "an image column's decoded size",
                    )?,
                    "the dataset's total decoded size",
                )?;
                crate::limits::within(
                    total_values,
                    "the dataset's total decoded size",
                    budget.max_values,
                )?;
                // A v3.0 dataset carries its frames *inside* the parquet file. A
                // declaration with no column is the other on-disk layout, and saying so
                // is more use than "column is missing" from the generic reader.
                if !table.has_column(key) {
                    return Err(TrainError::unsupported(format!(
                        "info.json declares camera {key:?} but {} has no such column; this \
                         reader decodes the embedded LeRobot v3.0 form only, where the frames \
                         are a struct<bytes: binary, path: string> column of the data file \
                         rather than separate files on disk",
                        path.display()
                    )));
                }
                let encoded = table.image_column(&path, key)?;
                if encoded.len() != table.rows() {
                    return Err(TrainError::column(
                        &path,
                        format!(
                            "image column {key:?} has {} rows but the file has {}",
                            encoded.len(),
                            table.rows()
                        ),
                    ));
                }
                let decoded = encoded
                    .into_iter()
                    .map(|(bytes, image_path)| {
                        DecodedImage::from_encoded(key, &bytes, image_path, shape)
                    })
                    .collect::<Result<Vec<_>>>()?;
                image_columns
                    .get_mut(key)
                    .expect("initialized above")
                    .extend(decoded);
            }
            timestamps.extend(table.f32_column(&path, "timestamp")?);
            frame_indices.extend(table.i64_column(&path, "frame_index")?);
            episode_indices.extend(table.i64_column(&path, "episode_index")?);
            absolute_indices.extend(table.i64_column(&path, "index")?);
            task_indices.extend(table.i64_column(&path, "task_index")?);
        }

        let dataset = Self {
            metadata,
            delta_indices,
            columns,
            image_columns,
            timestamps,
            frame_indices,
            episode_indices,
            absolute_indices,
            task_indices,
        };
        dataset.check_consistency()?;
        Ok(dataset)
    }

    fn check_consistency(&self) -> Result<()> {
        let rows = self.timestamps.len();
        for (key, values) in &self.columns {
            if values.len() != rows {
                return Err(TrainError::Metadata(format!(
                    "feature {key:?} has {} rows but timestamp has {rows}",
                    values.len()
                )));
            }
        }
        for (key, values) in &self.image_columns {
            if values.len() != rows {
                return Err(TrainError::Metadata(format!(
                    "image feature {key:?} has {} rows but timestamp has {rows}",
                    values.len()
                )));
            }
        }
        for (name, values) in [
            ("frame_index", &self.frame_indices),
            ("episode_index", &self.episode_indices),
            ("index", &self.absolute_indices),
            ("task_index", &self.task_indices),
        ] {
            if values.len() != rows {
                return Err(TrainError::Metadata(format!(
                    "column {name:?} has {} rows but timestamp has {rows}",
                    values.len()
                )));
            }
        }
        let declared = usize::try_from(self.metadata.total_frames()?).unwrap_or(usize::MAX);
        if declared != rows {
            return Err(TrainError::Metadata(format!(
                "info.json declares total_frames={declared} but the data files hold {rows}"
            )));
        }
        // Every frame's own `episode_index` must agree with the episode whose range
        // contains it. The two are independent sources of truth -- the frame row says
        // which episode it belongs to, and the episode table says which frames it owns
        // -- and `get` uses the *frame's* claim to pick the range it clamps the action
        // window against. When they disagree the window is clamped against a range the
        // frame is not inside, which is a silently wrong action chunk rather than an
        // error.
        for (row, declared) in self.episode_indices.iter().enumerate() {
            let absolute = self.absolute_indices[row];
            let owner = self.metadata.episode_of(absolute).ok_or_else(|| {
                TrainError::Metadata(format!(
                    "frame {row} has absolute index {absolute}, which no episode range contains"
                ))
            })?;
            if owner.episode_index != *declared {
                return Err(TrainError::Metadata(format!(
                    "frame {row} declares episode_index {declared} but its absolute index \
                     {absolute} falls in episode {}'s range {}..{}; the frame table and the \
                     episode table disagree about which episode owns it",
                    owner.episode_index, owner.dataset_from_index, owner.dataset_to_index
                )));
            }
        }

        // The reader maps an absolute frame index straight onto a row, which is
        // only valid when the data files are stored in absolute index order.
        // Upstream's writer guarantees it; a dataset that violates it would be
        // silently mis-indexed, so it is checked rather than assumed.
        for (row, index) in self.absolute_indices.iter().enumerate() {
            if *index != row as i64 {
                return Err(TrainError::unsupported(format!(
                    "the data files are not in absolute frame order (row {row} has index \
                     {index}); this slice reads whole, unfiltered datasets only"
                )));
            }
        }
        Ok(())
    }

    /// `__len__`.
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Whether the dataset holds no frames.
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// The metadata read from `meta/`.
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// The delta frame offsets in force, per key.
    pub fn delta_indices(&self) -> &IndexMap<String, Vec<i64>> {
        &self.delta_indices
    }

    /// `num_frames`.
    pub fn num_frames(&self) -> usize {
        self.len()
    }

    /// `num_episodes`.
    pub fn num_episodes(&self) -> usize {
        self.metadata.episodes.len()
    }

    /// `__getitem__`.
    pub fn get(&self, index: usize) -> Result<Frame> {
        if index >= self.len() {
            return Err(TrainError::Metadata(format!(
                "frame {index} is out of range for a dataset of {} frames",
                self.len()
            )));
        }
        let absolute = self.absolute_indices[index];
        let episode_index = self.episode_indices[index];
        let episode = self
            .metadata
            .episodes
            .iter()
            .find(|record| record.episode_index == episode_index)
            .ok_or_else(|| {
                TrainError::Metadata(format!(
                    "frame {index} names episode {episode_index}, which meta/episodes does not \
                     describe"
                ))
            })?;

        let mut windows: IndexMap<String, Vec<Vec<f32>>> = IndexMap::new();
        let mut padding: IndexMap<String, Vec<bool>> = IndexMap::new();
        for (key, values) in &self.columns {
            match self.delta_indices.get(key) {
                Some(deltas) => {
                    let window = query_window(
                        absolute,
                        episode.dataset_from_index,
                        episode.dataset_to_index,
                        deltas,
                    );
                    let mut rows = Vec::with_capacity(window.indices.len());
                    for target in &window.indices {
                        let row = usize::try_from(*target).map_err(|_| {
                            TrainError::Metadata(format!(
                                "the delta window of {key:?} reached negative index {target}"
                            ))
                        })?;
                        rows.push(values.get(row).cloned().ok_or_else(|| {
                            TrainError::Metadata(format!(
                                "the delta window of {key:?} reached row {row}, past the {} \
                                 rows read",
                                values.len()
                            ))
                        })?);
                    }
                    windows.insert(key.clone(), rows);
                    padding.insert(key.clone(), window.is_pad);
                }
                None => {
                    windows.insert(key.clone(), vec![values[index].clone()]);
                }
            }
        }

        let task_index = self.task_indices[index];
        let task = self
            .metadata
            .task(task_index)
            .ok_or_else(|| {
                TrainError::Metadata(format!(
                    "frame {index} names task_index {task_index}, which meta/tasks.parquet does \
                     not describe"
                ))
            })?
            .to_owned();
        let images = self
            .image_columns
            .iter()
            .map(|(key, values)| {
                values
                    .get(index)
                    .cloned()
                    .map(|image| (key.clone(), image))
                    .ok_or_else(|| {
                        TrainError::Metadata(format!(
                            "embedded image feature {key:?} reached row {index}, past the {} rows read",
                            values.len()
                        ))
                    })
            })
            .collect::<Result<IndexMap<_, _>>>()?;

        Ok(Frame {
            index: absolute,
            episode_index,
            frame_index: self.frame_indices[index],
            timestamp: self.timestamps[index],
            task_index,
            task,
            windows,
            images,
            padding,
        })
    }

    /// The `EpisodeAwareSampler` upstream's training loop builds for this dataset.
    pub fn sampler(
        &self,
        episodes: Option<&[i64]>,
        drop_n_last_frames: i64,
        shuffle: bool,
        seed: u64,
    ) -> Result<EpisodeAwareSampler> {
        Ok(EpisodeAwareSampler::new(
            &self.metadata.episode_from_indices(),
            &self.metadata.episode_to_indices(),
            episodes,
            0,
            drop_n_last_frames,
            shuffle,
            seed,
        )?)
    }

    /// Whether `action` carries a delta window, which ACT requires.
    pub fn has_action_window(&self) -> bool {
        self.delta_indices.contains_key(ACTION)
    }
}

/// The `(channels, height, width)` one embedded camera frame must decode to.
///
/// Derived from `info.json` rather than from the file, and bounded here, because it is
/// what every later allocation is sized from: the per-frame decode budget, the
/// reservation in [`crate::data::image::DecodedImage::from_encoded`], and the check
/// that a decoded cell is the frame the dataset says it is.
fn image_shape(key: &str, spec: &crate::data::meta::FeatureSpec) -> Result<(usize, usize, usize)> {
    use crate::data::image::CAMERA_CHANNELS;

    if spec.dtype != "image" {
        return Err(TrainError::Metadata(format!(
            "feature {key:?} is not an image"
        )));
    }
    if spec.shape.len() != 3 {
        return Err(TrainError::Metadata(format!(
            "image feature {key:?} declares shape {:?}; an embedded camera is \
             [channels, height, width]",
            spec.shape
        )));
    }
    // Channel-first, whichever way `info.json` spelled it: see
    // `FeatureSpec::policy_shape`, which is where upstream's `names`-keyed reorder
    // lives. The decoder allocates against this, so it has to agree with the shape
    // the policy config records for the same camera.
    let shape = spec.policy_shape();
    let mut dimensions = [0usize; 3];
    for (index, dimension) in shape.iter().enumerate() {
        dimensions[index] = usize::try_from(*dimension).map_err(|_| {
            TrainError::Metadata(format!(
                "image feature {key:?} has a dimension {dimension} that is not a valid extent"
            ))
        })?;
        if dimensions[index] == 0 {
            return Err(TrainError::Metadata(format!(
                "image feature {key:?} declares shape {:?}, which is empty; a frame with no \
                 pixels cannot be batched",
                spec.shape
            )));
        }
    }
    if dimensions[0] != CAMERA_CHANNELS {
        return Err(TrainError::unsupported(format!(
            "image feature {key:?} declares {} channels; an embedded camera decodes as RGB \
             with {CAMERA_CHANNELS}, and nothing reshapes a frame to fit a different count",
            dimensions[0]
        )));
    }
    crate::limits::within(
        dimensions[1],
        &format!("the height of image feature {key:?}"),
        crate::limits::MAX_IMAGE_EXTENT,
    )?;
    crate::limits::within(
        dimensions[2],
        &format!("the width of image feature {key:?}"),
        crate::limits::MAX_IMAGE_EXTENT,
    )?;
    crate::limits::within(
        crate::limits::checked_product(&dimensions, &format!("the size of image feature {key:?}"))?,
        &format!("the size of image feature {key:?}"),
        crate::limits::MAX_FEATURE_WIDTH,
    )?;
    Ok((dimensions[0], dimensions[1], dimensions[2]))
}

/// Columns `info.json` declares as features but that are per-frame scalars rather
/// than vectors, and that the reader therefore holds in dedicated fields.
const SCALAR_COLUMNS: [&str; 5] = [
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
];
