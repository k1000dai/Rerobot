//! Port of the `meta/info.json` slice of `lerobot.datasets`.
//!
//! Scope is one file of one dataset: the path constants, the `DatasetInfo`
//! dataclass, and reading and writing `meta/info.json` from a local directory.
//! `LeRobotDatasetMetadata`, tasks, stats, episodes, parquet, video, and the
//! Hub are **not** part of it and are not stubbed.

pub mod delta;
pub mod info;
pub mod io;
pub mod json;
pub mod sampler;
pub mod stats;

/// Max number of files per chunk (`DEFAULT_CHUNK_SIZE`).
///
/// Upstream's is a Python `int`; the value is small and fixed, so the constant
/// is an `i64` and widens to the [`num_bigint::BigInt`] field without loss.
pub const DEFAULT_CHUNK_SIZE: i64 = 1000;

/// Max size per data file, in megabytes (`DEFAULT_DATA_FILE_SIZE_IN_MB`).
pub const DEFAULT_DATA_FILE_SIZE_IN_MB: i64 = 100;

/// Max size per video file, in megabytes (`DEFAULT_VIDEO_FILE_SIZE_IN_MB`).
pub const DEFAULT_VIDEO_FILE_SIZE_IN_MB: i64 = 200;

/// `INFO_PATH` — the dataset-relative location of `info.json`.
pub const INFO_PATH: &str = "meta/info.json";

/// `STATS_PATH` — the dataset-relative location of `stats.json`.
///
/// The constant is ported because it is part of this slice's constant block;
/// nothing in Rerobot reads or writes that file yet.
pub const STATS_PATH: &str = "meta/stats.json";

/// `EPISODES_DIR`.
pub const EPISODES_DIR: &str = "meta/episodes";

/// `DATA_DIR`.
pub const DATA_DIR: &str = "data";

/// `VIDEO_DIR`.
pub const VIDEO_DIR: &str = "videos";

/// `CHUNK_FILE_PATTERN`.
///
/// The braces are Python `str.format` fields, kept verbatim. Rerobot does not
/// yet expand them; `{chunk_index:03d}` has no `format!` equivalent that a
/// caller could reuse by accident.
pub const CHUNK_FILE_PATTERN: &str = "chunk-{chunk_index:03d}/file-{file_index:03d}";

/// `IMAGE_FILE_PATTERN`.
pub const IMAGE_FILE_PATTERN: &str = "frame-{frame_index:06d}.png";

/// `DEPTH_FILE_PATTERN`.
pub const DEPTH_FILE_PATTERN: &str = "frame-{frame_index:06d}.tiff";

/// `DEFAULT_TASKS_PATH`.
pub const DEFAULT_TASKS_PATH: &str = "meta/tasks.parquet";

/// `DEFAULT_EPISODES_PATH` — `EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"`.
pub const DEFAULT_EPISODES_PATH: &str =
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet";

/// `DEFAULT_DATA_PATH` — `DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"`.
pub const DEFAULT_DATA_PATH: &str = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet";

/// `DEFAULT_VIDEO_PATH` — `VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"`.
pub const DEFAULT_VIDEO_PATH: &str =
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4";

/// `DEFAULT_IMAGE_PATH`.
pub const DEFAULT_IMAGE_PATH: &str =
    "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png";

/// `DEFAULT_DEPTH_PATH`.
pub const DEFAULT_DEPTH_PATH: &str =
    "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.tiff";
