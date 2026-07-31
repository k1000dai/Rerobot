//! The one error type the training slice reports.
//!
//! Every variant is either a port of an upstream exception or an explicit
//! statement that something upstream supports is *not* implemented here. There is
//! no variant that means "carried on anyway".

use rerobot_core::dataset::delta::DeltaTimestampError;
use rerobot_core::dataset::info::DatasetInfoError;
use rerobot_core::dataset::sampler::SamplerError;
use rerobot_core::dataset::stats::StatsError;
use rerobot_core::policy::act::ActConfigError;
use rerobot_core::policy::normalize::NormalizeError;
use std::fmt;
use std::path::{Path, PathBuf};

/// Why a training run could not start, could not proceed, or could not be saved.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainError {
    /// A file or directory the run needs could not be read or written.
    Io {
        /// Path involved.
        path: PathBuf,
        /// What the operating system said.
        message: String,
    },
    /// `meta/info.json` did not describe a dataset this slice can read.
    Metadata(String),
    /// A parquet file did not have the columns or arrow types required.
    Column {
        /// File involved.
        path: PathBuf,
        /// Which column, and what was wrong with it.
        message: String,
    },
    /// `meta/info.json` was malformed.
    DatasetInfo(DatasetInfoError),
    /// `meta/stats.json` was malformed.
    Stats(StatsError),
    /// The requested normalization could not be resolved against the statistics.
    Normalize(NormalizeError),
    /// The episode-aware sampler rejected the episode boundaries.
    Sampler(SamplerError),
    /// The action window did not land on the dataset's frame grid.
    DeltaTimestamps(DeltaTimestampError),
    /// The ACT configuration was rejected by upstream's `__post_init__`.
    ActConfig(ActConfigError),
    /// A tensor operation failed.
    ///
    /// `candle_core::Error` is neither `Clone` nor `PartialEq`, so it is rendered
    /// at the boundary. The message is candle's own, unmodified.
    Tensor(String),
    /// A checkpoint on disk did not hold what it must.
    Checkpoint {
        /// Directory involved.
        path: PathBuf,
        /// What was missing or inconsistent.
        message: String,
    },
    /// A training step produced a value it cannot continue from.
    ///
    /// Separate from [`TrainError::Metadata`] because it is a *runtime* condition
    /// rather than a malformed input, and a caller may want to tell the two apart:
    /// the first means "this run diverged", the second means "this run was never
    /// well formed".
    NonFinite {
        /// The step that produced it, counted from one.
        step: u64,
        /// Which quantity, e.g. `loss` or `grad_norm`.
        quantity: String,
        /// The value, rendered — `NaN`, `inf` or `-inf`.
        value: String,
    },
    /// The run asked for something upstream supports and this slice does not.
    ///
    /// This is the variant that keeps the port honest: nothing is quietly
    /// downgraded, so a config that would train a different model upstream fails
    /// here instead of training a different model.
    Unsupported(String),
}

impl TrainError {
    /// An [`TrainError::Io`] for `path` from a `std::io::Error`.
    pub fn io(path: impl AsRef<Path>, error: &std::io::Error) -> Self {
        Self::Io {
            path: path.as_ref().to_path_buf(),
            message: error.to_string(),
        }
    }

    /// An [`TrainError::Io`] for `path` from a message.
    pub fn io_message(path: impl AsRef<Path>, message: impl Into<String>) -> Self {
        Self::Io {
            path: path.as_ref().to_path_buf(),
            message: message.into(),
        }
    }

    /// A [`TrainError::Column`] for `path`.
    pub fn column(path: impl AsRef<Path>, message: impl Into<String>) -> Self {
        Self::Column {
            path: path.as_ref().to_path_buf(),
            message: message.into(),
        }
    }

    /// A [`TrainError::Checkpoint`] for `path`.
    pub fn checkpoint(path: impl AsRef<Path>, message: impl Into<String>) -> Self {
        Self::Checkpoint {
            path: path.as_ref().to_path_buf(),
            message: message.into(),
        }
    }

    /// A [`TrainError::Unsupported`].
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }
}

impl fmt::Display for TrainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(formatter, "{}: {message}", path.display()),
            Self::Metadata(message) => formatter.write_str(message),
            Self::Column { path, message } => write!(formatter, "{}: {message}", path.display()),
            Self::DatasetInfo(error) => write!(formatter, "meta/info.json: {error}"),
            Self::Stats(error) => write!(formatter, "{error}"),
            Self::Normalize(error) => write!(formatter, "{error}"),
            Self::Sampler(error) => write!(formatter, "{error}"),
            Self::DeltaTimestamps(error) => write!(formatter, "{error}"),
            Self::ActConfig(error) => write!(formatter, "{error}"),
            Self::Tensor(message) => write!(formatter, "tensor operation failed: {message}"),
            Self::Checkpoint { path, message } => {
                write!(formatter, "{}: {message}", path.display())
            }
            Self::NonFinite {
                step,
                quantity,
                value,
            } => write!(
                formatter,
                "step {step}: {quantity} is not finite ({value}); the step trained nothing, so \
                 the run stops rather than reporting success"
            ),
            Self::Unsupported(message) => write!(formatter, "unsupported in this slice: {message}"),
        }
    }
}

impl std::error::Error for TrainError {}

impl From<candle_core::Error> for TrainError {
    fn from(error: candle_core::Error) -> Self {
        Self::Tensor(error.to_string())
    }
}

impl From<StatsError> for TrainError {
    fn from(error: StatsError) -> Self {
        Self::Stats(error)
    }
}

impl From<NormalizeError> for TrainError {
    fn from(error: NormalizeError) -> Self {
        Self::Normalize(error)
    }
}

impl From<SamplerError> for TrainError {
    fn from(error: SamplerError) -> Self {
        Self::Sampler(error)
    }
}

impl From<DeltaTimestampError> for TrainError {
    fn from(error: DeltaTimestampError) -> Self {
        Self::DeltaTimestamps(error)
    }
}

impl From<ActConfigError> for TrainError {
    fn from(error: ActConfigError) -> Self {
        Self::ActConfig(error)
    }
}

impl From<DatasetInfoError> for TrainError {
    fn from(error: DatasetInfoError) -> Self {
        Self::DatasetInfo(error)
    }
}

/// Shorthand for this crate's results.
pub type Result<T> = std::result::Result<T, TrainError>;
