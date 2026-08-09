//! Port of `lerobot/datasets/io_utils.py`'s `load_stats` and
//! `cast_stats_to_numpy`: reading `meta/stats.json` from a local directory.
//!
//! Upstream's value domain is `dict[str, dict[str, np.ndarray]]`, produced by
//! `np.atleast_1d(np.array(value))` over the flattened JSON document. This port
//! narrows it in one deliberate way and says so rather than pretending
//! otherwise: a statistic must be a scalar or a flat list of numbers. Camera
//! features carry `(3, 1, 1)` statistics upstream, and reading those as a flat
//! vector would silently mis-shape them, so a nested list is
//! [`StatsError::NestedStatistic`]. This slice is state-only; when image
//! features land, so does the nested shape.
//!
//! Non-finite values are accepted, because `json.dump` writes the bare `NaN`,
//! `Infinity` and `-Infinity` tokens and a degenerate episode produces them.

use crate::dataset::io::load_json;
use crate::dataset::json::JsonLike;
use crate::dataset::STATS_PATH;
use indexmap::IndexMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Why `meta/stats.json` could not be turned into statistics.
#[derive(Debug, Clone, PartialEq)]
pub enum StatsError {
    /// The file could not be read or parsed.
    Parse {
        /// The path that was read.
        path: PathBuf,
        /// Rendering of the underlying [`crate::dataset::io::LoadError`].
        message: String,
    },
    /// The document's top level was not a JSON object.
    NotAnObject {
        /// Python type name of what was found.
        found: String,
    },
    /// A feature's value was not a JSON object of statistics.
    FeatureNotAnObject {
        /// Feature key.
        feature: String,
        /// Python type name of what was found.
        found: String,
    },
    /// A statistic contained a nested list; see the module documentation.
    NestedStatistic {
        /// Feature key.
        feature: String,
        /// Statistic name, e.g. `mean`.
        statistic: String,
    },
    /// A statistic contained something that is not a number.
    NotANumber {
        /// Feature key.
        feature: String,
        /// Statistic name.
        statistic: String,
        /// Python type name of what was found.
        found: String,
    },
}

impl fmt::Display for StatsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse { path, message } => {
                write!(formatter, "cannot read {}: {message}", path.display())
            }
            Self::NotAnObject { found } => write!(
                formatter,
                "{STATS_PATH} must hold a dict of per-feature statistics, found {found}"
            ),
            Self::FeatureNotAnObject { feature, found } => write!(
                formatter,
                "statistics for {feature:?} must be a dict, found {found}"
            ),
            Self::NestedStatistic { feature, statistic } => write!(
                formatter,
                "{feature:?} statistic {statistic:?} is nested; only scalar and flat \
                 statistics are supported by the state-only dataset slice"
            ),
            Self::NotANumber {
                feature,
                statistic,
                found,
            } => write!(
                formatter,
                "{feature:?} statistic {statistic:?} must be numeric, found {found}"
            ),
        }
    }
}

impl std::error::Error for StatsError {}

/// The statistics of one feature, in the order `stats.json` lists them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FeatureStats(IndexMap<String, Vec<f64>>);

impl FeatureStats {
    /// Construct statistics from their ordered names and values.
    ///
    /// This is primarily for a checkpoint processor state, whose safetensors
    /// representation is already flattened into `feature.statistic` entries.
    pub fn from_entries(entries: IndexMap<String, Vec<f64>>) -> Self {
        Self(entries)
    }

    /// A named statistic, or `None` when the feature does not carry it.
    pub fn get(&self, statistic: &str) -> Option<&[f64]> {
        self.0.get(statistic).map(Vec::as_slice)
    }

    /// `mean`.
    pub fn mean(&self) -> Option<&[f64]> {
        self.get("mean")
    }

    /// `std`.
    pub fn std(&self) -> Option<&[f64]> {
        self.get("std")
    }

    /// `min`.
    pub fn min(&self) -> Option<&[f64]> {
        self.get("min")
    }

    /// `max`.
    pub fn max(&self) -> Option<&[f64]> {
        self.get("max")
    }

    /// Statistic names, in file order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }
}

/// `meta/stats.json` as a typed value: per feature, per statistic, a flat vector.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DatasetStats(IndexMap<String, FeatureStats>);

impl DatasetStats {
    /// Construct a typed statistics document from ordered per-feature entries.
    ///
    /// The JSON loader remains the normal input boundary. This constructor exists
    /// for equivalent on-disk representations such as LeRobot processor
    /// safetensors, where the document has already been validated and flattened.
    pub fn from_entries(entries: IndexMap<String, FeatureStats>) -> Self {
        Self(entries)
    }

    /// The statistics of one feature, or `None` when it is absent.
    pub fn get(&self, feature: &str) -> Option<&FeatureStats> {
        self.0.get(feature)
    }

    /// Feature keys, in file order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    /// Whether any feature is present.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// How many features carry statistics.
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// `load_stats`: `None` when the file is absent, exactly as upstream.
pub fn load_stats(local_dir: &Path) -> Result<Option<DatasetStats>, StatsError> {
    let path = stats_path(local_dir);
    if !path.exists() {
        return Ok(None);
    }
    let document = load_json(&path).map_err(|error| StatsError::Parse {
        path: path.clone(),
        message: error.to_string(),
    })?;
    stats_from_value(&document).map(Some)
}

/// The dataset-relative location of `stats.json` joined onto `local_dir`.
pub fn stats_path(local_dir: &Path) -> PathBuf {
    local_dir.join(STATS_PATH)
}

/// `cast_stats_to_numpy` over an already-parsed document.
pub fn stats_from_value(document: &JsonLike) -> Result<DatasetStats, StatsError> {
    let JsonLike::Object(features) = document else {
        return Err(StatsError::NotAnObject {
            found: document.type_name().to_owned(),
        });
    };
    let mut out: IndexMap<String, FeatureStats> = IndexMap::with_capacity(features.len());
    for (feature, value) in features {
        out.insert(feature.clone(), feature_stats_from_value(feature, value)?);
    }
    Ok(DatasetStats(out))
}

fn feature_stats_from_value(feature: &str, value: &JsonLike) -> Result<FeatureStats, StatsError> {
    let JsonLike::Object(statistics) = value else {
        return Err(StatsError::FeatureNotAnObject {
            feature: feature.to_owned(),
            found: value.type_name().to_owned(),
        });
    };
    let mut out: IndexMap<String, Vec<f64>> = IndexMap::with_capacity(statistics.len());
    for (statistic, entry) in statistics {
        out.insert(
            statistic.clone(),
            statistic_vector(feature, statistic, entry)?,
        );
    }
    Ok(FeatureStats(out))
}

/// `np.atleast_1d(np.array(value))` for the scalar-or-flat-list domain.
fn statistic_vector(
    feature: &str,
    statistic: &str,
    value: &JsonLike,
) -> Result<Vec<f64>, StatsError> {
    match value {
        JsonLike::Array(items) | JsonLike::Tuple(items) => items
            .iter()
            .map(|item| match item {
                JsonLike::Array(_) | JsonLike::Tuple(_) | JsonLike::Object(_) => {
                    Err(StatsError::NestedStatistic {
                        feature: feature.to_owned(),
                        statistic: statistic.to_owned(),
                    })
                }
                scalar => scalar_number(feature, statistic, scalar),
            })
            .collect(),
        scalar => Ok(vec![scalar_number(feature, statistic, scalar)?]),
    }
}

fn scalar_number(feature: &str, statistic: &str, value: &JsonLike) -> Result<f64, StatsError> {
    match value {
        JsonLike::Float(number) => Ok(*number),
        // `np.array` of a Python `int` is exact; the widening here is lossy
        // above 2^53, and `docs/compatibility.md` records it. Statistics are
        // sample moments of `f32` data, so no real value reaches that.
        JsonLike::Int(integer) => Ok(bigint_to_f64(integer)),
        // `np.array(True)` is 1, and `count` is written as an integer, so a
        // boolean is numeric in NumPy's sense too.
        JsonLike::Bool(flag) => Ok(f64::from(u8::from(*flag))),
        other => Err(StatsError::NotANumber {
            feature: feature.to_owned(),
            statistic: statistic.to_owned(),
            found: other.type_name().to_owned(),
        }),
    }
}

fn bigint_to_f64(value: &num_bigint::BigInt) -> f64 {
    // `BigInt` has no infallible `f64` conversion, and `to_string().parse()` is
    // the one that rounds the way CPython's `float(int)` does for huge inputs.
    value
        .to_string()
        .parse::<f64>()
        .unwrap_or(if value.sign() == num_bigint::Sign::Minus {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        })
}
