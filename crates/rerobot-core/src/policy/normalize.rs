//! Port of `_NormalizationMixin._apply_transform` from
//! `lerobot/processor/normalize_processor.py`.
//!
//! Scope is the numeric transform: which statistics each
//! [`crate::types::NormalizationMode`] consumes, where the epsilon goes (it is *not* the same
//! place in every mode), the identity and pass-through rules, and the
//! `ValueError`s raised when a mode's statistics are absent. Arithmetic is in
//! `f32`, because that is the dtype the tensors carry and the statistics are cast
//! to before the subtraction happens.
//!
//! The surrounding `NormalizerProcessorStep` — the pipeline, the device and dtype
//! movement, `to()`, and the `EnvTransition` plumbing — is a separate, unported
//! slice.

use crate::dataset::stats::DatasetStats;
use crate::types::{NormalizationMode, PolicyFeature};
use indexmap::IndexMap;
use std::fmt;

/// `NormalizerProcessorStep.eps`, the default numerical-stability term.
pub const NORMALIZATION_EPS: f64 = 1e-8;

/// Why a [`Normalizer`] could not be built, or could not transform a value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizeError {
    /// The mode needs statistics the dataset does not carry.
    MissingStatistics {
        /// Feature key.
        key: String,
        /// Mode that was selected for it.
        mode: NormalizationMode,
    },
    /// A statistic's width disagrees with the feature's declared shape.
    StatisticsWidthMismatch {
        /// Feature key.
        key: String,
        /// Statistic name.
        statistic: String,
        /// Width the feature declares.
        expected: usize,
        /// Width the statistic has.
        found: usize,
    },
    /// A value's width disagrees with the feature's declared shape.
    WidthMismatch {
        /// Feature key.
        key: String,
        /// Width the feature declares.
        expected: usize,
        /// Width the value has.
        found: usize,
    },
}

impl fmt::Display for NormalizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingStatistics { key, mode } => {
                let (name, needs) = match mode {
                    NormalizationMode::MeanStd => ("MEAN_STD", "mean and std"),
                    NormalizationMode::MinMax => ("MIN_MAX", "min and max"),
                    NormalizationMode::Quantiles => ("QUANTILES", "q01 and q99"),
                    NormalizationMode::Quantile10 => ("QUANTILE10", "q10 and q90"),
                    NormalizationMode::Identity => ("IDENTITY", "no"),
                };
                write!(
                    formatter,
                    "{name} normalization mode requires {needs} stats, please update the \
                     dataset with the correct stats (feature {key:?})"
                )
            }
            Self::StatisticsWidthMismatch {
                key,
                statistic,
                expected,
                found,
            } => write!(
                formatter,
                "{key:?} statistic {statistic:?} has width {found} but the feature declares {expected}"
            ),
            Self::WidthMismatch {
                key,
                expected,
                found,
            } => write!(
                formatter,
                "{key:?} expects {expected} values, got {found}"
            ),
        }
    }
}

impl std::error::Error for NormalizeError {}

/// The resolved statistics of one normalized feature.
#[derive(Debug, Clone, PartialEq)]
struct Entry {
    mode: NormalizationMode,
    width: usize,
    /// Left operand: `mean`, `min`, `q01` or `q10` depending on the mode.
    low: Vec<f32>,
    /// Right operand: `std`, `max`, `q99` or `q90` depending on the mode.
    high: Vec<f32>,
}

/// `_NormalizationMixin` for a fixed set of features and statistics.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Normalizer {
    entries: IndexMap<String, Entry>,
}

impl Normalizer {
    /// Resolve `features` against `stats` under `norm_map`.
    ///
    /// A feature whose type is absent from `norm_map` gets
    /// [`NormalizationMode::Identity`], as upstream's
    /// `self.norm_map.get(feature_type, IDENTITY)` does. A feature with no
    /// statistics entry is skipped rather than rejected, matching
    /// `key not in self._tensor_stats`. A feature whose *mode* needs statistics
    /// the entry lacks is [`NormalizeError::MissingStatistics`].
    pub fn new(
        features: &IndexMap<String, PolicyFeature>,
        norm_map: &IndexMap<String, NormalizationMode>,
        stats: &DatasetStats,
    ) -> Result<Self, NormalizeError> {
        let mut entries = IndexMap::new();
        for (key, feature) in features {
            let mode = norm_map
                .get(feature.r#type.as_str())
                .copied()
                .unwrap_or(NormalizationMode::Identity);
            if mode == NormalizationMode::Identity {
                continue;
            }
            let Some(feature_stats) = stats.get(key) else {
                // `key not in self._tensor_stats` -> the tensor is returned as
                // it came in, without complaint.
                continue;
            };
            let (low_name, high_name) = statistic_names(mode);
            let (Some(low), Some(high)) =
                (feature_stats.get(low_name), feature_stats.get(high_name))
            else {
                return Err(NormalizeError::MissingStatistics {
                    key: key.clone(),
                    mode,
                });
            };
            let width = declared_width(feature);
            for (statistic, values) in [(low_name, low), (high_name, high)] {
                if values.len() != width {
                    return Err(NormalizeError::StatisticsWidthMismatch {
                        key: key.clone(),
                        statistic: statistic.to_owned(),
                        expected: width,
                        found: values.len(),
                    });
                }
            }
            entries.insert(
                key.clone(),
                Entry {
                    mode,
                    width,
                    low: low.iter().map(|value| *value as f32).collect(),
                    high: high.iter().map(|value| *value as f32).collect(),
                },
            );
        }
        Ok(Self { entries })
    }

    /// The mode a key is normalized under, or `None` when it passes through.
    pub fn mode(&self, key: &str) -> Option<NormalizationMode> {
        self.entries.get(key).map(|entry| entry.mode)
    }

    /// Keys this normalizer transforms, in feature declaration order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Forward transform. Unknown keys pass through unchanged.
    pub fn normalize(&self, key: &str, values: &[f32]) -> Result<Vec<f32>, NormalizeError> {
        self.apply(key, values, false)
    }

    /// Inverse transform. Unknown keys pass through unchanged.
    pub fn unnormalize(&self, key: &str, values: &[f32]) -> Result<Vec<f32>, NormalizeError> {
        self.apply(key, values, true)
    }

    fn apply(&self, key: &str, values: &[f32], inverse: bool) -> Result<Vec<f32>, NormalizeError> {
        let Some(entry) = self.entries.get(key) else {
            return Ok(values.to_vec());
        };
        if values.len() != entry.width {
            return Err(NormalizeError::WidthMismatch {
                key: key.to_owned(),
                expected: entry.width,
                found: values.len(),
            });
        }
        let eps = NORMALIZATION_EPS as f32;
        let transformed = values
            .iter()
            .zip(&entry.low)
            .zip(&entry.high)
            .map(|((value, low), high)| match entry.mode {
                NormalizationMode::MeanStd => {
                    // Forward divides by `std + eps`; the inverse multiplies by
                    // `std` alone. The asymmetry is upstream's.
                    if inverse {
                        value * high + low
                    } else {
                        (value - low) / (high + eps)
                    }
                }
                // MIN_MAX and the two quantile modes share the shape
                // `2 * (x - low) / (high - low) - 1`, differing only in which
                // statistics `low` and `high` are and in how a zero-width range
                // is patched: MIN_MAX substitutes eps for the denominator, and
                // so do the quantile modes.
                NormalizationMode::MinMax
                | NormalizationMode::Quantiles
                | NormalizationMode::Quantile10 => {
                    let denominator = high - low;
                    let denominator = if denominator == 0.0 { eps } else { denominator };
                    if inverse {
                        (value + 1.0) / 2.0 * denominator + low
                    } else {
                        2.0 * (value - low) / denominator - 1.0
                    }
                }
                // Unreachable: an identity entry is never inserted.
                NormalizationMode::Identity => *value,
            })
            .collect();
        Ok(transformed)
    }
}

/// Which pair of statistics a mode consumes, low then high.
fn statistic_names(mode: NormalizationMode) -> (&'static str, &'static str) {
    match mode {
        NormalizationMode::MeanStd => ("mean", "std"),
        NormalizationMode::MinMax => ("min", "max"),
        NormalizationMode::Quantiles => ("q01", "q99"),
        NormalizationMode::Quantile10 => ("q10", "q90"),
        NormalizationMode::Identity => ("", ""),
    }
}

/// The number of scalars one frame of a feature carries.
///
/// Upstream never asks: torch broadcasts the statistics over whatever shape
/// arrives. Here the declared shape is known, so a disagreement is a bug rather
/// than a broadcast. A dimension outside `usize` cannot describe an in-memory
/// frame, so it collapses to zero and the width check reports the mismatch.
fn declared_width(feature: &PolicyFeature) -> usize {
    feature
        .shape
        .iter()
        .map(|dimension| usize::try_from(dimension).unwrap_or(0))
        .product()
}
