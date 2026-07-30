//! Port of the delta-timestamp window slice of `lerobot.datasets`.
//!
//! Three upstream functions make up ACT's action chunk:
//!
//! * `datasets.factory.resolve_delta_timestamps` turns a policy config's
//!   `action_delta_indices` into seconds ([`action_delta_timestamps`]);
//! * `datasets.feature_utils.get_delta_indices` turns them back into frame
//!   offsets ([`get_delta_indices`]), and
//!   `feature_utils.check_delta_timestamps` refuses offsets that do not land on
//!   the frame grid ([`check_delta_timestamps`]);
//! * `datasets.dataset_reader.DatasetReader._get_query_indices` clamps a window
//!   to its episode and reports the clamped entries ([`query_window`]).
//!
//! The round trip through seconds is upstream's, not a simplification: the
//! configuration speaks in frames, the dataset's `delta_timestamps` speak in
//! seconds, and the reader speaks in frames again. Because `i / fps` is not
//! exact in binary64, the round trip is only lossless because upstream rounds,
//! and `round` there is Python's — half to even, not half away from zero.

use crate::dataset::json::python_float_repr;
use indexmap::IndexMap;
use std::fmt;

/// `TrainPipelineConfig.tolerance_s`, the default slack allowed when checking
/// that a delta timestamp lands on the frame grid.
pub const DEFAULT_TOLERANCE_S: f64 = 1e-4;

/// Python's `round(float)`: nearest, ties to even.
///
/// `f64::round` breaks ties away from zero, so it answers 1 where Python answers
/// 0 for `round(0.5)`. `get_delta_indices` is spelled `round(d * fps)`, and a
/// delta exactly between two frames is reachable, so the difference is
/// observable.
///
/// ```
/// use rerobot_core::dataset::delta::python_round_half_even;
///
/// assert_eq!(python_round_half_even(0.5), 0.0);
/// assert_eq!(python_round_half_even(1.5), 2.0);
/// assert_eq!(0.5f64.round(), 1.0);
/// ```
pub fn python_round_half_even(value: f64) -> f64 {
    let truncated = value.trunc();
    if (value - truncated).abs() == 0.5 {
        // Exactly between two integers: pick the even one. `floor` rather than
        // `trunc` so that negatives tie the same way (-1.5 -> -2, -0.5 -> 0).
        let lower = value.floor();
        if lower % 2.0 == 0.0 {
            lower
        } else {
            lower + 1.0
        }
    } else {
        // `f64::round` and Python agree away from ties.
        value.round()
    }
}

/// `round(value)`, saturating into `i64` rather than wrapping.
///
/// A Python `int` is unbounded, so upstream cannot overflow here. This port
/// saturates instead of wrapping, which is the difference between a nonsense
/// index and a silently negated one; `docs/compatibility.md` records it.
fn python_round_to_i64(value: f64) -> i64 {
    let rounded = python_round_half_even(value);
    if rounded.is_nan() {
        0
    } else if rounded >= i64::MAX as f64 {
        i64::MAX
    } else if rounded <= i64::MIN as f64 {
        i64::MIN
    } else {
        rounded as i64
    }
}

/// `[i / fps for i in range(chunk_size)]`, the ACT action window in seconds.
pub fn action_delta_timestamps(chunk_size: i64, fps: i64) -> Vec<f64> {
    (0..chunk_size.max(0))
        .map(|index| index as f64 / fps as f64)
        .collect()
}

/// `get_delta_indices`: `round(d * fps)` per key, preserving insertion order.
pub fn get_delta_indices(
    delta_timestamps: &IndexMap<String, Vec<f64>>,
    fps: i64,
) -> IndexMap<String, Vec<i64>> {
    delta_timestamps
        .iter()
        .map(|(key, deltas)| {
            let indices = deltas
                .iter()
                .map(|delta| python_round_to_i64(delta * fps as f64))
                .collect();
            (key.clone(), indices)
        })
        .collect()
}

/// The `ValueError` `check_delta_timestamps` raises, carrying exactly the
/// offending timestamps per key the way upstream's `pformat` dict does.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaTimestampError {
    /// Frame rate the timestamps were checked against.
    pub fps: i64,
    /// Per key, only the timestamps that fell outside the tolerance.
    pub outside_tolerance: IndexMap<String, Vec<f64>>,
}

impl fmt::Display for DeltaTimestampError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "The following delta_timestamps are found outside of tolerance range. \
             Please make sure they are multiples of 1/{} +/- tolerance and adjust \
             their values accordingly. {{",
            self.fps
        )?;
        for (index, (key, values)) in self.outside_tolerance.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(
                formatter,
                "{}: [",
                crate::dataset::info::python_str_repr(key)
            )?;
            for (position, value) in values.iter().enumerate() {
                if position > 0 {
                    formatter.write_str(", ")?;
                }
                formatter.write_str(&python_float_repr(*value))?;
            }
            formatter.write_str("]")?;
        }
        formatter.write_str("}")
    }
}

impl std::error::Error for DeltaTimestampError {}

/// `check_delta_timestamps`: every delta must be a multiple of `1/fps` to
/// within `tolerance_s`, measured as `abs(ts * fps - round(ts * fps)) / fps`.
pub fn check_delta_timestamps(
    delta_timestamps: &IndexMap<String, Vec<f64>>,
    fps: i64,
    tolerance_s: f64,
) -> Result<(), DeltaTimestampError> {
    let mut outside_tolerance: IndexMap<String, Vec<f64>> = IndexMap::new();
    for (key, deltas) in delta_timestamps {
        let offenders: Vec<f64> = deltas
            .iter()
            .copied()
            .filter(|delta| {
                let in_frames = delta * fps as f64;
                let deviation = (in_frames - python_round_half_even(in_frames)).abs() / fps as f64;
                !matches!(
                    deviation.partial_cmp(&tolerance_s),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )
            })
            .collect();
        if !offenders.is_empty() {
            outside_tolerance.insert(key.clone(), offenders);
        }
    }
    if outside_tolerance.is_empty() {
        Ok(())
    } else {
        Err(DeltaTimestampError {
            fps,
            outside_tolerance,
        })
    }
}

/// One episode-clamped delta window: the frame indices to read, and which of
/// them exist only because the window was clamped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryWindow {
    /// `max(ep_start, min(ep_end - 1, abs_idx + delta))` per delta.
    pub indices: Vec<i64>,
    /// `(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end)` per delta.
    ///
    /// This is what reaches the batch as `<key>_is_pad`.
    pub is_pad: Vec<bool>,
}

/// `DatasetReader._get_query_indices` for one key.
///
/// `ep_start` and `ep_end` are the episode's `dataset_from_index` and
/// `dataset_to_index`; the range is half-open, so the last readable frame is
/// `ep_end - 1`.
pub fn query_window(
    abs_idx: i64,
    ep_start: i64,
    ep_end: i64,
    delta_indices: &[i64],
) -> QueryWindow {
    let mut indices = Vec::with_capacity(delta_indices.len());
    let mut is_pad = Vec::with_capacity(delta_indices.len());
    for delta in delta_indices {
        // Saturating so that a hostile delta index cannot wrap the clamp into
        // pointing at another episode.
        let target = abs_idx.saturating_add(*delta);
        // `ep_end - 1` saturates rather than subtracting: the episode table is
        // attacker-controlled parquet, and `ep_end == i64::MIN` makes a plain
        // subtraction panic in a checked build and wrap to `i64::MAX` in release,
        // which would clamp the window onto entirely unrelated frames. A range that
        // degenerate is refused by the dataset reader; what this layer guarantees is
        // that it never wraps or panics on the way to that refusal.
        indices.push(ep_start.max(ep_end.saturating_sub(1).min(target)));
        is_pad.push(target < ep_start || target >= ep_end);
    }
    QueryWindow { indices, is_pad }
}
