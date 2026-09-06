//! Port of `lerobot.processor.rename_processor`
//! (`RenameObservationsProcessorStep`, `rename_stats`).
//!
//! Both functions are single-pass dict rebuilds, so key ordering and
//! last-write-wins collision behaviour are observable. Ordered maps are used
//! throughout to preserve them. The observation transform also carries a base
//! feature rename through the `_is_pad` and `_padding_mask` keys used by current
//! upstream temporal sampling.

use crate::types::PipelineFeatureType;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Insertion-ordered observation dict (`dict[str, Any]` upstream).
pub type Observation = IndexMap<String, Value>;

/// Insertion-ordered normalization-statistics dict (`dict[str, dict[str, Any]]`).
pub type Stats = IndexMap<String, Option<IndexMap<String, Value>>>;

/// The pipeline-feature aliases live in [`crate::processor`]; they are shared
/// with the other ported steps and re-exported here for the paths that named
/// them under this module first.
pub use super::{FeatureMap, PipelineFeatures};

/// Registry name upstream registers this step under.
pub const REGISTRY_NAME: &str = "rename_observations_processor";

/// A processor step that renames keys in an observation dictionary.
///
/// ```
/// use rerobot_core::processor::rename::{Observation, RenameObservationsProcessorStep};
/// use serde_json::json;
///
/// let step = RenameObservationsProcessorStep::new([("pixels", "observation.image")]);
///
/// let mut observation = Observation::new();
/// observation.insert("pixels".to_string(), json!([1, 2, 3]));
/// observation.insert("reward".to_string(), json!(1.0));
///
/// let renamed = step.observation(&observation);
/// assert_eq!(renamed["observation.image"], json!([1, 2, 3]));
/// assert_eq!(renamed["reward"], json!(1.0)); // unmapped keys are untouched
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RenameObservationsProcessorStep {
    /// Mapping from old key names to new key names.
    pub rename_map: IndexMap<String, String>,
}

impl RenameObservationsProcessorStep {
    /// Build a step from `(old, new)` pairs, preserving their order.
    pub fn new<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            rename_map: pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }

    /// Apply the rename map to an observation.
    ///
    /// Keys absent from the map keep their original name. Output order follows
    /// the input observation's iteration order; when two input keys collide on
    /// the same output key, the later one wins and the entry keeps the position
    /// of the earlier one (Python `dict` assignment semantics).
    ///
    /// A base feature mapping also renames its derived `_is_pad` and
    /// `_padding_mask` keys. An exact mapping for one of those derived keys wins.
    /// The map is applied exactly once per key, so a map like `{"a": "b", "b":
    /// "c"}` does not cascade `a -> b -> c`.
    pub fn observation(&self, observation: &Observation) -> Observation {
        let mut processed = Observation::with_capacity(observation.len());
        for (key, value) in observation {
            processed.insert(
                rename_key_with_metadata(key, &self.rename_map),
                value.clone(),
            );
        }
        processed
    }

    /// Serializable config, port of `get_config`.
    pub fn get_config(&self) -> Value {
        let map: Map<String, Value> = self
            .rename_map
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        let mut config = Map::new();
        config.insert("rename_map".to_string(), Value::Object(map));
        Value::Object(config)
    }

    /// Rename the observation-stage feature keys, port of `transform_features`.
    ///
    /// Returns `None` when the input has no `OBSERVATION` stage, mirroring the
    /// upstream `KeyError`.
    pub fn transform_features(&self, features: &PipelineFeatures) -> Option<PipelineFeatures> {
        let observation = features.get(&PipelineFeatureType::Observation)?;
        let renamed: FeatureMap = observation
            .iter()
            .map(|(k, v)| (self.rename_map.get(k).unwrap_or(k).clone(), v.clone()))
            .collect();
        let mut out = features.clone();
        out.insert(PipelineFeatureType::Observation, renamed);
        Some(out)
    }
}

/// Rename an observation key and, when applicable, its temporal padding metadata.
///
/// Current upstream preserves the relationship between a feature and the derived
/// `_is_pad`/`_padding_mask` keys. An exact mapping wins over the suffix rule, and
/// the lookup is performed once, so a destination that is itself another source
/// key is not traversed a second time.
fn rename_key_with_metadata(key: &str, rename_map: &IndexMap<String, String>) -> String {
    if let Some(mapped) = rename_map.get(key) {
        return mapped.clone();
    }
    for suffix in ["_is_pad", "_padding_mask"] {
        if let Some(base) = key.strip_suffix(suffix) {
            if let Some(mapped) = rename_map.get(base) {
                return format!("{mapped}{suffix}");
            }
        }
    }
    key.to_owned()
}

/// Rename the top-level keys of a statistics dict, port of `rename_stats`.
///
/// `None` sub-dicts become empty dicts. Returns an empty dict for empty input.
/// The result never aliases the input; upstream deep-copies for the same reason.
pub fn rename_stats(stats: &Stats, rename_map: &IndexMap<String, String>) -> Stats {
    if stats.is_empty() {
        return Stats::new();
    }
    let mut renamed = Stats::with_capacity(stats.len());
    for (old_key, sub_stats) in stats {
        let new_key = rename_map.get(old_key).unwrap_or(old_key);
        let sub = sub_stats.clone().unwrap_or_default();
        renamed.insert(new_key.clone(), Some(sub));
    }
    renamed
}
