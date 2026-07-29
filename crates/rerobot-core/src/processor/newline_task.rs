//! Port of `lerobot.processor.newline_task_processor.NewLineTaskProcessorStep`.
//!
//! The step exists because some tokenizers (PaliGemma) expect the prompt to end
//! with a newline. It rewrites the `task` entry of the complementary data and
//! nothing else, so key order, every other value, and the shape of a value it
//! does not recognise are all observable and preserved.
//!
//! Upstream returns the *same* dict object when `task` is absent or `None`, and
//! a shallow copy otherwise. A shallow Python copy still shares nested mutable
//! values, and `transform_features` aliases its whole input. This owned Rust API
//! deliberately does not reproduce those mutation-visible aliases: it returns
//! independent values. See `docs/compatibility.md` for this explicit boundary.

use super::{ComplementaryData, PipelineFeatures, ProcessorState};
use serde_json::Value;

/// Registry name upstream registers this step under.
///
/// Upstream keeps the SmolVLA-era name for backward compatibility with
/// serialized processor configs; it is deliberately not the class name. This
/// constant preserves the wire spelling only. Rerobot's processor registry and
/// pipeline-config reconstruction are not ported yet.
pub const REGISTRY_NAME: &str = "smolvla_new_line_processor";

/// A processor step that ensures the `task` description ends with a newline.
///
/// ```
/// use rerobot_core::processor::newline_task::NewLineTaskProcessorStep;
/// use rerobot_core::processor::ComplementaryData;
/// use serde_json::json;
///
/// let step = NewLineTaskProcessorStep;
///
/// let mut data = ComplementaryData::new();
/// data.insert("task".to_string(), json!("pick up the cube"));
/// data.insert("index".to_string(), json!(0));
///
/// let out = step.complementary_data(&data);
/// assert_eq!(out["task"], json!("pick up the cube\n"));
/// assert_eq!(out["index"], json!(0)); // every other key is untouched
/// assert_eq!(out.keys().collect::<Vec<_>>(), vec!["task", "index"]);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NewLineTaskProcessorStep;

impl NewLineTaskProcessorStep {
    /// Append a newline to the `task` entry, port of `complementary_data`.
    ///
    /// A string task gains a trailing `\n` unless it already ends with one; a
    /// list gains one per element, but only if *every* element is a string.
    /// Everything else — an absent `task`, a null one, a number, an object, and
    /// a list that mixes strings with anything else — is returned unchanged.
    ///
    /// ```
    /// use rerobot_core::processor::newline_task::NewLineTaskProcessorStep;
    /// use rerobot_core::processor::ComplementaryData;
    /// use serde_json::json;
    ///
    /// let step = NewLineTaskProcessorStep;
    /// let mut data = ComplementaryData::new();
    ///
    /// data.insert("task".to_string(), json!(["task1", "task2\n"]));
    /// assert_eq!(
    ///     step.complementary_data(&data)["task"],
    ///     json!(["task1\n", "task2\n"]),
    /// );
    ///
    /// // One non-string element and the whole list is left alone.
    /// data.insert("task".to_string(), json!(["task1", 1]));
    /// assert_eq!(step.complementary_data(&data)["task"], json!(["task1", 1]));
    ///
    /// // `endswith("\n")` is exact: a bare carriage return is not a newline.
    /// data.insert("task".to_string(), json!("pick\r"));
    /// assert_eq!(step.complementary_data(&data)["task"], json!("pick\r\n"));
    /// ```
    pub fn complementary_data(&self, complementary_data: &ComplementaryData) -> ComplementaryData {
        let updated = match complementary_data.get("task") {
            Some(Value::String(task)) => Some(Value::String(with_newline(task))),
            // `all(isinstance(t, str) for t in task)`, which is vacuously true
            // for an empty list and so rebuilds it as an empty list.
            Some(Value::Array(tasks)) => tasks
                .iter()
                .map(Value::as_str)
                .collect::<Option<Vec<&str>>>()
                .map(|tasks| {
                    Value::Array(
                        tasks
                            .into_iter()
                            .map(|task| Value::String(with_newline(task)))
                            .collect(),
                    )
                }),
            _ => None,
        };
        let mut processed = complementary_data.clone();
        if let Some(task) = updated {
            // `IndexMap::insert` on an existing key keeps its position, which is
            // what Python's `new_complementary_data["task"] = ...` does.
            processed.insert("task".to_string(), task);
        }
        processed
    }

    /// Serializable config, port of `ProcessorStep.get_config`.
    ///
    /// The step declares no configuration, so this is the base class's empty
    /// dict. Upstream's own test asserts on that value.
    ///
    /// ```
    /// use rerobot_core::processor::newline_task::NewLineTaskProcessorStep;
    /// use serde_json::json;
    ///
    /// assert_eq!(NewLineTaskProcessorStep.get_config(), json!({}));
    /// ```
    pub fn get_config(&self) -> Value {
        Value::Object(serde_json::Map::new())
    }

    /// Port of the base class's empty `state_dict`.
    pub fn state_dict(&self) -> ProcessorState {
        ProcessorState::new()
    }

    /// Port of the base class's no-op `load_state_dict`.
    ///
    /// Upstream ignores even a non-empty dictionary for a stateless step.
    pub fn load_state_dict(&mut self, _state: &ProcessorState) {}

    /// Port of the base class's no-op `reset`.
    pub fn reset(&mut self) {}

    /// Port of `transform_features`, which returns its input unchanged.
    ///
    /// The step rewrites values, never the feature contract, so the result is
    /// value-identical to the input and keeps its stage order.
    pub fn transform_features(&self, features: &PipelineFeatures) -> PipelineFeatures {
        features.clone()
    }
}

/// `task` with a trailing newline appended unless it already ends with one.
fn with_newline(task: &str) -> String {
    if task.ends_with('\n') {
        task.to_string()
    } else {
        format!("{task}\n")
    }
}
