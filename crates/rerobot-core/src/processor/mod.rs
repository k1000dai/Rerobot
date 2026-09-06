//! Ports of `lerobot.processor` steps that are pure data transforms.
//!
//! The ordered-map aliases the steps share live here: every one of them models
//! a Python `dict`, whose iteration order is observable in the values these
//! steps return.

use crate::types::{PipelineFeatureType, PolicyFeature};
use indexmap::IndexMap;
use serde_json::Value;

pub mod newline_task;
pub mod pipeline;
pub mod rename;

/// Insertion-ordered feature dict for one pipeline stage.
pub type FeatureMap = IndexMap<String, PolicyFeature>;

/// Feature dicts keyed by pipeline stage.
pub type PipelineFeatures = IndexMap<PipelineFeatureType, FeatureMap>;

/// Insertion-ordered complementary-data dict (`dict[str, Any]` upstream).
pub type ComplementaryData = IndexMap<String, Value>;

/// Stateless/stateful processor payloads at the current JSON-only boundary.
pub type ProcessorState = IndexMap<String, Value>;
