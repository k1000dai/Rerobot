//! A bounded native reconstruction of LeRobot's processor pipeline.
//!
//! The upstream `DataProcessorPipeline` is generic over arbitrary Python values,
//! supports a mutable registry, dynamic imports, Hub loading, and tensor state. A
//! pure-Rust crate cannot provide those Python-language capabilities by guessing.
//! This module therefore exposes the smaller, explicit JSON transition boundary
//! needed by the two stateless processor steps already ported in this crate:
//! [`rename_observations_processor`](crate::processor::rename) and
//! [`smolvla_new_line_processor`](crate::processor::newline_task).
//!
//! Config parsing is fail-closed: malformed documents, unknown registry names,
//! dynamic class paths, and state-bearing entries are errors rather than silently
//! becoming no-op steps. The accepted config shape is the same saved JSON shape
//! upstream uses (`name` plus an ordered `steps` array), but the executable step
//! set is deliberately limited and reported by [`ProcessorStepRegistry`].

use super::newline_task::NewLineTaskProcessorStep;
use super::rename::RenameObservationsProcessorStep;
use super::{ComplementaryData, PipelineFeatures};
use indexmap::IndexMap;
use serde_json::{Map, Value};
use std::error::Error;
use std::fmt;

/// An ordered JSON-only transition for the native processor boundary.
///
/// This is intentionally not a claim to implement upstream's complete
/// `EnvTransition`: tensors, arbitrary Python objects, actions, rewards, and
/// environment metadata are outside this JSON-focused slice. The fields here are
/// exactly the portions consumed by the two native stateless steps.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonTransition {
    /// Observation keys and JSON values, in insertion order.
    pub observation: super::rename::Observation,
    /// Complementary metadata, in insertion order.
    pub complementary_data: ComplementaryData,
}

impl JsonTransition {
    /// Construct a JSON transition from its ordered observation and metadata maps.
    pub fn new(
        observation: super::rename::Observation,
        complementary_data: ComplementaryData,
    ) -> Self {
        Self {
            observation,
            complementary_data,
        }
    }
}

/// Errors raised while reconstructing or applying a native processor pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessorPipelineError {
    /// The root JSON value was not an object.
    RootMustBeObject,
    /// A required top-level field was absent.
    MissingField {
        /// The absent field name.
        field: &'static str,
    },
    /// A field had a JSON type other than the one required by the saved format.
    WrongType {
        /// JSON path or field name.
        path: String,
        /// Human-readable expected type.
        expected: &'static str,
    },
    /// A step entry was malformed.
    InvalidStep {
        /// Zero-based step index.
        index: usize,
        /// Why the step entry is invalid.
        reason: String,
    },
    /// The config names a registry entry this native build does not provide.
    UnsupportedRegistryName {
        /// The unimplemented registry name.
        name: String,
    },
    /// The config names a dynamic Python class, which this native boundary cannot import.
    UnsupportedClass {
        /// The dynamic class path that cannot be imported.
        class: String,
    },
    /// A feature transform requires the upstream observation stage.
    MissingObservationStage {
        /// Zero-based step index that required the stage.
        step_index: usize,
    },
}

impl fmt::Display for ProcessorPipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RootMustBeObject => {
                f.write_str("processor pipeline config root must be an object")
            }
            Self::MissingField { field } => {
                write!(f, "processor pipeline config is missing {field:?}")
            }
            Self::WrongType { path, expected } => {
                write!(f, "processor pipeline config {path:?} must be {expected}")
            }
            Self::InvalidStep { index, reason } => {
                write!(f, "processor pipeline step {index} is invalid: {reason}")
            }
            Self::UnsupportedRegistryName { name } => {
                write!(
                    f,
                    "processor registry step {name:?} is not supported by this native boundary"
                )
            }
            Self::UnsupportedClass { class } => {
                write!(f, "processor class {class:?} cannot be dynamically imported by this native boundary")
            }
            Self::MissingObservationStage { step_index } => write!(
                f,
                "processor step {step_index} requires an OBSERVATION feature stage"
            ),
        }
    }
}

impl Error for ProcessorPipelineError {}

/// The stateless processor steps supported by the native JSON pipeline.
#[derive(Debug, Clone, PartialEq)]
enum NativeProcessorStep {
    /// Rename observation keys in one pass.
    Rename(RenameObservationsProcessorStep),
    /// Add a final LF to string task prompts.
    NewLine(NewLineTaskProcessorStep),
}

impl NativeProcessorStep {
    fn process(&self, transition: &JsonTransition) -> JsonTransition {
        match self {
            Self::Rename(step) => JsonTransition::new(
                step.observation(&transition.observation),
                transition.complementary_data.clone(),
            ),
            Self::NewLine(step) => JsonTransition::new(
                transition.observation.clone(),
                step.complementary_data(&transition.complementary_data),
            ),
        }
    }

    fn transform_features(&self, features: &PipelineFeatures) -> Option<PipelineFeatures> {
        match self {
            Self::Rename(step) => step.transform_features(features),
            Self::NewLine(step) => Some(step.transform_features(features)),
        }
    }
}

/// Resolver for the native subset of upstream's `ProcessorStepRegistry`.
///
/// The names and lookup precedence match the serialized upstream contract. Unlike
/// Python's process-global mutable registry this resolver has no registration or
/// dynamic import API; accepting a name here means the corresponding native
/// implementation is present and tested.
pub struct ProcessorStepRegistry;

impl ProcessorStepRegistry {
    /// Whether `name` names a processor implemented in this native boundary.
    pub fn contains(name: &str) -> bool {
        matches!(
            name,
            super::rename::REGISTRY_NAME | super::newline_task::REGISTRY_NAME
        )
    }

    /// The supported registry names, in the native resolver's stable order.
    pub fn names() -> &'static [&'static str] {
        &[
            super::rename::REGISTRY_NAME,
            super::newline_task::REGISTRY_NAME,
        ]
    }

    fn build(
        name: &str,
        config: Option<&Value>,
        step_index: usize,
    ) -> Result<NativeProcessorStep, ProcessorPipelineError> {
        if !Self::contains(name) {
            return Err(ProcessorPipelineError::UnsupportedRegistryName {
                name: name.to_owned(),
            });
        }
        let default_config = Value::Object(Map::new());
        let config = config.unwrap_or(&default_config);
        let object = config
            .as_object()
            .ok_or_else(|| ProcessorPipelineError::WrongType {
                path: format!("steps[{step_index}].config"),
                expected: "an object",
            })?;
        let allowed = if name == super::rename::REGISTRY_NAME {
            &["rename_map"][..]
        } else {
            &[][..]
        };
        if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
            return Err(ProcessorPipelineError::InvalidStep {
                index: step_index,
                reason: format!("config contains unexpected field {key:?}"),
            });
        }
        match name {
            super::rename::REGISTRY_NAME => {
                let rename_map = match object.get("rename_map") {
                    None => IndexMap::new(),
                    Some(Value::Object(entries)) => entries
                        .iter()
                        .map(|(old, new)| {
                            new.as_str()
                                .map(|new| (old.clone(), new.to_owned()))
                                .ok_or_else(|| ProcessorPipelineError::WrongType {
                                    path: format!("steps[{step_index}].config.rename_map[{old:?}]"),
                                    expected: "a string",
                                })
                        })
                        .collect::<Result<IndexMap<_, _>, _>>()?,
                    Some(_) => {
                        return Err(ProcessorPipelineError::WrongType {
                            path: format!("steps[{step_index}].config.rename_map"),
                            expected: "an object",
                        })
                    }
                };
                Ok(NativeProcessorStep::Rename(
                    RenameObservationsProcessorStep { rename_map },
                ))
            }
            super::newline_task::REGISTRY_NAME => {
                Ok(NativeProcessorStep::NewLine(NewLineTaskProcessorStep))
            }
            _ => Err(ProcessorPipelineError::UnsupportedRegistryName {
                name: name.to_owned(),
            }),
        }
    }
}

/// A sequential pipeline over [`JsonTransition`] values.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonProcessorPipeline {
    name: String,
    steps: Vec<NativeProcessorStep>,
}

impl JsonProcessorPipeline {
    /// Reconstruct a pipeline from an upstream-shaped JSON config.
    ///
    /// `steps` is required and must be an array. `name` is optional, matching
    /// `DataProcessorPipeline.from_config`, whose default is
    /// `"DataProcessorPipeline"`. A step with `registry_name` takes precedence
    /// over `class`, as it does upstream; a class-only entry is refused because
    /// dynamic Python imports have no safe Rust equivalent.
    pub fn from_config(config: &Value) -> Result<Self, ProcessorPipelineError> {
        let root = config
            .as_object()
            .ok_or(ProcessorPipelineError::RootMustBeObject)?;
        let steps = root
            .get("steps")
            .ok_or(ProcessorPipelineError::MissingField { field: "steps" })?
            .as_array()
            .ok_or_else(|| ProcessorPipelineError::WrongType {
                path: "steps".to_owned(),
                expected: "an array",
            })?;
        let name = match root.get("name") {
            None => "DataProcessorPipeline".to_owned(),
            Some(Value::String(name)) => name.clone(),
            Some(_) => {
                return Err(ProcessorPipelineError::WrongType {
                    path: "name".to_owned(),
                    expected: "a string",
                })
            }
        };

        let mut native_steps = Vec::with_capacity(steps.len());
        for (index, entry) in steps.iter().enumerate() {
            let entry = entry
                .as_object()
                .ok_or_else(|| ProcessorPipelineError::InvalidStep {
                    index,
                    reason: "expected an object".to_owned(),
                })?;
            if let Some(artifacts) = entry.get("artifacts") {
                match artifacts {
                    Value::Object(artifacts) if artifacts.is_empty() => {}
                    Value::Object(_) => {
                        return Err(ProcessorPipelineError::InvalidStep {
                            index,
                            reason: "artifacts are unsupported for the native stateless step set"
                                .to_owned(),
                        })
                    }
                    _ => {
                        return Err(ProcessorPipelineError::WrongType {
                            path: format!("steps[{index}].artifacts"),
                            expected: "an object",
                        })
                    }
                }
            }
            let registry_name = entry.get("registry_name");
            let identifier = registry_name
                .or_else(|| entry.get("class"))
                .ok_or_else(|| ProcessorPipelineError::InvalidStep {
                    index,
                    reason: "requires registry_name or class".to_owned(),
                })?;
            if let Some(name) = registry_name {
                let name = name
                    .as_str()
                    .ok_or_else(|| ProcessorPipelineError::WrongType {
                        path: format!("steps[{index}].registry_name"),
                        expected: "a string",
                    })?;
                native_steps.push(ProcessorStepRegistry::build(
                    name,
                    entry.get("config"),
                    index,
                )?);
            } else {
                let class =
                    identifier
                        .as_str()
                        .ok_or_else(|| ProcessorPipelineError::WrongType {
                            path: format!("steps[{index}].class"),
                            expected: "a string",
                        })?;
                return Err(ProcessorPipelineError::UnsupportedClass {
                    class: class.to_owned(),
                });
            }
            if entry
                .get("state_file")
                .is_some_and(|state_file| !state_file.is_null())
            {
                return Err(ProcessorPipelineError::InvalidStep {
                    index,
                    reason: "state_file is unsupported for the native stateless step set"
                        .to_owned(),
                });
            }
            // `class` is deliberately ignored when registry_name is present, matching
            // upstream's registry-first resolution precedence.
        }

        Ok(Self {
            name,
            steps: native_steps,
        })
    }

    /// Process one transition through every step in order.
    pub fn process(&self, input: &JsonTransition) -> JsonTransition {
        self.steps
            .iter()
            .fold(input.clone(), |transition, step| step.process(&transition))
    }

    /// Reset all stateful steps.
    ///
    /// The native registry currently contains only stateless steps, so this is
    /// intentionally a no-op. Keeping the lifecycle boundary explicit prevents
    /// callers from having to special-case this subset when a stateful native
    /// step is added later.
    pub fn reset(&mut self) {
        // Every currently supported native step is stateless.
    }

    /// Return the initial transition followed by each intermediate stage.
    pub fn step_through(&self, input: &JsonTransition) -> Vec<JsonTransition> {
        let mut states = Vec::with_capacity(self.steps.len().saturating_add(1));
        let mut transition = input.clone();
        states.push(transition.clone());
        for step in &self.steps {
            transition = step.process(&transition);
            states.push(transition.clone());
        }
        states
    }

    /// Apply all steps to an ordered feature description.
    pub fn transform_features(
        &self,
        input: &PipelineFeatures,
    ) -> Result<PipelineFeatures, ProcessorPipelineError> {
        let mut features = input.clone();
        for (index, step) in self.steps.iter().enumerate() {
            features = step
                .transform_features(&features)
                .ok_or(ProcessorPipelineError::MissingObservationStage { step_index: index })?;
        }
        Ok(features)
    }

    /// Serialize the supported pipeline back to the upstream config shape.
    ///
    /// Stateless native steps have no `state_file`, so a load/save round trip
    /// cannot accidentally claim that an unimplemented tensor state was preserved.
    pub fn get_config(&self) -> Value {
        let steps = self
            .steps
            .iter()
            .map(|step| {
                let (registry_name, config) = match step {
                    NativeProcessorStep::Rename(step) => {
                        (super::rename::REGISTRY_NAME, step.get_config())
                    }
                    NativeProcessorStep::NewLine(step) => {
                        (super::newline_task::REGISTRY_NAME, step.get_config())
                    }
                };
                let mut entry = Map::new();
                entry.insert(
                    "registry_name".to_owned(),
                    Value::String(registry_name.to_owned()),
                );
                entry.insert("config".to_owned(), config);
                Value::Object(entry)
            })
            .collect();
        let mut root = Map::new();
        root.insert("name".to_owned(), Value::String(self.name.clone()));
        root.insert("steps".to_owned(), Value::Array(steps));
        Value::Object(root)
    }
}
