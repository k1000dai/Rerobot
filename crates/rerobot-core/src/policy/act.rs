//! Configuration contract for upstream's Action Chunking Transformer policy.
//!
//! This module ports `lerobot.policies.act.configuration_act.ACTConfig`. It
//! deliberately stops at the configuration boundary: the ACT tensor model and
//! processor pipeline are separate, still-unported slices.

use crate::types::{FeatureType, NormalizationMode, PolicyFeature};
use indexmap::IndexMap;
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Upstream's AdamW optimizer preset returned by `ACTConfig`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdamWConfig {
    /// Learning rate.
    pub lr: f64,
    /// Decoupled weight decay.
    pub weight_decay: f64,
    /// Global gradient clipping norm inherited from `AdamWConfig`.
    pub grad_clip_norm: f64,
    /// First- and second-moment coefficients.
    pub betas: [f64; 2],
    /// Numerical stability term.
    pub eps: f64,
}

/// Python exception class represented by an ACT configuration failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActConfigErrorKind {
    /// Python `ValueError`.
    Value,
    /// Python `NotImplementedError`.
    NotImplemented,
}

/// Exact validation message and its upstream Python exception class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActConfigError {
    kind: ActConfigErrorKind,
    message: String,
}

impl ActConfigError {
    fn value(message: String) -> Self {
        Self {
            kind: ActConfigErrorKind::Value,
            message,
        }
    }

    fn not_implemented(message: &'static str) -> Self {
        Self {
            kind: ActConfigErrorKind::NotImplemented,
            message: message.into(),
        }
    }

    /// Upstream Python exception class represented by this error.
    pub fn kind(&self) -> ActConfigErrorKind {
        self.kind
    }
}

impl fmt::Display for ActConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ActConfigError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ActType {
    #[serde(rename = "act")]
    Act,
}

fn one() -> BigInt {
    BigInt::from(1)
}
fn hundred() -> BigInt {
    BigInt::from(100)
}
fn five_hundred_twelve() -> BigInt {
    BigInt::from(512)
}
fn eight() -> BigInt {
    BigInt::from(8)
}
fn three_thousand_two_hundred() -> BigInt {
    BigInt::from(3200)
}
fn four() -> BigInt {
    BigInt::from(4)
}
fn thirty_two() -> BigInt {
    BigInt::from(32)
}
fn default_input_features() -> Option<IndexMap<String, PolicyFeature>> {
    Some(IndexMap::new())
}
fn default_output_features() -> Option<IndexMap<String, PolicyFeature>> {
    Some(IndexMap::new())
}
fn default_push_to_hub() -> bool {
    true
}
fn default_normalization_mapping() -> IndexMap<FeatureType, NormalizationMode> {
    IndexMap::from([
        (FeatureType::Visual, NormalizationMode::MeanStd),
        (FeatureType::State, NormalizationMode::MeanStd),
        (FeatureType::Action, NormalizationMode::MeanStd),
    ])
}
fn default_vision_backbone() -> String {
    "resnet18".into()
}
fn default_backbone_weights() -> Option<String> {
    Some("ResNet18_Weights.IMAGENET1K_V1".into())
}
fn default_feedforward_activation() -> String {
    "relu".into()
}
fn default_true() -> bool {
    true
}
fn default_dropout() -> f64 {
    0.1
}
fn default_kl_weight() -> f64 {
    10.0
}
fn default_optimizer_lr() -> f64 {
    1e-5
}
fn default_optimizer_weight_decay() -> f64 {
    1e-4
}
fn default_dilation() -> PythonIntBool {
    PythonIntBool::Bool(false)
}

/// Upstream's unusually annotated dilation setting: `int = False`.
///
/// A freshly constructed Python config stores the boolean default and writes
/// JSON `false`. Draccus checkpoint decoding follows the annotation, accepts
/// Python-int inputs, and converts booleans to integer `0`/`1`. Both observable
/// forms are retained here.
#[derive(Debug, Clone)]
pub enum PythonIntBool {
    /// Fresh-constructor boolean spelling.
    Bool(bool),
    /// Draccus-decoded arbitrary-precision integer.
    Int(BigInt),
}

impl PartialEq for PythonIntBool {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(left), Self::Bool(right)) => left == right,
            (Self::Int(left), Self::Int(right)) => left == right,
            (Self::Bool(value), Self::Int(integer)) | (Self::Int(integer), Self::Bool(value)) => {
                integer == &BigInt::from(u8::from(*value))
            }
        }
    }
}

impl Eq for PythonIntBool {}

impl Serialize for PythonIntBool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Int(value) => bigint_wire::serialize(value, serializer),
        }
    }
}

impl<'de> Deserialize<'de> for PythonIntBool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        bigint_wire::deserialize(deserializer).map(Self::Int)
    }
}

/// Port of `lerobot.policies.act.configuration_act.ACTConfig`.
///
/// Integer configuration fields use [`BigInt`] because upstream accepts Python
/// `int`; validation therefore cannot silently narrow values to a machine word.
/// Rust values are independently owned. In contrast, a Python dataclass shallow
/// copy can continue sharing nested dictionaries and lists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActConfig {
    #[serde(rename = "type")]
    policy_type: ActType,
    /// Number of observation steps.
    #[serde(default = "one", with = "bigint_wire")]
    pub n_obs_steps: BigInt,
    /// Ordered input feature mapping; `None` represents JSON `null`.
    #[serde(default = "default_input_features")]
    pub input_features: Option<IndexMap<String, PolicyFeature>>,
    /// Ordered output feature mapping; `None` represents JSON `null`.
    #[serde(default = "default_output_features")]
    pub output_features: Option<IndexMap<String, PolicyFeature>>,
    /// Requested compute device. Device availability selection is not performed
    /// at this pure configuration boundary.
    #[serde(default)]
    pub device: Option<String>,
    /// Whether automatic mixed precision was requested.
    #[serde(default, with = "bool_wire")]
    pub use_amp: bool,
    /// Whether parameter-efficient fine tuning was used.
    #[serde(default, with = "bool_wire")]
    pub use_peft: bool,
    /// Whether checkpoints should be pushed to the Hub.
    #[serde(default = "default_push_to_hub", with = "bool_wire")]
    pub push_to_hub: bool,
    /// Optional Hub repository identifier.
    #[serde(default)]
    pub repo_id: Option<String>,
    /// Optional Hub privacy setting.
    #[serde(default, with = "optional_bool_wire")]
    pub private: Option<bool>,
    /// Optional Hub tags.
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Optional model license.
    #[serde(default)]
    pub license: Option<String>,
    /// Optional local or Hub pretrained source.
    #[serde(default)]
    pub pretrained_path: Option<String>,
    /// Optional pinned Hub revision.
    #[serde(default)]
    pub pretrained_revision: Option<String>,
    /// Predicted action chunk length.
    #[serde(default = "hundred", with = "bigint_wire")]
    pub chunk_size: BigInt,
    /// Number of predicted actions executed per invocation.
    #[serde(default = "hundred", with = "bigint_wire")]
    pub n_action_steps: BigInt,
    /// Feature normalization modes, preserving insertion order.
    #[serde(default = "default_normalization_mapping")]
    pub normalization_mapping: IndexMap<FeatureType, NormalizationMode>,
    /// Torchvision ResNet variant.
    #[serde(default = "default_vision_backbone")]
    pub vision_backbone: String,
    /// Optional torchvision pretrained weights identifier.
    #[serde(default = "default_backbone_weights")]
    pub pretrained_backbone_weights: Option<String>,
    /// Replace the final ResNet stride with dilation.
    #[serde(default = "default_dilation")]
    pub replace_final_stride_with_dilation: PythonIntBool,
    /// Use transformer pre-normalization.
    #[serde(default, with = "bool_wire")]
    pub pre_norm: bool,
    /// Transformer hidden dimension.
    #[serde(default = "five_hundred_twelve", with = "bigint_wire")]
    pub dim_model: BigInt,
    /// Attention head count.
    #[serde(default = "eight", with = "bigint_wire")]
    pub n_heads: BigInt,
    /// Feed-forward hidden dimension.
    #[serde(default = "three_thousand_two_hundred", with = "bigint_wire")]
    pub dim_feedforward: BigInt,
    /// Feed-forward activation name.
    #[serde(default = "default_feedforward_activation")]
    pub feedforward_activation: String,
    /// Transformer encoder layer count.
    #[serde(default = "four", with = "bigint_wire")]
    pub n_encoder_layers: BigInt,
    /// Transformer decoder layer count.
    #[serde(default = "one", with = "bigint_wire")]
    pub n_decoder_layers: BigInt,
    /// Enable the variational objective.
    #[serde(default = "default_true", with = "bool_wire")]
    pub use_vae: bool,
    /// VAE latent dimension.
    #[serde(default = "thirty_two", with = "bigint_wire")]
    pub latent_dim: BigInt,
    /// VAE encoder layer count.
    #[serde(default = "four", with = "bigint_wire")]
    pub n_vae_encoder_layers: BigInt,
    /// Optional temporal ensemble coefficient.
    #[serde(default, with = "optional_f64_wire")]
    pub temporal_ensemble_coeff: Option<f64>,
    /// Transformer dropout.
    #[serde(default = "default_dropout", with = "f64_wire")]
    pub dropout: f64,
    /// KL-divergence loss weight.
    #[serde(default = "default_kl_weight", with = "f64_wire")]
    pub kl_weight: f64,
    /// Main optimizer learning rate.
    #[serde(default = "default_optimizer_lr", with = "f64_wire")]
    pub optimizer_lr: f64,
    /// AdamW weight decay.
    #[serde(default = "default_optimizer_weight_decay", with = "f64_wire")]
    pub optimizer_weight_decay: f64,
    /// Vision backbone learning rate (retained although upstream's preset does
    /// not currently place it in a separate optimizer group).
    #[serde(default = "default_optimizer_lr", with = "f64_wire")]
    pub optimizer_lr_backbone: f64,
}

impl Default for ActConfig {
    fn default() -> Self {
        Self {
            policy_type: ActType::Act,
            n_obs_steps: one(),
            input_features: default_input_features(),
            output_features: default_output_features(),
            device: None,
            use_amp: false,
            use_peft: false,
            push_to_hub: true,
            repo_id: None,
            private: None,
            tags: None,
            license: None,
            pretrained_path: None,
            pretrained_revision: None,
            chunk_size: hundred(),
            n_action_steps: hundred(),
            normalization_mapping: default_normalization_mapping(),
            vision_backbone: default_vision_backbone(),
            pretrained_backbone_weights: default_backbone_weights(),
            replace_final_stride_with_dilation: default_dilation(),
            pre_norm: false,
            dim_model: five_hundred_twelve(),
            n_heads: eight(),
            dim_feedforward: three_thousand_two_hundred(),
            feedforward_activation: default_feedforward_activation(),
            n_encoder_layers: four(),
            n_decoder_layers: one(),
            use_vae: true,
            latent_dim: thirty_two(),
            n_vae_encoder_layers: four(),
            temporal_ensemble_coeff: None,
            dropout: default_dropout(),
            kl_weight: default_kl_weight(),
            optimizer_lr: default_optimizer_lr(),
            optimizer_weight_decay: default_optimizer_weight_decay(),
            optimizer_lr_backbone: default_optimizer_lr(),
        }
    }
}

impl ActConfig {
    /// Run upstream's `__post_init__` ACT-specific checks in the same order.
    ///
    /// Device selection and AMP warnings belong to the later runtime adapter and
    /// are intentionally not synthesized here.
    pub fn validate(&self) -> Result<(), ActConfigError> {
        if !self.vision_backbone.starts_with("resnet") {
            return Err(ActConfigError::value(format!(
                "`vision_backbone` must be one of the ResNet variants. Got {}.",
                self.vision_backbone
            )));
        }
        if self.temporal_ensemble_coeff.is_some() && self.n_action_steps > one() {
            return Err(ActConfigError::not_implemented(
                "`n_action_steps` must be 1 when using temporal ensembling. This is because the policy needs to be queried every step to compute the ensembled action.",
            ));
        }
        if self.n_action_steps > self.chunk_size {
            return Err(ActConfigError::value(format!(
                "The chunk size is the upper bound for the number of action steps per model invocation. Got {} for `n_action_steps` and {} for `chunk_size`.",
                self.n_action_steps, self.chunk_size
            )));
        }
        if self.n_obs_steps != one() {
            return Err(ActConfigError::value(format!(
                "Multiple observation steps not handled yet. Got `nobs_steps={}`",
                self.n_obs_steps
            )));
        }
        Ok(())
    }

    /// Validate that an image or environment-state input exists.
    pub fn validate_features(&self) -> Result<(), ActConfigError> {
        let valid = self.input_features.as_ref().is_some_and(|features| {
            features
                .values()
                .any(|feature| matches!(feature.r#type, FeatureType::Visual | FeatureType::Env))
        });
        if valid {
            Ok(())
        } else {
            Err(ActConfigError::value(
                "You must provide at least one image or the environment state among the inputs."
                    .into(),
            ))
        }
    }

    /// Upstream AdamW preset.
    pub fn optimizer_preset(&self) -> AdamWConfig {
        AdamWConfig {
            lr: self.optimizer_lr,
            weight_decay: self.optimizer_weight_decay,
            grad_clip_norm: 10.0,
            betas: [0.9, 0.999],
            eps: 1e-8,
        }
    }

    /// ACT has no scheduler preset upstream.
    pub fn scheduler_preset(&self) -> Option<()> {
        None
    }

    /// ACT does not request observation history indices.
    pub fn observation_delta_indices(&self) -> Option<&'static [BigInt]> {
        None
    }

    /// ACT does not request reward history indices.
    pub fn reward_delta_indices(&self) -> Option<&'static [BigInt]> {
        None
    }

    /// Lazy equivalent of Python's `list(range(chunk_size))`.
    ///
    /// Laziness preserves arbitrary-precision accepted values without turning a
    /// malformed or enormous config into an immediate machine-sized allocation.
    pub fn action_delta_indices(&self) -> ActionDeltaIndices {
        ActionDeltaIndices {
            next: BigInt::from(0),
            end: self.chunk_size.clone(),
        }
    }
}

/// Lazy arbitrary-precision action-index sequence.
#[derive(Debug, Clone)]
pub struct ActionDeltaIndices {
    next: BigInt,
    end: BigInt,
}

impl Iterator for ActionDeltaIndices {
    type Item = BigInt;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.end {
            return None;
        }
        let value = self.next.clone();
        self.next += 1;
        Some(value)
    }
}

mod bigint_wire {
    use num_bigint::BigInt;
    use serde::de::{self, Deserializer};
    use serde::ser::{Error as _, Serializer};
    use serde::{Deserialize, Serialize};
    use serde_json::value::RawValue;
    use std::str::FromStr;

    pub fn serialize<S>(value: &BigInt, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let raw = RawValue::from_string(value.to_string()).map_err(S::Error::custom)?;
        raw.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BigInt, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw: Box<RawValue> = Box::<RawValue>::deserialize(deserializer)?;
        let text = raw.get();
        let normalized = match text {
            // Draccus' integer decoder calls `int(value)`: Python bool is an
            // int subclass and therefore becomes 1/0 before ACTConfig stores it.
            "true" => return Ok(BigInt::from(1)),
            "false" => return Ok(BigInt::from(0)),
            quoted if quoted.starts_with('"') => {
                let value: String = serde_json::from_str(quoted).map_err(de::Error::custom)?;
                value.trim().to_owned()
            }
            bare => bare.to_owned(),
        };
        BigInt::from_str(&normalized).map_err(|_| {
            de::Error::custom(format!(
                "invalid Python integer `{text}`: expected an integer or a string accepted by int()"
            ))
        })
    }
}

mod f64_wire {
    use serde::de::{self, Deserializer};
    use serde::ser::{Error as _, Serializer};
    use serde::Deserialize;
    use serde_json::value::RawValue;

    pub fn serialize<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if !value.is_finite() {
            return Err(S::Error::custom(
                "non-finite Python float checkpoint output is not supported",
            ));
        }
        serializer.serialize_f64(*value)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw: Box<RawValue> = Box::<RawValue>::deserialize(deserializer)?;
        parse(raw.get()).map_err(de::Error::custom)
    }

    pub(super) fn parse(text: &str) -> Result<f64, String> {
        let (normalized, came_from_string) = match text {
            "true" => return Ok(1.0),
            "false" => return Ok(0.0),
            quoted if quoted.starts_with('"') => {
                let value: String =
                    serde_json::from_str(quoted).map_err(|error| error.to_string())?;
                (value.trim().to_owned(), true)
            }
            bare => (bare.to_owned(), false),
        };
        let value = normalized
            .parse::<f64>()
            .map_err(|_| format!("invalid Python float `{text}`"))?;
        // `float(10**10000)` raises OverflowError, while `float("1e10000")`
        // and a JSON exponent decoded as float produce infinity.
        let bare_integer = !came_from_string && !normalized.contains(['.', 'e', 'E']);
        if bare_integer && !value.is_finite() {
            return Err(format!(
                "Python integer `{text}` is too large to convert to float"
            ));
        }
        Ok(value)
    }
}

mod optional_f64_wire {
    use super::f64_wire;
    use serde::de::{self, Deserializer};
    use serde::ser::{Error as _, Serializer};
    use serde::Deserialize;
    use serde_json::value::RawValue;

    pub fn serialize<S>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            None => serializer.serialize_none(),
            Some(value) if value.is_finite() => serializer.serialize_some(value),
            Some(_) => Err(S::Error::custom(
                "non-finite Python float checkpoint output is not supported",
            )),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw: Box<RawValue> = Box::<RawValue>::deserialize(deserializer)?;
        if raw.get() == "null" {
            Ok(None)
        } else {
            f64_wire::parse(raw.get())
                .map(Some)
                .map_err(de::Error::custom)
        }
    }
}

mod bool_wire {
    use serde::de::{self, Deserializer};
    use serde::Deserialize;
    use serde::Serializer;
    use serde_json::value::RawValue;

    pub fn serialize<S>(value: &bool, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bool(*value)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<bool, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw: Box<RawValue> = Box::<RawValue>::deserialize(deserializer)?;
        parse(raw.get()).map_err(de::Error::custom)
    }

    pub(super) fn parse(text: &str) -> Result<bool, String> {
        let normalized = if text.starts_with('"') {
            serde_json::from_str::<String>(text).map_err(|error| error.to_string())?
        } else {
            text.to_owned()
        };
        match normalized.as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(format!("invalid Python bool `{text}`")),
        }
    }
}

mod optional_bool_wire {
    use super::bool_wire;
    use serde::de::{self, Deserializer};
    use serde::Deserialize;
    use serde::Serializer;
    use serde_json::value::RawValue;

    pub fn serialize<S>(value: &Option<bool>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(value) => serializer.serialize_some(value),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw: Box<RawValue> = Box::<RawValue>::deserialize(deserializer)?;
        if raw.get() == "null" {
            Ok(None)
        } else {
            bool_wire::parse(raw.get())
                .map(Some)
                .map_err(de::Error::custom)
        }
    }
}
