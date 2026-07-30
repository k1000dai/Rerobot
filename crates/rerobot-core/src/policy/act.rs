//! Configuration contract for upstream's Action Chunking Transformer policy.
//!
//! This module ports `lerobot.policies.act.configuration_act.ACTConfig`. It
//! deliberately stops at the configuration boundary: the ACT tensor model and
//! processor pipeline are separate, still-unported slices.

use crate::dataset::json::{dumps_pretty_ascii, loads, JsonLike, JsonObject};
use crate::policy::draccus::{
    decode_bool, decode_float, decode_int, pure_posix_path, python_str, DecodingError,
};
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
    /// Draccus' `DecodingError`/`ParsingError`, raised while reading a
    /// checkpoint rather than while validating a constructed config.
    Decoding,
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

    fn decoding(error: DecodingError) -> Self {
        Self {
            kind: ActConfigErrorKind::Decoding,
            message: error.to_string(),
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
fn default_normalization_mapping() -> IndexMap<String, NormalizationMode> {
    // Upstream writes the `FeatureType` *values* as the literal dict keys, but
    // the annotation is `dict[str, NormalizationMode]`, so the key domain is
    // every string and not just these three.
    IndexMap::from([
        (
            FeatureType::Visual.as_str().to_owned(),
            NormalizationMode::MeanStd,
        ),
        (
            FeatureType::State.as_str().to_owned(),
            NormalizationMode::MeanStd,
        ),
        (
            FeatureType::Action.as_str().to_owned(),
            NormalizationMode::MeanStd,
        ),
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
    #[serde(default, with = "optional_str_wire")]
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
    #[serde(default, with = "optional_str_wire")]
    pub repo_id: Option<String>,
    /// Optional Hub privacy setting.
    #[serde(default, with = "optional_bool_wire")]
    pub private: Option<bool>,
    /// Optional Hub tags.
    #[serde(default, with = "optional_str_list_wire")]
    pub tags: Option<Vec<String>>,
    /// Optional model license.
    #[serde(default, with = "optional_str_wire")]
    pub license: Option<String>,
    /// Optional local or Hub pretrained source.
    ///
    /// Upstream declares this `pathlib.Path | None`, so a decoded value is
    /// normalised by [`crate::policy::draccus::pure_posix_path`] and a non-string is refused
    /// rather than stringified.
    #[serde(default, with = "optional_path_wire")]
    pub pretrained_path: Option<String>,
    /// Optional pinned Hub revision.
    #[serde(default, with = "optional_str_wire")]
    pub pretrained_revision: Option<String>,
    /// Predicted action chunk length.
    #[serde(default = "hundred", with = "bigint_wire")]
    pub chunk_size: BigInt,
    /// Number of predicted actions executed per invocation.
    #[serde(default = "hundred", with = "bigint_wire")]
    pub n_action_steps: BigInt,
    /// Feature normalization modes, preserving insertion order.
    ///
    /// Upstream annotates this `dict[str, NormalizationMode]`, so the key is
    /// an arbitrary string and a checkpoint carrying one outside
    /// [`FeatureType`] loads rather than failing.
    #[serde(default = "default_normalization_mapping")]
    pub normalization_mapping: IndexMap<String, NormalizationMode>,
    /// Torchvision ResNet variant.
    #[serde(default = "default_vision_backbone", with = "str_wire")]
    pub vision_backbone: String,
    /// Optional torchvision pretrained weights identifier.
    #[serde(default = "default_backbone_weights", with = "optional_str_wire")]
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
    #[serde(default = "default_feedforward_activation", with = "str_wire")]
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

    /// Read a policy `config.json` the way `PreTrainedConfig.from_pretrained`
    /// does: CPython's `json.load`, then `draccus.parse(ACTConfig, ...)`.
    ///
    /// This is the checkpoint boundary, and it is wider than plain serde in
    /// three ways serde's data model cannot express:
    ///
    /// * duplicate object keys follow Python `dict` assignment — the last value
    ///   wins and the key keeps its first position — where serde raises;
    /// * the bare `NaN`, `Infinity` and `-Infinity` tokens CPython's writer
    ///   emits are accepted, where `serde_json` rejects the document;
    /// * integers keep every digit, because the reader never routes a numeric
    ///   token through an `f64`.
    ///
    /// Upstream strips the `"type"` tag before handing the document to
    /// Draccus, so it is required here and checked against this policy's
    /// registry name. Decoding ends with upstream's `__post_init__`, which
    /// `draccus.parse` reaches by constructing the dataclass: a checkpoint that
    /// would build an invalid config never yields one.
    ///
    /// ```
    /// use rerobot_core::policy::act::ActConfig;
    ///
    /// let config = ActConfig::from_checkpoint_json(
    ///     r#"{"type": "act", "chunk_size": 1, "chunk_size": 20, "n_action_steps": 20}"#,
    /// )
    /// .unwrap();
    /// assert_eq!(config.chunk_size, rerobot_core::BigInt::from(20));
    /// ```
    pub fn from_checkpoint_json(text: &str) -> Result<Self, ActConfigError> {
        let document = loads(text)
            .map_err(|error| ActConfigError::decoding(DecodingError::new(error.to_string())))?;
        Self::from_checkpoint_value(&document)
    }

    /// [`Self::from_checkpoint_json`] for an already-parsed document.
    pub fn from_checkpoint_value(document: &JsonLike) -> Result<Self, ActConfigError> {
        let config = decode_act_config(document).map_err(ActConfigError::decoding)?;
        config.validate()?;
        Ok(config)
    }

    /// Write this config the way `PreTrainedConfig._save_pretrained` does:
    /// `draccus.dump(self, f, indent=4)` under `draccus.config_type("json")`,
    /// which is `json.dump(encoded, f, indent=4)` with CPython's default
    /// `ensure_ascii=True`.
    ///
    /// The result is byte-identical to upstream's `config.json`: the registry
    /// tag first, then the dataclass fields in declaration order, four-space
    /// indent, `float.__repr__` spelling for every float, CPython's three
    /// non-finite tokens, `\uXXXX` escapes outside printable ASCII, and no
    /// trailing newline.
    ///
    /// ```
    /// use rerobot_core::policy::act::ActConfig;
    ///
    /// let mut config = ActConfig::default();
    /// config.device = Some("cpu".into());
    /// let text = config.to_checkpoint_json();
    /// assert!(text.starts_with("{\n    \"type\": \"act\",\n    \"n_obs_steps\": 1,\n"));
    /// // `json.dump` writes `repr(1e-05)`, not serde_json's `0.00001`.
    /// assert!(text.contains("\"optimizer_lr\": 1e-05,"));
    /// ```
    pub fn to_checkpoint_json(&self) -> String {
        dumps_pretty_ascii(&self.to_checkpoint_value())
    }

    /// [`Self::to_checkpoint_json`] without the final rendering step.
    pub fn to_checkpoint_value(&self) -> JsonLike {
        JsonLike::Object(encode_act_config(self))
    }
}

// ---------------------------------------------------------------------------
// Draccus checkpoint decoding
// ---------------------------------------------------------------------------

/// Pop a field, leaving the rest for the leftover-key check that
/// `decode_dataclass` ends with.
fn take(fields: &mut JsonObject, name: &str) -> Option<JsonLike> {
    fields.shift_remove(name)
}

fn try_clone_string(value: &str) -> Result<String, DecodingError> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(value.len())
        .map_err(|_| DecodingError::new("ACT string allocation failed"))?;
    cloned.push_str(value);
    Ok(cloned)
}

fn try_clone_value(value: &JsonLike) -> Result<JsonLike, DecodingError> {
    Ok(match value {
        JsonLike::Null => JsonLike::Null,
        JsonLike::Bool(value) => JsonLike::Bool(*value),
        JsonLike::Int(value) => JsonLike::Int(value.clone()),
        JsonLike::Float(value) => JsonLike::Float(*value),
        JsonLike::Str(value) => JsonLike::Str(try_clone_string(value)?),
        JsonLike::Array(values) | JsonLike::Tuple(values) => {
            let mut cloned = Vec::new();
            cloned
                .try_reserve_exact(values.len())
                .map_err(|_| DecodingError::new("ACT sequence allocation failed"))?;
            for value in values {
                cloned.push(try_clone_value(value)?);
            }
            if matches!(value, JsonLike::Array(_)) {
                JsonLike::Array(cloned)
            } else {
                JsonLike::Tuple(cloned)
            }
        }
        JsonLike::Object(values) => JsonLike::Object(try_clone_object(values)?),
    })
}

fn try_clone_object(values: &JsonObject) -> Result<JsonObject, DecodingError> {
    let mut cloned = JsonObject::new();
    cloned
        .try_reserve(values.len())
        .map_err(|_| DecodingError::new("ACT object allocation failed"))?;
    for (key, value) in values {
        cloned.insert(try_clone_string(key)?, try_clone_value(value)?);
    }
    Ok(cloned)
}

/// Decode a present field, tagging any failure with its key path.
fn field<T>(
    fields: &mut JsonObject,
    name: &str,
    default: T,
    decoder: impl FnOnce(&JsonLike) -> Result<T, DecodingError>,
) -> Result<T, DecodingError> {
    match take(fields, name) {
        None => Ok(default),
        Some(value) => decoder(&value).map_err(|error| error.under(name)),
    }
}

/// `decode_union(str, NoneType)` and friends: `None` short-circuits before the
/// inner decoder is ever reached.
fn optional<T>(
    value: &JsonLike,
    decoder: impl FnOnce(&JsonLike) -> Result<T, DecodingError>,
) -> Result<Option<T>, DecodingError> {
    match value {
        JsonLike::Null => Ok(None),
        other => decoder(other).map(Some),
    }
}

/// `decode_enum`: `cls(raw_value)`, then `cls[raw_value]`. Both spellings are
/// the member value here, because upstream's members name themselves.
fn decode_normalization_mode(value: &JsonLike) -> Result<NormalizationMode, DecodingError> {
    match value {
        JsonLike::Str(text) => text.parse().map_err(|_| {
            DecodingError::new(format!(
                "Couldn't parse '{text}' into an enum of type \
                 <enum 'NormalizationMode'>"
            ))
        }),
        other => Err(DecodingError::new(format!(
            "Couldn't parse '{}' into an enum of type <enum 'NormalizationMode'>",
            python_str(other)
        ))),
    }
}

fn decode_feature_type(value: &JsonLike) -> Result<FeatureType, DecodingError> {
    match value {
        JsonLike::Str(text) => text.parse().map_err(|_| {
            DecodingError::new(format!(
                "Couldn't parse '{text}' into an enum of type <enum 'FeatureType'>"
            ))
        }),
        other => Err(DecodingError::new(format!(
            "Couldn't parse '{}' into an enum of type <enum 'FeatureType'>",
            python_str(other)
        ))),
    }
}

/// `decode_dataclass(PolicyFeature, ...)`: two fields, both required, and a
/// leftover key is an error rather than a silently dropped extra.
fn decode_policy_feature(value: &JsonLike) -> Result<PolicyFeature, DecodingError> {
    let JsonLike::Object(map) = value else {
        return Err(DecodingError::new(format!(
            "Couldn't parse '{}' into a PolicyFeature",
            python_str(value)
        )));
    };
    let mut fields = try_clone_object(map)?;
    let feature_type = take(&mut fields, "type")
        .map(|raw| decode_feature_type(&raw).map_err(|error| error.under("type")))
        .transpose()?;
    let shape = take(&mut fields, "shape")
        .map(|raw| decode_shape(&raw).map_err(|error| error.under("shape")))
        .transpose()?;
    reject_extra_fields(&fields, "PolicyFeature")?;

    let missing: Vec<&str> = [("type", feature_type.is_none()), ("shape", shape.is_none())]
        .into_iter()
        .filter_map(|(name, absent)| absent.then_some(name))
        .collect();
    if !missing.is_empty() {
        let formatted: Vec<String> = missing.iter().map(|name| format!("`{name}`")).collect();
        return Err(DecodingError::new(format!(
            "Missing required field(s) {} for PolicyFeature",
            formatted.join(", ")
        )));
    }
    Ok(PolicyFeature {
        r#type: feature_type.expect("checked above"),
        shape: shape.expect("checked above"),
    })
}

/// `decode_tuple(int, Ellipsis)`. Draccus iterates strings by character and
/// mappings by key, in addition to ordinary JSON arrays.
fn decode_shape(value: &JsonLike) -> Result<Vec<BigInt>, DecodingError> {
    let mut decoded = Vec::new();
    let capacity = match value {
        JsonLike::Array(items) | JsonLike::Tuple(items) => items.len(),
        JsonLike::Str(text) => text.chars().count(),
        JsonLike::Object(map) => map.len(),
        other => {
            return Err(DecodingError::new(format!(
                "Value must not be None for conversion to a tuple: got '{}'",
                python_str(other)
            )))
        }
    };
    decoded
        .try_reserve_exact(capacity)
        .map_err(|_| DecodingError::new("ACT shape allocation failed"))?;

    match value {
        JsonLike::Array(items) | JsonLike::Tuple(items) => {
            for (index, item) in items.iter().enumerate() {
                decoded.push(decode_int(item).map_err(|error| error.under(index.to_string()))?);
            }
        }
        JsonLike::Str(text) => {
            for (index, character) in text.chars().enumerate() {
                let item = JsonLike::Str(character.to_string());
                decoded.push(decode_int(&item).map_err(|error| error.under(index.to_string()))?);
            }
        }
        JsonLike::Object(map) => {
            for (index, key) in map.keys().enumerate() {
                let item = JsonLike::Str(key.clone());
                decoded.push(decode_int(&item).map_err(|error| error.under(index.to_string()))?);
            }
        }
        _ => unreachable!("validated above"),
    }
    Ok(decoded)
}

fn is_pair(value: &JsonLike) -> bool {
    match value {
        JsonLike::Array(items) | JsonLike::Tuple(items) => items.len() == 2,
        JsonLike::Str(text) => text.chars().count() == 2,
        JsonLike::Object(map) => map.len() == 2,
        _ => false,
    }
}

/// Return fallibly cloned values yielded by an iterable used as one
/// `dict(...)` item.
fn decode_pair(value: &JsonLike) -> Result<Option<(JsonLike, JsonLike)>, DecodingError> {
    let pair = match value {
        JsonLike::Array(items) | JsonLike::Tuple(items) if items.len() == 2 => {
            Some((try_clone_value(&items[0])?, try_clone_value(&items[1])?))
        }
        JsonLike::Str(text) => {
            let mut characters = text.chars();
            let first = characters.next();
            let second = characters.next();
            match (first, second, characters.next()) {
                (Some(first), Some(second), None) => Some((
                    JsonLike::Str(first.to_string()),
                    JsonLike::Str(second.to_string()),
                )),
                _ => None,
            }
        }
        JsonLike::Object(map) if map.len() == 2 => {
            let mut keys = map.keys();
            Some((
                JsonLike::Str(try_clone_string(keys.next().expect("length checked"))?),
                JsonLike::Str(try_clone_string(keys.next().expect("length checked"))?),
            ))
        }
        _ => None,
    };
    Ok(pair)
}

/// `decode_dict(K, V)`, which keeps source order and accepts an iterable of
/// key/value pairs just like Python's `dict(raw_value)`.
fn decode_map<T>(
    value: &JsonLike,
    what: &str,
    mut decoder: impl FnMut(&JsonLike) -> Result<T, DecodingError>,
) -> Result<IndexMap<String, T>, DecodingError> {
    let mut decoded = IndexMap::new();
    match value {
        JsonLike::Object(map) => {
            decoded
                .try_reserve(map.len())
                .map_err(|_| DecodingError::new("ACT mapping allocation failed"))?;
            for (key, item) in map {
                let value = decoder(item).map_err(|error| error.under(key.clone()))?;
                decoded.insert(key.clone(), value);
            }
        }
        JsonLike::Array(items) | JsonLike::Tuple(items) => {
            decoded
                .try_reserve(items.len())
                .map_err(|_| DecodingError::new("ACT mapping allocation failed"))?;
            for (index, pair) in items.iter().enumerate() {
                let Some((raw_key, raw_value)) = decode_pair(pair)? else {
                    return Err(DecodingError::new(format!(
                        "Couldn't parse '{}' into a {what}",
                        python_str(value)
                    ))
                    .under(index.to_string()));
                };
                let key = python_str(&raw_key);
                let value = decoder(&raw_value).map_err(|error| error.under(key.clone()))?;
                decoded.insert(key, value);
            }
        }
        _ => {
            return Err(DecodingError::new(format!(
                "Couldn't parse '{}' into a {what}",
                python_str(value)
            )))
        }
    }
    Ok(decoded)
}

fn draccus_field_failure(
    value: &JsonLike,
    field: &str,
    field_type: &str,
    underlying: &str,
) -> DecodingError {
    DecodingError::new(format!(
        "Failed when parsing value='{}' into field \"<class 'lerobot.policies.act.configuration_act.ACTConfig'>.{field}\" of type {field_type}.\n\tUnderlying error is \"{underlying}\"",
        python_str(value)
    ))
}

fn unpack_failure(value: &JsonLike) -> String {
    let length = match value {
        JsonLike::Array(items) | JsonLike::Tuple(items) => Some(items.len()),
        JsonLike::Object(items) => Some(items.len()),
        JsonLike::Str(text) => Some(text.chars().count()),
        _ => None,
    };
    match length {
        Some(length) if length < 2 => {
            format!("ValueError: not enough values to unpack (expected 2, got {length})")
        }
        Some(_) => "ValueError: too many values to unpack (expected 2)".into(),
        None => format!(
            "TypeError: cannot unpack non-iterable {} object",
            value.type_name()
        ),
    }
}

fn python_key_error(value: &JsonLike) -> String {
    match value {
        JsonLike::Str(text) => format!("KeyError: '{text}'"),
        other => format!("KeyError: {}", python_str(other)),
    }
}

/// Preserve the enum decoder's leaked `KeyError` fallback and Python's pair
/// unpacking failures before `decode_dataclass` wraps the field.
fn decode_normalization_mapping(
    value: &JsonLike,
) -> Result<IndexMap<String, NormalizationMode>, DecodingError> {
    let field_type = "dict[str, lerobot.configs.types.NormalizationMode]";
    let mut decoded = IndexMap::new();
    match value {
        JsonLike::Object(map) => {
            decoded
                .try_reserve(map.len())
                .map_err(|_| DecodingError::new("ACT mapping allocation failed"))?;
            for (key, raw_value) in map {
                let mode = decode_normalization_mode(raw_value).map_err(|_| {
                    draccus_field_failure(
                        value,
                        "normalization_mapping",
                        field_type,
                        &python_key_error(raw_value),
                    )
                })?;
                decoded.insert(key.clone(), mode);
            }
        }
        JsonLike::Array(items) => {
            decoded
                .try_reserve(items.len())
                .map_err(|_| DecodingError::new("ACT mapping allocation failed"))?;
            for pair in items {
                let Some((raw_key, raw_value)) = decode_pair(pair)? else {
                    return Err(draccus_field_failure(
                        value,
                        "normalization_mapping",
                        field_type,
                        &unpack_failure(pair),
                    ));
                };
                let key = python_str(&raw_key);
                let mode = decode_normalization_mode(&raw_value).map_err(|_| {
                    draccus_field_failure(
                        value,
                        "normalization_mapping",
                        field_type,
                        &python_key_error(&raw_value),
                    )
                })?;
                decoded.insert(key, mode);
            }
        }
        other => {
            return Err(draccus_field_failure(
                value,
                "normalization_mapping",
                field_type,
                &format!(
                    "AttributeError: '{}' object has no attribute 'items'",
                    other.type_name()
                ),
            ));
        }
    }
    Ok(decoded)
}

fn scalar_feature_shape(value: &JsonLike) -> Option<(&str, &JsonLike)> {
    let JsonLike::Object(features) = value else {
        return None;
    };
    for (key, feature) in features {
        let JsonLike::Object(fields) = feature else {
            continue;
        };
        let Some(shape) = fields.get("shape") else {
            continue;
        };
        if matches!(
            shape,
            JsonLike::Bool(_) | JsonLike::Int(_) | JsonLike::Float(_)
        ) {
            return Some((key, shape));
        }
    }
    None
}

fn decode_optional_feature_map(
    value: &JsonLike,
) -> Result<Option<IndexMap<String, PolicyFeature>>, DecodingError> {
    if let JsonLike::Array(items) = value {
        if let Some(pair) = items.iter().find(|pair| !is_pair(pair)) {
            let failure = unpack_failure(pair);
            let message = failure
                .split_once(": ")
                .map_or(failure.as_str(), |(_, message)| message);
            return Err(DecodingError::new(format!(
                "Could not decode the value into any of the given types:\n    dict: {message}\n"
            )));
        }
    }
    optional(value, |inner| decode_map(inner, "dict", decode_policy_feature)).map_err(|error| {
        let Some((key, shape)) = scalar_feature_shape(value) else {
            return error;
        };
        DecodingError::new(format!(
            "Could not decode the value into any of the given types:\n    dict: `{key}.shape`: Failed when parsing value='{}' into field \"<class 'lerobot.configs.types.PolicyFeature'>.shape\" of type tuple[int, ...].\n         \tUnderlying error is \"TypeError: '{}' object is not iterable\"\n",
            python_str(shape),
            shape.type_name()
        ))
    })
}

/// `decode_dataclass`'s closing check: whatever is left was not a field.
fn reject_extra_fields(fields: &JsonObject, what: &str) -> Result<(), DecodingError> {
    if fields.is_empty() {
        return Ok(());
    }
    let capacity = fields
        .keys()
        .try_fold(0usize, |total, key| total.checked_add(key.len() + 4))
        .ok_or_else(|| DecodingError::new("ACT extra-field diagnostic is too large"))?;
    let mut formatted = String::new();
    formatted
        .try_reserve_exact(capacity)
        .map_err(|_| DecodingError::new("ACT extra-field diagnostic allocation failed"))?;
    for (index, key) in fields.keys().enumerate() {
        if index != 0 {
            formatted.push_str(", ");
        }
        formatted.push('`');
        formatted.push_str(key);
        formatted.push('`');
    }
    Err(DecodingError::new(format!(
        "The fields {formatted} are not valid for {what}"
    )))
}

fn decode_act_config(document: &JsonLike) -> Result<ActConfig, DecodingError> {
    let JsonLike::Object(map) = document else {
        return Err(DecodingError::new(format!(
            "Expected a dict for a choice class, got {}",
            python_str(document)
        )));
    };
    let mut fields = try_clone_object(map)?;

    // `from_pretrained` resolves and strips the registry tag before Draccus
    // ever sees the document.
    match take(&mut fields, "type") {
        None => return Err(DecodingError::new("Missing 'type' field in config.json")),
        Some(JsonLike::Str(tag)) if tag == "act" => {}
        Some(other) => {
            return Err(DecodingError::new(format!(
                "Policy type '{}' (from config.json) is not registered for ACTConfig",
                python_str(&other)
            )))
        }
    }

    let defaults = ActConfig::default();
    let config = ActConfig {
        policy_type: ActType::Act,
        n_obs_steps: field(&mut fields, "n_obs_steps", defaults.n_obs_steps, decode_int)?,
        input_features: field(
            &mut fields,
            "input_features",
            defaults.input_features,
            decode_optional_feature_map,
        )?,
        output_features: field(
            &mut fields,
            "output_features",
            defaults.output_features,
            decode_optional_feature_map,
        )?,
        device: field(&mut fields, "device", defaults.device, |value| {
            optional(value, |inner| Ok(python_str(inner)))
        })?,
        use_amp: field(&mut fields, "use_amp", defaults.use_amp, decode_bool)?,
        use_peft: field(&mut fields, "use_peft", defaults.use_peft, decode_bool)?,
        push_to_hub: field(
            &mut fields,
            "push_to_hub",
            defaults.push_to_hub,
            decode_bool,
        )?,
        repo_id: field(&mut fields, "repo_id", defaults.repo_id, |value| {
            optional(value, |inner| Ok(python_str(inner)))
        })?,
        private: field(&mut fields, "private", defaults.private, |value| {
            optional(value, decode_bool)
        })?,
        tags: field(&mut fields, "tags", defaults.tags, |value| {
            optional(value, decode_str_list)
        })?,
        license: field(&mut fields, "license", defaults.license, |value| {
            optional(value, |inner| Ok(python_str(inner)))
        })?,
        pretrained_path: field(
            &mut fields,
            "pretrained_path",
            defaults.pretrained_path,
            |value| optional(value, decode_path),
        )?,
        pretrained_revision: field(
            &mut fields,
            "pretrained_revision",
            defaults.pretrained_revision,
            |value| optional(value, |inner| Ok(python_str(inner))),
        )?,
        chunk_size: field(&mut fields, "chunk_size", defaults.chunk_size, decode_int)?,
        n_action_steps: field(
            &mut fields,
            "n_action_steps",
            defaults.n_action_steps,
            decode_int,
        )?,
        normalization_mapping: field(
            &mut fields,
            "normalization_mapping",
            defaults.normalization_mapping,
            decode_normalization_mapping,
        )?,
        vision_backbone: field(
            &mut fields,
            "vision_backbone",
            defaults.vision_backbone,
            |value| Ok(python_str(value)),
        )?,
        pretrained_backbone_weights: field(
            &mut fields,
            "pretrained_backbone_weights",
            defaults.pretrained_backbone_weights,
            |value| optional(value, |inner| Ok(python_str(inner))),
        )?,
        replace_final_stride_with_dilation: field(
            &mut fields,
            "replace_final_stride_with_dilation",
            defaults.replace_final_stride_with_dilation,
            |value| decode_int(value).map(PythonIntBool::Int),
        )?,
        pre_norm: field(&mut fields, "pre_norm", defaults.pre_norm, decode_bool)?,
        dim_model: field(&mut fields, "dim_model", defaults.dim_model, decode_int)?,
        n_heads: field(&mut fields, "n_heads", defaults.n_heads, decode_int)?,
        dim_feedforward: field(
            &mut fields,
            "dim_feedforward",
            defaults.dim_feedforward,
            decode_int,
        )?,
        feedforward_activation: field(
            &mut fields,
            "feedforward_activation",
            defaults.feedforward_activation,
            |value| Ok(python_str(value)),
        )?,
        n_encoder_layers: field(
            &mut fields,
            "n_encoder_layers",
            defaults.n_encoder_layers,
            decode_int,
        )?,
        n_decoder_layers: field(
            &mut fields,
            "n_decoder_layers",
            defaults.n_decoder_layers,
            decode_int,
        )?,
        use_vae: field(&mut fields, "use_vae", defaults.use_vae, decode_bool)?,
        latent_dim: field(&mut fields, "latent_dim", defaults.latent_dim, decode_int)?,
        n_vae_encoder_layers: field(
            &mut fields,
            "n_vae_encoder_layers",
            defaults.n_vae_encoder_layers,
            decode_int,
        )?,
        temporal_ensemble_coeff: field(
            &mut fields,
            "temporal_ensemble_coeff",
            defaults.temporal_ensemble_coeff,
            |value| optional(value, decode_float),
        )?,
        dropout: field(&mut fields, "dropout", defaults.dropout, decode_float)?,
        kl_weight: field(&mut fields, "kl_weight", defaults.kl_weight, decode_float)?,
        optimizer_lr: field(
            &mut fields,
            "optimizer_lr",
            defaults.optimizer_lr,
            decode_float,
        )?,
        optimizer_weight_decay: field(
            &mut fields,
            "optimizer_weight_decay",
            defaults.optimizer_weight_decay,
            decode_float,
        )?,
        optimizer_lr_backbone: field(
            &mut fields,
            "optimizer_lr_backbone",
            defaults.optimizer_lr_backbone,
            decode_float,
        )?,
    };
    reject_extra_fields(&fields, "ACTConfig")?;
    Ok(config)
}

/// `decode_list(str)`: the element type is a plain `str`, so `None` inside the
/// list becomes `'None'` rather than short-circuiting.
fn decode_str_list(value: &JsonLike) -> Result<Vec<String>, DecodingError> {
    match value {
        JsonLike::Array(items) | JsonLike::Tuple(items) => {
            let mut decoded = Vec::new();
            decoded
                .try_reserve_exact(items.len())
                .map_err(|_| DecodingError::new("ACT string-list allocation failed"))?;
            for item in items {
                decoded.push(match item {
                    JsonLike::Str(text) => try_clone_string(text)?,
                    other => python_str(other),
                });
            }
            Ok(decoded)
        }
        other => Err(DecodingError::new(format!(
            "The given value='{}' is not of a valid input for a list type",
            python_str(other)
        ))),
    }
}

/// `decode_from_init(Path, ...)`: `Path(...)` takes a string and nothing else.
fn decode_path(value: &JsonLike) -> Result<String, DecodingError> {
    match value {
        JsonLike::Str(text) => Ok(pure_posix_path(text)),
        other => Err(DecodingError::new(format!(
            "Couldn't parse '{}' into a Path",
            python_str(other)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Draccus checkpoint encoding
// ---------------------------------------------------------------------------

fn encode_features(features: &Option<IndexMap<String, PolicyFeature>>) -> JsonLike {
    match features {
        None => JsonLike::Null,
        Some(map) => JsonLike::Object(
            map.iter()
                .map(|(key, feature)| {
                    let mut encoded = JsonObject::new();
                    // `encode_enum` writes `obj.name`, which is the member
                    // value for every one of upstream's str-enums.
                    encoded.insert(
                        "type".to_string(),
                        JsonLike::Str(feature.r#type.as_str().to_string()),
                    );
                    encoded.insert(
                        "shape".to_string(),
                        JsonLike::Array(
                            feature
                                .shape
                                .iter()
                                .map(|dim| JsonLike::Int(dim.clone()))
                                .collect(),
                        ),
                    );
                    (key.clone(), JsonLike::Object(encoded))
                })
                .collect(),
        ),
    }
}

fn encode_optional_str(value: &Option<String>) -> JsonLike {
    match value {
        None => JsonLike::Null,
        Some(text) => JsonLike::Str(text.clone()),
    }
}

fn encode_act_config(config: &ActConfig) -> JsonObject {
    let mut out = JsonObject::new();
    // `encode_choice` puts the registry tag first, ahead of the fields.
    out.insert("type".to_string(), JsonLike::Str("act".to_string()));
    out.insert(
        "n_obs_steps".to_string(),
        JsonLike::Int(config.n_obs_steps.clone()),
    );
    out.insert(
        "input_features".to_string(),
        encode_features(&config.input_features),
    );
    out.insert(
        "output_features".to_string(),
        encode_features(&config.output_features),
    );
    out.insert("device".to_string(), encode_optional_str(&config.device));
    out.insert("use_amp".to_string(), JsonLike::Bool(config.use_amp));
    out.insert("use_peft".to_string(), JsonLike::Bool(config.use_peft));
    out.insert(
        "push_to_hub".to_string(),
        JsonLike::Bool(config.push_to_hub),
    );
    out.insert("repo_id".to_string(), encode_optional_str(&config.repo_id));
    out.insert(
        "private".to_string(),
        config.private.map_or(JsonLike::Null, JsonLike::Bool),
    );
    out.insert(
        "tags".to_string(),
        match &config.tags {
            None => JsonLike::Null,
            Some(items) => {
                JsonLike::Array(items.iter().map(|tag| JsonLike::Str(tag.clone())).collect())
            }
        },
    );
    out.insert("license".to_string(), encode_optional_str(&config.license));
    out.insert(
        "pretrained_path".to_string(),
        encode_optional_str(&config.pretrained_path),
    );
    out.insert(
        "pretrained_revision".to_string(),
        encode_optional_str(&config.pretrained_revision),
    );
    out.insert(
        "chunk_size".to_string(),
        JsonLike::Int(config.chunk_size.clone()),
    );
    out.insert(
        "n_action_steps".to_string(),
        JsonLike::Int(config.n_action_steps.clone()),
    );
    out.insert(
        "normalization_mapping".to_string(),
        JsonLike::Object(
            config
                .normalization_mapping
                .iter()
                .map(|(key, mode)| (key.clone(), JsonLike::Str(mode.as_str().to_string())))
                .collect(),
        ),
    );
    out.insert(
        "vision_backbone".to_string(),
        JsonLike::Str(config.vision_backbone.clone()),
    );
    out.insert(
        "pretrained_backbone_weights".to_string(),
        encode_optional_str(&config.pretrained_backbone_weights),
    );
    out.insert(
        "replace_final_stride_with_dilation".to_string(),
        match &config.replace_final_stride_with_dilation {
            PythonIntBool::Bool(flag) => JsonLike::Bool(*flag),
            PythonIntBool::Int(integer) => JsonLike::Int(integer.clone()),
        },
    );
    out.insert("pre_norm".to_string(), JsonLike::Bool(config.pre_norm));
    out.insert(
        "dim_model".to_string(),
        JsonLike::Int(config.dim_model.clone()),
    );
    out.insert("n_heads".to_string(), JsonLike::Int(config.n_heads.clone()));
    out.insert(
        "dim_feedforward".to_string(),
        JsonLike::Int(config.dim_feedforward.clone()),
    );
    out.insert(
        "feedforward_activation".to_string(),
        JsonLike::Str(config.feedforward_activation.clone()),
    );
    out.insert(
        "n_encoder_layers".to_string(),
        JsonLike::Int(config.n_encoder_layers.clone()),
    );
    out.insert(
        "n_decoder_layers".to_string(),
        JsonLike::Int(config.n_decoder_layers.clone()),
    );
    out.insert("use_vae".to_string(), JsonLike::Bool(config.use_vae));
    out.insert(
        "latent_dim".to_string(),
        JsonLike::Int(config.latent_dim.clone()),
    );
    out.insert(
        "n_vae_encoder_layers".to_string(),
        JsonLike::Int(config.n_vae_encoder_layers.clone()),
    );
    out.insert(
        "temporal_ensemble_coeff".to_string(),
        config
            .temporal_ensemble_coeff
            .map_or(JsonLike::Null, JsonLike::Float),
    );
    out.insert("dropout".to_string(), JsonLike::Float(config.dropout));
    out.insert("kl_weight".to_string(), JsonLike::Float(config.kl_weight));
    out.insert(
        "optimizer_lr".to_string(),
        JsonLike::Float(config.optimizer_lr),
    );
    out.insert(
        "optimizer_weight_decay".to_string(),
        JsonLike::Float(config.optimizer_weight_decay),
    );
    out.insert(
        "optimizer_lr_backbone".to_string(),
        JsonLike::Float(config.optimizer_lr_backbone),
    );
    out
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

/// Decode one serde `RawValue` token through a Draccus conversion.
///
/// The token is re-read into [`JsonLike`], the value domain `json.load`
/// produces, so the serde path and the checkpoint path share one decoder and
/// cannot drift apart. Unbounded integers survive because `RawValue` hands over
/// the verbatim source text rather than a machine-width `serde_json::Number`.
mod wire {
    use crate::dataset::json::{loads, JsonLike};
    use crate::policy::draccus::DecodingError;
    use serde::de::{self, Deserializer};
    use serde::Deserialize;
    use serde_json::value::RawValue;

    pub(super) fn read<'de, D>(deserializer: D) -> Result<(JsonLike, String), D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw: Box<RawValue> = Box::<RawValue>::deserialize(deserializer)?;
        let text = raw.get().to_owned();
        let value = loads(&text).map_err(de::Error::custom)?;
        Ok((value, text))
    }

    pub(super) fn decode<'de, D, T>(
        deserializer: D,
        decoder: impl FnOnce(&JsonLike) -> Result<T, DecodingError>,
    ) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (value, _) = read(deserializer)?;
        decoder(&value).map_err(de::Error::custom)
    }
}

mod bigint_wire {
    use super::wire;
    use crate::policy::draccus::decode_int;
    use num_bigint::BigInt;
    use serde::de::Deserializer;
    use serde::ser::{Error as _, Serializer};
    use serde::Serialize;
    use serde_json::value::RawValue;

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
        wire::decode(deserializer, decode_int)
    }
}

mod f64_wire {
    use super::wire;
    use crate::dataset::json::python_float_repr;
    use crate::policy::draccus::decode_float;
    use serde::de::Deserializer;
    use serde::ser::{Error as _, Serializer};
    use serde::Serialize;
    use serde_json::value::RawValue;

    /// `json.dump` writes `float.__repr__`, which is not serde_json's shortest
    /// form: CPython writes `1e-05` where serde_json writes `0.00001`. Both
    /// round-trip to the same double, but only one of them is the checkpoint
    /// byte sequence upstream produces.
    pub fn serialize<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if !value.is_finite() {
            return Err(S::Error::custom(
                "non-finite Python float output is not valid JSON; use \
                 `ActConfig::to_checkpoint_json` for CPython's NaN/Infinity tokens",
            ));
        }
        let raw = RawValue::from_string(python_float_repr(*value)).map_err(S::Error::custom)?;
        raw.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        wire::decode(deserializer, decode_float)
    }
}

mod optional_f64_wire {
    use super::{f64_wire, wire};
    use crate::dataset::json::JsonLike;
    use crate::policy::draccus::decode_float;
    use serde::de::{self, Deserializer};
    use serde::ser::{Error as _, Serializer};

    pub fn serialize<S>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            None => serializer.serialize_none(),
            Some(number) => {
                if !number.is_finite() {
                    return Err(S::Error::custom(
                        "non-finite Python float output is not valid JSON; use \
                         `ActConfig::to_checkpoint_json` for CPython's NaN/Infinity tokens",
                    ));
                }
                serializer.serialize_some(&OptionalFloat(*number))
            }
        }
    }

    /// Newtype so the `Some` arm reaches [`f64_wire::serialize`] and therefore
    /// CPython's spelling, which `serialize_some(&f64)` would bypass.
    struct OptionalFloat(f64);

    impl serde::Serialize for OptionalFloat {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            f64_wire::serialize(&self.0, serializer)
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // `decode_union` short-circuits on `None` before trying `float(...)`.
        let (value, _) = wire::read(deserializer)?;
        match value {
            JsonLike::Null => Ok(None),
            other => decode_float(&other).map(Some).map_err(de::Error::custom),
        }
    }
}

mod bool_wire {
    use super::wire;
    use crate::policy::draccus::decode_bool;
    use serde::de::Deserializer;
    use serde::Serializer;

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
        wire::decode(deserializer, decode_bool)
    }
}

mod optional_bool_wire {
    use super::wire;
    use crate::dataset::json::JsonLike;
    use crate::policy::draccus::decode_bool;
    use serde::de::{self, Deserializer};
    use serde::Serializer;

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
        let (value, _) = wire::read(deserializer)?;
        match value {
            JsonLike::Null => Ok(None),
            other => decode_bool(&other).map(Some).map_err(de::Error::custom),
        }
    }
}

/// `decode_from_init(str, ...)`: a field annotated `str` keeps `str(raw_value)`
/// of whatever JSON value it was given.
mod str_wire {
    use super::wire;
    use crate::policy::draccus::python_str;
    use serde::de::Deserializer;
    use serde::Serializer;

    pub fn serialize<S>(value: &str, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(value)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<String, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (value, _) = wire::read(deserializer)?;
        Ok(python_str(&value))
    }
}

/// `str | None`: `decode_union` returns `None` for `null` without ever calling
/// `str()`, so `null` and the string `"None"` stay distinguishable.
mod optional_str_wire {
    use super::wire;
    use crate::dataset::json::JsonLike;
    use crate::policy::draccus::python_str;
    use serde::de::Deserializer;
    use serde::Serializer;

    pub fn serialize<S>(value: &Option<String>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(text) => serializer.serialize_some(text),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (value, _) = wire::read(deserializer)?;
        match value {
            JsonLike::Null => Ok(None),
            other => Ok(Some(python_str(&other))),
        }
    }
}

/// `list[str] | None`: the *list* short-circuits on `null`, but its element
/// type is a plain `str`, so a `null` element becomes the string `"None"`.
mod optional_str_list_wire {
    use super::wire;
    use crate::dataset::json::JsonLike;
    use crate::policy::draccus::python_str;
    use serde::de::{self, Deserializer};
    use serde::Serializer;

    pub fn serialize<S>(value: &Option<Vec<String>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(items) => serializer.serialize_some(items),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (value, _) = wire::read(deserializer)?;
        match value {
            JsonLike::Null => Ok(None),
            JsonLike::Array(items) => Ok(Some(items.iter().map(python_str).collect())),
            other => Err(de::Error::custom(format!(
                "The given value='{}' is not of a valid input for a list type",
                python_str(&other)
            ))),
        }
    }
}

/// `Path | None`: `Path(...)` accepts only a string, and normalises it.
mod optional_path_wire {
    use super::wire;
    use crate::dataset::json::JsonLike;
    use crate::policy::draccus::{pure_posix_path, python_str};
    use serde::de::{self, Deserializer};
    use serde::Serializer;

    pub fn serialize<S>(value: &Option<String>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(text) => serializer.serialize_some(text),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (value, _) = wire::read(deserializer)?;
        match value {
            JsonLike::Null => Ok(None),
            JsonLike::Str(text) => Ok(Some(pure_posix_path(&text))),
            other => Err(de::Error::custom(format!(
                "Couldn't parse '{}' into a Path",
                python_str(&other)
            ))),
        }
    }
}
