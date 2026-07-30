//! Port of `lerobot.configs.types` and `lerobot.types` (str-backed enums).
//!
//! Upstream declares these as `class X(str, Enum)`, so the wire value is the
//! member value — not the member name — and lookup by value is exact and
//! case-sensitive. Both properties are preserved here.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

pub use num_bigint::BigInt;

/// Error returned when a wire string does not name a known enum member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseEnumError {
    /// Name of the enum that rejected the value.
    pub enum_name: &'static str,
    /// The rejected input, verbatim.
    pub value: String,
}

impl fmt::Display for ParseEnumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Mirrors CPython's `ValueError: '<v>' is not a valid <Enum>`.
        write!(f, "'{}' is not a valid {}", self.value, self.enum_name)
    }
}

impl std::error::Error for ParseEnumError {}

/// Port of `lerobot.configs.types.FeatureType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FeatureType {
    /// `STATE`
    State,
    /// `VISUAL`
    Visual,
    /// `ENV`
    Env,
    /// `ACTION`
    Action,
    /// `REWARD`
    Reward,
    /// `LANGUAGE`
    Language,
}

/// Port of `lerobot.configs.types.PipelineFeatureType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PipelineFeatureType {
    /// `ACTION`
    Action,
    /// `OBSERVATION`
    Observation,
}

/// Port of `lerobot.configs.types.NormalizationMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NormalizationMode {
    /// `MIN_MAX`
    MinMax,
    /// `MEAN_STD`
    MeanStd,
    /// `IDENTITY`
    Identity,
    /// `QUANTILES`
    Quantiles,
    /// `QUANTILE10`
    Quantile10,
}

/// Port of `lerobot.configs.types.RTCAttentionSchedule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RtcAttentionSchedule {
    /// `ZEROS`
    Zeros,
    /// `ONES`
    Ones,
    /// `LINEAR`
    Linear,
    /// `EXP`
    Exp,
}

/// Port of `lerobot.types.TransitionKey`. Note the lowercase wire values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionKey {
    /// `observation`
    Observation,
    /// `action`
    Action,
    /// `reward`
    Reward,
    /// `done`
    Done,
    /// `truncated`
    Truncated,
    /// `info`
    Info,
    /// `complementary_data`
    ComplementaryData,
}

/// Implements `as_str` / `all` / `Display` / `FromStr` from a member table
/// written in upstream declaration order.
macro_rules! str_enum {
    ($t:ident, $( $variant:ident => $wire:literal ),+ $(,)?) => {
        impl $t {
            /// Wire value, identical to the upstream `str` enum value.
            pub fn as_str(&self) -> &'static str {
                match self {
                    $( $t::$variant => $wire, )+
                }
            }

            /// All members, in upstream declaration order.
            pub fn all() -> &'static [Self] {
                &[ $( $t::$variant, )+ ]
            }
        }

        impl fmt::Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl FromStr for $t {
            type Err = ParseEnumError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $( $wire => Ok($t::$variant), )+
                    other => Err(ParseEnumError {
                        enum_name: stringify!($t),
                        value: other.to_string(),
                    }),
                }
            }
        }
    };
}

str_enum!(
    FeatureType,
    State => "STATE",
    Visual => "VISUAL",
    Env => "ENV",
    Action => "ACTION",
    Reward => "REWARD",
    Language => "LANGUAGE",
);

str_enum!(
    PipelineFeatureType,
    Action => "ACTION",
    Observation => "OBSERVATION",
);

str_enum!(
    NormalizationMode,
    MinMax => "MIN_MAX",
    MeanStd => "MEAN_STD",
    Identity => "IDENTITY",
    Quantiles => "QUANTILES",
    Quantile10 => "QUANTILE10",
);

str_enum!(
    RtcAttentionSchedule,
    Zeros => "ZEROS",
    Ones => "ONES",
    Linear => "LINEAR",
    Exp => "EXP",
);

str_enum!(
    TransitionKey,
    Observation => "observation",
    Action => "action",
    Reward => "reward",
    Done => "done",
    Truncated => "truncated",
    Info => "info",
    ComplementaryData => "complementary_data",
);

/// Port of `lerobot.configs.types.PolicyFeature`.
///
/// Upstream is `@dataclass class PolicyFeature: type: FeatureType; shape:
/// tuple[int, ...]` — two fields and no methods. This struct is deliberately
/// the same shape. It carried a `numel()` convenience for one revision; that
/// was removed because upstream has no such method, so the port could not
/// point at a behaviour to be compatible *with*, and its product overflowed.
/// Callers that want a product know their own shapes and can choose their own
/// overflow policy.
///
/// `shape` is a vector of [`BigInt`] rather than of `usize`, because a Python
/// `int` is signed and unbounded and both properties are used: `-1` is the
/// ordinary spelling of a dynamic axis, and nothing upstream clamps a
/// dimension to a machine word. The JSON wire form is the same bare decimal
/// integer `json.dumps` writes whenever CPython permits the conversion. CPython
/// 3.12 limits decimal integer conversion to 4,300 digits by default (unless
/// configured otherwise); serde accepts a superset and does not impose that
/// interpreter-wide denial-of-service guard.
///
/// ```
/// use rerobot_core::types::{BigInt, FeatureType, PolicyFeature};
///
/// let dynamic = PolicyFeature::new(FeatureType::State, [-1, 7]);
/// assert_eq!(dynamic.shape, vec![BigInt::from(-1), BigInt::from(7)]);
/// assert_eq!(
///     serde_json::to_string(&dynamic).unwrap(),
///     r#"{"type":"STATE","shape":[-1,7]}"#
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyFeature {
    /// Feature category.
    pub r#type: FeatureType,
    /// Shape, mirroring the upstream `tuple[int, ...]`.
    #[serde(with = "shape_wire")]
    pub shape: Vec<BigInt>,
}

impl PolicyFeature {
    /// Construct a feature from anything that yields dimensions.
    ///
    /// Every Rust integer converts into a [`BigInt`], so the ordinary call
    /// reads as it did before: `PolicyFeature::new(FeatureType::Visual, [3,
    /// 96, 96])`.
    pub fn new<I>(feature_type: FeatureType, shape: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<BigInt>,
    {
        Self {
            r#type: feature_type,
            shape: shape.into_iter().map(Into::into).collect(),
        }
    }
}

/// JSON wire form for `PolicyFeature::shape`: a sequence of bare decimal
/// integers of unbounded length.
///
/// `num-bigint`'s own `serde` impl writes a `(sign, [u32 limbs])` pair, which
/// is a stable format for round-tripping between Rust programs but is not the
/// integer upstream's `json.dumps` emits. The wire shape is therefore written
/// here rather than derived.
///
/// Both directions go through `serde_json::value::RawValue`, which carries the
/// verbatim JSON text of one value. That is what makes the exactness
/// independent of any fixed width: serialisation hands the serialiser the full
/// decimal expansion, and deserialisation parses the untouched source token
/// instead of letting the JSON parser round it into an `f64` first, which is
/// what an integer past `u64` would otherwise become.
mod shape_wire {
    use super::BigInt;
    use serde::de::{self, Deserialize, Deserializer};
    use serde::ser::{Error as _, SerializeSeq, Serializer};
    use serde_json::value::RawValue;

    pub(super) fn serialize<S>(shape: &[BigInt], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(shape.len()))?;
        for dim in shape {
            // `to_string` on a `BigInt` is a decimal integer, which is always
            // valid JSON, so the only way this fails is an allocator failure.
            let raw = RawValue::from_string(dim.to_string()).map_err(S::Error::custom)?;
            seq.serialize_element(&raw)?;
        }
        seq.end()
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<BigInt>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw: Vec<Box<RawValue>> = Vec::deserialize(deserializer)?;
        raw.iter()
            .map(|value| {
                let text = value.get();
                // The only decoder upstream ever applies to a `PolicyFeature`
                // is Draccus', and `tuple[int, ...]` decodes each item with
                // `decode_int`: a `float` is refused outright, and everything
                // else goes through Python's `int()`. So a bool and a string
                // that `int()` parses are both accepted, while `1.0`, `null`
                // and a nested array are not.
                let parsed = crate::dataset::json::loads(text)
                    .ok()
                    .and_then(|value| crate::policy::draccus::decode_int(&value).ok());
                parsed.ok_or_else(|| {
                    de::Error::custom(format!(
                        "invalid shape dimension `{text}`: expected a value Python's \
                         `int()` accepts, mirroring the upstream `tuple[int, ...]`"
                    ))
                })
            })
            .collect()
    }
}
