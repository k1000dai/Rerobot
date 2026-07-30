//! Behaviour parity tests for the str-backed enums ported from
//! `lerobot.configs.types` and `lerobot.types`.

use rerobot_core::types::{
    BigInt, FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature,
    RtcAttentionSchedule, TransitionKey,
};
use std::str::FromStr;

/// A shape from a list of `i64` literals, for the ordinary small cases.
fn shape(dims: [i64; 3]) -> Vec<BigInt> {
    dims.iter().map(|d| BigInt::from(*d)).collect()
}

#[test]
fn feature_type_wire_values_match_upstream() {
    assert_eq!(FeatureType::State.as_str(), "STATE");
    assert_eq!(FeatureType::Visual.as_str(), "VISUAL");
    assert_eq!(FeatureType::Env.as_str(), "ENV");
    assert_eq!(FeatureType::Action.as_str(), "ACTION");
    assert_eq!(FeatureType::Reward.as_str(), "REWARD");
    assert_eq!(FeatureType::Language.as_str(), "LANGUAGE");
}

#[test]
fn feature_type_declaration_order_matches_upstream() {
    let names: Vec<&str> = FeatureType::all().iter().map(|v| v.as_str()).collect();
    assert_eq!(
        names,
        vec!["STATE", "VISUAL", "ENV", "ACTION", "REWARD", "LANGUAGE"]
    );
}

#[test]
fn normalization_mode_wire_values_match_upstream() {
    let names: Vec<&str> = NormalizationMode::all()
        .iter()
        .map(|v| v.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["MIN_MAX", "MEAN_STD", "IDENTITY", "QUANTILES", "QUANTILE10"]
    );
}

#[test]
fn pipeline_feature_type_wire_values_match_upstream() {
    let names: Vec<&str> = PipelineFeatureType::all()
        .iter()
        .map(|v| v.as_str())
        .collect();
    assert_eq!(names, vec!["ACTION", "OBSERVATION"]);
}

#[test]
fn rtc_attention_schedule_wire_values_match_upstream() {
    let names: Vec<&str> = RtcAttentionSchedule::all()
        .iter()
        .map(|v| v.as_str())
        .collect();
    assert_eq!(names, vec!["ZEROS", "ONES", "LINEAR", "EXP"]);
}

#[test]
fn transition_key_wire_values_are_lowercase() {
    let names: Vec<&str> = TransitionKey::all().iter().map(|v| v.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "observation",
            "action",
            "reward",
            "done",
            "truncated",
            "info",
            "complementary_data"
        ]
    );
}

#[test]
fn display_matches_as_str() {
    assert_eq!(FeatureType::Visual.to_string(), "VISUAL");
    assert_eq!(
        TransitionKey::ComplementaryData.to_string(),
        "complementary_data"
    );
}

#[test]
fn from_str_round_trips_every_member() {
    for v in FeatureType::all() {
        assert_eq!(FeatureType::from_str(v.as_str()).unwrap(), *v);
    }
    for v in NormalizationMode::all() {
        assert_eq!(NormalizationMode::from_str(v.as_str()).unwrap(), *v);
    }
    for v in TransitionKey::all() {
        assert_eq!(TransitionKey::from_str(v.as_str()).unwrap(), *v);
    }
}

#[test]
fn from_str_is_case_sensitive_like_python_str_enums() {
    assert!(FeatureType::from_str("state").is_err());
    assert!(FeatureType::from_str("State").is_err());
    assert!(TransitionKey::from_str("OBSERVATION").is_err());
}

#[test]
fn from_str_rejects_unknown_and_empty_values() {
    let err = FeatureType::from_str("").unwrap_err();
    assert_eq!(err.enum_name, "FeatureType");
    assert_eq!(err.value, "");
    let err = NormalizationMode::from_str("MINMAX").unwrap_err();
    assert_eq!(err.to_string(), "'MINMAX' is not a valid NormalizationMode");
}

#[test]
fn from_str_rejects_the_rust_variant_spelling() {
    assert!(NormalizationMode::from_str("MinMax").is_err());
    assert!(TransitionKey::from_str("ComplementaryData").is_err());
}

#[test]
fn enums_serialize_to_their_wire_values() {
    assert_eq!(serde_json::to_string(&FeatureType::Env).unwrap(), "\"ENV\"");
    assert_eq!(
        serde_json::to_string(&NormalizationMode::Quantile10).unwrap(),
        "\"QUANTILE10\""
    );
    assert_eq!(
        serde_json::to_string(&TransitionKey::ComplementaryData).unwrap(),
        "\"complementary_data\""
    );
}

#[test]
fn enums_deserialize_from_their_wire_values() {
    let v: FeatureType = serde_json::from_str("\"LANGUAGE\"").unwrap();
    assert_eq!(v, FeatureType::Language);
    let v: PipelineFeatureType = serde_json::from_str("\"OBSERVATION\"").unwrap();
    assert_eq!(v, PipelineFeatureType::Observation);
    assert!(serde_json::from_str::<FeatureType>("\"language\"").is_err());
}

#[test]
fn policy_feature_serializes_with_upstream_field_names() {
    let f = PolicyFeature::new(FeatureType::Visual, [3, 96, 96]);
    assert_eq!(
        serde_json::to_string(&f).unwrap(),
        r#"{"type":"VISUAL","shape":[3,96,96]}"#
    );
    let back: PolicyFeature =
        serde_json::from_str(r#"{"type":"VISUAL","shape":[3,96,96]}"#).unwrap();
    assert_eq!(back, f);
}

#[test]
fn policy_feature_has_exactly_the_two_upstream_fields() {
    // Upstream is `@dataclass class PolicyFeature: type; shape`. The port used
    // to add a `numel()` convenience; it is gone, because there is no upstream
    // behaviour for it to match and its product overflowed. This test pins the
    // whole observable surface: construct, read both fields.
    let f = PolicyFeature::new(FeatureType::Visual, [3, 96, 96]);
    assert_eq!(f.r#type, FeatureType::Visual);
    assert_eq!(f.shape, shape([3, 96, 96]));
    let scalar = PolicyFeature::new(FeatureType::Reward, Vec::<BigInt>::new());
    assert!(scalar.shape.is_empty());
}

// --- the `tuple[int, ...]` domain ---------------------------------------
//
// Upstream's annotation is `tuple[int, ...]`, and a Python `int` is signed and
// unbounded. `-1` is the ordinary placeholder for a dynamic axis, and nothing
// upstream clamps a dimension to a machine word. The wire form is whatever
// `json.dumps` writes for that `int`: a bare decimal integer, of any length,
// identical on every platform.

#[test]
fn policy_feature_round_trips_a_negative_dimension_exactly() {
    // `json.dumps({"type": "STATE", "shape": [-1, 0, 1]})` upstream.
    let json = r#"{"type":"STATE","shape":[-1,0,1]}"#;
    let f: PolicyFeature = serde_json::from_str(json).unwrap();
    assert_eq!(f.shape, shape([-1, 0, 1]));
    assert_eq!(serde_json::to_string(&f).unwrap(), json);
    assert_eq!(f, PolicyFeature::new(FeatureType::State, [-1, 0, 1]));
}

#[test]
fn policy_feature_round_trips_dimensions_far_above_usize_max_exactly() {
    // 2**128 + 1 and its negation: past every fixed Rust integer width, and
    // written as literals rather than as `usize::MAX`-derived values so the
    // wire form is the same text on a 16-, 32-, 64- or 128-bit target.
    const HUGE: &str = "340282366920938463463374607431768211457";
    let json = format!(r#"{{"type":"STATE","shape":[{HUGE},-{HUGE}]}}"#);

    let f: PolicyFeature = serde_json::from_str(&json).unwrap();
    let big = BigInt::from_str(HUGE).unwrap();
    assert_eq!(f.shape, vec![big.clone(), -big.clone()]);
    assert_eq!(serde_json::to_string(&f).unwrap(), json);

    // And the value really is the arbitrary-precision one, not a float that
    // happens to print back: 2**128 + 1 differs from 2**128 by exactly one.
    assert_eq!(
        &f.shape[0] - BigInt::from(1),
        BigInt::from_str("340282366920938463463374607431768211456").unwrap()
    );
}

#[test]
fn policy_feature_shape_survives_a_thousand_digit_dimension() {
    // Python has no upper bound here, so neither does the port. `10**999`.
    let mut digits = String::from("1");
    digits.push_str(&"0".repeat(999));
    let json = format!(r#"{{"type":"ENV","shape":[{digits}]}}"#);

    let f: PolicyFeature = serde_json::from_str(&json).unwrap();
    assert_eq!(f.shape, vec![BigInt::from_str(&digits).unwrap()]);
    assert_eq!(serde_json::to_string(&f).unwrap(), json);
}

#[test]
fn policy_feature_rejects_malformed_json() {
    assert!(serde_json::from_str::<PolicyFeature>(r#"{"type":"VISUAL"}"#).is_err());
    assert!(serde_json::from_str::<PolicyFeature>(r#"{"type":"NOPE","shape":[1]}"#).is_err());
}

#[test]
fn policy_feature_rejects_unknown_fields_like_decode_dataclass() {
    // `draccus.parsers.decoding.decode_dataclass` collects the leftover keys
    // and raises `The fields `bogus` are not valid for PolicyFeature`. The
    // only JSON decoder upstream ever applies to a `PolicyFeature` is that
    // one, so an extra key is an error and not a silently dropped field.
    assert!(
        serde_json::from_str::<PolicyFeature>(r#"{"type":"STATE","shape":[7],"bogus":1}"#).is_err()
    );
    assert!(serde_json::from_str::<PolicyFeature>(r#"{"type":"STATE","shape":[7]}"#).is_ok());
}

#[test]
fn policy_feature_shape_follows_draccus_decode_int() {
    // `shape: tuple[int, ...]` decodes each item with `decode_int`, which
    // explicitly rejects a `float` and otherwise calls `int(raw_value)`. A
    // Python `bool` is an `int` subclass, and `int(str)` accepts surrounding
    // whitespace, a sign, PEP 515 underscores and Unicode decimal digits.
    for (good, expected) in [
        (r#"{"type":"STATE","shape":["1"]}"#, vec![1i64]),
        (r#"{"type":"STATE","shape":[true,false]}"#, vec![1, 0]),
        (r#"{"type":"STATE","shape":[" -1_0 "]}"#, vec![-10]),
        (r#"{"type":"STATE","shape":["١٢٣"]}"#, vec![123]),
    ] {
        let f: PolicyFeature = serde_json::from_str(good).unwrap();
        let expected: Vec<BigInt> = expected.into_iter().map(BigInt::from).collect();
        assert_eq!(f.shape, expected, "{good}");
    }

    // A Python `int` is not a `float`, and `int()` refuses the rest.
    for bad in [
        r#"{"type":"STATE","shape":[1.5]}"#,
        r#"{"type":"STATE","shape":[1.0]}"#,
        r#"{"type":"STATE","shape":[1e3]}"#,
        r#"{"type":"STATE","shape":[null]}"#,
        r#"{"type":"STATE","shape":[[1]]}"#,
        r#"{"type":"STATE","shape":["0x10"]}"#,
        r#"{"type":"STATE","shape":[""]}"#,
    ] {
        assert!(
            serde_json::from_str::<PolicyFeature>(bad).is_err(),
            "{bad} must not parse as a shape"
        );
    }
}
