// Copyright 2026 The Rerobot contributors
// SPDX-License-Identifier: Apache-2.0
//! Differential contract vectors for the Draccus checkpoint boundary of
//! upstream `ACTConfig`.
//!
//! Every expectation here was captured from CPython 3.12.13 running upstream
//! `f37be3edbee60f3a09a5183788b91eb19f0c07d1` with Draccus 0.10.0, through
//! `draccus.parse(ACTConfig, path)` and `draccus.dump(config, stream,
//! indent=4)` under `draccus.config_type("json")` — the two calls
//! `PreTrainedConfig.from_pretrained` and `_save_pretrained` make.

use num_bigint::BigInt;
use rerobot_core::policy::act::{ActConfig, ActConfigErrorKind};
use rerobot_core::policy::draccus::{pure_posix_path, python_int_from_str};
use rerobot_core::types::NormalizationMode;

fn bi(value: i64) -> BigInt {
    BigInt::from(value)
}

/// A checkpoint document with `patch` splicing over the defaults.
fn doc(patch: &str) -> String {
    format!(r#"{{"type":"act","device":"cpu"{patch}}}"#)
}

// ---------------------------------------------------------------------------
// (1) `normalization_mapping` is `dict[str, NormalizationMode]` upstream.
// ---------------------------------------------------------------------------

#[test]
fn normalization_mapping_accepts_any_string_key_like_upstream() {
    // Python: {'VISUAL': <MEAN_STD>, 'BOGUS': <MIN_MAX>} — the annotation is
    // `dict[str, NormalizationMode]`, so Draccus decodes the key with `str()`
    // and never checks it against `FeatureType`.
    let config = ActConfig::from_checkpoint_json(&doc(
        r#","normalization_mapping":{"VISUAL":"MEAN_STD","BOGUS":"MIN_MAX"}"#,
    ))
    .unwrap();
    assert_eq!(
        config.normalization_mapping.get("BOGUS"),
        Some(&NormalizationMode::MinMax)
    );
    assert_eq!(
        config.normalization_mapping.keys().collect::<Vec<_>>(),
        vec!["VISUAL", "BOGUS"]
    );

    // Case-sensitivity belongs to the value, not the key.
    let lower =
        ActConfig::from_checkpoint_json(&doc(r#","normalization_mapping":{"visual":"MEAN_STD"}"#))
            .unwrap();
    assert_eq!(
        lower.normalization_mapping.get("visual"),
        Some(&NormalizationMode::MeanStd)
    );

    // The value is still the exact upstream str-enum domain.
    assert!(
        ActConfig::from_checkpoint_json(&doc(r#","normalization_mapping":{"VISUAL":"nope"}"#))
            .is_err()
    );

    // Arbitrary keys survive the round trip, in insertion order.
    assert!(config.to_checkpoint_json().contains(
        "\"normalization_mapping\": {\n        \"VISUAL\": \"MEAN_STD\",\n        \"BOGUS\": \"MIN_MAX\"\n    }"
    ));

    // The same domain through plain serde, so the two decoders agree.
    let via_serde: ActConfig = serde_json::from_str(&doc(
        r#","normalization_mapping":{"VISUAL":"MEAN_STD","BOGUS":"MIN_MAX"}"#,
    ))
    .unwrap();
    assert_eq!(
        via_serde.normalization_mapping,
        config.normalization_mapping
    );
}

// ---------------------------------------------------------------------------
// (2) Draccus decodes every `str` field with `str(raw_value)`.
// ---------------------------------------------------------------------------

#[test]
fn string_fields_reproduce_draccus_str_coercion_with_python_spellings() {
    let config = ActConfig::from_checkpoint_json(&doc(
        r#","repo_id":5,"license":true,"pretrained_backbone_weights":123,"feedforward_activation":7,"pretrained_revision":1.0"#,
    ))
    .unwrap();
    assert_eq!(config.repo_id.as_deref(), Some("5"));
    assert_eq!(config.license.as_deref(), Some("True"));
    assert_eq!(config.pretrained_backbone_weights.as_deref(), Some("123"));
    assert_eq!(config.feedforward_activation, "7");
    assert_eq!(config.pretrained_revision.as_deref(), Some("1.0"));

    // `str | None` short-circuits on null before `str()` is ever called.
    let nulled = ActConfig::from_checkpoint_json(&doc(r#","license":null"#)).unwrap();
    assert_eq!(nulled.license, None);

    // `list[str]` has no such short-circuit: the element type is plain `str`,
    // so Python's `str(None)` applies.
    let tags = ActConfig::from_checkpoint_json(&doc(
        r#","tags":[1,true,null,1.5,1e21,[1,"a"],{"k":"v"}]"#,
    ))
    .unwrap();
    assert_eq!(
        tags.tags.unwrap(),
        vec![
            "1",
            "True",
            "None",
            "1.5",
            "1e+21",
            "[1, 'a']",
            "{'k': 'v'}"
        ]
    );

    // A non-string `vision_backbone` is stringified and then fails validation,
    // exactly as upstream's `__post_init__` does — not as a decode type error.
    let error = ActConfig::from_checkpoint_json(&doc(r#","vision_backbone":123"#)).unwrap_err();
    assert_eq!(error.kind(), ActConfigErrorKind::Value);
    assert_eq!(
        error.to_string(),
        "`vision_backbone` must be one of the ResNet variants. Got 123."
    );

    // Plain serde agrees with the checkpoint decoder.
    let via_serde: ActConfig =
        serde_json::from_str(&doc(r#","repo_id":5,"license":true"#)).unwrap();
    assert_eq!(via_serde.repo_id.as_deref(), Some("5"));
    assert_eq!(via_serde.license.as_deref(), Some("True"));
}

// ---------------------------------------------------------------------------
// (3) CPython's float wire spelling.
// ---------------------------------------------------------------------------

#[test]
fn float_fields_are_written_with_cpython_repr_spelling() {
    let mut config = ActConfig::default();
    config.device = Some("cpu".into());

    // `json.dump` writes `float.__repr__`. serde_json's shortest form is not
    // it: it prints `0.00001` where CPython prints `1e-05`.
    let checkpoint = config.to_checkpoint_json();
    assert!(checkpoint.contains("\"optimizer_lr\": 1e-05,"));
    assert!(checkpoint.contains("\"optimizer_lr_backbone\": 1e-05"));
    assert!(checkpoint.contains("\"optimizer_weight_decay\": 0.0001,"));
    assert!(checkpoint.contains("\"kl_weight\": 10.0,"));
    assert!(checkpoint.contains("\"dropout\": 0.1,"));

    // The serde writer must agree, or the two produce different checkpoints.
    let compact = serde_json::to_string(&config).unwrap();
    assert!(compact.contains(r#""optimizer_lr":1e-05"#));
    assert!(compact.contains(r#""optimizer_weight_decay":0.0001"#));

    config.temporal_ensemble_coeff = Some(1.5e-7);
    assert!(config
        .to_checkpoint_json()
        .contains("\"temporal_ensemble_coeff\": 1.5e-07,"));
    assert!(serde_json::to_string(&config)
        .unwrap()
        .contains(r#""temporal_ensemble_coeff":1.5e-07"#));
}

#[test]
fn default_checkpoint_json_is_byte_identical_to_upstream_draccus_dump() {
    // Captured verbatim from `draccus.dump(ACTConfig(device="cpu"), stream,
    // indent=4)` under `draccus.config_type("json")`: 1098 bytes, four-space
    // indent, `ensure_ascii=True`, and no trailing newline.
    let expected = include_str!("data/act_default_config.json");
    let mut config = ActConfig::default();
    config.device = Some("cpu".into());
    assert_eq!(config.to_checkpoint_json(), expected);
    assert_eq!(config.to_checkpoint_json().len(), 1098);
}

#[test]
fn checkpoint_writer_escapes_non_ascii_like_json_dump_ensure_ascii() {
    use rerobot_core::types::{FeatureType, PolicyFeature};
    let mut config = ActConfig::default();
    config.device = Some("cpu".into());
    config.input_features = Some(indexmap::IndexMap::from([(
        "observation.imagé.😀".to_string(),
        PolicyFeature::new(FeatureType::Visual, [3]),
    )]));
    // `json.dump` defaults to `ensure_ascii=True`, and writes astral planes as
    // a UTF-16 surrogate pair. Captured from upstream's own dump.
    assert!(config
        .to_checkpoint_json()
        .contains(r#""observation.imag\u00e9.\ud83d\ude00": {"#));
    assert!(!config.to_checkpoint_json().contains('é'));
}

// ---------------------------------------------------------------------------
// (4) Python's `int()` / `float()` string domains.
// ---------------------------------------------------------------------------

#[test]
fn integer_strings_follow_python_int_underscore_and_unicode_digit_domain() {
    // `n_action_steps` is pinned to 1 so `__post_init__` — which Draccus reaches
    // by constructing the dataclass — accepts every chunk size below.
    for (patch, expected) in [
        (r#","n_action_steps":1,"chunk_size":"1_000""#, 1000),
        (r#","n_action_steps":1,"chunk_size":"١٢٣""#, 123),
        (r#","n_action_steps":1,"chunk_size":"１００""#, 100),
        (r#","n_action_steps":1,"chunk_size":"１_０""#, 10),
        (r#","n_action_steps":1,"chunk_size":"𝟘𝟙""#, 1),
        (r#","n_action_steps":1,"chunk_size":"+1_0""#, 10),
        (r#","n_action_steps":1,"chunk_size":"  12  ""#, 12),
        (r#","n_action_steps":1,"chunk_size":"\t7\r\n""#, 7),
    ] {
        let config = ActConfig::from_checkpoint_json(&doc(patch)).unwrap();
        assert_eq!(config.chunk_size, bi(expected), "{patch}");
        let via_serde: ActConfig = serde_json::from_str(&doc(patch)).unwrap();
        assert_eq!(via_serde.chunk_size, bi(expected), "{patch} via serde");
    }

    // PEP 515: an underscore must sit between two digits.
    for patch in [
        r#","chunk_size":"_1""#,
        r#","chunk_size":"1_""#,
        r#","chunk_size":"1__0""#,
        r#","chunk_size":"0x10""#,
        r#","chunk_size":"""#,
        r#","chunk_size":"1 2""#,
    ] {
        assert!(
            ActConfig::from_checkpoint_json(&doc(patch)).is_err(),
            "{patch} must not decode"
        );
    }
}

#[test]
fn float_strings_follow_python_float_underscore_and_unicode_digit_domain() {
    for (patch, expected) in [
        (r#","dropout":"1_000.5""#, 1000.5),
        (r#","dropout":"1_0.0_1e1""#, 100.1),
        (r#","dropout":"1e1_0""#, 1e10),
        (r#","dropout":"１.５""#, 1.5),
        (r#","dropout":"١.5""#, 1.5),
        (r#","dropout":"𝟙.𝟝""#, 1.5),
        (r#","dropout":"  .5  ""#, 0.5),
        (r#","dropout":"\t.5\r\n""#, 0.5),
        (r#","dropout":"5.""#, 5.0),
    ] {
        let config = ActConfig::from_checkpoint_json(&doc(patch)).unwrap();
        assert_eq!(config.dropout, expected, "{patch}");
        let via_serde: ActConfig = serde_json::from_str(&doc(patch)).unwrap();
        assert_eq!(via_serde.dropout, expected, "{patch} via serde");
    }

    for patch in [
        r#","dropout":"1._5""#,
        r#","dropout":"._5""#,
        r#","dropout":"1.5_""#,
        r#","dropout":"nan_""#,
        r#","dropout":"0x10""#,
    ] {
        assert!(
            ActConfig::from_checkpoint_json(&doc(patch)).is_err(),
            "{patch} must not decode"
        );
    }
}

#[test]
fn numeric_strings_accept_cpythons_ascii_edge_whitespace() {
    let integer =
        ActConfig::from_checkpoint_json(&doc(r#","n_action_steps":1,"chunk_size":"\t7\r\n""#))
            .unwrap();
    assert_eq!(integer.chunk_size, bi(7));

    let float = ActConfig::from_checkpoint_json(&doc(r#","dropout":"\t.5\r\n""#)).unwrap();
    assert_eq!(float.dropout, 0.5);
}

#[test]
fn every_unicode_15_decimal_digit_has_cpythons_value() {
    // Zero code points from UnicodeData.txt 15.0.0. Together these 68
    // ten-code-point runs cover every character whose general category is Nd.
    let zeroes = [
        0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
        0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
        0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0,
        0xFF10, 0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0,
        0x11650, 0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60,
        0x16AC0, 0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0,
        0x1E950, 0x1FBF0,
    ];
    for zero in zeroes {
        for value in 0..10 {
            let digit = char::from_u32(zero + value).unwrap().to_string();
            assert_eq!(
                python_int_from_str(&digit),
                Some(BigInt::from(value)),
                "U+{:04X}",
                zero + value
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (5) Draccus mapping and iterable tuple coercion.
// ---------------------------------------------------------------------------

#[test]
fn mapping_fields_accept_json_pair_sequences_like_python_dict() {
    let config = ActConfig::from_checkpoint_json(&doc(
        r#","normalization_mapping":[["BOGUS","MIN_MAX"]],"input_features":[],"output_features":[]"#,
    ))
    .unwrap();
    assert_eq!(
        config.normalization_mapping.get("BOGUS"),
        Some(&NormalizationMode::MinMax)
    );
    assert!(config.input_features.as_ref().unwrap().is_empty());
    assert!(config.output_features.as_ref().unwrap().is_empty());
}

#[test]
fn feature_shape_iterates_strings_and_mapping_keys_like_draccus() {
    let from_string = ActConfig::from_checkpoint_json(&doc(
        r#","input_features":{"k":{"type":"STATE","shape":"𝟘𝟙"}}"#,
    ))
    .unwrap();
    assert_eq!(
        from_string.input_features.as_ref().unwrap()["k"].shape,
        vec![bi(0), bi(1)]
    );

    let from_keys = ActConfig::from_checkpoint_json(&doc(
        r#","input_features":{"k":{"type":"STATE","shape":{"7":1}}}"#,
    ))
    .unwrap();
    assert_eq!(
        from_keys.input_features.as_ref().unwrap()["k"].shape,
        vec![bi(7)]
    );
}

#[test]
fn malformed_iterables_keep_draccus_field_level_error_shapes() {
    let wrong_length =
        ActConfig::from_checkpoint_json(r#"{"type":"act","normalization_mapping":[["VISUAL"]]}"#)
            .unwrap_err();
    assert_eq!(
        wrong_length.to_string(),
        "`normalization_mapping`: Failed when parsing value='[['VISUAL']]' into field \"<class 'lerobot.policies.act.configuration_act.ACTConfig'>.normalization_mapping\" of type dict[str, lerobot.configs.types.NormalizationMode].\n\tUnderlying error is \"ValueError: not enough values to unpack (expected 2, got 1)\""
    );

    let mapping_pair = ActConfig::from_checkpoint_json(
        r#"{"type":"act","normalization_mapping":[{"VISUAL":"MEAN_STD","STATE":"MEAN_STD"}]}"#,
    )
    .unwrap_err();
    assert_eq!(
        mapping_pair.to_string(),
        "`normalization_mapping`: Failed when parsing value='[{'VISUAL': 'MEAN_STD', 'STATE': 'MEAN_STD'}]' into field \"<class 'lerobot.policies.act.configuration_act.ACTConfig'>.normalization_mapping\" of type dict[str, lerobot.configs.types.NormalizationMode].\n\tUnderlying error is \"KeyError: 'STATE'\""
    );

    let scalar_mapping =
        ActConfig::from_checkpoint_json(r#"{"type":"act","normalization_mapping":1}"#).unwrap_err();
    assert_eq!(
        scalar_mapping.to_string(),
        "`normalization_mapping`: Failed when parsing value='1' into field \"<class 'lerobot.policies.act.configuration_act.ACTConfig'>.normalization_mapping\" of type dict[str, lerobot.configs.types.NormalizationMode].\n\tUnderlying error is \"AttributeError: 'int' object has no attribute 'items'\""
    );

    let scalar_shape = ActConfig::from_checkpoint_json(
        r#"{"type":"act","input_features":{"k":{"type":"STATE","shape":1}}}"#,
    )
    .unwrap_err();
    assert_eq!(
        scalar_shape.to_string(),
        "`input_features`: Could not decode the value into any of the given types:\n    dict: `k.shape`: Failed when parsing value='1' into field \"<class 'lerobot.configs.types.PolicyFeature'>.shape\" of type tuple[int, ...].\n         \tUnderlying error is \"TypeError: 'int' object is not iterable\"\n"
    );

    for field in ["input_features", "output_features"] {
        let malformed = ActConfig::from_checkpoint_json(&format!(
            r#"{{"type":"act","{field}":[["only-one-item"]]}}"#
        ))
        .unwrap_err();
        assert_eq!(
            malformed.to_string(),
            format!(
                "`{field}`: Could not decode the value into any of the given types:\n    dict: not enough values to unpack (expected 2, got 1)\n"
            )
        );
    }
}

// ---------------------------------------------------------------------------
// (6) `pretrained_path` is a `pathlib.Path` upstream.
// ---------------------------------------------------------------------------

#[test]
fn pretrained_path_is_normalised_like_pathlib_purepath() {
    for (input, expected) in [
        ("a//b", "a/b"),
        ("", "."),
        ("foo/", "foo"),
        ("./x", "x"),
        (".", "."),
        ("a/./b//c/", "a/b/c"),
        ("../a", "../a"),
        ("/abs//path/", "/abs/path"),
        ("//net/share", "//net/share"),
        ("///three", "/three"),
    ] {
        let config =
            ActConfig::from_checkpoint_json(&doc(&format!(r#","pretrained_path":"{input}""#)))
                .unwrap();
        assert_eq!(
            config.pretrained_path.as_deref(),
            Some(expected),
            "Path({input:?})"
        );
        assert_eq!(pure_posix_path(input), expected);
    }

    // `Path(...)` rejects every non-string, so this field gets no `str()`
    // coercion; `None` still passes straight through.
    assert!(ActConfig::from_checkpoint_json(&doc(r#","pretrained_path":123"#)).is_err());
    assert!(ActConfig::from_checkpoint_json(&doc(r#","pretrained_path":true"#)).is_err());
    assert_eq!(
        ActConfig::from_checkpoint_json(&doc(r#","pretrained_path":null"#))
            .unwrap()
            .pretrained_path,
        None
    );
}

// ---------------------------------------------------------------------------
// (6) Nested `PolicyFeature` follows `decode_dataclass` + `decode_int`.
// ---------------------------------------------------------------------------

#[test]
fn nested_policy_feature_rejects_unknown_fields_and_coerces_shape_integers() {
    let error = ActConfig::from_checkpoint_json(&doc(
        r#","input_features":{"observation.state":{"type":"STATE","shape":[7],"bogus":1}}"#,
    ))
    .unwrap_err();
    assert_eq!(error.kind(), ActConfigErrorKind::Decoding);
    assert!(
        error
            .to_string()
            .contains("The fields `bogus` are not valid for PolicyFeature"),
        "{error}"
    );

    let config = ActConfig::from_checkpoint_json(&doc(
        r#","input_features":{"k":{"type":"STATE","shape":["7",true,false," 1_0 "]}}"#,
    ))
    .unwrap();
    assert_eq!(
        config.input_features.as_ref().unwrap()["k"].shape,
        vec![bi(7), bi(1), bi(0), bi(10)]
    );

    for bad in [
        r#","input_features":{"k":{"type":"STATE","shape":[1.0]}}"#,
        r#","input_features":{"k":{"type":"STATE","shape":[null]}}"#,
        r#","input_features":{"k":{"type":"STATE","shape":[[1]]}}"#,
        r#","input_features":{"k":{"type":"STATE"}}"#,
    ] {
        assert!(
            ActConfig::from_checkpoint_json(&doc(bad)).is_err(),
            "{bad} must not decode"
        );
    }
}

// ---------------------------------------------------------------------------
// (7) Duplicate object keys follow Python `dict` assignment.
// ---------------------------------------------------------------------------

#[test]
fn duplicate_checkpoint_keys_take_the_last_value_like_python_dict() {
    let config =
        ActConfig::from_checkpoint_json(r#"{"type":"act","chunk_size":1,"chunk_size":200}"#)
            .unwrap();
    assert_eq!(config.chunk_size, bi(200));

    // The key keeps the position it was first inserted at, as Python does.
    let features = ActConfig::from_checkpoint_json(
        r#"{"type":"act","normalization_mapping":{"VISUAL":"MEAN_STD","STATE":"MIN_MAX","VISUAL":"IDENTITY"}}"#,
    )
    .unwrap();
    assert_eq!(
        features
            .normalization_mapping
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect::<Vec<_>>(),
        vec![
            ("VISUAL", NormalizationMode::Identity),
            ("STATE", NormalizationMode::MinMax),
        ]
    );
}

// ---------------------------------------------------------------------------
// (8) Non-finite floats round-trip through the checkpoint API.
// ---------------------------------------------------------------------------

#[test]
fn nonfinite_floats_round_trip_through_the_checkpoint_api() {
    // `json.load` accepts the three bare tokens CPython's writer emits.
    let config = ActConfig::from_checkpoint_json(&doc(
        r#","dropout":NaN,"kl_weight":Infinity,"optimizer_lr":-Infinity"#,
    ))
    .unwrap();
    assert!(config.dropout.is_nan());
    assert_eq!(config.kl_weight, f64::INFINITY);
    assert_eq!(config.optimizer_lr, f64::NEG_INFINITY);

    let text = config.to_checkpoint_json();
    assert!(text.contains("\"dropout\": NaN,"));
    assert!(text.contains("\"kl_weight\": Infinity,"));
    assert!(text.contains("\"optimizer_lr\": -Infinity,"));

    // And back again, so the non-finite tokens survive a full read/write
    // cycle. The dilation field is the one upstream widens from the
    // constructor's `false` to Draccus' `0` on the first read; from the second
    // pass on the cycle is a fixed point.
    let reread = ActConfig::from_checkpoint_json(&text)
        .unwrap()
        .to_checkpoint_json();
    assert_eq!(
        reread,
        text.replace(
            r#""replace_final_stride_with_dilation": false,"#,
            r#""replace_final_stride_with_dilation": 0,"#
        )
    );
    assert_eq!(
        ActConfig::from_checkpoint_json(&reread)
            .unwrap()
            .to_checkpoint_json(),
        reread
    );

    // `temporal_ensemble_coeff` is `float | None`; `null` is still None.
    let coeff = ActConfig::from_checkpoint_json(&doc(
        r#","n_action_steps":1,"temporal_ensemble_coeff":Infinity"#,
    ))
    .unwrap();
    assert_eq!(coeff.temporal_ensemble_coeff, Some(f64::INFINITY));
    assert!(coeff
        .to_checkpoint_json()
        .contains("\"temporal_ensemble_coeff\": Infinity,"));
}

// ---------------------------------------------------------------------------
// Checkpoint-boundary behaviour shared with upstream `from_pretrained`.
// ---------------------------------------------------------------------------

#[test]
fn checkpoint_decoding_runs_upstream_post_init_validation() {
    // `draccus.parse` constructs the dataclass, so `__post_init__` runs and a
    // checkpoint that would build an invalid config never yields one.
    let error = ActConfig::from_checkpoint_json(&doc(r#","n_obs_steps":2"#)).unwrap_err();
    assert_eq!(error.kind(), ActConfigErrorKind::Value);
    assert_eq!(
        error.to_string(),
        "Multiple observation steps not handled yet. Got `nobs_steps=2`"
    );

    let error =
        ActConfig::from_checkpoint_json(&doc(r#","temporal_ensemble_coeff":0.01"#)).unwrap_err();
    assert_eq!(error.kind(), ActConfigErrorKind::NotImplemented);
}

#[test]
fn checkpoint_decoding_rejects_unknown_fields_and_requires_the_registry_tag() {
    let error = ActConfig::from_checkpoint_json(&doc(r#","future_field":true"#)).unwrap_err();
    assert_eq!(error.kind(), ActConfigErrorKind::Decoding);
    assert!(
        error
            .to_string()
            .contains("The fields `future_field` are not valid for ACTConfig"),
        "{error}"
    );

    assert!(ActConfig::from_checkpoint_json(r#"{"chunk_size":5}"#).is_err());
    assert!(ActConfig::from_checkpoint_json(r#"{"type":"diffusion"}"#).is_err());
    assert!(ActConfig::from_checkpoint_json("{").is_err());
}

#[test]
fn checkpoint_round_trip_reproduces_upstreams_own_dilation_widening() {
    // Reading upstream's `config.json` and writing it back is not the identity
    // upstream either: `replace_final_stride_with_dilation` is annotated `int`,
    // so Draccus turns the freshly-constructed `false` into `0`, and the next
    // dump writes `0`. Confirmed by running that exact cycle upstream.
    let text = include_str!("data/act_default_config.json");
    let config = ActConfig::from_checkpoint_json(text).unwrap();
    assert_eq!(config.chunk_size, bi(100));
    assert_eq!(config.device.as_deref(), Some("cpu"));

    let rewritten = config.to_checkpoint_json();
    assert_eq!(
        rewritten,
        text.replace(
            r#""replace_final_stride_with_dilation": false,"#,
            r#""replace_final_stride_with_dilation": 0,"#
        )
    );
    // Every other field survives byte for byte, and the cycle is a fixed point
    // from the second pass on.
    assert_eq!(
        ActConfig::from_checkpoint_json(&rewritten)
            .unwrap()
            .to_checkpoint_json(),
        rewritten
    );
}
