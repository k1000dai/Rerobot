// Copyright 2026 The Rerobot contributors
// SPDX-License-Identifier: Apache-2.0
//! Differential contract vectors for upstream `ACTConfig`.

use indexmap::IndexMap;
use num_bigint::BigInt;
use rerobot_core::policy::act::{ActConfig, ActConfigErrorKind, PythonIntBool};
use rerobot_core::types::{FeatureType, NormalizationMode, PolicyFeature};
use serde_json::json;

fn bi(value: i64) -> BigInt {
    BigInt::from(value)
}

#[test]
fn defaults_match_upstream_act_config() {
    let config = ActConfig::default();

    assert_eq!(config.n_obs_steps, bi(1));
    assert_eq!(config.chunk_size, bi(100));
    assert_eq!(config.n_action_steps, bi(100));
    // Upstream annotates this `dict[str, NormalizationMode]`; the three literal
    // keys it writes happen to be `FeatureType` values, but the key domain is
    // every string. See `tests/act_checkpoint.rs`.
    assert_eq!(
        config.normalization_mapping,
        IndexMap::from([
            (
                FeatureType::Visual.as_str().to_owned(),
                NormalizationMode::MeanStd
            ),
            (
                FeatureType::State.as_str().to_owned(),
                NormalizationMode::MeanStd
            ),
            (
                FeatureType::Action.as_str().to_owned(),
                NormalizationMode::MeanStd
            ),
        ])
    );
    assert_eq!(config.vision_backbone, "resnet18");
    assert_eq!(
        config.pretrained_backbone_weights.as_deref(),
        Some("ResNet18_Weights.IMAGENET1K_V1")
    );
    assert_eq!(
        config.replace_final_stride_with_dilation,
        PythonIntBool::Bool(false)
    );
    assert!(!config.pre_norm);
    assert_eq!(config.dim_model, bi(512));
    assert_eq!(config.n_heads, bi(8));
    assert_eq!(config.dim_feedforward, bi(3200));
    assert_eq!(config.feedforward_activation, "relu");
    assert_eq!(config.n_encoder_layers, bi(4));
    assert_eq!(config.n_decoder_layers, bi(1));
    assert!(config.use_vae);
    assert_eq!(config.latent_dim, bi(32));
    assert_eq!(config.n_vae_encoder_layers, bi(4));
    assert_eq!(config.temporal_ensemble_coeff, None);
    assert_eq!(config.dropout, 0.1);
    assert_eq!(config.kl_weight, 10.0);
    assert_eq!(config.optimizer_lr, 1e-5);
    assert_eq!(config.optimizer_weight_decay, 1e-4);
    assert_eq!(config.optimizer_lr_backbone, 1e-5);
    config.validate().unwrap();
}

#[test]
fn validation_preserves_upstream_error_precedence_and_exact_messages() {
    let mut config = ActConfig::default();
    config.vision_backbone = "vit".into();
    config.temporal_ensemble_coeff = Some(0.01);
    config.n_action_steps = bi(101);
    config.chunk_size = bi(100);
    config.n_obs_steps = bi(2);

    let error = config.validate().unwrap_err();
    assert_eq!(error.kind(), ActConfigErrorKind::Value);
    assert_eq!(
        error.to_string(),
        "`vision_backbone` must be one of the ResNet variants. Got vit."
    );

    config.vision_backbone = "resnet18".into();
    let error = config.validate().unwrap_err();
    assert_eq!(error.kind(), ActConfigErrorKind::NotImplemented);
    assert_eq!(
        error.to_string(),
        "`n_action_steps` must be 1 when using temporal ensembling. This is because the policy needs to be queried every step to compute the ensembled action."
    );

    config.temporal_ensemble_coeff = None;
    let error = config.validate().unwrap_err();
    assert_eq!(error.kind(), ActConfigErrorKind::Value);
    assert_eq!(
        error.to_string(),
        "The chunk size is the upper bound for the number of action steps per model invocation. Got 101 for `n_action_steps` and 100 for `chunk_size`."
    );

    config.n_action_steps = bi(100);
    let error = config.validate().unwrap_err();
    assert_eq!(
        error.to_string(),
        "Multiple observation steps not handled yet. Got `nobs_steps=2`"
    );
}

#[test]
fn signed_arbitrary_precision_step_counts_are_accepted_before_value_validation() {
    let huge: BigInt = format!("1{}", "0".repeat(500)).parse().unwrap();
    let mut config = ActConfig::default();
    config.chunk_size = huge.clone();
    config.n_action_steps = huge.clone();
    config.validate().unwrap();

    config.chunk_size = bi(-2);
    config.n_action_steps = bi(-1);
    assert_eq!(
        config.validate().unwrap_err().to_string(),
        "The chunk size is the upper bound for the number of action steps per model invocation. Got -1 for `n_action_steps` and -2 for `chunk_size`."
    );
}

#[test]
fn action_delta_indices_match_python_range_without_eager_allocation() {
    let mut config = ActConfig::default();
    config.chunk_size = bi(4);
    assert_eq!(
        config.action_delta_indices().collect::<Vec<_>>(),
        vec![bi(0), bi(1), bi(2), bi(3)]
    );

    config.chunk_size = bi(0);
    assert_eq!(config.action_delta_indices().next(), None);
    config.chunk_size = bi(-10);
    assert_eq!(config.action_delta_indices().next(), None);
}

#[test]
fn feature_validation_accepts_visual_or_environment_and_rejects_state_only() {
    let mut config = ActConfig::default();
    assert_eq!(
        config.validate_features().unwrap_err().to_string(),
        "You must provide at least one image or the environment state among the inputs."
    );

    config.input_features = Some(IndexMap::from([(
        "observation.environment_state".into(),
        PolicyFeature::new(FeatureType::Env, [7]),
    )]));
    config.validate_features().unwrap();

    config.input_features = Some(IndexMap::from([(
        "observation.state".into(),
        PolicyFeature::new(FeatureType::State, [7]),
    )]));
    assert!(config.validate_features().is_err());

    config.input_features = Some(IndexMap::from([(
        "any.visual.key".into(),
        PolicyFeature::new(FeatureType::Visual, [3, 4, 5]),
    )]));
    config.validate_features().unwrap();
}

#[test]
fn presets_and_delta_indices_match_upstream() {
    let config = ActConfig::default();
    assert_eq!(
        config.optimizer_preset(),
        rerobot_core::policy::act::AdamWConfig {
            lr: 1e-5,
            weight_decay: 1e-4,
            grad_clip_norm: 10.0,
            betas: [0.9, 0.999],
            eps: 1e-8,
        }
    );
    assert_eq!(config.scheduler_preset(), None);
    assert_eq!(config.observation_delta_indices(), None);
    assert_eq!(config.reward_delta_indices(), None);
}

#[test]
fn checkpoint_config_json_matches_upstream_field_order_and_types() {
    let mut config = ActConfig::default();
    config.device = Some("cpu".into());
    config.push_to_hub = false;
    config.input_features = Some(IndexMap::from([(
        "observation.state".into(),
        PolicyFeature::new(FeatureType::State, [7]),
    )]));
    config.output_features = Some(IndexMap::from([(
        "action".into(),
        PolicyFeature::new(FeatureType::Action, [7]),
    )]));

    let wire = serde_json::to_value(&config).unwrap();
    assert_eq!(wire["type"], "act");
    assert_eq!(wire["n_obs_steps"], 1);
    assert_eq!(
        wire["input_features"]["observation.state"]["shape"],
        json!([7])
    );
    assert_eq!(
        wire["normalization_mapping"],
        json!({
            "VISUAL": "MEAN_STD",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD"
        })
    );
    assert_eq!(wire["replace_final_stride_with_dilation"], false);
    assert_eq!(wire["pretrained_path"], serde_json::Value::Null);

    let text = serde_json::to_string(&config).unwrap();
    assert!(
        text.starts_with(r#"{"type":"act","n_obs_steps":1,"input_features":{"observation.state""#)
    );
    let decoded: ActConfig = serde_json::from_str(&text).unwrap();
    assert_eq!(decoded, config);
}

#[test]
fn checkpoint_wire_round_trips_huge_and_negative_python_integers() {
    let huge: BigInt = format!("-1{}", "0".repeat(1000)).parse().unwrap();
    let mut config = ActConfig::default();
    config.chunk_size = huge.clone();
    config.n_action_steps = huge.clone();
    config.dim_model = huge.clone();

    let wire = serde_json::to_string(&config).unwrap();
    assert!(wire.contains(&format!(r#""chunk_size":{huge}"#)));
    assert!(wire.contains(&format!(r#""dim_model":{huge}"#)));
    let decoded: ActConfig = serde_json::from_str(&wire).unwrap();
    assert_eq!(decoded, config);
}

#[test]
fn checkpoint_wire_distinguishes_absent_null_wrong_type_and_unknown_fields() {
    let base = serde_json::to_value(ActConfig::default()).unwrap();

    let mut missing_type = base.clone();
    missing_type.as_object_mut().unwrap().remove("type");
    assert!(serde_json::from_value::<ActConfig>(missing_type).is_err());

    let mut unknown_type = base.clone();
    unknown_type["type"] = json!("diffusion");
    assert!(serde_json::from_value::<ActConfig>(unknown_type).is_err());

    let mut missing = base.clone();
    missing.as_object_mut().unwrap().remove("chunk_size");
    assert_eq!(
        serde_json::from_value::<ActConfig>(missing)
            .unwrap()
            .chunk_size,
        bi(100)
    );

    let mut null = base.clone();
    null["chunk_size"] = serde_json::Value::Null;
    assert!(serde_json::from_value::<ActConfig>(null).is_err());

    let mut numeric_string = base.clone();
    numeric_string["chunk_size"] = json!(" +00100 ");
    assert_eq!(
        serde_json::from_value::<ActConfig>(numeric_string)
            .unwrap()
            .chunk_size,
        bi(100)
    );

    let mut python_bool = base.clone();
    python_bool["chunk_size"] = json!(true);
    assert_eq!(
        serde_json::from_value::<ActConfig>(python_bool)
            .unwrap()
            .chunk_size,
        bi(1)
    );

    let mut wrong = base.clone();
    wrong["chunk_size"] = json!(1.0);
    assert!(serde_json::from_value::<ActConfig>(wrong).is_err());

    let mut float_string = base.clone();
    float_string["dropout"] = json!(" 1.25 ");
    assert_eq!(
        serde_json::from_value::<ActConfig>(float_string)
            .unwrap()
            .dropout,
        1.25
    );

    let mut float_bool = base.clone();
    float_bool["optimizer_lr"] = json!(true);
    assert_eq!(
        serde_json::from_value::<ActConfig>(float_bool)
            .unwrap()
            .optimizer_lr,
        1.0
    );

    let mut optional_float_string = base.clone();
    optional_float_string["temporal_ensemble_coeff"] = json!("0.01");
    assert_eq!(
        serde_json::from_value::<ActConfig>(optional_float_string)
            .unwrap()
            .temporal_ensemble_coeff,
        Some(0.01)
    );

    let mut bool_string = base.clone();
    bool_string["pre_norm"] = json!("true");
    bool_string["use_vae"] = json!("false");
    let decoded = serde_json::from_value::<ActConfig>(bool_string).unwrap();
    assert!(decoded.pre_norm);
    assert!(!decoded.use_vae);
    assert!(
        serde_json::from_str::<ActConfig>(r#"{"type":"act","pre_norm":"\u0074rue"}"#)
            .unwrap()
            .pre_norm
    );

    let mut optional_bool_string = base.clone();
    optional_bool_string["private"] = json!("false");
    assert!(!serde_json::from_value::<ActConfig>(optional_bool_string)
        .unwrap()
        .private
        .unwrap());

    let mut invalid_bool = base.clone();
    invalid_bool["pre_norm"] = json!(1);
    assert!(serde_json::from_value::<ActConfig>(invalid_bool).is_err());

    let mut dilation_integer = base.clone();
    dilation_integer["replace_final_stride_with_dilation"] = json!("-2");
    assert_eq!(
        serde_json::from_value::<ActConfig>(dilation_integer)
            .unwrap()
            .replace_final_stride_with_dilation,
        PythonIntBool::Int(bi(-2))
    );

    let mut dilation_bool = base.clone();
    dilation_bool["replace_final_stride_with_dilation"] = json!(true);
    assert_eq!(
        serde_json::from_value::<ActConfig>(dilation_bool)
            .unwrap()
            .replace_final_stride_with_dilation,
        PythonIntBool::Int(bi(1))
    );

    let mut unknown = base;
    unknown["future_field"] = json!(true);
    assert!(serde_json::from_value::<ActConfig>(unknown).is_err());
}

#[test]
fn nonfinite_float_checkpoint_output_fails_instead_of_silently_becoming_null() {
    let mut config = ActConfig::default();
    config.dropout = f64::NAN;
    assert!(serde_json::to_string(&config).is_err());

    config.dropout = 0.1;
    config.temporal_ensemble_coeff = Some(f64::INFINITY);
    assert!(serde_json::to_string(&config).is_err());
}
