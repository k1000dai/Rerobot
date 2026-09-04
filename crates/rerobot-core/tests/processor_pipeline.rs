//! Vertical tests for the JSON processor-pipeline reconstruction boundary.
//!
//! The upstream oracle is `lerobot.processor.pipeline.DataProcessorPipeline` at
//! commit f37be3edbee60f3a09a5183788b91eb19f0c07d1. This file intentionally
//! covers only the JSON transition domain implemented by the native port.

use rerobot_core::processor::pipeline::{
    JsonProcessorPipeline, JsonTransition, ProcessorPipelineError, ProcessorStepRegistry,
};
use rerobot_core::processor::{FeatureMap, PipelineFeatures};
use rerobot_core::types::{FeatureType, PipelineFeatureType, PolicyFeature};
use serde_json::json;

fn transition() -> JsonTransition {
    let mut observation = rerobot_core::processor::rename::Observation::new();
    observation.insert("pixels".to_owned(), json!([1, 2, 3]));
    observation.insert("observation.state".to_owned(), json!([0.5, 1.5]));

    let mut complementary_data = rerobot_core::processor::ComplementaryData::new();
    complementary_data.insert("task".to_owned(), json!("pick up the cube"));
    complementary_data.insert("index".to_owned(), json!(7));
    JsonTransition::new(observation, complementary_data)
}

#[test]
fn from_config_reconstructs_registered_steps_and_runs_them_in_order() {
    let config = json!({
        "name": "Policy Preprocessor",
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {"pixels": "observation.image"}}
            },
            {"registry_name": "smolvla_new_line_processor", "config": {}}
        ]
    });

    let pipeline = JsonProcessorPipeline::from_config(&config).unwrap();
    let output = pipeline.process(&transition());

    assert_eq!(output.observation["observation.image"], json!([1, 2, 3]));
    assert!(!output.observation.contains_key("pixels"));
    assert_eq!(
        output.complementary_data["task"],
        json!("pick up the cube\n")
    );
    assert_eq!(output.complementary_data["index"], json!(7));
}

#[test]
fn step_through_includes_the_unmodified_input_then_each_stage() {
    let config = json!({
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {"pixels": "observation.image"}}
            },
            {"registry_name": "smolvla_new_line_processor", "config": {}}
        ]
    });
    let pipeline = JsonProcessorPipeline::from_config(&config).unwrap();
    let states = pipeline.step_through(&transition());

    assert_eq!(states.len(), 3);
    assert!(states[0].observation.contains_key("pixels"));
    assert!(states[1].observation.contains_key("observation.image"));
    assert_eq!(
        states[1].complementary_data["task"],
        json!("pick up the cube")
    );
    assert_eq!(
        states[2].complementary_data["task"],
        json!("pick up the cube\n")
    );
}

#[test]
fn transform_features_applies_each_step_without_changing_stage_order() {
    let mut observation = FeatureMap::new();
    observation.insert(
        "pixels".to_owned(),
        PolicyFeature::new(FeatureType::Visual, [3, 32, 32]),
    );
    let mut action = FeatureMap::new();
    action.insert(
        "action".to_owned(),
        PolicyFeature::new(FeatureType::Action, [2]),
    );
    let mut features = PipelineFeatures::new();
    features.insert(PipelineFeatureType::Observation, observation);
    features.insert(PipelineFeatureType::Action, action);

    let config = json!({
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {"pixels": "observation.image"}}
            },
            {"registry_name": "smolvla_new_line_processor", "config": {}}
        ]
    });
    let pipeline = JsonProcessorPipeline::from_config(&config).unwrap();
    let output = pipeline.transform_features(&features).unwrap();

    assert!(output[&PipelineFeatureType::Observation].contains_key("observation.image"));
    assert_eq!(
        output.keys().copied().collect::<Vec<_>>(),
        vec![
            PipelineFeatureType::Observation,
            PipelineFeatureType::Action
        ]
    );
}

#[test]
fn empty_pipeline_is_valid_and_is_an_identity() {
    let input = transition();
    let pipeline = JsonProcessorPipeline::from_config(&json!({"steps": []})).unwrap();
    assert_eq!(pipeline.process(&input), input);
    assert_eq!(pipeline.step_through(&input), vec![input]);
}

#[test]
fn supported_pipeline_config_round_trips_in_upstream_order() {
    let config = json!({
        "name": "Policy Preprocessor",
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {"pixels": "observation.image"}}
            },
            {"registry_name": "smolvla_new_line_processor", "config": {}}
        ]
    });
    let pipeline = JsonProcessorPipeline::from_config(&config).unwrap();
    assert_eq!(pipeline.get_config(), config);
}

#[test]
fn malformed_and_unsupported_configs_are_distinguished() {
    let missing_steps = JsonProcessorPipeline::from_config(&json!({})).unwrap_err();
    assert!(matches!(
        missing_steps,
        ProcessorPipelineError::MissingField { field } if field == "steps"
    ));

    let unknown = JsonProcessorPipeline::from_config(&json!({
        "steps": [{"registry_name": "not_a_real_step", "config": {}}]
    }))
    .unwrap_err();
    assert!(matches!(
        unknown,
        ProcessorPipelineError::UnsupportedRegistryName { name } if name == "not_a_real_step"
    ));

    let class_only = JsonProcessorPipeline::from_config(&json!({
        "steps": [{"class": "lerobot.processor.rename.RenameObservationsProcessorStep"}]
    }))
    .unwrap_err();
    assert!(matches!(
        class_only,
        ProcessorPipelineError::UnsupportedClass { class }
            if class == "lerobot.processor.rename.RenameObservationsProcessorStep"
    ));

    let wrong_config_type = JsonProcessorPipeline::from_config(&json!({
        "steps": [{"registry_name": "smolvla_new_line_processor", "config": null}]
    }))
    .unwrap_err();
    assert!(matches!(
        wrong_config_type,
        ProcessorPipelineError::WrongType { path, expected }
            if path == "steps[0].config" && expected == "an object"
    ));

    let unknown_with_bad_config = JsonProcessorPipeline::from_config(&json!({
        "steps": [{"registry_name": "not_a_real_step", "config": null}]
    }))
    .unwrap_err();
    assert!(matches!(
        unknown_with_bad_config,
        ProcessorPipelineError::UnsupportedRegistryName { name } if name == "not_a_real_step"
    ));

    let stateful = JsonProcessorPipeline::from_config(&json!({
        "steps": [{
            "registry_name": "smolvla_new_line_processor",
            "state_file": "step_0.safetensors"
        }]
    }))
    .unwrap_err();
    assert!(matches!(
        stateful,
        ProcessorPipelineError::InvalidStep { index: 0, reason }
            if reason.contains("state_file is unsupported")
    ));
}

#[test]
fn a_saved_step_rejects_unknown_constructor_configuration() {
    let error = JsonProcessorPipeline::from_config(&json!({
        "steps": [{
            "registry_name": "smolvla_new_line_processor",
            "config": {"unexpected": true}
        }]
    }))
    .unwrap_err();
    assert!(matches!(
        error,
        ProcessorPipelineError::InvalidStep { index: 0, .. }
    ));
}

#[test]
fn feature_transform_reports_a_missing_observation_stage() {
    let pipeline = JsonProcessorPipeline::from_config(&json!({
        "steps": [{
            "registry_name": "rename_observations_processor",
            "config": {"rename_map": {"pixels": "observation.image"}}
        }]
    }))
    .unwrap();
    let features = PipelineFeatures::new();
    let error = pipeline.transform_features(&features).unwrap_err();
    assert!(matches!(
        error,
        ProcessorPipelineError::MissingObservationStage { step_index: 0 }
    ));
}

#[test]
fn registry_exposes_only_the_native_steps_and_keeps_upstream_names() {
    assert!(ProcessorStepRegistry::contains(
        "rename_observations_processor"
    ));
    assert!(ProcessorStepRegistry::contains(
        "smolvla_new_line_processor"
    ));
    assert!(!ProcessorStepRegistry::contains("device_processor"));
}

#[test]
fn null_state_file_is_treated_as_absent_for_stateless_steps() {
    let pipeline = JsonProcessorPipeline::from_config(&json!({
        "steps": [{
            "registry_name": "rename_observations_processor",
            "state_file": null,
        }]
    }))
    .expect("null state_file is the serialized spelling of no state");

    assert_eq!(
        pipeline.get_config(),
        json!({
            "name": "DataProcessorPipeline",
            "steps": [{
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {}},
            }]
        })
    );
}

#[test]
fn reset_is_a_noop_for_the_stateless_native_steps() {
    let mut pipeline = JsonProcessorPipeline::from_config(&json!({
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {"pixels": "observation.image"}}
            },
            {"registry_name": "smolvla_new_line_processor", "config": {}}
        ]
    }))
    .unwrap();
    let input = transition();
    let before = pipeline.process(&input);
    pipeline.reset();
    assert_eq!(pipeline.process(&input), before);
}
