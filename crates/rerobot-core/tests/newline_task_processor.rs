//! Behaviour parity tests for the newline task processor, derived from upstream
//! `tests/processor/test_smolvla_processor.py` and from direct Python 3.12
//! probes of `lerobot.processor.newline_task_processor` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1.

use rerobot_core::processor::newline_task::{NewLineTaskProcessorStep, REGISTRY_NAME};
use rerobot_core::processor::{ComplementaryData, FeatureMap, PipelineFeatures, ProcessorState};
use rerobot_core::types::{FeatureType, PipelineFeatureType, PolicyFeature};
use serde_json::json;

fn data(pairs: &[(&str, serde_json::Value)]) -> ComplementaryData {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

#[test]
fn complementary_data_without_a_task_key_is_returned_unchanged() {
    let step = NewLineTaskProcessorStep;
    let input = data(&[("other", json!("data"))]);
    assert_eq!(step.complementary_data(&input), input);
}

#[test]
fn a_task_string_without_a_trailing_newline_gets_one() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!("pick up the cube"))]));
    assert_eq!(out["task"], json!("pick up the cube\n"));
}

#[test]
fn a_task_string_that_already_ends_with_a_newline_is_left_alone() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!("pick up the cube\n"))]));
    assert_eq!(out["task"], json!("pick up the cube\n"));
}

#[test]
fn every_string_in_a_task_list_gets_a_newline_unless_it_has_one() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!(["task1", "task2\n", "task3"]))]));
    assert_eq!(out["task"], json!(["task1\n", "task2\n", "task3\n"]));
}

#[test]
fn a_task_list_that_is_not_all_strings_is_left_alone() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!(["a", 1]))]));
    assert_eq!(out["task"], json!(["a", 1]));
}

#[test]
fn get_config_is_the_empty_config_of_the_base_class() {
    // Upstream `ProcessorStep.get_config` returns `{}` and this step does not
    // override it; `test_smolvla_newline_processor_state_dict` asserts on it.
    assert_eq!(NewLineTaskProcessorStep.get_config(), json!({}));
}

#[test]
fn transform_features_returns_its_input_unchanged_in_order() {
    let step = NewLineTaskProcessorStep;
    let mut observation = FeatureMap::new();
    observation.insert(
        "observation.state".to_string(),
        PolicyFeature::new(FeatureType::State, vec![10]),
    );
    let mut features = PipelineFeatures::new();
    features.insert(PipelineFeatureType::Observation, observation);
    features.insert(PipelineFeatureType::Action, FeatureMap::new());

    let out = step.transform_features(&features);
    assert_eq!(out, features);
    assert_eq!(
        out.keys().collect::<Vec<_>>(),
        vec![
            &PipelineFeatureType::Observation,
            &PipelineFeatureType::Action
        ]
    );
}

#[test]
fn a_null_task_is_left_alone() {
    let step = NewLineTaskProcessorStep;
    let input = data(&[("task", json!(null))]);
    assert_eq!(step.complementary_data(&input), input);
}

#[test]
fn an_empty_task_string_becomes_a_lone_newline() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!(""))]));
    assert_eq!(out["task"], json!("\n"));
}

#[test]
fn a_task_string_ending_in_crlf_is_left_alone() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!("pick\r\n"))]));
    assert_eq!(out["task"], json!("pick\r\n"));
}

#[test]
fn a_task_string_ending_in_a_bare_carriage_return_gets_a_newline() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!("pick\r"))]));
    assert_eq!(out["task"], json!("pick\r\n"));
}

#[test]
fn a_multibyte_task_string_gets_a_newline_after_its_last_code_point() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!("タスクを実行 🧪"))]));
    assert_eq!(out["task"], json!("タスクを実行 🧪\n"));
}

#[test]
fn unicode_line_and_next_line_controls_are_not_lf() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!(["line\u{2028}", "line\u{0085}"]))]));
    assert_eq!(out["task"], json!(["line\u{2028}\n", "line\u{0085}\n"]));
}

#[test]
fn an_empty_task_list_stays_an_empty_list() {
    // Python's `all(...)` over an empty list is `True`, so the list branch is
    // taken and rebuilds an empty list.
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!([]))]));
    assert_eq!(out["task"], json!([]));
}

#[test]
fn an_empty_string_inside_a_task_list_becomes_a_lone_newline() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!(["", "a\n"]))]));
    assert_eq!(out["task"], json!(["\n", "a\n"]));
}

#[test]
fn a_task_list_of_booleans_is_left_alone() {
    // `isinstance(True, str)` is False upstream, so the list is not all strings.
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!([true, false]))]));
    assert_eq!(out["task"], json!([true, false]));
}

#[test]
fn a_nested_task_list_is_left_alone() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[("task", json!([["a"]]))]));
    assert_eq!(out["task"], json!([["a"]]));
}

#[test]
fn a_scalar_task_is_left_alone() {
    let step = NewLineTaskProcessorStep;
    let input = data(&[("task", json!(5))]);
    assert_eq!(step.complementary_data(&input), input);
}

#[test]
fn an_object_task_is_left_alone() {
    let step = NewLineTaskProcessorStep;
    let input = data(&[("task", json!({"a": "b"}))]);
    assert_eq!(step.complementary_data(&input), input);
}

#[test]
fn every_other_key_keeps_its_value_and_its_position() {
    let step = NewLineTaskProcessorStep;
    let out = step.complementary_data(&data(&[
        ("z", json!(1)),
        ("task", json!("go")),
        ("a", json!([1, 2])),
        ("index", json!(null)),
    ]));
    assert_eq!(
        out.keys().collect::<Vec<_>>(),
        vec!["z", "task", "a", "index"]
    );
    assert_eq!(out["z"], json!(1));
    assert_eq!(out["task"], json!("go\n"));
    assert_eq!(out["a"], json!([1, 2]));
    assert_eq!(out["index"], json!(null));
}

#[test]
fn the_source_complementary_data_is_not_modified() {
    let step = NewLineTaskProcessorStep;
    let input = data(&[("task", json!(["a", "b"]))]);
    let out = step.complementary_data(&input);
    assert_eq!(input["task"], json!(["a", "b"]));
    assert_eq!(out["task"], json!(["a\n", "b\n"]));
}

#[test]
fn the_result_outlives_the_input_it_was_built_from() {
    let step = NewLineTaskProcessorStep;
    let input = data(&[("task", json!("go"))]);
    let out = step.complementary_data(&input);
    drop(input);
    assert_eq!(out["task"], json!("go\n"));
}

#[test]
fn registry_name_matches_upstream() {
    assert_eq!(REGISTRY_NAME, "smolvla_new_line_processor");
}

#[test]
fn stateless_lifecycle_matches_the_processor_step_base_class() {
    let mut step = NewLineTaskProcessorStep;
    assert!(step.state_dict().is_empty());

    let mut arbitrary_state = ProcessorState::new();
    arbitrary_state.insert("ignored".to_string(), json!(123));
    step.load_state_dict(&arbitrary_state);
    step.reset();

    assert!(step.state_dict().is_empty());
}
