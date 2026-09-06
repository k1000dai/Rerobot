//! Behaviour parity tests for the rename processor, derived from upstream
//! `tests/processor/test_rename_processor.py` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1, with the current upstream suffix
//! extension pinned by `origin/main` commit `3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`.

use indexmap::IndexMap;
use rerobot_core::processor::rename::{
    rename_stats, FeatureMap, Observation, PipelineFeatures, RenameObservationsProcessorStep,
    Stats, REGISTRY_NAME,
};
use rerobot_core::types::{FeatureType, PipelineFeatureType, PolicyFeature};
use serde_json::json;

fn obs(pairs: &[(&str, serde_json::Value)]) -> Observation {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

fn map(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

#[test]
fn registry_name_matches_upstream() {
    assert_eq!(REGISTRY_NAME, "rename_observations_processor");
}

#[test]
fn basic_renaming() {
    let step = RenameObservationsProcessorStep::new([
        ("observation.state", "observation.robot_state"),
        ("observation.image", "observation.camera"),
    ]);
    let out = step.observation(&obs(&[
        ("observation.state", json!([1, 2, 3])),
        ("observation.image", json!("img")),
    ]));
    assert_eq!(
        out.keys().collect::<Vec<_>>(),
        vec!["observation.robot_state", "observation.camera"]
    );
    assert_eq!(out["observation.robot_state"], json!([1, 2, 3]));
}

#[test]
fn empty_rename_map_is_identity() {
    let step = RenameObservationsProcessorStep::default();
    let input = obs(&[("a", json!(1)), ("b", json!(2))]);
    assert_eq!(step.observation(&input), input);
}

#[test]
fn empty_observation_stays_empty() {
    let step = RenameObservationsProcessorStep::new([("a", "b")]);
    assert!(step.observation(&Observation::new()).is_empty());
}

#[test]
fn keys_absent_from_the_map_are_kept_verbatim() {
    let step = RenameObservationsProcessorStep::new([("pixels", "observation.image")]);
    let out = step.observation(&obs(&[
        ("pixels", json!("p")),
        ("reward", json!(1.0)),
        ("info", json!({"episode": 1})),
    ]));
    assert_eq!(out["observation.image"], json!("p"));
    assert_eq!(out["reward"], json!(1.0));
    assert_eq!(out["info"], json!({"episode": 1}));
    assert!(!out.contains_key("pixels"));
}

#[test]
fn an_explicit_padding_mapping_takes_precedence_over_the_derived_suffix_rule() {
    let step = RenameObservationsProcessorStep::new([
        ("observation.state", "obs.state"),
        ("observation.state_is_pad", "explicit.pad"),
    ]);
    let out = step.observation(&obs(&[
        ("observation.state", json!([1.0])),
        ("observation.state_is_pad", json!([false])),
        ("observation.state_padding_mask", json!([true])),
    ]));

    assert_eq!(
        out.keys().collect::<Vec<_>>(),
        vec!["obs.state", "explicit.pad", "obs.state_padding_mask",]
    );
    assert_eq!(out["explicit.pad"], json!([false]));
    assert_eq!(out["obs.state_padding_mask"], json!([true]));
}

#[test]
fn current_upstream_renames_temporal_padding_metadata_with_its_feature() {
    // The moved upstream implementation preserves the two derived keys when a
    // feature is renamed. Exact-key mappings still take precedence over this
    // suffix rule, and the rule is one-pass rather than cascading.
    let step = RenameObservationsProcessorStep::new([("observation.state", "obs.state")]);
    let out = step.observation(&obs(&[
        ("observation.state", json!([1.0])),
        ("observation.state_is_pad", json!([false])),
        ("observation.state_padding_mask", json!([true])),
    ]));

    assert_eq!(
        out.keys().cloned().collect::<Vec<_>>(),
        vec!["obs.state", "obs.state_is_pad", "obs.state_padding_mask"]
    );
    assert_eq!(out["obs.state_is_pad"], json!([false]));
    assert_eq!(out["obs.state_padding_mask"], json!([true]));
}

#[test]
fn overlapping_rename_chain_does_not_cascade() {
    // {"a": "b", "b": "c"} applied to {"a": 1, "b": 2, "x": 3}
    let step = RenameObservationsProcessorStep::new([("a", "b"), ("b", "c")]);
    let out = step.observation(&obs(&[("a", json!(1)), ("b", json!(2)), ("x", json!(3))]));
    assert!(!out.contains_key("a"));
    assert_eq!(out["b"], json!(1));
    assert_eq!(out["c"], json!(2));
    assert_eq!(out["x"], json!(3));
}

#[test]
fn colliding_output_keys_keep_the_last_value_at_the_first_position() {
    // Python dict assignment: `processed["b"] = 0` then `processed["b"] = 1`.
    let step = RenameObservationsProcessorStep::new([("a", "b")]);
    let out = step.observation(&obs(&[("b", json!(0)), ("a", json!(1))]));
    assert_eq!(out.len(), 1);
    assert_eq!(out["b"], json!(1));
    assert_eq!(out.keys().collect::<Vec<_>>(), vec!["b"]);
}

#[test]
fn renaming_a_key_onto_itself_is_a_no_op() {
    let step = RenameObservationsProcessorStep::new([("a", "a")]);
    let out = step.observation(&obs(&[("a", json!(7))]));
    assert_eq!(out["a"], json!(7));
    assert_eq!(out.len(), 1);
}

#[test]
fn rename_map_entries_for_missing_keys_are_ignored() {
    let step = RenameObservationsProcessorStep::new([("missing", "renamed")]);
    let out = step.observation(&obs(&[("present", json!(1))]));
    assert_eq!(out.len(), 1);
    assert!(!out.contains_key("renamed"));
}

#[test]
fn value_types_are_preserved_including_nested_structures() {
    let step = RenameObservationsProcessorStep::new([("nested", "renamed")]);
    let nested = json!({"a": [1, 2, {"b": null}], "c": 1.5, "d": true});
    let out = step.observation(&obs(&[("nested", nested.clone())]));
    assert_eq!(out["renamed"], nested);
}

#[test]
fn chained_steps_compose() {
    let first = RenameObservationsProcessorStep::new([("a", "b")]);
    let second = RenameObservationsProcessorStep::new([("b", "c")]);
    let out = second.observation(&first.observation(&obs(&[("a", json!(1))])));
    assert_eq!(out["c"], json!(1));
}

#[test]
fn get_config_round_trips_through_json() {
    let step = RenameObservationsProcessorStep::new([("a", "b"), ("c", "d")]);
    let cfg = step.get_config();
    assert_eq!(cfg, json!({"rename_map": {"a": "b", "c": "d"}}));
    let restored: RenameObservationsProcessorStep = serde_json::from_value(cfg).unwrap();
    assert_eq!(restored, step);
}

#[test]
fn get_config_of_an_empty_step_is_an_empty_map() {
    let cfg = RenameObservationsProcessorStep::default().get_config();
    assert_eq!(cfg, json!({"rename_map": {}}));
}

#[test]
fn get_config_preserves_rename_map_order() {
    let step = RenameObservationsProcessorStep::new([("z", "1"), ("a", "2"), ("m", "3")]);
    assert_eq!(
        serde_json::to_string(&step.get_config()).unwrap(),
        r#"{"rename_map":{"z":"1","a":"2","m":"3"}}"#
    );
}

fn features(pairs: &[(&str, FeatureType, &[usize])]) -> PipelineFeatures {
    let observation: FeatureMap = pairs
        .iter()
        .map(|(k, t, s)| ((*k).to_string(), PolicyFeature::new(*t, s.to_vec())))
        .collect();
    let mut out = PipelineFeatures::new();
    out.insert(PipelineFeatureType::Action, FeatureMap::new());
    out.insert(PipelineFeatureType::Observation, observation);
    out
}

#[test]
fn transform_features_renames_observation_keys_only() {
    let step = RenameObservationsProcessorStep::new([("observation.state", "obs.state")]);
    let input = features(&[
        ("observation.state", FeatureType::State, &[4]),
        ("observation.image", FeatureType::Visual, &[3, 64, 64]),
    ]);
    let out = step.transform_features(&input).unwrap();
    let o = &out[&PipelineFeatureType::Observation];
    assert_eq!(
        o.keys().collect::<Vec<_>>(),
        vec!["obs.state", "observation.image"]
    );
    assert_eq!(
        o["obs.state"],
        PolicyFeature::new(FeatureType::State, vec![4])
    );
    assert!(out[&PipelineFeatureType::Action].is_empty());
}

#[test]
fn transform_features_handles_overlapping_keys_like_the_observation_path() {
    let step = RenameObservationsProcessorStep::new([("a", "b"), ("b", "c")]);
    let input = features(&[
        ("a", FeatureType::State, &[1]),
        ("b", FeatureType::Action, &[2]),
    ]);
    let out = step.transform_features(&input).unwrap();
    let o = &out[&PipelineFeatureType::Observation];
    assert_eq!(o["b"], PolicyFeature::new(FeatureType::State, vec![1]));
    assert_eq!(o["c"], PolicyFeature::new(FeatureType::Action, vec![2]));
}

#[test]
fn transform_features_keeps_unmapped_padding_metadata_exact() {
    // Current upstream's feature declaration transform only maps exact feature
    // names; the suffix extension is for runtime observation dictionaries.
    let step = RenameObservationsProcessorStep::new([("observation.state", "obs.state")]);
    let input = features(&[
        ("observation.state", FeatureType::State, &[1]),
        ("observation.state_is_pad", FeatureType::State, &[1]),
    ]);
    let out = step.transform_features(&input).unwrap();
    let observation = &out[&PipelineFeatureType::Observation];

    assert!(observation.contains_key("obs.state"));
    assert!(observation.contains_key("observation.state_is_pad"));
    assert!(!observation.contains_key("obs.state_is_pad"));
}

#[test]
fn transform_features_without_an_observation_stage_returns_none() {
    let step = RenameObservationsProcessorStep::new([("a", "b")]);
    let mut input = PipelineFeatures::new();
    input.insert(PipelineFeatureType::Action, FeatureMap::new());
    assert!(step.transform_features(&input).is_none());
}

/// One `(feature_name, Some(sub_stats) | None)` literal, as written in tests.
type StatsLiteral<'a> = (&'a str, Option<&'a [(&'a str, serde_json::Value)]>);

fn stats(pairs: &[StatsLiteral<'_>]) -> Stats {
    pairs
        .iter()
        .map(|(k, v)| {
            let sub = v.map(|entries| {
                entries
                    .iter()
                    .map(|(sk, sv)| ((*sk).to_string(), sv.clone()))
                    .collect()
            });
            ((*k).to_string(), sub)
        })
        .collect()
}

#[test]
fn rename_stats_renames_top_level_keys() {
    let s = stats(&[
        (
            "observation.state",
            Some(&[("mean", json!([0.0])), ("std", json!([1.0]))]),
        ),
        ("action", Some(&[("mean", json!([0.0]))])),
    ]);
    let renamed = rename_stats(
        &s,
        &map(&[("observation.state", "observation.robot_state")]),
    );
    assert!(renamed.contains_key("observation.robot_state"));
    assert!(!renamed.contains_key("observation.state"));
    assert!(renamed.contains_key("action"));
    assert_eq!(
        renamed["observation.robot_state"].as_ref().unwrap()["mean"],
        json!([0.0])
    );
}

#[test]
fn rename_stats_of_empty_input_is_empty() {
    assert!(rename_stats(&Stats::new(), &map(&[("a", "b")])).is_empty());
}

#[test]
fn rename_stats_turns_none_sub_stats_into_empty_maps() {
    let s = stats(&[("a", None)]);
    let renamed = rename_stats(&s, &map(&[("a", "b")]));
    assert_eq!(renamed["b"].as_ref().unwrap().len(), 0);
}

#[test]
fn rename_stats_preserves_order_and_leaves_unmapped_keys_alone() {
    let s = stats(&[
        ("z", Some(&[("mean", json!(1))])),
        ("a", Some(&[("mean", json!(2))])),
    ]);
    let renamed = rename_stats(&s, &map(&[("a", "aa")]));
    assert_eq!(renamed.keys().collect::<Vec<_>>(), vec!["z", "aa"]);
}

#[test]
fn rename_stats_collisions_keep_the_last_value() {
    let s = stats(&[
        ("b", Some(&[("mean", json!(1))])),
        ("a", Some(&[("mean", json!(2))])),
    ]);
    let renamed = rename_stats(&s, &map(&[("a", "b")]));
    assert_eq!(renamed.len(), 1);
    assert_eq!(renamed["b"].as_ref().unwrap()["mean"], json!(2));
}

#[test]
fn rename_stats_result_is_independent_of_the_input() {
    let s = stats(&[("a", Some(&[("mean", json!([0.0]))]))]);
    let renamed = rename_stats(&s, &map(&[("a", "b")]));
    drop(s);
    assert_eq!(renamed["b"].as_ref().unwrap()["mean"], json!([0.0]));
}
