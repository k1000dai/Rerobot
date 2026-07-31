//! Behaviour parity tests for `_NormalizationMixin._apply_transform` in
//! `lerobot/processor/normalize_processor.py` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1.
//!
//! Scope is the numeric transform itself, in `f32`, because that is the dtype
//! the tensors carry: all four supported modes, the `eps` placement (which is
//! *not* the same in every mode), the identity/pass-through rules, and the
//! `ValueError`s raised when a mode's statistics are absent. The processor
//! pipeline, device movement and the `dtype` re-cast around it are separate,
//! unported slices.

use indexmap::IndexMap;
use rerobot_core::dataset::stats::{stats_from_value, DatasetStats};
use rerobot_core::policy::normalize::{NormalizeError, Normalizer, NORMALIZATION_EPS};
use rerobot_core::types::{FeatureType, NormalizationMode, PolicyFeature};

fn stats(text: &str) -> DatasetStats {
    stats_from_value(&rerobot_core::dataset::json::loads(text).unwrap()).unwrap()
}

fn features(pairs: &[(&str, FeatureType, i64)]) -> IndexMap<String, PolicyFeature> {
    pairs
        .iter()
        .map(|(key, feature_type, dim)| {
            (
                (*key).to_owned(),
                PolicyFeature::new(*feature_type, [rerobot_core::BigInt::from(*dim)]),
            )
        })
        .collect()
}

fn mean_std_map() -> IndexMap<String, NormalizationMode> {
    IndexMap::from([
        (
            FeatureType::State.as_str().to_owned(),
            NormalizationMode::MeanStd,
        ),
        (
            FeatureType::Env.as_str().to_owned(),
            NormalizationMode::MeanStd,
        ),
        (
            FeatureType::Action.as_str().to_owned(),
            NormalizationMode::MeanStd,
        ),
    ])
}

/// The fixture dataset's statistics for its three state-only features.
fn fixture_stats() -> DatasetStats {
    stats(
        r#"{
        "observation.state": {
            "min": [0.0, 0.0], "max": [1.0, 1.0],
            "mean": [0.4375, 0.5625],
            "std": [0.36975499987602234, 0.36975499987602234],
            "count": [4]
        },
        "observation.environment_state": {
            "min": [10.0, -4.0], "max": [13.0, -1.0],
            "mean": [11.5, -2.5],
            "std": [1.1180340051651, 1.1180340051651],
            "count": [4]
        },
        "action": {
            "min": [-0.5, -0.5], "max": [0.5, 0.5],
            "mean": [0.0625, -0.0625],
            "std": [0.36975499987602234, 0.36975499987602234],
            "count": [4]
        }
    }"#,
    )
}

fn fixture_normalizer() -> Normalizer {
    Normalizer::new(
        &features(&[
            ("observation.state", FeatureType::State, 2),
            ("observation.environment_state", FeatureType::Env, 2),
            ("action", FeatureType::Action, 2),
        ]),
        &mean_std_map(),
        &fixture_stats(),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// MEAN_STD
// ---------------------------------------------------------------------------

#[test]
fn the_epsilon_is_upstreams_and_lands_on_the_denominator_only() {
    assert_eq!(NORMALIZATION_EPS, 1e-8);

    let normalizer = fixture_normalizer();
    // (x - mean) / (std + eps), computed in f32 exactly as the tensors are.
    let mean = 0.4375f32;
    let std = 0.369_755_f32;
    let expected = (0.0f32 - mean) / (std + NORMALIZATION_EPS as f32);
    let actual = normalizer
        .normalize("observation.state", &[0.0, 1.0])
        .unwrap();
    assert_eq!(actual[0], expected);
}

#[test]
fn mean_std_normalizes_the_first_fixture_frame() {
    let normalizer = fixture_normalizer();
    let state = normalizer
        .normalize("observation.state", &[0.0, 1.0])
        .unwrap();
    let env = normalizer
        .normalize("observation.environment_state", &[10.0, -1.0])
        .unwrap();
    let action = normalizer.normalize("action", &[0.5, -0.5]).unwrap();

    // Signs and magnitudes: frame 0's state is below the mean on axis 0 and
    // above it on axis 1; the env state is at its minimum on axis 0.
    assert!(state[0] < 0.0 && state[1] > 0.0);
    assert!(env[0] < 0.0 && env[1] > 0.0);
    assert!(action[0] > 0.0 && action[1] < 0.0);
    assert!(state
        .iter()
        .chain(&env)
        .chain(&action)
        .all(|v| v.is_finite()));
}

#[test]
fn unnormalizing_uses_std_without_the_epsilon_so_it_is_not_an_exact_inverse() {
    // Upstream: forward divides by `std + eps`, inverse multiplies by `std`.
    // The asymmetry is deliberate there and is preserved here.
    let normalizer = fixture_normalizer();
    let normalized = normalizer.normalize("action", &[0.5, -0.5]).unwrap();
    let restored = normalizer.unnormalize("action", &normalized).unwrap();
    assert!((restored[0] - 0.5).abs() < 1e-6);
    assert!((restored[1] + 0.5).abs() < 1e-6);
}

#[test]
fn a_zero_standard_deviation_is_absorbed_by_the_epsilon_rather_than_dividing_by_zero() {
    let normalizer = Normalizer::new(
        &features(&[("observation.state", FeatureType::State, 1)]),
        &mean_std_map(),
        &stats(r#"{"observation.state": {"mean": [3.0], "std": [0.0]}}"#),
    )
    .unwrap();
    let out = normalizer.normalize("observation.state", &[4.0]).unwrap();
    assert!(out[0].is_finite());
    assert_eq!(out[0], 1.0f32 / NORMALIZATION_EPS as f32);
}

#[test]
fn missing_mean_or_std_is_the_upstream_value_error() {
    let error = Normalizer::new(
        &features(&[("action", FeatureType::Action, 2)]),
        &mean_std_map(),
        &stats(r#"{"action": {"mean": [0.0, 0.0]}}"#),
    )
    .unwrap_err();
    assert_eq!(
        error,
        NormalizeError::MissingStatistics {
            key: "action".into(),
            mode: NormalizationMode::MeanStd
        }
    );
    assert!(
        error
            .to_string()
            .contains("MEAN_STD normalization mode requires mean and std stats"),
        "message drifted from upstream: {error}"
    );
}

// ---------------------------------------------------------------------------
// The other three numeric modes
// ---------------------------------------------------------------------------

#[test]
fn min_max_maps_the_range_onto_minus_one_to_one() {
    let map = IndexMap::from([(
        FeatureType::Action.as_str().to_owned(),
        NormalizationMode::MinMax,
    )]);
    let normalizer = Normalizer::new(
        &features(&[("action", FeatureType::Action, 1)]),
        &map,
        &stats(r#"{"action": {"min": [-2.0], "max": [2.0]}}"#),
    )
    .unwrap();
    assert_eq!(normalizer.normalize("action", &[-2.0]).unwrap(), vec![-1.0]);
    assert_eq!(normalizer.normalize("action", &[0.0]).unwrap(), vec![0.0]);
    assert_eq!(normalizer.normalize("action", &[2.0]).unwrap(), vec![1.0]);
    assert_eq!(normalizer.unnormalize("action", &[1.0]).unwrap(), vec![2.0]);
}

#[test]
fn min_max_substitutes_the_epsilon_for_a_zero_width_range_so_the_minimum_maps_to_minus_one() {
    let map = IndexMap::from([(
        FeatureType::Action.as_str().to_owned(),
        NormalizationMode::MinMax,
    )]);
    let normalizer = Normalizer::new(
        &features(&[("action", FeatureType::Action, 1)]),
        &map,
        &stats(r#"{"action": {"min": [5.0], "max": [5.0]}}"#),
    )
    .unwrap();
    assert_eq!(normalizer.normalize("action", &[5.0]).unwrap(), vec![-1.0]);
}

#[test]
fn the_quantile_modes_use_their_own_pairs_of_statistics() {
    for (mode, low, high) in [
        (NormalizationMode::Quantiles, "q01", "q99"),
        (NormalizationMode::Quantile10, "q10", "q90"),
    ] {
        let map = IndexMap::from([(FeatureType::Action.as_str().to_owned(), mode)]);
        let normalizer = Normalizer::new(
            &features(&[("action", FeatureType::Action, 1)]),
            &map,
            &stats(&format!(
                r#"{{"action": {{"{low}": [-4.0], "{high}": [4.0]}}}}"#
            )),
        )
        .unwrap();
        assert_eq!(normalizer.normalize("action", &[-4.0]).unwrap(), vec![-1.0]);
        assert_eq!(normalizer.normalize("action", &[4.0]).unwrap(), vec![1.0]);
        assert_eq!(normalizer.unnormalize("action", &[0.0]).unwrap(), vec![0.0]);

        let missing = Normalizer::new(
            &features(&[("action", FeatureType::Action, 1)]),
            &map,
            &stats(r#"{"action": {"mean": [0.0]}}"#),
        )
        .unwrap_err();
        assert_eq!(
            missing,
            NormalizeError::MissingStatistics {
                key: "action".into(),
                mode
            }
        );
    }
}

// ---------------------------------------------------------------------------
// Pass-through and refusal rules
// ---------------------------------------------------------------------------

#[test]
fn identity_leaves_values_alone_and_needs_no_statistics() {
    let map = IndexMap::from([(
        FeatureType::Action.as_str().to_owned(),
        NormalizationMode::Identity,
    )]);
    let normalizer = Normalizer::new(
        &features(&[("action", FeatureType::Action, 2)]),
        &map,
        &stats("{}"),
    )
    .unwrap();
    assert_eq!(
        normalizer.normalize("action", &[1.5, -2.5]).unwrap(),
        vec![1.5, -2.5]
    );
    assert_eq!(normalizer.mode("action"), None);
}

#[test]
fn a_feature_type_absent_from_the_map_defaults_to_identity() {
    // `self.norm_map.get(feature_type, NormalizationMode.IDENTITY)`.
    let normalizer = Normalizer::new(
        &features(&[("observation.state", FeatureType::State, 2)]),
        &IndexMap::new(),
        &fixture_stats(),
    )
    .unwrap();
    assert_eq!(
        normalizer
            .normalize("observation.state", &[9.0, 9.0])
            .unwrap(),
        vec![9.0, 9.0]
    );
}

#[test]
fn a_feature_without_statistics_passes_through_untouched() {
    // `if norm_mode == IDENTITY or key not in self._tensor_stats: return tensor`.
    let normalizer = Normalizer::new(
        &features(&[("observation.state", FeatureType::State, 2)]),
        &mean_std_map(),
        &stats("{}"),
    )
    .unwrap();
    assert_eq!(
        normalizer
            .normalize("observation.state", &[9.0, 8.0])
            .unwrap(),
        vec![9.0, 8.0]
    );
    assert_eq!(normalizer.mode("observation.state"), None);
}

#[test]
fn an_unregistered_key_passes_through_untouched() {
    let normalizer = fixture_normalizer();
    assert_eq!(normalizer.normalize("reward", &[1.0]).unwrap(), vec![1.0]);
}

#[test]
fn a_value_of_the_wrong_width_is_refused_rather_than_broadcast() {
    // Torch would broadcast a (1,) statistic over a (3,) tensor; the feature's
    // declared shape is known here, so a mismatch is a bug, not a broadcast.
    let normalizer = fixture_normalizer();
    let error = normalizer
        .normalize("action", &[1.0, 2.0, 3.0])
        .unwrap_err();
    assert_eq!(
        error,
        NormalizeError::WidthMismatch {
            key: "action".into(),
            expected: 2,
            found: 3
        }
    );
}

#[test]
fn the_statistics_width_has_to_match_the_declared_feature_width() {
    let error = Normalizer::new(
        &features(&[("action", FeatureType::Action, 2)]),
        &mean_std_map(),
        &stats(r#"{"action": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}}"#),
    )
    .unwrap_err();
    assert_eq!(
        error,
        NormalizeError::StatisticsWidthMismatch {
            key: "action".into(),
            statistic: "mean".into(),
            expected: 2,
            found: 3
        }
    );
}

#[test]
fn the_registered_mode_is_reported_for_every_normalized_feature() {
    let normalizer = fixture_normalizer();
    for key in [
        "observation.state",
        "observation.environment_state",
        "action",
    ] {
        assert_eq!(normalizer.mode(key), Some(NormalizationMode::MeanStd));
    }
}
