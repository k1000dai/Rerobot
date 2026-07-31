//! Behaviour parity tests for `lerobot/datasets/io_utils.py`'s `load_stats` and
//! `cast_stats_to_numpy` at commit f37be3edbee60f3a09a5183788b91eb19f0c07d1,
//! against the `meta/stats.json` written by upstream itself.
//!
//! Each test works inside its own directory under the system temporary
//! directory and removes it afterwards.

use rerobot_core::dataset::stats::{load_stats, stats_from_value, DatasetStats, StatsError};
use rerobot_core::dataset::STATS_PATH;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rerobot-dataset-stats-{}-{label}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(path.join("meta")).expect("cannot create the test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write_stats(&self, text: &str) {
        std::fs::write(self.0.join(STATS_PATH), text).expect("cannot write stats.json");
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The `observation.state` and `action` blocks of the committed fixture's
/// `meta/stats.json`, written by upstream `LeRobotDataset.save_episode`.
const FIXTURE_STATS: &str = r#"{
    "observation.state": {
        "min": [0.0, 0.0],
        "max": [1.0, 1.0],
        "mean": [0.4375, 0.5625],
        "std": [0.36975499987602234, 0.36975499987602234],
        "count": [4],
        "q01": [-1.000000013351432e-10, -1.000000013351432e-10]
    },
    "action": {
        "min": [-0.5, -0.5],
        "max": [0.5, 0.5],
        "mean": [0.0625, -0.0625],
        "std": [0.36975499987602234, 0.36975499987602234],
        "count": [4]
    }
}"#;

fn fixture() -> DatasetStats {
    stats_from_value(&rerobot_core::dataset::json::loads(FIXTURE_STATS).unwrap()).unwrap()
}

// ---------------------------------------------------------------------------
// `load_stats`
// ---------------------------------------------------------------------------

#[test]
fn an_absent_stats_file_is_none_not_an_error() {
    let dir = TempDir::new("absent");
    assert_eq!(load_stats(dir.path()), Ok(None));
}

#[test]
fn stats_are_read_from_the_upstream_path() {
    let dir = TempDir::new("present");
    dir.write_stats(FIXTURE_STATS);
    let stats = load_stats(dir.path()).unwrap().expect("stats.json exists");
    assert_eq!(
        stats.keys().collect::<Vec<_>>(),
        vec!["observation.state", "action"]
    );
}

#[test]
fn a_malformed_stats_file_is_a_parse_error_naming_the_path() {
    let dir = TempDir::new("malformed");
    dir.write_stats("{not json");
    let error = load_stats(dir.path()).unwrap_err();
    assert!(
        matches!(error, StatsError::Parse { .. }),
        "unexpected error: {error}"
    );
    assert!(
        error.to_string().contains("stats.json"),
        "message does not name the file: {error}"
    );
}

// ---------------------------------------------------------------------------
// The value domain
// ---------------------------------------------------------------------------

#[test]
fn each_feature_carries_the_statistics_upstream_writes() {
    let stats = fixture();
    let state = stats
        .get("observation.state")
        .expect("the feature is present");
    assert_eq!(state.mean(), Some(&[0.4375, 0.5625][..]));
    assert_eq!(state.min(), Some(&[0.0, 0.0][..]));
    assert_eq!(state.max(), Some(&[1.0, 1.0][..]));
    assert_eq!(
        state.std(),
        Some(&[0.36975499987602234, 0.36975499987602234][..])
    );
    assert_eq!(state.get("count"), Some(&[4.0][..]));
    assert_eq!(state.get("q01"), Some(&[-1.000000013351432e-10; 2][..]));
    assert_eq!(state.get("q99"), None);
}

#[test]
fn insertion_order_is_preserved_for_features_and_for_their_statistics() {
    let stats = fixture();
    assert_eq!(
        stats
            .get("observation.state")
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        vec!["min", "max", "mean", "std", "count", "q01"]
    );
}

#[test]
fn a_scalar_statistic_becomes_a_one_element_vector_like_atleast_1d() {
    let value =
        rerobot_core::dataset::json::loads(r#"{"reward": {"mean": 0.5, "count": 4}}"#).unwrap();
    let stats = stats_from_value(&value).unwrap();
    let reward = stats.get("reward").unwrap();
    assert_eq!(reward.mean(), Some(&[0.5][..]));
    assert_eq!(reward.get("count"), Some(&[4.0][..]));
}

#[test]
fn an_unknown_feature_is_absent_rather_than_defaulted() {
    let stats = fixture();
    assert!(stats.get("observation.images.top").is_none());
}

// ---------------------------------------------------------------------------
// The boundaries this slice declares
// ---------------------------------------------------------------------------

#[test]
fn nested_image_statistics_are_refused_instead_of_being_silently_flattened() {
    // A camera feature's stats are shape (3, 1, 1) upstream. This slice is
    // state-only and says so rather than reading them as a flat vector.
    let value = rerobot_core::dataset::json::loads(
        r#"{"observation.images.top": {"mean": [[[0.5]], [[0.4]], [[0.3]]]}}"#,
    )
    .unwrap();
    let error = stats_from_value(&value).unwrap_err();
    assert_eq!(
        error,
        StatsError::NestedStatistic {
            feature: "observation.images.top".into(),
            statistic: "mean".into()
        }
    );
}

#[test]
fn a_non_numeric_statistic_is_refused() {
    let value = rerobot_core::dataset::json::loads(r#"{"action": {"mean": ["nope"]}}"#).unwrap();
    let error = stats_from_value(&value).unwrap_err();
    assert_eq!(
        error,
        StatsError::NotANumber {
            feature: "action".into(),
            statistic: "mean".into(),
            found: "str".into()
        }
    );
}

#[test]
fn the_document_has_to_be_an_object_of_objects() {
    let error = stats_from_value(&rerobot_core::dataset::json::loads("[]").unwrap()).unwrap_err();
    assert_eq!(
        error,
        StatsError::NotAnObject {
            found: "list".into()
        }
    );

    let error = stats_from_value(&rerobot_core::dataset::json::loads(r#"{"action": 1}"#).unwrap())
        .unwrap_err();
    assert_eq!(
        error,
        StatsError::FeatureNotAnObject {
            feature: "action".into(),
            found: "int".into()
        }
    );
}

#[test]
fn non_finite_statistics_survive_because_cpython_writes_them() {
    // `json.dump` emits the bare `NaN` / `Infinity` tokens, and a degenerate
    // episode can produce them. Reading must not turn that into a parse error.
    let value =
        rerobot_core::dataset::json::loads(r#"{"action": {"std": [NaN, Infinity, -Infinity]}}"#)
            .unwrap();
    let stats = stats_from_value(&value).unwrap();
    let std = stats.get("action").unwrap().std().unwrap();
    assert!(std[0].is_nan());
    assert_eq!(std[1], f64::INFINITY);
    assert_eq!(std[2], f64::NEG_INFINITY);
}
