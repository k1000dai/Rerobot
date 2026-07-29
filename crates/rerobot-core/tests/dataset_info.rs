//! Behaviour parity tests for `DatasetInfo` and the `meta/info.json` constants.
//!
//! Expectations come from running upstream's own `DatasetInfo` source under
//! CPython 3.12.13 — the dataclass and its constant block extracted verbatim
//! from `src/lerobot/datasets/utils.py` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1 — not from reading it.

use indexmap::IndexMap;
use num_bigint::BigInt;
use rerobot_core::dataset::info::{
    python_str_repr, unknown_fields_warning, DatasetInfo, DatasetInfoError, Feature,
};
use rerobot_core::dataset::json::{dumps_pretty, loads, JsonLike, JsonObject};
use rerobot_core::dataset::{
    CHUNK_FILE_PATTERN, DATA_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH, DEFAULT_DEPTH_PATH, DEFAULT_EPISODES_PATH, DEFAULT_IMAGE_PATH,
    DEFAULT_TASKS_PATH, DEFAULT_VIDEO_FILE_SIZE_IN_MB, DEFAULT_VIDEO_PATH, DEPTH_FILE_PATTERN,
    EPISODES_DIR, IMAGE_FILE_PATTERN, INFO_PATH, STATS_PATH, VIDEO_DIR,
};
use std::str::FromStr;
use std::sync::{Mutex, Once};

struct ThreadCaptureLogger;

static LOGGER: ThreadCaptureLogger = ThreadCaptureLogger;
static INSTALL_LOGGER: Once = Once::new();
static WARNINGS: Mutex<Vec<(std::thread::ThreadId, String)>> = Mutex::new(Vec::new());

impl log::Log for ThreadCaptureLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            WARNINGS
                .lock()
                .unwrap()
                .push((std::thread::current().id(), record.args().to_string()));
        }
    }

    fn flush(&self) {}
}

fn install_capture_logger() {
    INSTALL_LOGGER.call_once(|| {
        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Warn);
    });
}

fn features(pairs: &[(&str, &[(&str, JsonLike)])]) -> IndexMap<String, Feature> {
    pairs
        .iter()
        .map(|(name, entries)| {
            let feature: Feature = entries
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect();
            ((*name).to_string(), feature)
        })
        .collect()
}

fn ints(values: &[i64]) -> JsonLike {
    JsonLike::Array(
        values
            .iter()
            .map(|v| JsonLike::Int(BigInt::from(*v)))
            .collect(),
    )
}

fn raw(text: &str) -> JsonObject {
    loads(text).unwrap().as_object().unwrap().clone()
}

fn minimal() -> DatasetInfo {
    DatasetInfo::new("v3.0", 30, IndexMap::new()).unwrap()
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn the_path_constants_are_upstreams_byte_for_byte() {
    assert_eq!(INFO_PATH, "meta/info.json");
    assert_eq!(STATS_PATH, "meta/stats.json");
    assert_eq!(EPISODES_DIR, "meta/episodes");
    assert_eq!(DATA_DIR, "data");
    assert_eq!(VIDEO_DIR, "videos");
    assert_eq!(
        CHUNK_FILE_PATTERN,
        "chunk-{chunk_index:03d}/file-{file_index:03d}"
    );
    assert_eq!(IMAGE_FILE_PATTERN, "frame-{frame_index:06d}.png");
    assert_eq!(DEPTH_FILE_PATTERN, "frame-{frame_index:06d}.tiff");
    assert_eq!(DEFAULT_TASKS_PATH, "meta/tasks.parquet");
}

#[test]
fn the_composed_path_templates_match_the_concatenations_upstream_writes() {
    assert_eq!(
        DEFAULT_EPISODES_PATH,
        format!("{EPISODES_DIR}/{CHUNK_FILE_PATTERN}.parquet")
    );
    assert_eq!(
        DEFAULT_DATA_PATH,
        format!("{DATA_DIR}/{CHUNK_FILE_PATTERN}.parquet")
    );
    assert_eq!(
        DEFAULT_VIDEO_PATH,
        format!("{VIDEO_DIR}/{{video_key}}/{CHUNK_FILE_PATTERN}.mp4")
    );
    assert_eq!(
        DEFAULT_IMAGE_PATH,
        format!("images/{{image_key}}/episode-{{episode_index:06d}}/{IMAGE_FILE_PATTERN}")
    );
    assert_eq!(
        DEFAULT_DEPTH_PATH,
        format!("images/{{image_key}}/episode-{{episode_index:06d}}/{DEPTH_FILE_PATTERN}")
    );
}

#[test]
fn the_three_size_constants_are_upstreams_values() {
    assert_eq!(DEFAULT_CHUNK_SIZE, 1000);
    assert_eq!(DEFAULT_DATA_FILE_SIZE_IN_MB, 100);
    assert_eq!(DEFAULT_VIDEO_FILE_SIZE_IN_MB, 200);
}

// ---------------------------------------------------------------------------
// Construction and defaults
// ---------------------------------------------------------------------------

#[test]
fn the_field_names_are_upstreams_dataclass_order() {
    assert_eq!(
        DatasetInfo::FIELD_NAMES,
        [
            "codebase_version",
            "fps",
            "features",
            "total_episodes",
            "total_frames",
            "total_tasks",
            "chunks_size",
            "data_files_size_in_mb",
            "video_files_size_in_mb",
            "data_path",
            "video_path",
            "robot_type",
            "splits",
            "tools",
        ]
    );
}

#[test]
fn the_eleven_defaulted_fields_take_upstreams_defaults() {
    let info = minimal();
    assert_eq!(info.total_episodes, BigInt::from(0));
    assert_eq!(info.total_frames, BigInt::from(0));
    assert_eq!(info.total_tasks, BigInt::from(0));
    assert_eq!(info.chunks_size, BigInt::from(1000));
    assert_eq!(info.data_files_size_in_mb, BigInt::from(100));
    assert_eq!(info.video_files_size_in_mb, BigInt::from(200));
    assert_eq!(info.data_path, DEFAULT_DATA_PATH);
    assert_eq!(info.video_path.as_deref(), Some(DEFAULT_VIDEO_PATH));
    assert_eq!(info.robot_type, None);
    assert!(info.splits.is_empty());
    assert_eq!(info.tools, None);
}

#[test]
fn to_dict_writes_every_field_in_declaration_order_with_tools_dropped() {
    let info = DatasetInfo::new(
        "v3.0",
        30,
        features(&[(
            "a",
            &[
                ("dtype", JsonLike::Str("float32".to_string())),
                ("shape", ints(&[1, 2])),
                ("names", JsonLike::Null),
            ],
        )]),
    )
    .unwrap();

    // Byte-identical to `json.dumps(info.to_dict(), indent=4, ensure_ascii=False)`.
    assert_eq!(
        dumps_pretty(&JsonLike::Object(info.to_dict())),
        r#"{
    "codebase_version": "v3.0",
    "fps": 30,
    "features": {
        "a": {
            "dtype": "float32",
            "shape": [
                1,
                2
            ],
            "names": null
        }
    },
    "total_episodes": 0,
    "total_frames": 0,
    "total_tasks": 0,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 200,
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "robot_type": null,
    "splits": {}
}"#
    );
}

#[test]
fn an_unset_video_path_is_written_as_null_but_an_unset_tools_key_is_absent() {
    let mut info = minimal();
    info.video_path = None;
    info.robot_type = Some("so101".to_string());
    info.splits.insert("train".to_string(), "0:100".to_string());
    let dict = info.to_dict();
    assert_eq!(dict["video_path"], JsonLike::Null);
    assert_eq!(dict["robot_type"], JsonLike::Str("so101".to_string()));
    assert!(!dict.contains_key("tools"));
}

#[test]
fn an_empty_tools_list_is_kept_because_only_none_means_undeclared() {
    let mut info = minimal();
    info.tools = Some(vec![]);
    assert_eq!(info.to_dict()["tools"], JsonLike::Array(vec![]));
}

#[test]
fn a_declared_tools_list_survives_to_dict_unchanged() {
    let tool = raw(r#"{"type": "function", "function": {"name": "say"}}"#);
    let mut info = minimal();
    info.tools = Some(vec![tool.clone()]);
    assert_eq!(
        info.to_dict()["tools"],
        JsonLike::Array(vec![JsonLike::Object(tool)])
    );
}

#[test]
fn to_dict_hands_back_an_independent_dict_each_time() {
    // `dataclasses.asdict` deep-copies, so mutating the result cannot reach
    // back into the info. An owned Rust return has the same property.
    let info =
        DatasetInfo::new("v3.0", 30, features(&[("a", &[("shape", ints(&[1, 2]))])])).unwrap();
    let mut first = info.to_dict();
    let second = info.to_dict();
    assert_eq!(first, second);
    first.insert(
        "codebase_version".to_string(),
        JsonLike::Str("mutated".to_string()),
    );
    assert_eq!(info.to_dict(), second);
}

// ---------------------------------------------------------------------------
// `__post_init__`: shape coercion
// ---------------------------------------------------------------------------

#[test]
fn a_list_valued_feature_shape_becomes_a_tuple_in_memory() {
    let info =
        DatasetInfo::new("v3.0", 30, features(&[("a", &[("shape", ints(&[1, 2]))])])).unwrap();
    assert_eq!(
        info.features["a"]["shape"],
        JsonLike::Tuple(vec![
            JsonLike::Int(BigInt::from(1)),
            JsonLike::Int(BigInt::from(2))
        ])
    );
    // And is therefore *not* equal to the list it came from, exactly as
    // `info.features["a"]["shape"] == [1, 2]` is `False` in Python.
    assert_ne!(info.features["a"]["shape"], ints(&[1, 2]));
}

#[test]
fn only_the_top_level_shape_key_is_coerced() {
    // Observed upstream: outer `shape` is a tuple, `names` stays a list, and a
    // `shape` nested inside another dict stays a list.
    let inner = raw(r#"{"shape": [9]}"#);
    let info = DatasetInfo::new(
        "v3.0",
        30,
        features(&[(
            "a",
            &[
                ("shape", ints(&[1, 2])),
                (
                    "names",
                    JsonLike::Array(vec![JsonLike::Str("x".to_string())]),
                ),
                ("info", JsonLike::Object(inner)),
            ],
        )]),
    )
    .unwrap();
    assert!(matches!(info.features["a"]["shape"], JsonLike::Tuple(_)));
    assert!(matches!(info.features["a"]["names"], JsonLike::Array(_)));
    let nested = info.features["a"]["info"].as_object().unwrap();
    assert!(matches!(nested["shape"], JsonLike::Array(_)));
}

#[test]
fn a_shape_that_is_not_a_list_is_left_exactly_as_it_is() {
    let info = DatasetInfo::new(
        "v3.0",
        30,
        features(&[
            ("b", &[("shape", JsonLike::Str("xy".to_string()))]),
            ("c", &[("noshape", JsonLike::Int(BigInt::from(1)))]),
        ]),
    )
    .unwrap();
    assert_eq!(info.features["b"]["shape"], JsonLike::Str("xy".to_string()));
    assert!(!info.features["c"].contains_key("shape"));
}

#[test]
fn an_empty_shape_list_becomes_an_empty_tuple_and_returns_to_an_empty_list() {
    let info = DatasetInfo::new("v3.0", 30, features(&[("d", &[("shape", ints(&[]))])])).unwrap();
    assert_eq!(info.features["d"]["shape"], JsonLike::Tuple(vec![]));
    let dict = info.to_dict();
    let out = dict["features"].as_object().unwrap()["d"]
        .as_object()
        .unwrap();
    assert_eq!(out["shape"], JsonLike::Array(vec![]));
}

#[test]
fn to_dict_turns_the_tuple_shape_back_into_a_list() {
    let info =
        DatasetInfo::new("v3.0", 30, features(&[("a", &[("shape", ints(&[1, 2]))])])).unwrap();
    let dict = info.to_dict();
    let out = dict["features"].as_object().unwrap()["a"]
        .as_object()
        .unwrap();
    assert_eq!(out["shape"], ints(&[1, 2]));
}

// ---------------------------------------------------------------------------
// `__post_init__`: validation
// ---------------------------------------------------------------------------

#[test]
fn each_of_the_four_counters_is_rejected_when_not_positive_with_upstreams_message() {
    for (field, message) in [
        ("fps", "fps must be positive, got 0"),
        ("chunks_size", "chunks_size must be positive, got 0"),
        (
            "data_files_size_in_mb",
            "data_files_size_in_mb must be positive, got 0",
        ),
        (
            "video_files_size_in_mb",
            "video_files_size_in_mb must be positive, got 0",
        ),
    ] {
        let mut info = minimal();
        match field {
            "fps" => info.fps = BigInt::from(0),
            "chunks_size" => info.chunks_size = BigInt::from(0),
            "data_files_size_in_mb" => info.data_files_size_in_mb = BigInt::from(0),
            _ => info.video_files_size_in_mb = BigInt::from(0),
        }
        let error = info.post_init().unwrap_err();
        assert_eq!(
            error,
            DatasetInfoError::NotPositive {
                field,
                value: BigInt::from(0)
            }
        );
        assert_eq!(error.to_string(), message);
    }
}

#[test]
fn a_negative_counter_is_reported_with_its_own_value() {
    let error = DatasetInfo::new("v3.0", -1, IndexMap::new()).unwrap_err();
    assert_eq!(error.to_string(), "fps must be positive, got -1");
}

#[test]
fn fps_is_checked_before_the_other_three() {
    let mut info = minimal();
    info.fps = BigInt::from(0);
    info.chunks_size = BigInt::from(0);
    assert_eq!(
        info.post_init().unwrap_err().to_string(),
        "fps must be positive, got 0"
    );
}

#[test]
fn the_shape_coercion_happens_before_validation_can_reject_the_info() {
    // Upstream coerces every shape and only then checks `fps`, so a rejected
    // info has already had its features rewritten. Rerobot owns its copy, so
    // the rewrite is not observable — which is the point of this test.
    let mut info = minimal();
    info.features = features(&[("a", &[("shape", ints(&[1, 2]))])]);
    info.fps = BigInt::from(0);
    assert!(info.post_init().is_err());
    assert!(matches!(info.features["a"]["shape"], JsonLike::Tuple(_)));
}

#[test]
fn an_unbounded_counter_is_neither_narrowed_nor_rejected() {
    let huge = BigInt::from(2).pow(200);
    let mut info = DatasetInfo::new("v3.0", huge.clone(), IndexMap::new()).unwrap();
    info.total_frames = huge.clone();
    info.chunks_size = huge.clone();
    info.post_init().unwrap();
    assert_eq!(info.fps, huge);
    assert_eq!(
        dumps_pretty(&info.to_dict()["fps"]),
        "1606938044258990275541962092341162602522202993782792835301376"
    );
}

#[test]
fn an_unbounded_shape_dimension_survives_the_tuple_round_trip() {
    let huge = BigInt::from_str("340282366920938463463374607431768211457").unwrap();
    let info = DatasetInfo::new(
        "v3.0",
        30,
        features(&[(
            "a",
            &[("shape", JsonLike::Array(vec![JsonLike::Int(huge.clone())]))],
        )]),
    )
    .unwrap();
    assert_eq!(
        info.features["a"]["shape"],
        JsonLike::Tuple(vec![JsonLike::Int(huge.clone())])
    );
    let dict = info.to_dict();
    let out = dict["features"].as_object().unwrap()["a"]
        .as_object()
        .unwrap();
    assert_eq!(out["shape"], JsonLike::Array(vec![JsonLike::Int(huge)]));
}

#[test]
fn assigning_a_field_does_not_re_run_post_init_just_as_a_dataclass_does_not() {
    let mut info = minimal();
    info.fps = BigInt::from(0);
    // No validation happens on assignment, so `to_dict` still works and writes
    // the invalid value — upstream behaves identically.
    assert_eq!(info.to_dict()["fps"], JsonLike::Int(BigInt::from(0)));
}

// ---------------------------------------------------------------------------
// `from_dict`
// ---------------------------------------------------------------------------

#[test]
fn from_dict_reads_the_three_required_fields_and_defaults_the_rest() {
    let info = DatasetInfo::from_dict(&raw(
        r#"{"codebase_version": "v3.1", "fps": 30, "features": {}}"#,
    ))
    .unwrap();
    assert_eq!(info.codebase_version, "v3.1");
    assert_eq!(info.fps, BigInt::from(30));
    assert_eq!(info.chunks_size, BigInt::from(1000));
    assert_eq!(info.tools, None);
}

#[test]
fn from_dict_reports_every_missing_required_field_with_cpythons_wording() {
    for (text, message) in [
        (
            "{}",
            "DatasetInfo.__init__() missing 3 required positional arguments: \
             'codebase_version', 'fps', and 'features'",
        ),
        (
            r#"{"fps": 30}"#,
            "DatasetInfo.__init__() missing 2 required positional arguments: \
             'codebase_version' and 'features'",
        ),
        (
            r#"{"codebase_version": "v"}"#,
            "DatasetInfo.__init__() missing 2 required positional arguments: 'fps' and 'features'",
        ),
        (
            r#"{"codebase_version": "v", "fps": 30}"#,
            "DatasetInfo.__init__() missing 1 required positional argument: 'features'",
        ),
    ] {
        let error = DatasetInfo::from_dict(&raw(text)).unwrap_err();
        assert_eq!(error.to_string(), message, "for {text}");
    }
}

#[test]
fn from_dict_ignores_unknown_keys_but_they_can_be_asked_for_and_come_back_sorted() {
    let data = raw(r#"{"codebase_version": "v3.0", "fps": 30, "features": {},
             "total_videos": 3, "aaa": 1, "Zed": 2, "é": 3}"#);
    assert!(DatasetInfo::from_dict(&data).is_ok());
    // Python's `sorted()` orders by code point, so uppercase precedes
    // lowercase and non-ASCII sorts last.
    assert_eq!(
        DatasetInfo::unknown_fields(&data),
        vec!["Zed", "aaa", "total_videos", "é"]
    );
}

#[test]
fn a_document_with_only_known_keys_has_no_unknown_fields_and_no_warning() {
    let data = raw(r#"{"codebase_version": "v3.0", "fps": 30, "features": {}}"#);
    assert!(DatasetInfo::unknown_fields(&data).is_empty());
    assert_eq!(unknown_fields_warning(&[]), None);
}

#[test]
fn the_unknown_field_warning_text_is_upstreams() {
    assert_eq!(
        unknown_fields_warning(&["total_videos".to_string()]).unwrap(),
        "Unknown fields in DatasetInfo: ['total_videos']. These will be ignored."
    );
    assert_eq!(
        unknown_fields_warning(&[
            "Zed".to_string(),
            "aaa".to_string(),
            "it's".to_string(),
            "total_videos".to_string(),
            "é".to_string(),
        ])
        .unwrap(),
        "Unknown fields in DatasetInfo: ['Zed', 'aaa', \"it's\", 'total_videos', 'é']. \
         These will be ignored."
    );
}

#[test]
fn from_dict_logs_unknown_fields_before_a_later_construction_error() {
    install_capture_logger();
    let thread = std::thread::current().id();
    WARNINGS
        .lock()
        .unwrap()
        .retain(|(owner, _)| *owner != thread);

    let error = DatasetInfo::from_dict(&raw(r#"{"unknown": 1}"#)).unwrap_err();
    assert!(matches!(error, DatasetInfoError::MissingRequiredFields(_)));
    let warnings: Vec<String> = WARNINGS
        .lock()
        .unwrap()
        .iter()
        .filter(|(owner, _)| *owner == thread)
        .map(|(_, message)| message.clone())
        .collect();
    assert_eq!(
        warnings,
        ["Unknown fields in DatasetInfo: ['unknown']. These will be ignored."]
    );
}

#[test]
fn python_str_repr_picks_the_quote_python_picks() {
    assert_eq!(python_str_repr("plain"), "'plain'");
    assert_eq!(python_str_repr("it's"), "\"it's\"");
    assert_eq!(python_str_repr("say \"hi\""), "'say \"hi\"'");
    assert_eq!(python_str_repr("both'\""), "'both\\'\"'");
    assert_eq!(python_str_repr("é"), "'é'");
    assert_eq!(python_str_repr("tab\there"), "'tab\\there'");
    assert_eq!(python_str_repr("nl\nhere"), "'nl\\nhere'");
    assert_eq!(python_str_repr("\u{85}"), "'\\x85'");
    assert_eq!(python_str_repr("\u{a0}"), "'\\xa0'");
    assert_eq!(python_str_repr("\u{200b}"), "'\\u200b'");
    assert_eq!(python_str_repr("\u{2028}"), "'\\u2028'");
}

#[test]
fn from_dict_reads_a_null_optional_as_unset_and_a_missing_one_as_its_default() {
    let info = DatasetInfo::from_dict(&raw(r#"{"codebase_version": "v", "fps": 1, "features": {},
             "video_path": null, "robot_type": null}"#))
    .unwrap();
    assert_eq!(info.video_path, None);
    assert_eq!(info.robot_type, None);

    let info = DatasetInfo::from_dict(&raw(
        r#"{"codebase_version": "v", "fps": 1, "features": {}}"#,
    ))
    .unwrap();
    assert_eq!(info.video_path.as_deref(), Some(DEFAULT_VIDEO_PATH));
}

#[test]
fn from_dict_keeps_extra_feature_entries_and_their_order() {
    let info = DatasetInfo::from_dict(&raw(r#"{"codebase_version": "v", "fps": 1,
             "features": {"z": {"zz": 1, "dtype": "float32", "shape": [2], "aa": null},
                          "a": {"shape": [1]}}}"#))
    .unwrap();
    assert_eq!(info.features.keys().collect::<Vec<_>>(), vec!["z", "a"]);
    assert_eq!(
        info.features["z"].keys().collect::<Vec<_>>(),
        vec!["zz", "dtype", "shape", "aa"]
    );
}

#[test]
fn from_dict_rejects_a_value_outside_the_typed_domain_naming_both_types() {
    for (text, field, expected, found) in [
        (
            r#"{"codebase_version": 5, "fps": 1, "features": {}}"#,
            "codebase_version",
            "str",
            "int",
        ),
        (
            r#"{"codebase_version": "v", "fps": "30", "features": {}}"#,
            "fps",
            "int",
            "str",
        ),
        (
            r#"{"codebase_version": "v", "fps": 30.5, "features": {}}"#,
            "fps",
            "int",
            "float",
        ),
        (
            r#"{"codebase_version": "v", "fps": true, "features": {}}"#,
            "fps",
            "int",
            "bool",
        ),
        (
            r#"{"codebase_version": "v", "fps": null, "features": {}}"#,
            "fps",
            "int",
            "NoneType",
        ),
        (
            r#"{"codebase_version": "v", "fps": 1, "features": []}"#,
            "features",
            "dict",
            "list",
        ),
        (
            r#"{"codebase_version": "v", "fps": 1, "features": {"a": 5}}"#,
            "features.a",
            "dict",
            "int",
        ),
        (
            r#"{"codebase_version": "v", "fps": 1, "features": {}, "splits": {"train": 1}}"#,
            "splits.train",
            "str",
            "int",
        ),
        (
            r#"{"codebase_version": "v", "fps": 1, "features": {}, "tools": [1]}"#,
            "tools.0",
            "dict",
            "int",
        ),
        (
            r#"{"codebase_version": "v", "fps": 1, "features": {}, "video_path": 7}"#,
            "video_path",
            "str",
            "int",
        ),
    ] {
        let error = DatasetInfo::from_dict(&raw(text)).unwrap_err();
        assert_eq!(
            error,
            DatasetInfoError::WrongType {
                field: field.to_string(),
                expected,
                found,
            },
            "for {text}"
        );
    }
}

#[test]
fn a_missing_required_field_is_reported_before_a_wrong_type() {
    // `cls(**data)` raises for the missing argument before `__post_init__`
    // ever runs, so that is the error a caller sees first.
    let error = DatasetInfo::from_dict(&raw(r#"{"fps": "not an int"}"#)).unwrap_err();
    assert!(matches!(error, DatasetInfoError::MissingRequiredFields(_)));
}

#[test]
fn from_dict_still_validates_the_counters() {
    let error = DatasetInfo::from_dict(&raw(
        r#"{"codebase_version": "v", "fps": 0, "features": {}}"#,
    ))
    .unwrap_err();
    assert_eq!(error.to_string(), "fps must be positive, got 0");
}

#[test]
fn post_init_counter_errors_precede_unrelated_typed_boundary_errors() {
    for (document, expected) in [
        (
            r#"{"codebase_version": "v", "fps": 0, "features": {}, "chunks_size": "bad"}"#,
            "fps must be positive, got 0",
        ),
        (
            r#"{"codebase_version": "v", "fps": 30, "features": {}, "chunks_size": 0, "data_files_size_in_mb": "bad"}"#,
            "chunks_size must be positive, got 0",
        ),
        (
            r#"{"codebase_version": "v", "fps": 30, "features": {}, "chunks_size": 1, "data_files_size_in_mb": 0, "video_files_size_in_mb": "bad"}"#,
            "data_files_size_in_mb must be positive, got 0",
        ),
        (
            r#"{"codebase_version": "v", "fps": 0, "features": {}, "splits": 7}"#,
            "fps must be positive, got 0",
        ),
    ] {
        let error = DatasetInfo::from_dict(&raw(document)).unwrap_err();
        assert_eq!(error.to_string(), expected, "document: {document}");
    }
}

#[test]
fn from_dict_then_to_dict_round_trips_a_full_document() {
    let text = r#"{
    "codebase_version": "v3.0",
    "fps": 30,
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [
                6
            ],
            "names": [
                "clé"
            ]
        }
    },
    "total_episodes": 12,
    "total_frames": 3600,
    "total_tasks": 2,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 200,
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": null,
    "robot_type": "so101",
    "splits": {
        "train": "0:12"
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "say"
            }
        }
    ]
}"#;
    let info = DatasetInfo::from_dict(&raw(text)).unwrap();
    assert_eq!(dumps_pretty(&JsonLike::Object(info.to_dict())), text);
}
