//! Behaviour parity tests for reading and writing `meta/info.json` on a local
//! filesystem, derived from `lerobot/datasets/io_utils.py` and
//! `lerobot/utils/io_utils.py` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1.
//!
//! Every test works inside a directory of its own under the system temporary
//! directory and removes it afterwards, so the suite stays runnable in
//! parallel and leaves nothing behind.

use indexmap::IndexMap;
use num_bigint::BigInt;
use rerobot_core::dataset::info::{DatasetInfo, Feature};
use rerobot_core::dataset::io::{
    info_path, load_info, load_json, write_info, write_json, LoadError, LoadInfoError,
};
use rerobot_core::dataset::json::{loads, JsonLike, MAX_JSON_INPUT_BYTES};
use rerobot_core::dataset::INFO_PATH;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

/// A unique directory that deletes itself when the test ends.
struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rerobot-dataset-io-{}-{label}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("cannot create the test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn features(name: &str, shape: &[i64]) -> IndexMap<String, Feature> {
    let mut feature = Feature::new();
    feature.insert("dtype".to_string(), JsonLike::Str("float32".to_string()));
    feature.insert(
        "shape".to_string(),
        JsonLike::Array(
            shape
                .iter()
                .map(|d| JsonLike::Int(BigInt::from(*d)))
                .collect(),
        ),
    );
    IndexMap::from([(name.to_string(), feature)])
}

fn sample() -> DatasetInfo {
    DatasetInfo::new("v3.0", 30, features("observation.state", &[6])).unwrap()
}

// ---------------------------------------------------------------------------
// `info_path`
// ---------------------------------------------------------------------------

#[test]
fn the_info_path_is_the_constant_joined_onto_the_dataset_root() {
    let root = Path::new("datasets").join("demo");
    assert_eq!(info_path(&root), root.join("meta").join("info.json"));
    // The constant keeps upstream's POSIX spelling; joining is what makes it
    // native, so the two must agree on every platform.
    assert_eq!(info_path(&root), root.join(INFO_PATH));
}

// ---------------------------------------------------------------------------
// `write_json` / `load_json`
// ---------------------------------------------------------------------------

#[test]
fn write_json_creates_every_missing_parent_directory() {
    // `fpath.parent.mkdir(exist_ok=True, parents=True)`.
    let temp = TempDir::new("mkdir");
    let path = temp.path().join("a").join("b").join("c").join("out.json");
    write_json(&loads(r#"{"a": 1}"#).unwrap(), &path).unwrap();
    assert!(path.is_file());
}

#[test]
fn write_json_writes_four_space_json_with_no_trailing_newline() {
    let temp = TempDir::new("format");
    let path = temp.path().join("out.json");
    write_json(&loads(r#"{"a": 1, "b": [2]}"#).unwrap(), &path).unwrap();
    let written = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        written,
        "{\n    \"a\": 1,\n    \"b\": [\n        2\n    ]\n}"
    );
    assert!(!written.ends_with('\n'));
}

#[test]
fn write_json_writes_utf8_without_escaping_non_ascii() {
    // `ensure_ascii=False`, so the bytes on disk are the characters.
    let temp = TempDir::new("utf8");
    let path = temp.path().join("out.json");
    let value = loads("{\"clé\": \"héllo ✅ 😀\"}").unwrap();
    write_json(&value, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        String::from_utf8(bytes).unwrap(),
        "{\n    \"clé\": \"héllo ✅ 😀\"\n}"
    );
    assert_eq!(load_json(&path).unwrap(), value);
}

#[test]
fn write_json_overwrites_an_existing_file_rather_than_appending() {
    let temp = TempDir::new("overwrite");
    let path = temp.path().join("out.json");
    write_json(
        &loads(r#"{"long": "aaaaaaaaaaaaaaaaaaaa"}"#).unwrap(),
        &path,
    )
    .unwrap();
    write_json(&loads(r#"{"a": 1}"#).unwrap(), &path).unwrap();
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "{\n    \"a\": 1\n}"
    );
}

#[test]
fn load_json_of_a_missing_file_is_an_io_error_and_not_a_panic() {
    let temp = TempDir::new("missing");
    let error = load_json(&temp.path().join("nope.json")).unwrap_err();
    let LoadError::Io { path, source } = error else {
        panic!("expected an IO error")
    };
    assert_eq!(path, temp.path().join("nope.json"));
    assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn load_json_of_a_directory_is_an_io_error_and_not_a_panic() {
    let temp = TempDir::new("isdir");
    assert!(matches!(
        load_json(temp.path()).unwrap_err(),
        LoadError::Io { .. }
    ));
}

#[test]
fn load_json_of_malformed_content_carries_cpythons_message_and_coordinates() {
    let temp = TempDir::new("malformed");
    let path = temp.path().join("bad.json");
    std::fs::write(&path, "{\"a\": 1,\n \"b\": }").unwrap();
    let LoadError::Parse { source, .. } = load_json(&path).unwrap_err() else {
        panic!("expected a parse error")
    };
    assert_eq!(source.msg, "Expecting value");
    // CPython: msg='Expecting value' pos=15 line=2 col=7
    assert_eq!((source.line, source.column, source.position), (2, 7, 15));
}

#[test]
fn load_json_of_invalid_utf8_is_an_io_error_and_not_a_panic() {
    let temp = TempDir::new("badutf8");
    let path = temp.path().join("bad.json");
    std::fs::write(&path, [0x7b, 0x22, 0xff, 0x22, 0x7d]).unwrap();
    assert!(matches!(
        load_json(&path).unwrap_err(),
        LoadError::Io { .. }
    ));
}

#[test]
fn load_json_accepts_any_top_level_value_just_as_json_load_does() {
    let temp = TempDir::new("toplevel");
    let path = temp.path().join("v.json");
    for (text, expected) in [
        (
            "[1, 2]",
            JsonLike::Array(vec![
                JsonLike::Int(BigInt::from(1)),
                JsonLike::Int(BigInt::from(2)),
            ]),
        ),
        ("null", JsonLike::Null),
        ("3", JsonLike::Int(BigInt::from(3))),
    ] {
        std::fs::write(&path, text).unwrap();
        assert_eq!(load_json(&path).unwrap(), expected, "for {text}");
    }
}

// ---------------------------------------------------------------------------
// `write_info` / `load_info`
// ---------------------------------------------------------------------------

#[test]
fn write_info_puts_the_file_at_meta_info_json_creating_the_directory() {
    let temp = TempDir::new("writeinfo");
    let root = temp.path().join("dataset");
    write_info(&sample(), &root).unwrap();
    assert!(root.join("meta").join("info.json").is_file());
}

#[test]
fn write_info_writes_exactly_what_json_dump_would_write() {
    let temp = TempDir::new("writebytes");
    let root = temp.path().join("dataset");
    write_info(&sample(), &root).unwrap();
    assert_eq!(
        std::fs::read_to_string(info_path(&root)).unwrap(),
        r#"{
    "codebase_version": "v3.0",
    "fps": 30,
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [
                6
            ]
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
fn an_info_round_trips_through_the_filesystem_unchanged() {
    let temp = TempDir::new("roundtrip");
    let root = temp.path().join("dataset");
    let mut info = sample();
    info.robot_type = Some("so101".to_string());
    info.total_frames = BigInt::from(2).pow(200);
    info.video_path = None;
    info.splits.insert("train".to_string(), "0:12".to_string());
    info.tools = Some(vec![loads(r#"{"type": "function"}"#)
        .unwrap()
        .as_object()
        .unwrap()
        .clone()]);
    info.post_init().unwrap();

    write_info(&info, &root).unwrap();
    assert_eq!(load_info(&root).unwrap(), info);
}

#[test]
fn a_non_ascii_info_round_trips_byte_for_byte() {
    let temp = TempDir::new("nonascii");
    let root = temp.path().join("dataset");
    let mut info = DatasetInfo::new("v3.0", 30, features("observation.clé", &[2])).unwrap();
    info.robot_type = Some("bras-à-café ✅".to_string());
    write_info(&info, &root).unwrap();

    let written = std::fs::read_to_string(info_path(&root)).unwrap();
    assert!(written.contains("\"observation.clé\""));
    assert!(written.contains("\"bras-à-café ✅\""));
    assert_eq!(load_info(&root).unwrap(), info);
}

#[test]
fn load_info_reads_back_the_tuple_shape_post_init_produces() {
    let temp = TempDir::new("shape");
    let root = temp.path().join("dataset");
    write_info(&sample(), &root).unwrap();
    let loaded = load_info(&root).unwrap();
    assert_eq!(
        loaded.features["observation.state"]["shape"],
        JsonLike::Tuple(vec![JsonLike::Int(BigInt::from(6))])
    );
}

#[test]
fn load_info_of_a_missing_dataset_is_an_error_and_not_a_panic() {
    let temp = TempDir::new("noinfo");
    let error = load_info(&temp.path().join("absent")).unwrap_err();
    assert!(matches!(error, LoadInfoError::Load(LoadError::Io { .. })));
}

#[test]
fn load_info_of_malformed_json_reports_the_parse_failure() {
    let temp = TempDir::new("badinfo");
    let root = temp.path().join("dataset");
    std::fs::create_dir_all(root.join("meta")).unwrap();
    std::fs::write(info_path(&root), "{not json}").unwrap();
    assert!(matches!(
        load_info(&root).unwrap_err(),
        LoadInfoError::Load(LoadError::Parse { .. })
    ));
}

#[test]
fn load_info_of_a_non_object_document_says_so() {
    let temp = TempDir::new("notobject");
    let root = temp.path().join("dataset");
    std::fs::create_dir_all(root.join("meta")).unwrap();
    std::fs::write(info_path(&root), "[1, 2]").unwrap();
    let LoadInfoError::NotAnObject { found, .. } = load_info(&root).unwrap_err() else {
        panic!("expected a NotAnObject error")
    };
    assert_eq!(found, "list");
}

#[test]
fn load_info_surfaces_a_missing_required_field_with_upstreams_message() {
    let temp = TempDir::new("incomplete");
    let root = temp.path().join("dataset");
    std::fs::create_dir_all(root.join("meta")).unwrap();
    std::fs::write(info_path(&root), r#"{"fps": 30}"#).unwrap();
    let error = load_info(&root).unwrap_err();
    assert!(matches!(error, LoadInfoError::Info(_)));
    assert_eq!(
        error.to_string(),
        "DatasetInfo.__init__() missing 2 required positional arguments: \
         'codebase_version' and 'features'"
    );
}

#[test]
fn load_info_surfaces_an_invalid_counter_with_upstreams_message() {
    let temp = TempDir::new("badfps");
    let root = temp.path().join("dataset");
    std::fs::create_dir_all(root.join("meta")).unwrap();
    std::fs::write(
        info_path(&root),
        r#"{"codebase_version": "v3.0", "fps": 0, "features": {}}"#,
    )
    .unwrap();
    assert_eq!(
        load_info(&root).unwrap_err().to_string(),
        "fps must be positive, got 0"
    );
}

#[test]
fn load_info_ignores_a_v2_era_extra_key_and_still_reports_it_on_request() {
    let temp = TempDir::new("legacy");
    let root = temp.path().join("dataset");
    std::fs::create_dir_all(root.join("meta")).unwrap();
    std::fs::write(
        info_path(&root),
        r#"{"codebase_version": "v2.1", "fps": 30, "features": {}, "total_videos": 7}"#,
    )
    .unwrap();
    let info = load_info(&root).unwrap();
    assert_eq!(info.codebase_version, "v2.1");

    // The raw document is still available for the unknown-field question,
    // which `load_info` deliberately does not answer on its own.
    let JsonLike::Object(raw) = load_json(&info_path(&root)).unwrap() else {
        panic!("expected an object")
    };
    assert_eq!(DatasetInfo::unknown_fields(&raw), vec!["total_videos"]);
}

#[test]
fn writing_into_a_path_blocked_by_a_file_is_an_error_and_not_a_panic() {
    let temp = TempDir::new("blocked");
    let blocker = temp.path().join("dataset");
    std::fs::write(&blocker, "not a directory").unwrap();
    // `meta/` cannot be created underneath a regular file.
    assert!(write_info(&sample(), &blocker).is_err());
}

#[test]
fn load_json_rejects_an_oversized_file_before_reading_it_into_memory() {
    let temp = TempDir::new("oversized");
    let path = temp.path().join("huge.json");
    let file = std::fs::File::create(&path).unwrap();
    file.set_len(MAX_JSON_INPUT_BYTES as u64 + 1).unwrap();

    let error = load_json(&path).unwrap_err();
    assert!(matches!(
        error,
        LoadError::ResourceLimit {
            limit: MAX_JSON_INPUT_BYTES,
            actual,
            ..
        } if actual == MAX_JSON_INPUT_BYTES as u64 + 1
    ));
}
