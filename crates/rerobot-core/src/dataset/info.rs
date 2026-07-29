//! Port of `lerobot.datasets.utils.DatasetInfo`, the typed representation of
//! `meta/info.json`.
//!
//! Upstream is a `@dataclass` with three required fields, eleven defaulted
//! ones, a `__post_init__` that coerces feature shapes and validates four
//! counters, and `to_dict` / `from_dict` either side of the JSON file. All of
//! that is ported. The deprecated dict-style compatibility layer
//! (`__getitem__`, `__setitem__`, `__contains__`, `get`) is **not**: it exists
//! upstream to keep un-migrated `info["key"]` call-sites working while emitting
//! a `DeprecationWarning`, and a Rust port has no such call-sites to keep
//! working.
//!
//! Every integer field is a [`BigInt`], because a Python `int` is unbounded and
//! nothing upstream clamps one. See `docs/compatibility.md` for the two places
//! where the typed port is deliberately narrower than the dataclass.

use super::json::{JsonLike, JsonObject};
use indexmap::IndexMap;
use num_bigint::BigInt;
use std::fmt;
use unicode_general_category::{get_general_category, GeneralCategory};

/// One entry of `DatasetInfo.features` — upstream's `dict[str, dict]` inner
/// `dict`, whose contents are unconstrained beyond the `shape` key.
pub type Feature = JsonObject;

/// Typed representation of `meta/info.json`.
///
/// The fields are public and in upstream's declaration order, which is the
/// order [`DatasetInfo::to_dict`] writes them in.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetInfo {
    /// `codebase_version` — required.
    pub codebase_version: String,
    /// `fps` — required, and validated positive.
    pub fps: BigInt,
    /// `features` — required.
    pub features: IndexMap<String, Feature>,
    /// `total_episodes`, defaulting to zero.
    pub total_episodes: BigInt,
    /// `total_frames`, defaulting to zero.
    pub total_frames: BigInt,
    /// `total_tasks`, defaulting to zero.
    pub total_tasks: BigInt,
    /// `chunks_size`, validated positive.
    pub chunks_size: BigInt,
    /// `data_files_size_in_mb`, validated positive.
    pub data_files_size_in_mb: BigInt,
    /// `video_files_size_in_mb`, validated positive.
    pub video_files_size_in_mb: BigInt,
    /// `data_path` template.
    pub data_path: String,
    /// `video_path` template; `None` for a dataset with no videos.
    pub video_path: Option<String>,
    /// `robot_type`.
    pub robot_type: Option<String>,
    /// `splits`.
    pub splits: IndexMap<String, String>,
    /// `tools` — OpenAI-style tool schemas. `None` means undeclared/unset and
    /// drops the key from [`DatasetInfo::to_dict`] rather than writing `null`;
    /// `Some(vec![])` explicitly declares an empty tool list.
    pub tools: Option<Vec<JsonObject>>,
}

/// Why a [`DatasetInfo`] could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum DatasetInfoError {
    /// `cls(**data)` had no value for one or more fields without a default.
    MissingRequiredFields(Vec<&'static str>),
    /// A field held a value outside this port's typed domain.
    ///
    /// Upstream's dataclass performs no type checking, so this has no
    /// counterpart there; see `docs/compatibility.md`.
    WrongType {
        /// Dotted path of the offending field, e.g. `features.state.shape`.
        field: String,
        /// The Python type the port requires.
        expected: &'static str,
        /// The Python type actually present.
        found: &'static str,
    },
    /// One of `__post_init__`'s four positivity checks failed.
    NotPositive {
        /// Field name, as it appears in upstream's message.
        field: &'static str,
        /// The rejected value.
        value: BigInt,
    },
}

impl fmt::Display for DatasetInfoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequiredFields(fields) => {
                let plural = if fields.len() == 1 { "" } else { "s" };
                write!(
                    f,
                    "DatasetInfo.__init__() missing {} required positional argument{plural}: {}",
                    fields.len(),
                    join_python_style(fields)
                )
            }
            Self::WrongType {
                field,
                expected,
                found,
            } => write!(
                f,
                "'{field}' must be {expected}, found {found} \
                 (Rerobot's typed boundary; upstream's dataclass does not check)"
            ),
            Self::NotPositive { field, value } => {
                write!(f, "{field} must be positive, got {value}")
            }
        }
    }
}

impl std::error::Error for DatasetInfoError {}

/// CPython's argument list: `'a'`, `'a' and 'b'`, `'a', 'b', and 'c'`.
fn join_python_style(fields: &[&'static str]) -> String {
    let quoted: Vec<String> = fields.iter().map(|name| format!("'{name}'")).collect();
    match quoted.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [first, second] => format!("{first} and {second}"),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
    }
}

/// A `WrongType` naming the Python type actually present.
fn wrong_type(
    field: impl Into<String>,
    expected: &'static str,
    found: &JsonLike,
) -> DatasetInfoError {
    DatasetInfoError::WrongType {
        field: field.into(),
        expected,
        found: found.type_name(),
    }
}

/// `data[key]` as a `str`, or `None` when the key is absent.
fn read_str(data: &JsonObject, key: &'static str) -> Result<Option<String>, DatasetInfoError> {
    match data.get(key) {
        None => Ok(None),
        Some(JsonLike::Str(text)) => Ok(Some(text.clone())),
        Some(other) => Err(wrong_type(key, "str", other)),
    }
}

/// `data[key]` as an `int`, or `None` when the key is absent.
///
/// A JSON `true` is rejected rather than read as `1`. Python's `bool` is an
/// `int` subclass, so upstream would accept it and write it back out as
/// `true`; carrying that through would mean every counter remembering whether
/// it had been spelled as a bool. See `docs/compatibility.md`.
fn read_int(data: &JsonObject, key: &'static str) -> Result<Option<BigInt>, DatasetInfoError> {
    match data.get(key) {
        None => Ok(None),
        Some(JsonLike::Int(value)) => Ok(Some(value.clone())),
        Some(other) => Err(wrong_type(key, "int", other)),
    }
}

/// `data[key]` as `str | None`. The outer `Option` is presence, the inner one
/// is the value, so an explicit `null` and an absent key stay distinguishable.
#[allow(clippy::option_option)]
fn read_optional_str(
    data: &JsonObject,
    key: &'static str,
) -> Result<Option<Option<String>>, DatasetInfoError> {
    match data.get(key) {
        None => Ok(None),
        Some(JsonLike::Null) => Ok(Some(None)),
        Some(JsonLike::Str(text)) => Ok(Some(Some(text.clone()))),
        Some(other) => Err(wrong_type(key, "str", other)),
    }
}

impl DatasetInfo {
    /// The 14 field names, in upstream's declaration order.
    pub const FIELD_NAMES: [&'static str; 14] = [
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
    ];

    /// Construct from the three required fields, defaulting the other eleven,
    /// then run `__post_init__`.
    pub fn new(
        codebase_version: impl Into<String>,
        fps: impl Into<BigInt>,
        features: IndexMap<String, Feature>,
    ) -> Result<Self, DatasetInfoError> {
        let mut info = Self {
            codebase_version: codebase_version.into(),
            fps: fps.into(),
            features,
            total_episodes: BigInt::from(0),
            total_frames: BigInt::from(0),
            total_tasks: BigInt::from(0),
            chunks_size: BigInt::from(super::DEFAULT_CHUNK_SIZE),
            data_files_size_in_mb: BigInt::from(super::DEFAULT_DATA_FILE_SIZE_IN_MB),
            video_files_size_in_mb: BigInt::from(super::DEFAULT_VIDEO_FILE_SIZE_IN_MB),
            data_path: super::DEFAULT_DATA_PATH.to_string(),
            video_path: Some(super::DEFAULT_VIDEO_PATH.to_string()),
            robot_type: None,
            splits: IndexMap::new(),
            tools: None,
        };
        info.post_init()?;
        Ok(info)
    }

    /// Port of `__post_init__`: coerce every list-valued feature `shape` to a
    /// tuple, then validate the four positive counters in upstream's order.
    ///
    /// Public because upstream's is reachable: assigning to a dataclass field
    /// does not re-run it, so a caller who edits `chunks_size` is in exactly
    /// the unvalidated state upstream leaves them in, and this is how they get
    /// back out of it.
    pub fn post_init(&mut self) -> Result<(), DatasetInfoError> {
        // "Coerce feature shapes from list to tuple - JSON deserialisation
        // returns lists, but the rest of the codebase expects tuples." Only the
        // feature dict\'s own `shape` key is touched; anything nested deeper
        // keeps whatever it already was.
        for feature in self.features.values_mut() {
            if let Some(shape @ JsonLike::Array(_)) = feature.get_mut("shape") {
                let JsonLike::Array(dimensions) = std::mem::replace(shape, JsonLike::Null) else {
                    unreachable!("just matched an array")
                };
                *shape = JsonLike::Tuple(dimensions);
            }
        }

        // The four checks, in upstream\'s order, so the first complaint about
        // a doubly-invalid info is the one upstream makes.
        for (field, value) in [
            ("fps", &self.fps),
            ("chunks_size", &self.chunks_size),
            ("data_files_size_in_mb", &self.data_files_size_in_mb),
            ("video_files_size_in_mb", &self.video_files_size_in_mb),
        ] {
            if *value <= BigInt::from(0) {
                return Err(DatasetInfoError::NotPositive {
                    field,
                    value: value.clone(),
                });
            }
        }
        Ok(())
    }

    /// Port of `from_dict`: take the known keys, ignore the rest.
    ///
    /// Unknown keys are logged at warning level before construction, including
    /// when a later validation step fails, as upstream does.
    pub fn from_dict(data: &JsonObject) -> Result<Self, DatasetInfoError> {
        if let Some(message) = unknown_fields_warning(&Self::unknown_fields(data)) {
            log::warn!("{message}");
        }
        // `cls(**{...})` raises for absent required arguments before the body
        // of `__post_init__` runs, so this precedes every other complaint.
        let missing: Vec<&'static str> = Self::FIELD_NAMES[..3]
            .iter()
            .copied()
            .filter(|name| !data.contains_key(*name))
            .collect();
        if !missing.is_empty() {
            return Err(DatasetInfoError::MissingRequiredFields(missing));
        }

        let features = match &data["features"] {
            JsonLike::Object(map) => {
                let mut features = IndexMap::new();
                for (name, value) in map {
                    match value {
                        JsonLike::Object(feature) => {
                            features.insert(name.clone(), feature.clone());
                        }
                        other => return Err(wrong_type(format!("features.{name}"), "dict", other)),
                    }
                }
                features
            }
            other => return Err(wrong_type("features", "dict", other)),
        };

        // Upstream runs shape coercion and these four `__post_init__` checks
        // before it ever observes unrelated fields such as `splits`. Preserve
        // that meaningful error precedence even though this typed Rust
        // boundary must eventually validate those otherwise-unchecked fields.
        let fps = read_int(data, "fps")?.ok_or_else(|| wrong_type("fps", "int", &data["fps"]))?;
        if fps <= BigInt::from(0) {
            return Err(DatasetInfoError::NotPositive {
                field: "fps",
                value: fps,
            });
        }
        let chunks_size = read_int(data, "chunks_size")?
            .unwrap_or_else(|| BigInt::from(super::DEFAULT_CHUNK_SIZE));
        if chunks_size <= BigInt::from(0) {
            return Err(DatasetInfoError::NotPositive {
                field: "chunks_size",
                value: chunks_size,
            });
        }
        let data_files_size_in_mb = read_int(data, "data_files_size_in_mb")?
            .unwrap_or_else(|| BigInt::from(super::DEFAULT_DATA_FILE_SIZE_IN_MB));
        if data_files_size_in_mb <= BigInt::from(0) {
            return Err(DatasetInfoError::NotPositive {
                field: "data_files_size_in_mb",
                value: data_files_size_in_mb,
            });
        }
        let video_files_size_in_mb = read_int(data, "video_files_size_in_mb")?
            .unwrap_or_else(|| BigInt::from(super::DEFAULT_VIDEO_FILE_SIZE_IN_MB));
        if video_files_size_in_mb <= BigInt::from(0) {
            return Err(DatasetInfoError::NotPositive {
                field: "video_files_size_in_mb",
                value: video_files_size_in_mb,
            });
        }

        let splits = match data.get("splits") {
            None => IndexMap::new(),
            Some(JsonLike::Object(map)) => {
                let mut splits = IndexMap::new();
                for (name, value) in map {
                    match value {
                        JsonLike::Str(text) => {
                            splits.insert(name.clone(), text.clone());
                        }
                        other => return Err(wrong_type(format!("splits.{name}"), "str", other)),
                    }
                }
                splits
            }
            Some(other) => return Err(wrong_type("splits", "dict", other)),
        };

        let tools = match data.get("tools") {
            None | Some(JsonLike::Null) => None,
            Some(JsonLike::Array(items)) => {
                let mut tools = Vec::with_capacity(items.len());
                for (index, item) in items.iter().enumerate() {
                    match item {
                        JsonLike::Object(tool) => tools.push(tool.clone()),
                        other => return Err(wrong_type(format!("tools.{index}"), "dict", other)),
                    }
                }
                Some(tools)
            }
            Some(other) => return Err(wrong_type("tools", "list", other)),
        };

        let mut info = Self {
            codebase_version: read_str(data, "codebase_version")?
                .ok_or_else(|| wrong_type("codebase_version", "str", &data["codebase_version"]))?,
            fps,
            features,
            total_episodes: read_int(data, "total_episodes")?.unwrap_or_else(|| BigInt::from(0)),
            total_frames: read_int(data, "total_frames")?.unwrap_or_else(|| BigInt::from(0)),
            total_tasks: read_int(data, "total_tasks")?.unwrap_or_else(|| BigInt::from(0)),
            chunks_size,
            data_files_size_in_mb,
            video_files_size_in_mb,
            data_path: read_str(data, "data_path")?
                .unwrap_or_else(|| super::DEFAULT_DATA_PATH.to_string()),
            video_path: read_optional_str(data, "video_path")?
                .unwrap_or_else(|| Some(super::DEFAULT_VIDEO_PATH.to_string())),
            robot_type: read_optional_str(data, "robot_type")?.unwrap_or(None),
            splits,
            tools,
        };
        info.post_init()?;
        Ok(info)
    }

    /// The keys of `data` that are not [`DatasetInfo::FIELD_NAMES`], sorted.
    ///
    /// `sorted()` on Python `str`s orders by code point, which is what Rust's
    /// `str: Ord` does, so the two agree exactly.
    pub fn unknown_fields(data: &JsonObject) -> Vec<String> {
        let mut unknown: Vec<String> = data
            .keys()
            .filter(|key| !Self::FIELD_NAMES.contains(&key.as_str()))
            .cloned()
            .collect();
        unknown.sort();
        unknown
    }

    /// Port of `to_dict`: a JSON-serialisable dict in field order, with tuple
    /// shapes turned back into lists and `tools` dropped when unset.
    pub fn to_dict(&self) -> JsonObject {
        let features = self
            .features
            .iter()
            .map(|(name, feature)| {
                let mut feature = feature.clone();
                // "Converts tuple shapes back to lists so `json.dump` can
                // handle them." Only a tuple is rewritten; a shape that was
                // never a list stays whatever it is.
                if let Some(shape @ JsonLike::Tuple(_)) = feature.get_mut("shape") {
                    let JsonLike::Tuple(dimensions) = std::mem::replace(shape, JsonLike::Null)
                    else {
                        unreachable!("just matched a tuple")
                    };
                    *shape = JsonLike::Array(dimensions);
                }
                (name.clone(), JsonLike::Object(feature))
            })
            .collect();

        let optional_str = |value: &Option<String>| match value {
            Some(text) => JsonLike::Str(text.clone()),
            None => JsonLike::Null,
        };

        let mut dict = JsonObject::new();
        dict.insert(
            "codebase_version".to_string(),
            JsonLike::Str(self.codebase_version.clone()),
        );
        dict.insert("fps".to_string(), JsonLike::Int(self.fps.clone()));
        dict.insert("features".to_string(), JsonLike::Object(features));
        dict.insert(
            "total_episodes".to_string(),
            JsonLike::Int(self.total_episodes.clone()),
        );
        dict.insert(
            "total_frames".to_string(),
            JsonLike::Int(self.total_frames.clone()),
        );
        dict.insert(
            "total_tasks".to_string(),
            JsonLike::Int(self.total_tasks.clone()),
        );
        dict.insert(
            "chunks_size".to_string(),
            JsonLike::Int(self.chunks_size.clone()),
        );
        dict.insert(
            "data_files_size_in_mb".to_string(),
            JsonLike::Int(self.data_files_size_in_mb.clone()),
        );
        dict.insert(
            "video_files_size_in_mb".to_string(),
            JsonLike::Int(self.video_files_size_in_mb.clone()),
        );
        dict.insert(
            "data_path".to_string(),
            JsonLike::Str(self.data_path.clone()),
        );
        dict.insert("video_path".to_string(), optional_str(&self.video_path));
        dict.insert("robot_type".to_string(), optional_str(&self.robot_type));
        dict.insert(
            "splits".to_string(),
            JsonLike::Object(
                self.splits
                    .iter()
                    .map(|(k, v)| (k.clone(), JsonLike::Str(v.clone())))
                    .collect(),
            ),
        );
        // "Drops `tools` when unset so existing datasets keep a clean
        // info.json." An empty list is set, and is kept.
        if let Some(tools) = &self.tools {
            dict.insert(
                "tools".to_string(),
                JsonLike::Array(tools.iter().cloned().map(JsonLike::Object).collect()),
            );
        }
        dict
    }
}

/// The text of upstream's `logger.warning` for a non-empty unknown-field list.
///
/// [`DatasetInfo::from_dict`] emits this message through the `log` facade at
/// warning level. Applications retain control over logger handlers and filters.
pub fn unknown_fields_warning(unknown: &[String]) -> Option<String> {
    if unknown.is_empty() {
        return None;
    }
    let rendered: Vec<String> = unknown.iter().map(|key| python_str_repr(key)).collect();
    Some(format!(
        "Unknown fields in DatasetInfo: [{}]. These will be ignored.",
        rendered.join(", ")
    ))
}

/// Port of CPython 3.12's `repr()` for a Python `str`, as it appears inside the
/// warning's list. The Unicode general-category table is pinned to Unicode
/// 15.0, the version bundled by CPython 3.12.
pub fn python_str_repr(value: &str) -> String {
    // CPython prefers an apostrophe, and switches to a double quote only when
    // that would need escaping and this would not.
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote);
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if !python_is_printable(c) => match c as u32 {
                code @ 0..=0xff => out.push_str(&format!("\\x{code:02x}")),
                code @ 0x100..=0xffff => out.push_str(&format!("\\u{code:04x}")),
                code => out.push_str(&format!("\\U{code:08x}")),
            },
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// CPython's `_PyUnicode_IsPrintable`: categories Cc, Cf, Cs, Co, Cn, Zl, Zp
/// and Zs are non-printable, except for ordinary ASCII space U+0020.
fn python_is_printable(ch: char) -> bool {
    if ch == ' ' {
        return true;
    }
    !matches!(
        get_general_category(ch),
        GeneralCategory::Control
            | GeneralCategory::Format
            | GeneralCategory::Surrogate
            | GeneralCategory::PrivateUse
            | GeneralCategory::Unassigned
            | GeneralCategory::LineSeparator
            | GeneralCategory::ParagraphSeparator
            | GeneralCategory::SpaceSeparator
    )
}
