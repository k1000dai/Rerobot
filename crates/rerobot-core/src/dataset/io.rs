//! Port of the four local-filesystem functions behind `meta/info.json`:
//! `load_json` / `write_json` from `lerobot/utils/io_utils.py`, and
//! `load_info` / `write_info` from `lerobot/datasets/io_utils.py`.
//!
//! Local directories only. Upstream's Hub download path, the snapshot cache
//! and every other file under `meta/` are outside this slice.

use super::info::{DatasetInfo, DatasetInfoError};
use super::json::{self, JsonLike, ParseError};
use super::INFO_PATH;
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Why a JSON file could not be read.
#[derive(Debug)]
pub enum LoadError {
    /// The file could not be opened or read, or held invalid UTF-8.
    Io {
        /// The path that was being read.
        path: PathBuf,
        /// The underlying failure.
        source: std::io::Error,
    },
    /// The file was read but is not the JSON CPython would accept.
    Parse {
        /// The path that was being read.
        path: PathBuf,
        /// The failure, carrying CPython's message and coordinates.
        source: ParseError,
    },
    /// The file exceeds the explicit metadata resource budget.
    ResourceLimit {
        /// The path that was being read.
        path: PathBuf,
        /// Maximum accepted bytes.
        limit: usize,
        /// Bytes observed, capped at `limit + 1` if the file grew while read.
        actual: u64,
    },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "cannot read {}: {source}", path.display()),
            Self::Parse { path, source } => write!(f, "cannot parse {}: {source}", path.display()),
            Self::ResourceLimit {
                path,
                limit,
                actual,
            } => write!(
                f,
                "cannot read {}: metadata JSON is {actual} bytes, limit is {limit}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            Self::ResourceLimit { .. } => None,
        }
    }
}

/// Why `meta/info.json` could not be turned into a [`DatasetInfo`].
#[derive(Debug)]
pub enum LoadInfoError {
    /// The file could not be read or parsed.
    Load(LoadError),
    /// The file parsed, but its top-level value is not an object.
    ///
    /// Upstream reaches `cls(**raw)`, which raises `TypeError` for anything
    /// that is not a mapping; this is that failure, named.
    NotAnObject {
        /// The path that was being read.
        path: PathBuf,
        /// The Python type of the top-level value.
        found: &'static str,
    },
    /// The object parsed but does not describe a valid `DatasetInfo`.
    Info(DatasetInfoError),
}

impl fmt::Display for LoadInfoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(source) => source.fmt(f),
            Self::NotAnObject { path, found } => write!(
                f,
                "{} must hold a JSON object, found {found}",
                path.display()
            ),
            Self::Info(source) => source.fmt(f),
        }
    }
}

impl std::error::Error for LoadInfoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Load(source) => Some(source),
            Self::NotAnObject { .. } => None,
            Self::Info(source) => Some(source),
        }
    }
}

/// Port of `load_json`.
///
/// Upstream's `open(fpath)` decodes with the process's locale encoding;
/// Rerobot always decodes UTF-8. See `docs/compatibility.md` — on every
/// platform where that locale *is* UTF-8 the two agree, and where it is not,
/// guessing an encoding is worse than naming one.
pub fn load_json(path: &Path) -> Result<JsonLike, LoadError> {
    let file = std::fs::File::open(path).map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = file.metadata().map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() > json::MAX_JSON_INPUT_BYTES as u64 {
        return Err(LoadError::ResourceLimit {
            path: path.to_path_buf(),
            limit: json::MAX_JSON_INPUT_BYTES,
            actual: metadata.len(),
        });
    }

    let mut bytes = Vec::new();
    let mut limited = file.take(json::MAX_JSON_INPUT_BYTES as u64 + 1);
    let mut chunk = [0u8; 8192];
    loop {
        let read = limited.read(&mut chunk).map_err(|source| LoadError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if read == 0 {
            break;
        }
        bytes
            .try_reserve_exact(read)
            .map_err(|source| LoadError::Io {
                path: path.to_path_buf(),
                source: std::io::Error::new(std::io::ErrorKind::OutOfMemory, source.to_string()),
            })?;
        bytes.extend_from_slice(&chunk[..read]);
    }
    if bytes.len() > json::MAX_JSON_INPUT_BYTES {
        return Err(LoadError::ResourceLimit {
            path: path.to_path_buf(),
            limit: json::MAX_JSON_INPUT_BYTES,
            actual: bytes.len() as u64,
        });
    }
    let text = String::from_utf8(bytes).map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, source),
    })?;
    json::loads(&text).map_err(|source| LoadError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

/// Port of `write_json`, which creates the parent directories first.
///
/// The bytes are exactly what `json.dump(data, f, indent=4,
/// ensure_ascii=False)` produces: four-space indentation, literal non-ASCII,
/// and no trailing newline.
pub fn write_json(data: &JsonLike, path: &Path) -> Result<(), std::io::Error> {
    // `fpath.parent.mkdir(exist_ok=True, parents=True)`. A path with no parent
    // component is already relative to the working directory.
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, json::dumps_pretty(data))
}

/// Port of `write_info`: `write_json(info.to_dict(), local_dir / INFO_PATH)`.
pub fn write_info(info: &DatasetInfo, local_dir: &Path) -> Result<(), std::io::Error> {
    write_json(&JsonLike::Object(info.to_dict()), &info_path(local_dir))
}

/// Port of `load_info`: `DatasetInfo.from_dict(load_json(local_dir / INFO_PATH))`.
///
/// Unknown top-level keys are ignored and logged at warning level by
/// [`DatasetInfo::from_dict`], as upstream does. Callers that also need the
/// sorted field list can call [`load_json`] and [`DatasetInfo::unknown_fields`].
pub fn load_info(local_dir: &Path) -> Result<DatasetInfo, LoadInfoError> {
    let path = info_path(local_dir);
    let raw = load_json(&path).map_err(LoadInfoError::Load)?;
    let JsonLike::Object(data) = raw else {
        return Err(LoadInfoError::NotAnObject {
            path,
            found: raw.type_name(),
        });
    };
    DatasetInfo::from_dict(&data).map_err(LoadInfoError::Info)
}

/// The `meta/info.json` path inside `local_dir`.
///
/// `INFO_PATH` keeps upstream's POSIX spelling; joining it is what makes the
/// result native, and `Path::join` accepts the forward slashes on Windows too.
pub fn info_path(local_dir: &Path) -> PathBuf {
    local_dir.join(INFO_PATH)
}
