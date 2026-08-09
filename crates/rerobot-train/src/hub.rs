//! Native Hugging Face dataset snapshot resolution.
//!
//! The Python LeRobot implementation falls back from a local dataset root to
//! `snapshot_download`. This module provides the same boundary without Python:
//! it lists the dataset tree through the Hub API, streams each file over rustls,
//! stages the result beside the destination, and publishes it with one rename.

use crate::error::{Result, TrainError};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The dataset format revision used by LeRobot v3 datasets.
pub const DEFAULT_REVISION: &str = "v3.0";
const DEFAULT_HF_BASE_URL: &str = "https://huggingface.co";
const MAX_TREE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FILES: usize = 100_000;
const TREE_PAGE_LIMIT: usize = 1_000;
const MAX_FILE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// A validated Hub snapshot path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubSnapshot {
    root: PathBuf,
}

impl HubSnapshot {
    /// Validate and convert a Hub tree path into a relative filesystem path.
    pub fn relative_path(raw: &str) -> Result<PathBuf> {
        if raw.is_empty() || raw.contains('\\') {
            return Err(TrainError::Metadata(format!(
                "Hub snapshot path {raw:?} is empty or contains a backslash"
            )));
        }
        let path = Path::new(raw);
        if path.is_absolute()
            || path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(TrainError::Metadata(format!(
                "Hub snapshot path {raw:?} is not a safe relative path"
            )));
        }
        if path
            .components()
            .any(|component| matches!(component, Component::CurDir))
        {
            return Err(TrainError::Metadata(format!(
                "Hub snapshot path {raw:?} contains a current-directory component"
            )));
        }
        Ok(path.to_path_buf())
    }

    /// The local root published for this snapshot.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Validate a Hub dataset identifier such as `lerobot/pusht`.
pub fn validate_repo_id(repo_id: &str) -> Result<()> {
    let mut parts = repo_id.split('/');
    let Some(namespace) = parts.next() else {
        return Err(TrainError::Metadata("dataset.repo_id is empty".to_owned()));
    };
    let Some(name) = parts.next() else {
        return Err(TrainError::Metadata(format!(
            "dataset.repo_id {repo_id:?} must have the form namespace/name"
        )));
    };
    if parts.next().is_some()
        || namespace.is_empty()
        || name.is_empty()
        || namespace == "."
        || namespace == ".."
        || name == "."
        || name == ".."
        || repo_id.starts_with('/')
        || repo_id.contains('\\')
        || repo_id.contains('\0')
    {
        return Err(TrainError::Metadata(format!(
            "dataset.repo_id {repo_id:?} is not a safe namespace/name identifier"
        )));
    }
    Ok(())
}

/// Resolve the revision-safe cache root used when no explicit dataset directory
/// was supplied.
pub fn cache_root(home: &Path, repo_id: &str, revision: &str) -> Result<PathBuf> {
    validate_repo_id(repo_id)?;
    if revision.is_empty() || revision == "." || revision == ".." || revision.contains('\0') {
        return Err(TrainError::Metadata(format!(
            "dataset revision {revision:?} is empty or unsafe"
        )));
    }
    let repo_key = format!(
        "datasets--{}--{}",
        repo_id.split('/').next().unwrap(),
        repo_id.split('/').nth(1).unwrap()
    );
    Ok(home
        .join("hub")
        .join(repo_key)
        .join("snapshots")
        .join(encode_component(revision)))
}

/// The default local Hub cache location, honoring Hugging Face's environment
/// variables where they are available.
pub fn default_cache_home() -> PathBuf {
    if let Some(path) = std::env::var_os("HF_LEROBOT_HOME") {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("HF_HOME") {
        return PathBuf::from(path).join("lerobot");
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/huggingface/lerobot")
}

/// The default dataset directory for a Hub-backed training command.
pub fn default_dataset_root(repo_id: &str) -> Result<PathBuf> {
    cache_root(&default_cache_home(), repo_id, DEFAULT_REVISION)
}

/// Whether the local root has the minimum files the native reader needs.
pub fn is_complete_dataset(root: &Path) -> bool {
    root.is_dir()
        && root.join("meta/info.json").is_file()
        && root.join("meta/tasks.parquet").is_file()
        && root.join("meta/episodes").is_dir()
        && has_parquet_under(&root.join("data"))
}

/// Resolve an optional explicit root, downloading a missing dataset from the Hub.
///
/// An explicit root is used exactly as requested. Without one, the caller should
/// pass [`default_dataset_root`], which points at a revision-safe cache location.
pub fn resolve_dataset_root(
    repo_id: &str,
    requested_root: &Path,
    revision: Option<&str>,
) -> Result<PathBuf> {
    if is_complete_dataset(requested_root) {
        return Ok(requested_root.to_path_buf());
    }
    let revision = revision.unwrap_or(DEFAULT_REVISION);
    HubDownloader::new(DEFAULT_HF_BASE_URL).download(repo_id, requested_root, revision)
}

/// An injectable Hub downloader. The base URL is public so integration tests can
/// use a local HTTP server without touching the real Hub.
pub struct HubDownloader {
    base_url: String,
    agent: ureq::Agent,
    token: Option<String>,
}

impl HubDownloader {
    /// Create a downloader using `base_url`, normally `https://huggingface.co`.
    pub fn new(base_url: &str) -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(120)))
            .build()
            .into();
        Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            agent,
            token: std::env::var("HF_TOKEN")
                .ok()
                .or_else(|| std::env::var("HUGGINGFACE_HUB_TOKEN").ok()),
        }
    }

    /// Download a complete dataset snapshot into `destination` atomically.
    pub fn download(&self, repo_id: &str, destination: &Path, revision: &str) -> Result<PathBuf> {
        validate_repo_id(repo_id)?;
        let target = destination.to_path_buf();
        if target.exists() {
            if is_complete_dataset(&target) {
                return Ok(target);
            }
            let mut entries =
                fs::read_dir(&target).map_err(|error| TrainError::io(&target, &error))?;
            if entries.next().is_some() {
                return Err(TrainError::io_message(
                    &target,
                    "dataset root exists but is incomplete; refusing to overwrite it",
                ));
            }
            fs::remove_dir(&target).map_err(|error| TrainError::io(&target, &error))?;
        }

        let tree = self.list_files(repo_id, revision)?;
        if tree.is_empty() {
            return Err(TrainError::io_message(
                &target,
                "Hugging Face dataset tree is empty",
            ));
        }
        if tree.len() > MAX_FILES {
            return Err(TrainError::unsupported(format!(
                "Hugging Face dataset has {} files, exceeding the native limit {MAX_FILES}",
                tree.len()
            )));
        }
        if tree
            .iter()
            .any(|path| path.starts_with("videos/") || path.ends_with(".mp4"))
        {
            return Err(TrainError::unsupported(
                "the Hub dataset contains video shards; this native reader supports embedded PNG/JPEG columns, not MP4 downloads",
            ));
        }

        let parent = target.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|error| TrainError::io(parent, &error))?;
        let staging = parent.join(format!(
            ".{}.staging-{}-{}",
            target
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("dataset"),
            std::process::id(),
            unique_suffix()
        ));
        fs::create_dir(&staging).map_err(|error| TrainError::io(&staging, &error))?;

        let result = self.download_tree(repo_id, revision, &tree, &staging);
        if let Err(error) = result {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
        if !is_complete_dataset(&staging) {
            let _ = fs::remove_dir_all(&staging);
            return Err(TrainError::io_message(
                &target,
                "Hub snapshot downloaded but does not contain a readable LeRobot v3 dataset",
            ));
        }
        fs::rename(&staging, &target).map_err(|error| {
            let _ = fs::remove_dir_all(&staging);
            TrainError::io(&target, &error)
        })?;
        Ok(target)
    }

    fn list_files(&self, repo_id: &str, revision: &str) -> Result<Vec<String>> {
        let first_url = format!(
            "{}/api/datasets/{}/{}/tree/{}?recursive=true&expand=false&limit={TREE_PAGE_LIMIT}",
            self.base_url,
            encode_segment(repo_id.split('/').next().unwrap()),
            encode_segment(repo_id.split('/').nth(1).unwrap()),
            encode_component(revision),
        );
        let mut paths = Vec::new();
        let mut next_url = Some(first_url);
        while let Some(url) = next_url.take() {
            let mut request = self.agent.get(&url);
            if let Some(token) = &self.token {
                request = request.header("Authorization", &format!("Bearer {token}"));
            }
            let mut response = request.call().map_err(|error| {
                TrainError::io_message(
                    Path::new(repo_id),
                    format!("Hugging Face tree request failed: {error}"),
                )
            })?;
            let link = response
                .headers()
                .get("link")
                .and_then(|value| value.to_str().ok())
                .and_then(parse_next_link);
            let bytes = response
                .body_mut()
                .with_config()
                .limit(MAX_TREE_BYTES)
                .read_to_vec()
                .map_err(|error| {
                    TrainError::io_message(
                        Path::new(repo_id),
                        format!("cannot read Hugging Face tree: {error}"),
                    )
                })?;
            let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
                TrainError::io_message(
                    Path::new(repo_id),
                    format!("Hugging Face tree is not valid JSON: {error}"),
                )
            })?;
            let entries = json.as_array().ok_or_else(|| {
                TrainError::io_message(
                    Path::new(repo_id),
                    "Hugging Face tree response is not an array",
                )
            })?;
            for entry in entries {
                if entry.get("type").and_then(serde_json::Value::as_str) != Some("file") {
                    continue;
                }
                let raw = entry
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        TrainError::io_message(
                            Path::new(repo_id),
                            "Hugging Face tree file has no path",
                        )
                    })?;
                let relative = HubSnapshot::relative_path(raw)?;
                paths.push(
                    relative
                        .to_string_lossy()
                        .replace(std::path::MAIN_SEPARATOR, "/"),
                );
                if paths.len() > MAX_FILES {
                    return Err(TrainError::unsupported(format!(
                        "Hugging Face dataset contains more than {MAX_FILES} files"
                    )));
                }
            }
            next_url = link;
        }
        Ok(paths)
    }

    fn download_tree(
        &self,
        repo_id: &str,
        revision: &str,
        paths: &[String],
        staging: &Path,
    ) -> Result<()> {
        for raw_path in paths {
            let relative = HubSnapshot::relative_path(raw_path)?;
            let output = staging.join(&relative);
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent).map_err(|error| TrainError::io(parent, &error))?;
            }
            let url = format!(
                "{}/datasets/{}/{}/resolve/{}/{}",
                self.base_url,
                encode_segment(repo_id.split('/').next().unwrap()),
                encode_segment(repo_id.split('/').nth(1).unwrap()),
                encode_component(revision),
                raw_path
                    .split('/')
                    .map(encode_segment)
                    .collect::<Vec<_>>()
                    .join("/")
            );
            let mut request = self.agent.get(&url);
            if let Some(token) = &self.token {
                request = request.header("Authorization", &format!("Bearer {token}"));
            }
            let mut response = request.call().map_err(|error| {
                TrainError::io_message(
                    &output,
                    format!("Hugging Face file request failed: {error}"),
                )
            })?;
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&output)
                .map_err(|error| TrainError::io(&output, &error))?;
            let mut reader = response
                .body_mut()
                .with_config()
                .limit(MAX_FILE_BYTES + 1)
                .reader();
            let copied = io::copy(&mut reader, &mut file)
                .map_err(|error| TrainError::io(&output, &error))?;
            file.flush()
                .map_err(|error| TrainError::io(&output, &error))?;
            if copied > MAX_FILE_BYTES {
                return Err(TrainError::unsupported(format!(
                    "Hugging Face file {raw_path:?} exceeds the {MAX_FILE_BYTES}-byte limit"
                )));
            }
        }
        Ok(())
    }
}

fn has_parquet_under(root: &Path) -> bool {
    let Ok(entries) = fs::read_dir(root) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        path.is_dir() && has_parquet_under(&path)
            || path.extension().is_some_and(|ext| ext == "parquet")
    })
}

fn encode_segment(value: &str) -> String {
    value
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
                char::from(byte).to_string()
            } else {
                format!("%{:02X}", byte)
            }
        })
        .collect()
}

fn encode_component(value: &str) -> String {
    encode_segment(value)
}

fn parse_next_link(header: &str) -> Option<String> {
    header.split(',').find_map(|part| {
        let (url, attributes) = part.split_once(';')?;
        let is_next = attributes.split(';').any(|attribute| {
            attribute.trim().eq_ignore_ascii_case("rel=\"next\"")
                || attribute.trim().eq_ignore_ascii_case("rel=next")
        });
        if !is_next {
            return None;
        }
        Some(
            url.trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_owned(),
        )
    })
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

#[cfg(test)]
mod tests {
    use super::{encode_segment, parse_next_link};

    #[test]
    fn parses_hugging_face_next_link() {
        assert_eq!(
            parse_next_link("<https://huggingface.co/next?page=2>; rel=\"next\", <https://huggingface.co/prev>; rel=\"prev\""),
            Some("https://huggingface.co/next?page=2".to_owned())
        );
    }

    #[test]
    fn url_segments_escape_slashes_and_spaces() {
        assert_eq!(
            encode_segment("feature/branch name"),
            "feature%2Fbranch%20name"
        );
    }
}
