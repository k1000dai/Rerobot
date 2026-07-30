//! Port of the checkpoint layout in `lerobot/common/train_utils.py`.
//!
//! ```text
//! <output_dir>/checkpoints/
//!   000001/
//!     pretrained_model/
//!       config.json          <- the policy config, byte-identical to upstream's
//!       model.safetensors    <- the policy state dict, upstream's tensor names
//!       train_config.json    <- the run config
//!     training_state/
//!       training_step.json
//!       rng_state.safetensors
//!       optimizer_state.safetensors
//!       optimizer_param_groups.json
//!   last -> 000001
//! ```
//!
//! Two things differ from upstream, both recorded in `docs/compatibility.md`:
//!
//! * `rng_state.safetensors` holds Rerobot's one-word generator state, not
//!   Python's, NumPy's and PyTorch's three (see [`rerobot_core::random`]);
//! * `checkpoints/last` is a directory symlink where the platform allows one and a
//!   one-line text file naming the target where it does not, because
//!   `std::os::windows::fs::symlink_dir` needs a privilege an ordinary user does
//!   not have. [`read_last_checkpoint`] accepts both.

use crate::error::{Result, TrainError};
use rerobot_core::dataset::json::{dumps_pretty_ascii, loads, JsonLike, JsonObject};
use rerobot_core::random::SplitMix64;
use std::path::{Path, PathBuf};

/// `CHECKPOINTS_DIR`.
pub const CHECKPOINTS_DIR: &str = "checkpoints";
/// `LAST_CHECKPOINT_LINK`.
pub const LAST_CHECKPOINT_LINK: &str = "last";
/// `PRETRAINED_MODEL_DIR`.
pub const PRETRAINED_MODEL_DIR: &str = "pretrained_model";
/// `TRAINING_STATE_DIR`.
pub const TRAINING_STATE_DIR: &str = "training_state";
/// `TRAINING_STEP`.
pub const TRAINING_STEP: &str = "training_step.json";
/// `RNG_STATE`.
pub const RNG_STATE: &str = "rng_state.safetensors";
/// `OPTIMIZER_STATE`.
pub const OPTIMIZER_STATE: &str = "optimizer_state.safetensors";
/// `OPTIMIZER_PARAM_GROUPS`.
pub const OPTIMIZER_PARAM_GROUPS: &str = "optimizer_param_groups.json";
/// The policy weights file `save_pretrained` writes.
pub const MODEL_FILE: &str = "model.safetensors";
/// The policy config file `_save_pretrained` writes.
pub const CONFIG_FILE: &str = "config.json";
/// `TRAIN_CONFIG_NAME`.
pub const TRAIN_CONFIG_NAME: &str = "train_config.json";

/// The key Rerobot's generator state is stored under in `rng_state.safetensors`.
///
/// Deliberately *not* one of upstream's three (`random_state`,
/// `numpy_random_state`, `torch_random_state`): a reader expecting those must fail
/// to find them rather than find something that looks like them and is not.
pub const RERBOT_RNG_KEY: &str = "rerobot_splitmix64_state";

/// `get_step_identifier`: zero-padded to at least six digits.
pub fn step_identifier(step: u64, total_steps: u64) -> String {
    let digits = total_steps.to_string().len().max(6);
    format!("{step:0digits$}")
}

/// `get_step_checkpoint_dir`.
pub fn step_checkpoint_dir(output_dir: &Path, total_steps: u64, step: u64) -> PathBuf {
    output_dir
        .join(CHECKPOINTS_DIR)
        .join(step_identifier(step, total_steps))
}

/// How `checkpoints/last` was recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LastCheckpointKind {
    /// A relative directory symlink, as upstream writes.
    Symlink,
    /// A one-line text file naming the target directory.
    PortableFile,
}

/// `update_last_checkpoint`: point `checkpoints/last` at `checkpoint_dir`.
///
/// Tries a symlink first and falls back to the portable file when the platform
/// refuses one. The fallback is reported rather than hidden so a caller can say
/// which happened.
pub fn update_last_checkpoint(checkpoint_dir: &Path) -> Result<LastCheckpointKind> {
    let parent = checkpoint_dir.parent().ok_or_else(|| {
        TrainError::checkpoint(checkpoint_dir, "has no parent directory to link from")
    })?;
    let target = checkpoint_dir
        .file_name()
        .ok_or_else(|| TrainError::checkpoint(checkpoint_dir, "has no final path component"))?;
    let link = parent.join(LAST_CHECKPOINT_LINK);
    remove_last_marker(&link)?;

    match symlink_dir(Path::new(target), &link) {
        Ok(()) => Ok(LastCheckpointKind::Symlink),
        Err(_) => {
            write_portable_marker(parent, target)?;
            Ok(LastCheckpointKind::PortableFile)
        }
    }
}

/// [`update_last_checkpoint`], forced to one representation.
///
/// Exists so that both branches are reachable on one platform, which is the only
/// way the portable fallback is tested anywhere but Windows.
pub fn write_last_checkpoint(
    checkpoint_dir: &Path,
    kind: LastCheckpointKind,
) -> Result<LastCheckpointKind> {
    match kind {
        LastCheckpointKind::Symlink => update_last_checkpoint(checkpoint_dir),
        LastCheckpointKind::PortableFile => {
            let parent = checkpoint_dir.parent().ok_or_else(|| {
                TrainError::checkpoint(checkpoint_dir, "has no parent directory to link from")
            })?;
            let target = checkpoint_dir.file_name().ok_or_else(|| {
                TrainError::checkpoint(checkpoint_dir, "has no final path component")
            })?;
            write_portable_marker(parent, target)?;
            Ok(LastCheckpointKind::PortableFile)
        }
    }
}

/// Resolve `checkpoints/last` to the directory it names.
pub fn read_last_checkpoint(checkpoints_dir: &Path) -> Result<PathBuf> {
    let link = checkpoints_dir.join(LAST_CHECKPOINT_LINK);
    let metadata =
        std::fs::symlink_metadata(&link).map_err(|error| TrainError::io(&link, &error))?;
    if metadata.file_type().is_symlink() {
        // `symlink_metadata` describes the link, not its target, so the target has
        // to be read out and re-anchored: upstream writes a *relative* link.
        let target = std::fs::read_link(&link).map_err(|error| TrainError::io(&link, &error))?;
        let resolved = if target.is_absolute() {
            target
        } else {
            checkpoints_dir.join(target)
        };
        if !resolved.is_dir() {
            return Err(TrainError::checkpoint(
                &link,
                format!("points at {}, which is not a directory", resolved.display()),
            ));
        }
        return Ok(resolved);
    }
    if metadata.is_dir() {
        // Not a link at all but a real directory under the reserved name.
        return Ok(link);
    }
    let contents = std::fs::read_to_string(&link).map_err(|error| TrainError::io(&link, &error))?;
    let name = contents.trim();
    if name.is_empty() {
        return Err(TrainError::checkpoint(&link, "names no checkpoint"));
    }
    let resolved = checkpoints_dir.join(name);
    if !resolved.is_dir() {
        return Err(TrainError::checkpoint(
            &link,
            format!("names {name:?}, which is not a directory"),
        ));
    }
    Ok(resolved)
}

/// Write the portable marker into `parent`, naming `target`, atomically.
///
/// A temporary file in the *same* directory, then `rename`. Both halves matter:
///
/// * **`rename`, not `write`.** The previous version unlinked the marker and then
///   called `std::fs::write` on the reserved path. `write` opens the path, and opening
///   *follows a symlink* — so an attacker who planted one between the unlink and the
///   open had the marker's content written into any file they could name, truncating
///   it. `rename` replaces a name; it never follows what is already there. There is no
///   longer an unlink at all, so the window it opened is gone rather than narrowed.
/// * **same directory.** `rename` is only atomic within a filesystem, and a temporary
///   in the system temp directory may be on another one. The name is prefixed so a
///   stray temporary from a killed process is recognisable, and it is removed on the
///   error path so a failed write leaves nothing behind.
///
/// A real directory at the reserved path is still refused first, because `rename` onto
/// one fails with a platform-specific errno rather than a message that says what to do.
fn write_portable_marker(parent: &Path, target: &std::ffi::OsStr) -> Result<()> {
    let link = parent.join(LAST_CHECKPOINT_LINK);
    refuse_non_marker(&link)?;

    // Unique per process so two concurrent writers cannot collide on the temporary.
    let temporary = parent.join(format!(
        ".{LAST_CHECKPOINT_LINK}.{}.tmp",
        std::process::id()
    ));
    let contents = format!("{}\n", target.to_string_lossy());
    std::fs::write(&temporary, contents).map_err(|error| TrainError::io(&temporary, &error))?;
    match std::fs::rename(&temporary, &link) {
        Ok(()) => Ok(()),
        Err(error) => {
            // Leave nothing behind on the failure path.
            let _ = std::fs::remove_file(&temporary);
            Err(TrainError::io(&link, &error))
        }
    }
}

/// Refuse anything at the reserved path that is not a marker this code may replace.
///
/// A symlink and a regular file are both replaceable by `rename`; a directory is not,
/// and a caller-controlled tree there must never be removed.
fn refuse_non_marker(link: &Path) -> Result<()> {
    let Ok(metadata) = std::fs::symlink_metadata(link) else {
        return Ok(());
    };
    let kind = metadata.file_type();
    if kind.is_symlink() || kind.is_file() {
        return Ok(());
    }
    if kind.is_dir() {
        return Err(TrainError::checkpoint(
            link,
            "is a real directory, not a checkpoint marker; refusing to delete it. Move or \
             remove it by hand if it is not wanted -- maintaining the marker must never \
             recursively delete a directory",
        ));
    }
    Err(TrainError::checkpoint(
        link,
        format!(
            "is neither a symlink, a regular file nor a directory ({kind:?}); refusing to \
             replace it"
        ),
    ))
}

/// Clear whatever stands where the `last` marker goes — but only if it is a marker.
///
/// This used to `remove_dir_all` a real directory found at the reserved path, which
/// made maintaining a one-line marker into a recursive delete of a caller-controlled
/// tree. It never needed to be: the marker is either a symlink or a small regular
/// file, and both are removed by unlinking. A symlink is unlinked rather than
/// followed, so its target survives too.
///
/// Anything else at that path is a refusal. It is a reserved name in a directory this
/// process owns, so finding a real directory there means either the caller put
/// something valuable in the way or something is racing to have it deleted; neither
/// is a case for removing it.
fn remove_last_marker(link: &Path) -> Result<()> {
    let Ok(metadata) = std::fs::symlink_metadata(link) else {
        // Nothing there, which is the common case.
        return Ok(());
    };
    let kind = metadata.file_type();
    if kind.is_symlink() {
        // Unlinks the link. `remove_file` never follows one.
        return std::fs::remove_file(link).map_err(|error| TrainError::io(link, &error));
    }
    if kind.is_dir() {
        return Err(TrainError::checkpoint(
            link,
            "is a real directory, not a checkpoint marker; refusing to delete it. Move or \
             remove it by hand if it is not wanted -- maintaining the marker must never \
             recursively delete a directory",
        ));
    }
    if kind.is_file() {
        return std::fs::remove_file(link).map_err(|error| TrainError::io(link, &error));
    }
    Err(TrainError::checkpoint(
        link,
        format!(
            "is neither a symlink, a regular file nor a directory ({kind:?}); refusing to \
             replace it"
        ),
    ))
}

#[cfg(unix)]
fn symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(windows)]
fn symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(target, link)
}

#[cfg(not(any(unix, windows)))]
fn symlink_dir(_target: &Path, _link: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "this platform has no directory symlinks",
    ))
}

/// `training_step.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingStep {
    /// `step`.
    pub step: u64,
    /// `num_processes`, always one here.
    pub num_processes: u64,
    /// `batch_size`.
    pub batch_size: usize,
}

impl TrainingStep {
    /// The JSON object upstream's `save_training_step` writes.
    pub fn to_json(self) -> JsonLike {
        let mut object = JsonObject::new();
        object.insert("step".into(), int(self.step));
        object.insert("num_processes".into(), int(self.num_processes));
        object.insert("batch_size".into(), int(self.batch_size as u64));
        JsonLike::Object(object)
    }

    /// Read `training_step.json`.
    pub fn read(training_state_dir: &Path) -> Result<Self> {
        let path = training_state_dir.join(TRAINING_STEP);
        let text = std::fs::read_to_string(&path).map_err(|error| TrainError::io(&path, &error))?;
        let document = loads(&text).map_err(|error| {
            TrainError::checkpoint(&path, format!("is not valid JSON: {error}"))
        })?;
        let JsonLike::Object(object) = document else {
            return Err(TrainError::checkpoint(&path, "is not a JSON object"));
        };
        let field = |name: &str| -> Result<u64> {
            match object.get(name) {
                Some(JsonLike::Int(value)) => u64::try_from(value)
                    .map_err(|_| TrainError::checkpoint(&path, format!("{name} is out of range"))),
                Some(_) => Err(TrainError::checkpoint(
                    &path,
                    format!("{name} is not an integer"),
                )),
                None => Err(TrainError::checkpoint(&path, format!("has no {name}"))),
            }
        };
        Ok(Self {
            step: field("step")?,
            num_processes: field("num_processes").unwrap_or(1),
            batch_size: field("batch_size").unwrap_or(0) as usize,
        })
    }
}

fn int(value: u64) -> JsonLike {
    JsonLike::Int(num_bigint::BigInt::from(value))
}

/// Write `rng_state.safetensors`.
pub fn write_rng_state(training_state_dir: &Path, rng: &SplitMix64) -> Result<()> {
    let path = training_state_dir.join(RNG_STATE);
    // Stored as `i64` because safetensors has no unsigned 64-bit dtype; the
    // conversion is a bit-cast, so the round trip is exact for every state.
    let tensor = candle_core::Tensor::new(&[rng.state() as i64], &candle_core::Device::Cpu)?;
    let mut tensors = std::collections::HashMap::new();
    tensors.insert(RERBOT_RNG_KEY.to_owned(), tensor);
    candle_core::safetensors::save(&tensors, &path)?;
    Ok(())
}

/// Read `rng_state.safetensors`.
pub fn read_rng_state(training_state_dir: &Path) -> Result<SplitMix64> {
    let path = training_state_dir.join(RNG_STATE);
    let tensors = candle_core::safetensors::load(&path, &candle_core::Device::Cpu)?;

    // An extra tensor means this is not a file this reader understands. A checkpoint
    // carrying upstream's `torch_random_state` alongside would otherwise restore one
    // generator and silently drop three.
    let mut unexpected: Vec<&str> = tensors
        .keys()
        .filter(|key| key.as_str() != RERBOT_RNG_KEY)
        .map(String::as_str)
        .collect();
    if !unexpected.is_empty() {
        unexpected.sort_unstable();
        return Err(TrainError::checkpoint(
            &path,
            format!(
                "holds tensors this reader does not understand: {}. Rerobot's generator state \
                 is the single tensor {RERBOT_RNG_KEY:?}",
                unexpected.join(", ")
            ),
        ));
    }

    let tensor = tensors.get(RERBOT_RNG_KEY).ok_or_else(|| {
        TrainError::checkpoint(&path, format!("has no {RERBOT_RNG_KEY:?} tensor"))
    })?;
    // The dtype is checked, not converted: the state is a `u64` bit-cast to `i64`, so
    // reading the same bits as a float silently loses the low bits of a large state.
    if tensor.dtype() != candle_core::DType::I64 {
        return Err(TrainError::checkpoint(
            &path,
            format!(
                "{RERBOT_RNG_KEY:?} has dtype {:?} but the generator state is one I64 word",
                tensor.dtype()
            ),
        ));
    }
    // Exactly one element, rather than "the first element of whatever is here": a
    // two-element tensor used to restore half a state and report success.
    if tensor.dims() != [1] {
        return Err(TrainError::checkpoint(
            &path,
            format!(
                "{RERBOT_RNG_KEY:?} has shape {:?} but the generator state is exactly one element",
                tensor.dims()
            ),
        ));
    }
    let values = tensor.to_vec1::<i64>()?;
    let state = values
        .first()
        .ok_or_else(|| TrainError::checkpoint(&path, format!("{RERBOT_RNG_KEY:?} is empty")))?;
    Ok(SplitMix64::from_state(*state as u64))
}

/// Write a JSON document the way `lerobot.utils.io_utils.write_json` does:
/// `json.dump(..., indent=4)` with CPython's default `ensure_ascii=True`.
pub fn write_json(value: &JsonLike, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| TrainError::io(parent, &error))?;
    }
    std::fs::write(path, dumps_pretty_ascii(value)).map_err(|error| TrainError::io(path, &error))
}
