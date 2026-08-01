//! Which compute device `--policy.device` names, and whether this build has it.
//!
//! Upstream resolves the device inside `PreTrainedConfig.__post_init__`, where
//! `torch.cuda.is_available()` decides and a missing GPU silently downgrades the
//! run to CPU with a warning. Nothing here downgrades: a device that was asked
//! for and cannot be provided is an error, because a run that reports success
//! after training somewhere other than where it was told to is the failure this
//! port exists to avoid.
//!
//! The decision is split in two so that the half that does not touch hardware is
//! testable everywhere:
//!
//! * [`parse`] turns the string into a [`DeviceSpec`], refusing a spelling this
//!   *build* cannot serve. It allocates nothing and initializes nothing, so
//!   [`crate::config::TrainConfig::validate`] can call it before a dataset is
//!   opened.
//! * [`DeviceSpec::open`] initializes the hardware, and is where a machine with
//!   no working GPU is caught.

use crate::error::{Result, TrainError};
use candle_core::Device;

/// Whether this build compiled candle's CUDA backend.
///
/// `false` for a default build; `true` with the crate's `cuda` feature. Public
/// because it is the honest answer to "can this binary use a GPU?", which no
/// amount of probing at runtime can otherwise establish.
pub const CUDA_COMPILED: bool = cfg!(feature = "cuda");

/// A device this build knows how to name.
///
/// Naming one is not the same as having one: [`DeviceSpec::open`] is where a
/// CUDA ordinal meets an actual driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSpec {
    /// `cpu`, the default.
    Cpu,
    /// `cuda` or `cuda:0`, carrying the ordinal.
    Cuda(usize),
}

impl DeviceSpec {
    /// Initialize the device.
    ///
    /// # Errors
    ///
    /// [`TrainError::Device`] when the CUDA driver, runtime or GPU is missing or
    /// refuses the ordinal. There is deliberately no fallback to the CPU: the
    /// caller asked for a GPU.
    pub fn open(&self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda(ordinal) => open_cuda(*ordinal),
        }
    }

    /// How this spec is spelled on the command line.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda(_) => "cuda",
        }
    }
}

/// Resolve `--policy.device` to a spec, without touching any hardware.
///
/// `None` is upstream's unset field and means the CPU, which is what
/// [`crate::config::TrainConfig::new`] fills in.
///
/// # Errors
///
/// [`TrainError::Unsupported`] for a device this build cannot provide — either
/// because it is not ported at all (`mps`, `xpu`), or because it is CUDA and the
/// `cuda` feature was off when this binary was compiled.
pub fn parse(spec: Option<&str>) -> Result<DeviceSpec> {
    let Some(spec) = spec else {
        return Ok(DeviceSpec::Cpu);
    };
    if spec == "cpu" {
        return Ok(DeviceSpec::Cpu);
    }
    if let Some(ordinal) = cuda_ordinal(spec) {
        if !CUDA_COMPILED {
            return Err(TrainError::unsupported(format!(
                "policy.device = {spec:?}; this binary was built without CUDA support, so only \
                 \"cpu\" is accepted. Rebuild with `--features cuda` (for example `cargo build \
                 --release -p rerobot-cli --features cuda`) on a machine with the NVIDIA CUDA \
                 toolkit to train on a GPU."
            )));
        }
        return match ordinal {
            // Bare `cuda` means the default/current CUDA device upstream; this
            // slice deliberately resolves that to its only supported ordinal,
            // device 0.
            None | Some(0) => Ok(DeviceSpec::Cuda(0)),
            // A spelled-out ordinal this slice does not select is refused rather
            // than quietly remapped to GPU 0, which is a different GPU.
            _ => Err(TrainError::unsupported(format!(
                "policy.device = {spec:?}; this slice opens CUDA device 0 only, so pass \"cuda\" \
                 or \"cuda:0\""
            ))),
        };
    }
    Err(TrainError::unsupported(if CUDA_COMPILED {
        format!("policy.device = {spec:?}; this binary accepts \"cpu\", \"cuda\" and \"cuda:0\"")
    } else {
        format!(
            "policy.device = {spec:?}; this binary was built without CUDA support, so only \
             \"cpu\" is accepted"
        )
    }))
}

/// [`parse`] followed by [`DeviceSpec::open`].
///
/// # Errors
///
/// Whatever either step reports; see both.
pub fn resolve(spec: Option<&str>) -> Result<Device> {
    parse(spec)?.open()
}

/// The ordinal in a `cuda` spelling, or `None` when this is not one.
///
/// `Some(None)` is bare `cuda`, which torch reads as "the current device" and
/// this slice reads as device 0. `cuda:` and `cuda:x` are not cuda spellings at
/// all, so they fall through to the general refusal rather than to the ordinal
/// one — the user did not name a GPU, they mistyped something.
fn cuda_ordinal(spec: &str) -> Option<Option<usize>> {
    if spec == "cuda" {
        return Some(None);
    }
    let rest = spec.strip_prefix("cuda:")?;
    // Not `str::parse`, which accepts a leading `+`.
    if rest.is_empty() || !rest.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    // An ordinal too large for `usize` is still an ordinal; it is refused below
    // with the "device 0 only" message rather than mistaken for a typo.
    Some(Some(rest.parse().unwrap_or(usize::MAX)))
}

#[cfg(feature = "cuda")]
fn open_cuda(ordinal: usize) -> Result<Device> {
    Device::new_cuda(ordinal).map_err(|error| {
        TrainError::Device(format!(
            "CUDA device {ordinal} could not be initialized: {error}. This binary was built with \
             CUDA support and the run asked for a GPU, so it stops here rather than falling back \
             to the CPU; pass --policy.device=cpu to train on the CPU instead."
        ))
    })
}

#[cfg(not(feature = "cuda"))]
fn open_cuda(ordinal: usize) -> Result<Device> {
    // Unreachable through `parse`, which refuses every CUDA spelling in this
    // build. Reachable by a library caller that constructed the spec by hand, and
    // it must say the same thing rather than panic.
    Err(TrainError::Device(format!(
        "CUDA device {ordinal} was requested but this binary was built without CUDA support; \
         rebuild with `--features cuda`"
    )))
}
