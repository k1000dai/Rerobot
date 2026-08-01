//! `--policy.device`: which spellings are accepted, and what a refusal says.
//!
//! Everything here runs on CPU. The tests that need an NVIDIA GPU live in
//! `tests/cuda_smoke.rs`, which does not compile at all without the `cuda`
//! feature — so a green run of this file is never evidence that a GPU was
//! exercised.

use rerobot_train::device::{self, DeviceSpec};
use rerobot_train::error::TrainError;

#[test]
fn cpu_is_the_default_and_resolves_to_the_cpu_device() {
    assert_eq!(device::parse(None).unwrap(), DeviceSpec::Cpu);
    assert_eq!(device::parse(Some("cpu")).unwrap(), DeviceSpec::Cpu);
    assert!(device::resolve(Some("cpu")).unwrap().is_cpu());
    assert!(device::resolve(None).unwrap().is_cpu());
}

#[test]
fn a_device_this_build_cannot_provide_is_refused_rather_than_downgraded() {
    let error = device::parse(Some("mps")).unwrap_err();
    assert!(matches!(error, TrainError::Unsupported(_)));
    assert!(
        error.to_string().contains("\"mps\""),
        "the refusal must name the value: {error}"
    );
}

/// The whole point of the feature gate: a default build says so, names the
/// rebuild, and does not quietly train on the CPU instead.
#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_is_refused_by_a_binary_built_without_it() {
    for spelling in ["cuda", "cuda:0"] {
        let error = device::parse(Some(spelling)).unwrap_err();
        assert!(matches!(error, TrainError::Unsupported(_)));
        let message = error.to_string();
        assert!(
            message.contains("built without CUDA support"),
            "{spelling}: {message}"
        );
        assert!(
            message.contains("--features cuda"),
            "{spelling} must name the rebuild: {message}"
        );
        // `resolve` must refuse for the same reason rather than falling back.
        assert!(device::resolve(Some(spelling)).is_err());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn both_upstream_spellings_select_cuda_device_zero() {
    assert_eq!(device::parse(Some("cuda")).unwrap(), DeviceSpec::Cuda(0));
    assert_eq!(device::parse(Some("cuda:0")).unwrap(), DeviceSpec::Cuda(0));
}

/// Only ordinal 0 is selected, and a different one is refused rather than
/// silently rounded down to the GPU the user did not ask for.
#[cfg(feature = "cuda")]
#[test]
fn a_non_zero_cuda_ordinal_is_refused_rather_than_remapped() {
    let error = device::parse(Some("cuda:1")).unwrap_err();
    assert!(matches!(error, TrainError::Unsupported(_)));
    assert!(
        error.to_string().contains("device 0"),
        "unexpected error: {error}"
    );
}
