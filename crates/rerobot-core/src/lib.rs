#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

/// The unbounded signed integer this crate uses wherever upstream holds a
/// Python `int` — `PolicyFeature::shape` and `ActionInterpolator`'s multiplier.
/// Re-exported so callers need not depend on `num-bigint` themselves.
pub use num_bigint::BigInt;

pub mod action_interpolator;
pub mod byte_count;
pub mod dataset;
pub mod processor;
pub mod ring_buffer;
pub mod rollout;
pub mod sysinfo;
pub mod types;

/// Upstream package version this crate is ported against.
pub const UPSTREAM_VERSION: &str = "0.6.1";

/// Upstream git commit this crate is ported against.
pub const UPSTREAM_COMMIT: &str = "f37be3edbee60f3a09a5183788b91eb19f0c07d1";
