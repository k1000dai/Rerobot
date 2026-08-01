#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod config;
pub mod data;
pub mod device;
pub mod error;
pub mod limits;
pub mod model;
pub mod optim;
pub mod processor;
pub mod run;

/// The tensor runtime this crate is built on.
///
/// Re-exported because candle's types appear in this crate's public API —
/// [`candle_core::Device`] is an argument to [`model::act::ActModel::new`] and
/// [`candle_core::Tensor`] is what a forward pass returns — and a caller cannot
/// name them without depending on the exact same candle version, which the
/// `=0.9.1` pin in this crate's manifest makes awkward to do by hand.
pub use candle_core;

/// Re-exported for the same reason: [`data::dataset::StateOnlyDataset::load`]
/// takes an [`indexmap::IndexMap`] of delta timestamps.
pub use indexmap;

/// Upstream package version this crate is ported against.
pub const UPSTREAM_VERSION: &str = rerobot_core::UPSTREAM_VERSION;

/// Upstream git commit this crate is ported against.
pub const UPSTREAM_COMMIT: &str = rerobot_core::UPSTREAM_COMMIT;
