//! Reading a state-only LeRobot v3.0 dataset from a local directory.
//!
//! Layered deliberately: [`parquet`] knows arrow types and nothing about
//! LeRobot, [`meta`] knows `meta/` and nothing about tensors, [`dataset`] knows
//! frames and delta windows, and [`batch`] is the only layer that touches candle.

/// Collating frames into tensors.
pub mod batch;
/// The frame-level dataset and its delta windows.
pub mod dataset;
/// Everything under `meta/`.
pub mod meta;
/// The narrow parquet reader the slice needs.
pub mod parquet;
