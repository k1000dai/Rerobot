//! Reading a local LeRobot v3.0 dataset from a local directory, including embedded
//! PNG/JPEG camera columns.
//!
//! Layered deliberately: [`parquet`] knows arrow types and nothing about
//! LeRobot, [`meta`] knows `meta/` and nothing about tensors, [`dataset`] knows
//! frames and delta windows, and [`batch`] and [`image`] are the only layers that
//! touch candle.

/// Collating frames into tensors.
pub mod batch;
/// The frame-level dataset and its delta windows.
pub mod dataset;
/// The camera-tensor contract and its normalization.
pub mod image;
/// Everything under `meta/`.
pub mod meta;
/// The narrow parquet reader the slice needs.
pub mod parquet;
