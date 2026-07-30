//! Policy configuration and model contracts.

/// Action Chunking Transformer configuration.
pub mod act;
/// The Draccus value conversions a checkpoint `config.json` is decoded through.
pub mod draccus;
/// The mean/std, min/max and quantile normalization transforms.
pub mod normalize;
