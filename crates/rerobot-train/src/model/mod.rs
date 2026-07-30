//! The ACT tensor model.
//!
//! Split so that the parts with independent contracts are independently testable:
//! [`ops`] is "PyTorch operators on candle", [`params`] is "upstream's
//! `state_dict` names and torch's initialization distributions", and [`act`] is
//! the architecture that composes them.

/// The Action Chunking Transformer itself.
pub mod act;
/// The PyTorch operators it is built from.
pub mod ops;
/// The named parameter store and its initialization.
pub mod params;
