//! Ports of `lerobot.rollout` units that are pure state, with no IO, no
//! hardware, and no policy inference.
//!
//! Only the DAgger *event state machine* lives here. The rollout strategies
//! themselves — the policy loop, teleoperator handover, dataset recording, and
//! the keyboard/pedal listeners that feed this state machine — are not ported.
//!
//! `RolloutRingBuffer`, the other ported piece of this upstream family, is at
//! [`crate::ring_buffer`]; it predates this module and its path is kept so that
//! it does not move under callers.

pub mod dagger;
