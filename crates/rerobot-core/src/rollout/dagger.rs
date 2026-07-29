//! Port of the DAgger event state machine in
//! `lerobot.rollout.strategies.dagger` (`DAggerPhase`, `_DAGGER_TRANSITIONS`,
//! `DAggerEvents` — lines 83-159 of the pinned upstream module).
//!
//! DAgger alternates between a policy driving the robot and a human correcting
//! it. Input-device threads request phase changes; the main loop consumes them.
//! This module is that hand-off and nothing else.
//!
//! # What is *not* here
//!
//! The rest of `dagger.py` is not ported: `DAggerStrategy` itself, the keyboard
//! and pedal listeners that call [`DAggerEvents::request_transition`], the
//! teleoperator handover, episode recording and hub upload, and the policy
//! inference loop. None of them are stubbed or simulated. This module holds no
//! robot state, performs no IO, and starts no threads of its own.
//!
//! # Transitions
//!
//! The four valid `(phase, event)` pairs, and no others:
//!
//! | From | Event | To |
//! | --- | --- | --- |
//! | `autonomous` | `pause_resume` | `paused` |
//! | `paused` | `pause_resume` | `autonomous` |
//! | `paused` | `correction` | `correcting` |
//! | `correcting` | `correction` | `paused` |
//!
//! A request is validated twice — once against the phase observed under the
//! lock when it is made, and again against the phase at consume time — because
//! the phase can be moved by [`DAggerEvents::set_phase`] or [`DAggerEvents::reset`]
//! in between. There is exactly one pending slot, so a later valid request
//! overwrites an earlier one.
//!
//! # Upstream behaviour reproduced deliberately
//!
//! * [`DAggerEvents::reset`] clears `upload_requested` and **not**
//!   `stop_recording`: a session stopped with ESC stays stopped across a reset.
//! * An invalid request is ignored *without* clearing a valid pending request.
//! * A pending request invalidated by a phase change is still consumed — it
//!   yields `None` and is dropped rather than held until its phase returns.
//!
//! # Compatibility boundaries
//!
//! * Upstream's `DAggerPhase` is a plain `enum.Enum`, not a `str`-backed one
//!   like those in [`crate::types`]. Its members therefore have no JSON wire
//!   form — `json.dumps` refuses them — so this port deliberately implements no
//!   `serde` support: there is no upstream serialization to be compatible with.
//!   [`DAggerPhase::as_str`] and [`FromStr`] are the member `.value` and
//!   upstream's by-value lookup `DAggerPhase("paused")`; [`fmt::Display`] is
//!   that same value, which is *not* Python's `str(DAggerPhase.PAUSED)`
//!   (`"DAggerPhase.PAUSED"`).
//! * [`EventFlag`] ports the three `threading.Event` operations DAgger actually
//!   uses — `set`, `clear`, `is_set`. `wait()` and its timeout are not ported,
//!   because nothing in the upstream DAgger path blocks on these events.
//! * The state lock is a [`std::sync::Mutex`] whose poisoning is *recovered*,
//!   not propagated. A Python `threading.Lock` has no poison flag, so a panic
//!   in one thread must not turn every later call into a panic here either.
//! * `request_transition` takes an arbitrary `&str`, exactly like the Python
//!   method: unknown names are ignored rather than rejected.

use crate::types::ParseEnumError;
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

/// Observable phases of a DAgger episode.
///
/// The member values are upstream's exactly: `autonomous` (policy driving),
/// `paused` (engine paused, awaiting input) and `correcting` (human driving via
/// teleop). Upstream is a plain `enum.Enum`, so these values have no JSON wire
/// form and this type intentionally has no `serde` impl; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DAggerPhase {
    /// `autonomous`
    Autonomous,
    /// `paused`
    Paused,
    /// `correcting`
    Correcting,
}

impl DAggerPhase {
    /// Upstream member value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Autonomous => "autonomous",
            Self::Paused => "paused",
            Self::Correcting => "correcting",
        }
    }

    /// All members, in upstream declaration order.
    pub fn all() -> &'static [Self] {
        &[Self::Autonomous, Self::Paused, Self::Correcting]
    }
}

impl fmt::Display for DAggerPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DAggerPhase {
    type Err = ParseEnumError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "autonomous" => Ok(Self::Autonomous),
            "paused" => Ok(Self::Paused),
            "correcting" => Ok(Self::Correcting),
            other => Err(ParseEnumError {
                enum_name: "DAggerPhase",
                value: other.to_string(),
            }),
        }
    }
}

/// Session-level flag, standing in for the `threading.Event`s DAgger uses.
///
/// Only the three operations the upstream DAgger path uses are ported —
/// `set`, `clear` and `is_set`. `Event.wait()` is not: nothing in that path
/// blocks on these flags. Backed by an `AtomicBool`, so it needs no lock.
///
/// ```
/// use rerobot_core::rollout::dagger::EventFlag;
///
/// let flag = EventFlag::new();
/// assert!(!flag.is_set());
/// flag.set();
/// assert!(flag.is_set());
/// flag.clear();
/// assert!(!flag.is_set());
/// ```
#[derive(Debug, Default)]
pub struct EventFlag(AtomicBool);

impl EventFlag {
    /// A cleared flag, like a freshly constructed `threading.Event`.
    pub fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    /// Set the flag. Idempotent, like `Event.set`.
    pub fn set(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Clear the flag. Idempotent, like `Event.clear`.
    pub fn clear(&self) {
        self.0.store(false, Ordering::SeqCst);
    }

    /// Whether the flag is set.
    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Upstream's event name for the pause/resume control, spelled exactly as the
/// keyboard and pedal handlers spell it.
///
/// [`DAggerEvents::request_transition`] takes an arbitrary `&str`, like the
/// Python method it ports; this constant only prevents spelling drift. The
/// input-device handlers that produce these names are not ported.
pub const PAUSE_RESUME_EVENT: &str = "pause_resume";

/// Upstream's event name for the correction control.
pub const CORRECTION_EVENT: &str = "correction";

/// Port of `_DAGGER_TRANSITIONS`: the next phase for a `(phase, event)` pair,
/// or `None` when the pair is not one of the four valid transitions.
fn transition(phase: DAggerPhase, event: &str) -> Option<DAggerPhase> {
    match (phase, event) {
        (DAggerPhase::Autonomous, "pause_resume") => Some(DAggerPhase::Paused),
        (DAggerPhase::Paused, "pause_resume") => Some(DAggerPhase::Autonomous),
        (DAggerPhase::Paused, "correction") => Some(DAggerPhase::Correcting),
        (DAggerPhase::Correcting, "correction") => Some(DAggerPhase::Paused),
        _ => None,
    }
}

#[derive(Debug)]
struct State {
    phase: DAggerPhase,
    pending: Option<String>,
}

/// Thread-safe container for DAgger input device events.
///
/// Upstream's keyboard/pedal threads write transition requests; the main loop
/// consumes them. Every method takes `&self` and is safe to call concurrently:
/// the phase and the pending request live behind one lock and are always read
/// and written together, so a request can never be validated against a phase
/// other than the one it is stored or applied against.
///
/// ```
/// use rerobot_core::rollout::dagger::{DAggerEvents, DAggerPhase, CORRECTION_EVENT, PAUSE_RESUME_EVENT};
///
/// let events = DAggerEvents::new();
/// assert_eq!(events.phase(), DAggerPhase::Autonomous);
///
/// // The input-device thread asks; the main loop applies.
/// events.request_transition(PAUSE_RESUME_EVENT);
/// assert_eq!(events.phase(), DAggerPhase::Autonomous); // not yet
/// assert_eq!(
///     events.consume_transition(),
///     Some((DAggerPhase::Autonomous, DAggerPhase::Paused))
/// );
///
/// // A request that is not a transition from the current phase is ignored.
/// events.request_transition("upload");
/// events.request_transition(CORRECTION_EVENT); // valid from PAUSED
/// assert_eq!(
///     events.consume_transition(),
///     Some((DAggerPhase::Paused, DAggerPhase::Correcting))
/// );
/// assert_eq!(events.consume_transition(), None); // consumed exactly once
/// ```
#[derive(Debug)]
pub struct DAggerEvents {
    state: Mutex<State>,
    /// Stop the whole session. Set by ESC upstream, and — deliberately — not
    /// cleared by [`DAggerEvents::reset`].
    pub stop_recording: EventFlag,
    /// Push the dataset to the hub on demand. Cleared by
    /// [`DAggerEvents::reset`].
    pub upload_requested: EventFlag,
}

impl DAggerEvents {
    /// A fresh event container: phase [`DAggerPhase::Autonomous`], nothing
    /// pending, both flags clear.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                phase: DAggerPhase::Autonomous,
                pending: None,
            }),
            stop_recording: EventFlag::default(),
            upload_requested: EventFlag::default(),
        }
    }

    /// Take the state lock, recovering it if a previous holder panicked.
    ///
    /// A Python `threading.Lock` has no poison flag, so upstream keeps working
    /// after a panic inside a critical section. Recovering here keeps that
    /// behaviour instead of turning every later call into a panic; see the
    /// module docs.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Current phase of the DAgger state machine, upstream's `phase` property.
    pub fn phase(&self) -> DAggerPhase {
        self.lock().phase
    }

    /// Overwrite the phase, upstream's `phase` property setter.
    pub fn set_phase(&self, value: DAggerPhase) {
        self.lock().phase = value;
    }

    /// Request a phase transition, upstream's `request_transition`.
    ///
    /// Called from the input-device threads. The request is stored only if it
    /// is a valid transition from the phase observed under the same lock
    /// acquisition, which is what prevents impossible state changes. An
    /// invalid or unknown `event` is ignored, and — like upstream — does *not*
    /// clear a valid request that is already pending. A later valid request
    /// replaces an earlier one: there is a single pending slot.
    pub fn request_transition(&self, event: &str) {
        let mut state = self.lock();
        if transition(state.phase, event).is_some() {
            state.pending = Some(event.to_string());
        }
    }

    /// Consume a pending transition, upstream's `consume_transition`.
    ///
    /// Called from the main loop. Takes and clears the pending request, then
    /// re-checks it against the phase as it is *now*: the phase may have moved
    /// since the request was made. A request that is no longer valid returns
    /// `None` and leaves the phase untouched, but is still consumed — upstream
    /// clears `_pending_transition` before the table lookup, so an invalidated
    /// request is dropped rather than held.
    ///
    /// Returns `(old_phase, new_phase)` when the phase actually moved.
    pub fn consume_transition(&self) -> Option<(DAggerPhase, DAggerPhase)> {
        let mut state = self.lock();
        let pending = state.pending.take()?;
        let new_phase = transition(state.phase, &pending)?;
        let old_phase = state.phase;
        state.phase = new_phase;
        Some((old_phase, new_phase))
    }

    /// Reset all transient state for a fresh session.
    ///
    /// Sets the phase back to [`DAggerPhase::Autonomous`] and drops any pending
    /// request in one lock acquisition, then clears `upload_requested`.
    ///
    /// `stop_recording` is deliberately **not** cleared, because upstream does
    /// not clear it: a session stopped with ESC stays stopped across a reset.
    pub fn reset(&self) {
        {
            let mut state = self.lock();
            state.phase = DAggerPhase::Autonomous;
            state.pending = None;
        }
        // Upstream clears this one *outside* the lock, and clears
        // `stop_recording` nowhere: a session stopped with ESC stays stopped.
        self.upload_requested.clear();
    }
}

impl Default for DAggerEvents {
    fn default() -> Self {
        Self::new()
    }
}

/// Poisoning is only reachable from inside this module, because the state lock
/// is private and no public method runs caller code while holding it. These
/// tests therefore live here rather than in `tests/dagger.rs`.
#[cfg(test)]
mod tests {
    use super::*;

    /// Panic inside the critical section and catch the unwind so the test can
    /// verify that every public operation recovers the poisoned lock.
    fn poison(events: &DAggerEvents) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut state = events.state.lock().expect("not poisoned yet");
            state.phase = DAggerPhase::Paused;
            panic!("poisoning the state lock");
        }));
        assert!(result.is_err(), "the poisoning panic did not unwind");
        assert!(events.state.is_poisoned(), "the lock was not poisoned");
    }

    #[test]
    fn a_poisoned_state_lock_still_serves_every_public_operation() {
        let events = DAggerEvents::new();
        poison(&events);

        // A `threading.Lock` has no poison flag, so none of this may panic.
        assert_eq!(events.phase(), DAggerPhase::Paused);
        events.request_transition(CORRECTION_EVENT);
        assert_eq!(
            events.consume_transition(),
            Some((DAggerPhase::Paused, DAggerPhase::Correcting))
        );
        events.set_phase(DAggerPhase::Paused);
        events.reset();
        assert_eq!(events.phase(), DAggerPhase::Autonomous);
        assert_eq!(events.consume_transition(), None);
    }

    #[test]
    fn a_poisoned_state_lock_keeps_the_writes_the_panicking_section_made() {
        let events = DAggerEvents::new();
        poison(&events);
        // `into_inner` on the poison error, not a silent reset to the default.
        assert_eq!(events.phase(), DAggerPhase::Paused);
    }
}
