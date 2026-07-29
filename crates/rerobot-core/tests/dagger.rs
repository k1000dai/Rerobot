//! Behaviour parity tests for the DAgger event state machine, derived from
//! `lerobot/rollout/strategies/dagger.py` lines 83-159 at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1.

use rerobot_core::rollout::dagger::{
    DAggerEvents, DAggerPhase, EventFlag, CORRECTION_EVENT, PAUSE_RESUME_EVENT,
};
use std::str::FromStr;
use std::sync::Barrier;

#[test]
fn phase_values_are_the_upstream_member_values() {
    assert_eq!(DAggerPhase::Autonomous.as_str(), "autonomous");
    assert_eq!(DAggerPhase::Paused.as_str(), "paused");
    assert_eq!(DAggerPhase::Correcting.as_str(), "correcting");
}

#[test]
fn phases_are_listed_in_upstream_declaration_order() {
    assert_eq!(
        DAggerPhase::all(),
        &[
            DAggerPhase::Autonomous,
            DAggerPhase::Paused,
            DAggerPhase::Correcting
        ]
    );
}

#[test]
fn phase_display_is_the_member_value_not_pythons_repr() {
    // Upstream is a plain `enum.Enum`, so Python's own `str()` would render
    // `DAggerPhase.AUTONOMOUS`. `Display` here is the member *value*, which is
    // the string the state machine is defined in terms of; see the module docs.
    assert_eq!(DAggerPhase::Autonomous.to_string(), "autonomous");
    assert_eq!(DAggerPhase::Correcting.to_string(), "correcting");
}

#[test]
fn a_phase_is_looked_up_by_value_exactly() {
    // `DAggerPhase("paused")` is the upstream lookup; it is by value, and it is
    // case-sensitive.
    assert_eq!(DAggerPhase::from_str("paused"), Ok(DAggerPhase::Paused));
    let err = DAggerPhase::from_str("PAUSED").unwrap_err();
    assert_eq!(err.enum_name, "DAggerPhase");
    assert_eq!(err.value, "PAUSED");
    assert_eq!(err.to_string(), "'PAUSED' is not a valid DAggerPhase");
    // Member *names* are not values, which is what upstream's `Enum` call does.
    assert!(DAggerPhase::from_str("AUTONOMOUS").is_err());
    assert!(DAggerPhase::from_str("").is_err());
}

#[test]
fn a_fresh_events_object_starts_autonomous() {
    assert_eq!(DAggerEvents::new().phase(), DAggerPhase::Autonomous);
}

#[test]
fn a_fresh_events_object_has_nothing_pending() {
    assert_eq!(DAggerEvents::new().consume_transition(), None);
}

#[test]
fn the_session_flags_start_clear() {
    let events = DAggerEvents::new();
    assert!(!events.stop_recording.is_set());
    assert!(!events.upload_requested.is_set());
}

#[test]
fn a_flag_reports_set_and_clear_independently_of_the_other_flag() {
    let events = DAggerEvents::new();

    events.stop_recording.set();
    assert!(events.stop_recording.is_set());
    assert!(!events.upload_requested.is_set());

    events.stop_recording.set(); // setting a set flag is idempotent
    assert!(events.stop_recording.is_set());

    events.stop_recording.clear();
    assert!(!events.stop_recording.is_set());
    events.stop_recording.clear(); // clearing a clear flag is idempotent
    assert!(!events.stop_recording.is_set());

    assert!(!EventFlag::new().is_set());
}

#[test]
fn pause_resume_moves_autonomous_to_paused() {
    let events = DAggerEvents::new();
    events.request_transition("pause_resume");
    // Requesting does not move the machine; only consuming does.
    assert_eq!(events.phase(), DAggerPhase::Autonomous);

    assert_eq!(
        events.consume_transition(),
        Some((DAggerPhase::Autonomous, DAggerPhase::Paused))
    );
    assert_eq!(events.phase(), DAggerPhase::Paused);
}

#[test]
fn pause_resume_moves_paused_back_to_autonomous() {
    let events = DAggerEvents::new();
    events.set_phase(DAggerPhase::Paused);
    events.request_transition("pause_resume");
    assert_eq!(
        events.consume_transition(),
        Some((DAggerPhase::Paused, DAggerPhase::Autonomous))
    );
    assert_eq!(events.phase(), DAggerPhase::Autonomous);
}

#[test]
fn correction_moves_paused_to_correcting() {
    let events = DAggerEvents::new();
    events.set_phase(DAggerPhase::Paused);
    events.request_transition("correction");
    assert_eq!(
        events.consume_transition(),
        Some((DAggerPhase::Paused, DAggerPhase::Correcting))
    );
    assert_eq!(events.phase(), DAggerPhase::Correcting);
}

#[test]
fn correction_moves_correcting_back_to_paused() {
    let events = DAggerEvents::new();
    events.set_phase(DAggerPhase::Correcting);
    events.request_transition("correction");
    assert_eq!(
        events.consume_transition(),
        Some((DAggerPhase::Correcting, DAggerPhase::Paused))
    );
    assert_eq!(events.phase(), DAggerPhase::Paused);
}

/// The five `(phase, event)` pairs the upstream table deliberately omits.
const INVALID_PAIRS: &[(DAggerPhase, &str)] = &[
    (DAggerPhase::Autonomous, "correction"),
    (DAggerPhase::Correcting, "pause_resume"),
    (DAggerPhase::Autonomous, "upload"),
    (DAggerPhase::Paused, "upload"),
    (DAggerPhase::Correcting, "upload"),
];

#[test]
fn a_transition_that_is_not_in_the_table_is_never_enqueued() {
    for (phase, event) in INVALID_PAIRS {
        let events = DAggerEvents::new();
        events.set_phase(*phase);
        events.request_transition(event);
        assert_eq!(
            events.consume_transition(),
            None,
            "({phase}, {event}) is not an upstream transition"
        );
        assert_eq!(events.phase(), *phase, "({phase}, {event}) moved the phase");
    }
}

#[test]
fn reset_returns_the_machine_to_a_fresh_session() {
    let events = DAggerEvents::new();
    events.set_phase(DAggerPhase::Correcting);
    events.request_transition("correction");

    events.reset();

    assert_eq!(events.phase(), DAggerPhase::Autonomous);
    assert_eq!(events.consume_transition(), None, "pending survived reset");
}

#[test]
fn reset_clears_upload_requested_but_deliberately_not_stop_recording() {
    // Upstream's `reset` clears `upload_requested` and says nothing about
    // `stop_recording`, so a session stopped with ESC stays stopped across a
    // reset. That asymmetry is ported, not corrected.
    let events = DAggerEvents::new();
    events.upload_requested.set();
    events.stop_recording.set();

    events.reset();

    assert!(!events.upload_requested.is_set());
    assert!(
        events.stop_recording.is_set(),
        "upstream reset does not clear stop_recording"
    );
}

#[test]
fn a_later_valid_request_overwrites_the_pending_one() {
    let events = DAggerEvents::new();
    events.set_phase(DAggerPhase::Paused);
    events.request_transition("pause_resume");
    events.request_transition("correction"); // both valid from PAUSED

    assert_eq!(
        events.consume_transition(),
        Some((DAggerPhase::Paused, DAggerPhase::Correcting)),
        "the last valid request wins"
    );
}

#[test]
fn an_invalid_request_does_not_clear_a_valid_pending_one() {
    let events = DAggerEvents::new();
    events.request_transition("pause_resume"); // valid from AUTONOMOUS
    events.request_transition("correction"); // invalid from AUTONOMOUS
    events.request_transition("nonsense");
    events.request_transition("");

    assert_eq!(
        events.consume_transition(),
        Some((DAggerPhase::Autonomous, DAggerPhase::Paused)),
        "an ignored request must not drop the pending one"
    );
}

#[test]
fn consuming_a_transition_clears_it() {
    let events = DAggerEvents::new();
    events.request_transition("pause_resume");

    assert!(events.consume_transition().is_some());
    assert_eq!(
        events.consume_transition(),
        None,
        "a consumed request must not fire twice"
    );
    assert_eq!(events.phase(), DAggerPhase::Paused);
}

#[test]
fn a_pending_request_invalidated_by_a_phase_change_is_consumed_and_yields_nothing() {
    let events = DAggerEvents::new();
    events.request_transition("pause_resume"); // valid from AUTONOMOUS
    events.set_phase(DAggerPhase::Correcting); // ... but not from CORRECTING

    assert_eq!(
        events.consume_transition(),
        None,
        "the request is revalidated against the phase at consume time"
    );
    assert_eq!(
        events.phase(),
        DAggerPhase::Correcting,
        "an invalidated request must not move the phase"
    );

    // It was still consumed: it must not fire later, once its phase returns.
    events.set_phase(DAggerPhase::Autonomous);
    assert_eq!(
        events.consume_transition(),
        None,
        "an invalidated request must be dropped, not held"
    );
    assert_eq!(events.phase(), DAggerPhase::Autonomous);
}

#[test]
fn the_event_names_are_the_upstream_spellings_and_are_what_the_machine_accepts() {
    assert_eq!(PAUSE_RESUME_EVENT, "pause_resume");
    assert_eq!(CORRECTION_EVENT, "correction");

    let events = DAggerEvents::new();
    events.request_transition(PAUSE_RESUME_EVENT);
    assert_eq!(
        events.consume_transition(),
        Some((DAggerPhase::Autonomous, DAggerPhase::Paused))
    );
    events.request_transition(CORRECTION_EVENT);
    assert_eq!(
        events.consume_transition(),
        Some((DAggerPhase::Paused, DAggerPhase::Correcting))
    );
}

#[test]
fn the_full_upstream_cycle_runs_end_to_end() {
    let events = DAggerEvents::new();
    let cycle = [
        (
            PAUSE_RESUME_EVENT,
            DAggerPhase::Autonomous,
            DAggerPhase::Paused,
        ),
        (
            CORRECTION_EVENT,
            DAggerPhase::Paused,
            DAggerPhase::Correcting,
        ),
        (
            CORRECTION_EVENT,
            DAggerPhase::Correcting,
            DAggerPhase::Paused,
        ),
        (
            PAUSE_RESUME_EVENT,
            DAggerPhase::Paused,
            DAggerPhase::Autonomous,
        ),
    ];

    for (event, old, new) in cycle {
        assert_eq!(events.phase(), old);
        events.request_transition(event);
        assert_eq!(events.consume_transition(), Some((old, new)));
        assert_eq!(events.phase(), new);
    }

    assert_eq!(
        events.phase(),
        DAggerPhase::Autonomous,
        "back where it began"
    );
    assert_eq!(events.consume_transition(), None);
}

/// The four upstream transitions, as a lookup for the concurrency tests.
fn is_upstream_transition(old: DAggerPhase, new: DAggerPhase) -> bool {
    matches!(
        (old, new),
        (DAggerPhase::Autonomous, DAggerPhase::Paused)
            | (DAggerPhase::Paused, DAggerPhase::Autonomous)
            | (DAggerPhase::Paused, DAggerPhase::Correcting)
            | (DAggerPhase::Correcting, DAggerPhase::Paused)
    )
}

#[test]
fn the_event_container_is_shareable_across_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DAggerEvents>();
    assert_send_sync::<EventFlag>();
}

#[test]
fn concurrent_requests_only_ever_yield_a_consistent_chain_of_table_transitions() {
    // Two input threads issue the two requests valid from PAUSED at the same
    // instant. The one-slot semantics allow either to win, but the result must
    // be one complete upstream transition rather than torn state.
    let events = DAggerEvents::new();
    events.set_phase(DAggerPhase::Paused);
    let start = Barrier::new(3);

    std::thread::scope(|scope| {
        let events = &events;
        let start = &start;
        for event in [PAUSE_RESUME_EVENT, CORRECTION_EVENT] {
            scope.spawn(move || {
                start.wait();
                events.request_transition(event);
            });
        }
        start.wait();
    });

    let pair = events
        .consume_transition()
        .expect("one valid request must win");
    assert_eq!(pair.0, DAggerPhase::Paused);
    assert!(is_upstream_transition(pair.0, pair.1));
    assert_eq!(events.phase(), pair.1);
    assert_eq!(events.consume_transition(), None);
}

#[test]
fn concurrent_consumers_never_replay_or_invent_a_transition() {
    // Exactly one of two simultaneous consumers may take the one pending slot.
    // Both workers are intrinsically finite, including every panic path.
    let events = DAggerEvents::new();
    events.request_transition(PAUSE_RESUME_EVENT);
    let start = Barrier::new(3);

    let results = std::thread::scope(|scope| {
        let events = &events;
        let start = &start;
        let consumers: Vec<_> = (0..2)
            .map(|_| {
                scope.spawn(move || {
                    start.wait();
                    events.consume_transition()
                })
            })
            .collect();

        start.wait();
        consumers
            .into_iter()
            .map(|consumer| consumer.join().expect("consumer thread panicked"))
            .collect::<Vec<_>>()
    });

    assert_eq!(results.iter().filter(|result| result.is_some()).count(), 1);
    assert!(results.contains(&Some((DAggerPhase::Autonomous, DAggerPhase::Paused))));
    assert_eq!(events.phase(), DAggerPhase::Paused);
    assert_eq!(events.consume_transition(), None);
}

#[test]
fn flags_survive_concurrent_setting_clearing_and_reading() {
    const THREADS: usize = 8;
    const ROUNDS: usize = 5_000;

    let events = DAggerEvents::new();
    std::thread::scope(|scope| {
        let events = &events;
        for t in 0..THREADS {
            scope.spawn(move || {
                for _ in 0..ROUNDS {
                    if t % 2 == 0 {
                        events.upload_requested.set();
                    } else {
                        events.upload_requested.clear();
                    }
                    events.stop_recording.set();
                    let _ = events.stop_recording.is_set();
                }
            });
        }
    });

    // Every thread sets `stop_recording` and none clears it, so it is set.
    assert!(events.stop_recording.is_set());
    // `upload_requested` raced, so only the last write is known — but a final
    // uncontended write must still be observable, which a lost update is not.
    events.upload_requested.set();
    assert!(events.upload_requested.is_set());
    events.upload_requested.clear();
    assert!(!events.upload_requested.is_set());
}

#[test]
fn a_reset_racing_requests_still_leaves_a_fresh_machine() {
    // A correction requested before reset is cleared by reset; requested after
    // reset it is invalid from AUTONOMOUS. Either interleaving therefore ends in
    // the same fresh state. Both workers perform one operation and always exit.
    let events = DAggerEvents::new();
    events.set_phase(DAggerPhase::Correcting);
    events.upload_requested.set();
    let start = Barrier::new(3);

    std::thread::scope(|scope| {
        let events = &events;
        let start = &start;
        scope.spawn(move || {
            start.wait();
            events.request_transition(CORRECTION_EVENT);
        });
        scope.spawn(move || {
            start.wait();
            events.reset();
        });
        start.wait();
    });

    assert_eq!(events.phase(), DAggerPhase::Autonomous);
    assert_eq!(events.consume_transition(), None);
    assert!(!events.upload_requested.is_set());
}
