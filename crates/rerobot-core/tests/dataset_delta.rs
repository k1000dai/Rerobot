//! Behaviour parity tests for the delta-timestamp window slice of
//! `lerobot.datasets`, at commit f37be3edbee60f3a09a5183788b91eb19f0c07d1:
//!
//! * `feature_utils.get_delta_indices`,
//! * `feature_utils.check_delta_timestamps`,
//! * `dataset_reader.DatasetReader._get_query_indices`.
//!
//! These three are the whole of ACT's action-chunk construction: the config's
//! `action_delta_indices` become timestamps in `datasets.factory`, the reader
//! turns them back into frame indices clamped to the episode, and the clamped
//! entries are reported through `action_is_pad`.

use indexmap::IndexMap;
use rerobot_core::dataset::delta::{
    action_delta_timestamps, check_delta_timestamps, get_delta_indices, python_round_half_even,
    query_window, DEFAULT_TOLERANCE_S,
};

fn map(pairs: &[(&str, &[f64])]) -> IndexMap<String, Vec<f64>> {
    pairs
        .iter()
        .map(|(key, values)| ((*key).to_owned(), values.to_vec()))
        .collect()
}

// ---------------------------------------------------------------------------
// `round`, the Python builtin
// ---------------------------------------------------------------------------

#[test]
fn rounding_is_pythons_half_to_even_not_rusts_half_away_from_zero() {
    // `f64::round` answers 1, 3, -1 and -3 here; Python's `round` answers the
    // even neighbour. `get_delta_indices` is spelled `round(d * fps)`, so the
    // difference is observable whenever a delta lands exactly between frames.
    assert_eq!(python_round_half_even(0.5), 0.0);
    assert_eq!(python_round_half_even(1.5), 2.0);
    assert_eq!(python_round_half_even(2.5), 2.0);
    assert_eq!(python_round_half_even(3.5), 4.0);
    assert_eq!(python_round_half_even(-0.5), 0.0);
    assert_eq!(python_round_half_even(-1.5), -2.0);
    assert_eq!(python_round_half_even(-2.5), -2.0);
}

#[test]
fn rounding_away_from_ties_is_ordinary_nearest() {
    assert_eq!(python_round_half_even(0.4999), 0.0);
    assert_eq!(python_round_half_even(0.5001), 1.0);
    assert_eq!(python_round_half_even(-0.5001), -1.0);
    assert_eq!(python_round_half_even(7.0), 7.0);
}

// ---------------------------------------------------------------------------
// `get_delta_indices`
// ---------------------------------------------------------------------------

#[test]
fn act_action_timestamps_are_the_chunk_range_divided_by_fps() {
    // `datasets.factory.resolve_delta_timestamps`:
    //     delta_timestamps[ACTION] = [i / ds_meta.fps for i in cfg.action_delta_indices]
    // and ACT's `action_delta_indices` is `list(range(chunk_size))`.
    assert_eq!(action_delta_timestamps(2, 10), vec![0.0, 0.1]);
    // `i / fps`, evaluated in binary64 exactly as Python's `/` does — not a
    // repeated addition of 1/fps, which would drift (0.1 + 0.2 is not 0.3).
    assert_eq!(
        action_delta_timestamps(5, 10),
        vec![0.0, 0.1, 0.2, 0.3, 0.4]
    );
    assert_ne!(action_delta_timestamps(4, 10)[3], 0.1 + 0.1 + 0.1);
    assert_eq!(action_delta_timestamps(0, 10), Vec::<f64>::new());
}

#[test]
fn delta_indices_round_trip_the_chunk_range_at_ten_fps() {
    let timestamps = map(&[("action", &action_delta_timestamps(2, 10))]);
    let indices = get_delta_indices(&timestamps, 10);
    assert_eq!(indices["action"], vec![0, 1]);
}

#[test]
fn delta_indices_survive_the_float_error_that_dividing_by_fps_introduces() {
    // 0.30000000000000004 * 10 is 3.0000000000000004, not 3.
    let timestamps = map(&[("action", &action_delta_timestamps(100, 30))]);
    let indices = get_delta_indices(&timestamps, 30);
    assert_eq!(indices["action"], (0..100).collect::<Vec<i64>>());
}

#[test]
fn delta_indices_keep_negative_history_offsets() {
    let timestamps = map(&[("observation.state", &[-0.2, -0.1, 0.0][..])]);
    let indices = get_delta_indices(&timestamps, 10);
    assert_eq!(indices["observation.state"], vec![-2, -1, 0]);
}

#[test]
fn delta_indices_preserve_key_insertion_order() {
    let timestamps = map(&[("action", &[0.0][..]), ("observation.state", &[0.0][..])]);
    let indices = get_delta_indices(&timestamps, 10);
    assert_eq!(
        indices.keys().collect::<Vec<_>>(),
        vec!["action", "observation.state"]
    );
}

// ---------------------------------------------------------------------------
// `check_delta_timestamps`
// ---------------------------------------------------------------------------

#[test]
fn timestamps_that_are_multiples_of_the_frame_period_are_within_tolerance() {
    let timestamps = map(&[("action", &action_delta_timestamps(4, 10))]);
    assert_eq!(
        check_delta_timestamps(&timestamps, 10, DEFAULT_TOLERANCE_S),
        Ok(())
    );
}

#[test]
fn a_timestamp_off_the_frame_grid_is_reported_with_only_the_offending_values() {
    let timestamps = map(&[
        ("action", &[0.0, 0.1][..]),
        ("observation.state", &[0.0, 0.05, 0.2][..]),
    ]);
    let error = check_delta_timestamps(&timestamps, 10, DEFAULT_TOLERANCE_S)
        .expect_err("0.05 s is half a frame at 10 fps");
    assert_eq!(
        error.outside_tolerance.keys().collect::<Vec<_>>(),
        vec!["observation.state"]
    );
    assert_eq!(error.outside_tolerance["observation.state"], vec![0.05]);
}

#[test]
fn the_tolerance_is_measured_in_seconds_after_dividing_by_fps() {
    // Upstream: `abs(ts * fps - round(ts * fps)) / fps <= tolerance_s`. At
    // 10 fps, 0.101 s is 1.01 frames, so the deviation is 0.01 / 10 = 0.001 s —
    // 0.0010000000000000009 once binary64 has had its say, which is why the
    // accepting bound below is above 0.001 and not equal to it.
    let timestamps = map(&[("action", &[0.101][..])]);
    assert!(check_delta_timestamps(&timestamps, 10, 0.0011).is_ok());
    assert!(check_delta_timestamps(&timestamps, 10, 0.0009).is_err());
    // The default tolerance is a tenth of that, so it rejects.
    assert!(check_delta_timestamps(&timestamps, 10, DEFAULT_TOLERANCE_S).is_err());
}

#[test]
fn the_error_message_names_the_frame_rate_like_upstream() {
    let timestamps = map(&[("action", &[0.05][..])]);
    let error = check_delta_timestamps(&timestamps, 10, DEFAULT_TOLERANCE_S).unwrap_err();
    let rendered = error.to_string();
    assert!(
        rendered.contains("multiples of 1/10"),
        "message did not name the frame rate: {rendered}"
    );
    assert!(
        rendered.contains("action"),
        "message did not name the key: {rendered}"
    );
}

// ---------------------------------------------------------------------------
// `_get_query_indices`
// ---------------------------------------------------------------------------

#[test]
fn a_window_wholly_inside_the_episode_is_unclamped_and_unpadded() {
    let window = query_window(1, 0, 4, &[0, 1]);
    assert_eq!(window.indices, vec![1, 2]);
    assert_eq!(window.is_pad, vec![false, false]);
}

#[test]
fn a_window_running_past_the_episode_end_clamps_to_the_last_frame_and_pads() {
    // Frame 3 of a four-frame episode: `max(0, min(3, 3 + 1)) == 3`, and
    // `3 + 1 >= 4` marks it padded.
    let window = query_window(3, 0, 4, &[0, 1]);
    assert_eq!(window.indices, vec![3, 3]);
    assert_eq!(window.is_pad, vec![false, true]);
}

#[test]
fn every_frame_of_the_four_frame_fixture_produces_the_upstream_window() {
    let expected = [
        (vec![0, 1], vec![false, false]),
        (vec![1, 2], vec![false, false]),
        (vec![2, 3], vec![false, false]),
        (vec![3, 3], vec![false, true]),
    ];
    for (frame, (indices, is_pad)) in expected.iter().enumerate() {
        let window = query_window(frame as i64, 0, 4, &[0, 1]);
        assert_eq!(&window.indices, indices, "frame {frame} indices");
        assert_eq!(&window.is_pad, is_pad, "frame {frame} padding");
    }
}

#[test]
fn a_window_running_before_the_episode_start_clamps_to_the_first_frame_and_pads() {
    let window = query_window(0, 0, 4, &[-2, -1, 0]);
    assert_eq!(window.indices, vec![0, 0, 0]);
    assert_eq!(window.is_pad, vec![true, true, false]);
}

#[test]
fn clamping_is_relative_to_the_episode_not_the_dataset() {
    // Second episode of a dataset, occupying absolute frames 4..8.
    let window = query_window(4, 4, 8, &[-1, 0, 4]);
    assert_eq!(window.indices, vec![4, 4, 7]);
    assert_eq!(window.is_pad, vec![true, false, true]);
}

#[test]
fn a_chunk_longer_than_the_episode_pads_every_frame_past_the_end() {
    let window = query_window(0, 0, 2, &[0, 1, 2, 3]);
    assert_eq!(window.indices, vec![0, 1, 1, 1]);
    assert_eq!(window.is_pad, vec![false, false, true, true]);
}

#[test]
fn a_single_frame_episode_clamps_the_whole_chunk_onto_itself() {
    let window = query_window(0, 0, 1, &[0, 1]);
    assert_eq!(window.indices, vec![0, 0]);
    assert_eq!(window.is_pad, vec![false, true]);
}

#[test]
fn a_degenerate_episode_range_cannot_overflow_the_clamp() {
    // `query_window` computes `ep_end - 1`. With `ep_end == i64::MIN` that subtraction
    // underflows: a panic in a checked build and a wrap to `i64::MAX` in release,
    // which would silently clamp a window onto the wrong frames. The episode table is
    // attacker-controlled parquet, so the value is reachable.
    //
    // Saturating is the right answer rather than an error, because this is the
    // arithmetic layer: a range this malformed is refused by
    // `StateOnlyDataset::load`, and what this function must guarantee is only that it
    // never wraps or panics on the way there.
    let window = query_window(0, i64::MIN, i64::MIN, &[0, 1]);
    assert_eq!(window.indices.len(), 2);
    assert!(
        window.is_pad.iter().all(|padded| *padded),
        "an empty episode has no unpadded frame"
    );

    // The other three corners of the same arithmetic.
    let window = query_window(i64::MAX, 0, 4, &[0, 1]);
    assert!(window.indices.iter().all(|index| *index == 3));
    assert!(window.is_pad.iter().all(|padded| *padded));

    let window = query_window(i64::MIN, 0, 4, &[-1, 0]);
    assert!(window.indices.iter().all(|index| *index == 0));
    assert!(window.is_pad.iter().all(|padded| *padded));

    let window = query_window(0, i64::MAX, i64::MAX, &[0]);
    assert_eq!(window.indices, vec![i64::MAX]);
    assert_eq!(window.is_pad, vec![true]);

    // And the delta itself may be extreme.
    let window = query_window(0, 0, 4, &[i64::MIN, i64::MAX]);
    assert_eq!(window.indices, vec![0, 3]);
    assert_eq!(window.is_pad, vec![true, true]);
}

#[test]
fn an_empty_delta_list_produces_an_empty_window() {
    let window = query_window(0, 0, 4, &[]);
    assert!(window.indices.is_empty());
    assert!(window.is_pad.is_empty());
}
