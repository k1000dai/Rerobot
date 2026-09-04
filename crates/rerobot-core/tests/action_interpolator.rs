//! Behaviour parity tests for `ActionInterpolator`, derived from upstream
//! `tests/policies/rtc/test_action_interpolator.py` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1.

use rerobot_core::action_interpolator::{ActionInterpolator, InterpolatorError};
use rerobot_core::BigInt;

fn interp(multiplier: i64) -> ActionInterpolator<f32> {
    ActionInterpolator::new(multiplier).expect("valid multiplier")
}

/// `2**exponent` as the unbounded integer upstream would have been handed.
fn two_pow(exponent: u32) -> BigInt {
    BigInt::from(2).pow(exponent)
}

#[test]
fn multiplier_1_is_not_enabled() {
    assert!(!interp(1).enabled());
    assert_eq!(*interp(1).multiplier(), BigInt::from(1));
}

#[test]
fn multiplier_2_is_enabled() {
    assert!(interp(2).enabled());
}

#[test]
fn multiplier_0_is_rejected() {
    assert_eq!(
        ActionInterpolator::<f32>::new(0).unwrap_err(),
        InterpolatorError::InvalidMultiplier(BigInt::from(0))
    );
}

#[test]
fn negative_multiplier_is_rejected() {
    assert_eq!(
        ActionInterpolator::<f32>::new(-5).unwrap_err(),
        InterpolatorError::InvalidMultiplier(BigInt::from(-5))
    );
}

#[test]
fn invalid_multiplier_message_matches_upstream_wording() {
    let err = ActionInterpolator::<f32>::new(0).unwrap_err();
    assert_eq!(err.to_string(), "multiplier must be >= 1, got 0");
}

#[test]
fn needs_new_action_true_initially() {
    assert!(interp(2).needs_new_action());
}

#[test]
fn needs_new_action_false_after_add() {
    let mut i = interp(2);
    i.add(&[1.0]).unwrap();
    assert!(!i.needs_new_action());
}

#[test]
fn needs_new_action_true_after_buffer_exhausted() {
    let mut i = interp(2);
    i.add(&[1.0]).unwrap();
    i.get();
    assert!(i.needs_new_action());
}

#[test]
fn needs_new_action_true_only_after_all_interpolated_steps_consumed() {
    let mut i = interp(2);
    i.add(&[0.0]).unwrap();
    i.get();
    i.add(&[1.0]).unwrap();
    i.get();
    assert!(!i.needs_new_action());
    i.get();
    assert!(i.needs_new_action());
}

#[test]
fn passthrough_single_action_returned_as_is() {
    let mut i = interp(1);
    i.add(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(i.get(), Some([1.0f32, 2.0, 3.0].as_slice()));
    assert_eq!(i.get(), None);
}

#[test]
fn passthrough_sequential_actions_never_interpolate() {
    let mut i = interp(1);
    for step in 0..3 {
        let v = [step as f32];
        i.add(&v).unwrap();
        assert_eq!(i.get(), Some(v.as_slice()));
        assert_eq!(i.get(), None);
    }
}

#[test]
fn first_action_is_not_interpolated() {
    let mut i = interp(2);
    i.add(&[0.0, 0.0]).unwrap();
    assert_eq!(i.get(), Some([0.0f32, 0.0].as_slice()));
    assert_eq!(i.get(), None);
}

#[test]
fn second_action_produces_two_steps_at_2x() {
    let mut i = interp(2);
    i.add(&[0.0, 0.0]).unwrap();
    i.get();
    i.add(&[2.0, 4.0]).unwrap();
    assert_eq!(i.get(), Some([1.0f32, 2.0].as_slice()));
    assert_eq!(i.get(), Some([2.0f32, 4.0].as_slice()));
    assert_eq!(i.get(), None);
}

#[test]
fn three_consecutive_actions_at_2x() {
    let mut i = interp(2);
    i.add(&[0.0]).unwrap();
    assert_eq!(i.get(), Some([0.0f32].as_slice()));
    i.add(&[4.0]).unwrap();
    assert_eq!(i.get(), Some([2.0f32].as_slice()));
    assert_eq!(i.get(), Some([4.0f32].as_slice()));
    i.add(&[10.0]).unwrap();
    assert_eq!(i.get(), Some([7.0f32].as_slice()));
    assert_eq!(i.get(), Some([10.0f32].as_slice()));
}

#[test]
fn three_steps_at_3x() {
    let mut i = interp(3);
    i.add(&[0.0, 0.0]).unwrap();
    i.get();
    i.add(&[3.0, 6.0]).unwrap();
    assert_eq!(i.get(), Some([1.0f32, 2.0].as_slice()));
    assert_eq!(i.get(), Some([2.0f32, 4.0].as_slice()));
    assert_eq!(i.get(), Some([3.0f32, 6.0].as_slice()));
    assert_eq!(i.get(), None);
}

#[test]
fn last_interpolated_step_equals_target_exactly() {
    let mut i = interp(3);
    i.add(&[10.0]).unwrap();
    i.get();
    i.add(&[100.0]).unwrap();
    i.get();
    i.get();
    assert_eq!(i.get(), Some([100.0f32].as_slice()));
}

#[test]
fn six_dof_interpolation_halves_the_target() {
    let mut i = interp(2);
    i.add(&[0.0; 6]).unwrap();
    i.get();
    let target = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    i.add(&target).unwrap();
    assert_eq!(i.get(), Some([0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0].as_slice()));
    assert_eq!(i.get(), Some(target.as_slice()));
}

#[test]
fn reset_clears_buffer_and_previous_action() {
    let mut i = interp(2);
    i.add(&[1.0]).unwrap();
    i.reset();
    assert!(i.needs_new_action());
    assert_eq!(i.get(), None);
    // No previous action after reset, so the next add passes through.
    i.add(&[9.0]).unwrap();
    assert_eq!(i.get(), Some([9.0f32].as_slice()));
    assert_eq!(i.get(), None);
}

#[test]
fn get_returns_none_before_any_add() {
    let mut i = interp(2);
    assert_eq!(i.get(), None);
}

#[test]
fn control_interval_divides_by_multiplier() {
    assert_eq!(interp(1).get_control_interval(30.0).unwrap(), 1.0 / 30.0);
    assert_eq!(interp(2).get_control_interval(30.0).unwrap(), 1.0 / 60.0);
    assert_eq!(interp(3).get_control_interval(30.0).unwrap(), 1.0 / 90.0);
    assert_eq!(interp(2).get_control_interval(60.0).unwrap(), 1.0 / 120.0);
}

#[test]
fn control_loop_emits_multiplier_times_actions_after_the_first() {
    let mut i = interp(3);
    let mut emitted = 0usize;
    for step in 0..4 {
        if i.needs_new_action() {
            i.add(&[step as f32]).unwrap();
        }
        while i.get().is_some() {
            emitted += 1;
        }
    }
    // First add passes through (1), the remaining three yield 3 each.
    assert_eq!(emitted, 1 + 3 * 3);
}

#[test]
fn interpolated_sequence_is_monotonic_for_monotonic_inputs() {
    let mut i = interp(4);
    i.add(&[0.0]).unwrap();
    i.get();
    let mut last = 0.0f32;
    for target in [1.0f32, 2.0, 3.0] {
        i.add(&[target]).unwrap();
        while let Some(a) = i.get() {
            let v = a[0];
            assert!(v > last, "expected {v} > {last}");
            last = v;
        }
    }
    assert_eq!(last, 3.0);
}

#[test]
fn non_broadcastable_shapes_are_rejected() {
    // torch: `The size of tensor a (2) must match the size of tensor b (3) at
    // non-singleton dimension 0`, raised by `action - self._prev`.
    let mut i = interp(2);
    i.add(&[0.0, 0.0, 0.0]).unwrap();
    i.get();
    assert_eq!(
        i.add(&[1.0, 2.0]).unwrap_err(),
        InterpolatorError::NotBroadcastable {
            prev_len: 3,
            action_len: 2
        }
    );
}

#[test]
fn shape_checks_do_not_apply_before_a_previous_action_exists() {
    let mut i = interp(2);
    i.add(&[0.0, 0.0]).unwrap();
    i.reset();
    i.add(&[1.0])
        .expect("no previous action, so no shape constraint");
}

// --- broadcasting -------------------------------------------------------
//
// Upstream computes `self._prev + t * (action - self._prev)` on 1-D
// `torch.Tensor`s, so a length-1 operand broadcasts against a length-N one in
// either direction. Every expected value below was produced by executing the
// pinned upstream `ActionInterpolator` under `torch` 2.13.0 with
// `torch.float32` tensors.

#[test]
fn a_length_one_previous_broadcasts_against_a_length_n_action() {
    let mut i = interp(2);
    i.add(&[5.0]).unwrap();
    i.get();
    i.add(&[0.0, 10.0]).unwrap();
    assert_eq!(i.get(), Some([2.5f32, 7.5].as_slice()));
    assert_eq!(i.get(), Some([0.0f32, 10.0].as_slice()));
    assert_eq!(i.get(), None);
}

#[test]
fn a_length_n_previous_broadcasts_against_a_length_one_action() {
    let mut i = interp(2);
    i.add(&[0.0, 10.0]).unwrap();
    i.get();
    i.add(&[4.0]).unwrap();
    assert_eq!(i.get(), Some([2.0f32, 7.0].as_slice()));
    assert_eq!(i.get(), Some([4.0f32, 4.0].as_slice()));
    assert_eq!(i.get(), None);
}

#[test]
fn broadcasting_updates_the_previous_action_to_the_new_shape() {
    // `self._prev = action.clone()`: prev takes the *action's* length, not the
    // broadcast result's length.
    let mut i = interp(2);
    i.add(&[0.0, 10.0]).unwrap();
    i.get();
    i.add(&[4.0]).unwrap(); // prev is now length 1
    while i.get().is_some() {}
    // A length-3 action is therefore legal now: 1 broadcasts against 3.
    i.add(&[1.0, 2.0, 3.0])
        .expect("prev is length 1, so it broadcasts against length 3");
    assert_eq!(i.get(), Some([2.5f32, 3.0, 3.5].as_slice()));
    assert_eq!(i.get(), Some([1.0f32, 2.0, 3.0].as_slice()));
}

#[test]
fn a_length_one_previous_against_an_empty_action_yields_empty_steps() {
    // torch broadcasts (1,) against (0,) to (0,).
    let mut i = interp(2);
    i.add(&[1.0]).unwrap();
    i.get();
    i.add(&[]).unwrap();
    assert_eq!(i.get(), Some([].as_slice()));
    assert_eq!(i.get(), Some([].as_slice()));
    assert_eq!(i.get(), None);
}

#[test]
fn an_empty_previous_against_a_length_one_action_yields_empty_steps() {
    let mut i = interp(2);
    i.add(&[]).unwrap();
    i.get();
    i.add(&[1.0]).unwrap();
    assert_eq!(i.get(), Some([].as_slice()));
    assert_eq!(i.get(), Some([].as_slice()));
    assert_eq!(i.get(), None);
}

#[test]
fn a_longer_action_against_an_empty_previous_is_not_broadcastable() {
    let mut i = interp(2);
    i.add(&[]).unwrap();
    i.get();
    assert_eq!(
        i.add(&[1.0, 2.0]).unwrap_err(),
        InterpolatorError::NotBroadcastable {
            prev_len: 0,
            action_len: 2
        }
    );
}

#[test]
fn an_empty_action_against_a_longer_previous_is_not_broadcastable() {
    let mut i = interp(2);
    i.add(&[1.0, 2.0, 3.0]).unwrap();
    i.get();
    assert_eq!(
        i.add(&[]).unwrap_err(),
        InterpolatorError::NotBroadcastable {
            prev_len: 3,
            action_len: 0
        }
    );
}

#[test]
fn broadcast_error_message_matches_the_torch_runtime_error() {
    let mut i = interp(2);
    i.add(&[0.0, 0.0, 0.0]).unwrap();
    let err = i.add(&[1.0, 2.0]).unwrap_err();
    assert_eq!(
        err.to_string(),
        "The size of tensor a (2) must match the size of tensor b (3) \
         at non-singleton dimension 0"
    );
}

#[test]
fn three_step_broadcast_values_match_torch_bit_for_bit() {
    // Upstream under torch 2.13.0, float32, multiplier 3, prev [2.0],
    // action [1.0, 4.0, 7.0].
    let mut i = interp(3);
    i.add(&[2.0]).unwrap();
    i.get();
    i.add(&[1.0, 4.0, 7.0]).unwrap();
    assert_eq!(
        i.get(),
        Some([1.666_666_6f32, 2.666_666_7, 3.666_666_7].as_slice())
    );
    assert_eq!(
        i.get(),
        Some([1.333_333_3f32, 3.333_333_5, 5.333_333_5].as_slice())
    );
    assert_eq!(i.get(), Some([1.0f32, 4.0, 7.0].as_slice()));
    assert_eq!(i.get(), None);
}

// --- state after a failed `add` -----------------------------------------

#[test]
fn a_failed_add_clears_the_buffer_before_reporting_the_error() {
    // Upstream assigns `self._buffer = []` *before* the loop that raises, so
    // the not-yet-consumed tail is gone once the error surfaces.
    let mut i = interp(3);
    i.add(&[0.0, 0.0, 0.0]).unwrap();
    assert!(!i.needs_new_action());
    i.add(&[1.0, 2.0]).unwrap_err();
    assert_eq!(i.get(), None, "the buffer must be empty after the error");
    assert!(i.needs_new_action());
}

#[test]
fn a_failed_add_leaves_the_previous_action_untouched() {
    // `self._prev` is only reassigned after the loop, so the failed action is
    // not remembered and the next `add` is checked against the old prev.
    let mut i = interp(2);
    i.add(&[0.0, 0.0, 0.0]).unwrap();
    i.get();
    i.add(&[1.0, 2.0]).unwrap_err();

    // Still length 3, verified against upstream: prev [0,0,0] + action [9.0].
    i.add(&[9.0]).unwrap();
    assert_eq!(i.get(), Some([4.5f32, 4.5, 4.5].as_slice()));
    assert_eq!(i.get(), Some([9.0f32, 9.0, 9.0].as_slice()));
    assert_eq!(i.get(), None);
}

#[test]
fn a_failed_add_does_not_make_a_previously_illegal_shape_legal() {
    let mut i = interp(2);
    i.add(&[0.0, 0.0, 0.0]).unwrap();
    i.add(&[1.0, 2.0]).unwrap_err();
    assert_eq!(
        i.add(&[1.0, 2.0]).unwrap_err(),
        InterpolatorError::NotBroadcastable {
            prev_len: 3,
            action_len: 2
        }
    );
}

#[test]
fn empty_action_is_accepted_and_stays_empty() {
    let mut i = interp(2);
    i.add(&[]).unwrap();
    assert_eq!(i.get(), Some([].as_slice()));
    i.add(&[]).unwrap();
    assert_eq!(i.get(), Some([].as_slice()));
    assert_eq!(i.get(), Some([].as_slice()));
    assert_eq!(i.get(), None);
}

#[test]
fn f32_and_f64_differ_at_the_one_third_weight() {
    let mut a: ActionInterpolator<f32> = ActionInterpolator::new(3).unwrap();
    a.add(&[0.0]).unwrap();
    a.get();
    a.add(&[1.0]).unwrap();
    let thirty_two = a.get().unwrap()[0];

    let mut b: ActionInterpolator<f64> = ActionInterpolator::new(3).unwrap();
    b.add(&[0.0]).unwrap();
    b.get();
    b.add(&[1.0]).unwrap();
    let sixty_four = b.get().unwrap()[0];

    assert_eq!(thirty_two, 1.0f32 / 3.0);
    assert_eq!(sixty_four, 1.0f64 / 3.0);
    assert_ne!(f64::from(thirty_two), sixty_four);
}

#[test]
fn f32_weight_is_narrowed_before_multiplication() {
    // PyTorch casts a Python `float` scalar to the tensor's dtype before the
    // op, so t = 1/3 is narrowed to f32 *first*. These operands were found by
    // search precisely because the two orders disagree on them, and the
    // expected value is what upstream produced under torch 2.13.0.
    // Shortest round-trip spellings of the f64 values the oracle printed
    // (15.420589447021484, -20.663904190063477, 3.392423629760742 and
    // 3.3924245834350586); each parses to the identical f32.
    const PREV: f32 = 15.420_589;
    const ACTION: f32 = -20.663_904;
    const NARROW_FIRST: f32 = 3.392_423_6;
    const WIDE_THEN_NARROW: f32 = 3.392_424_6;
    assert_ne!(NARROW_FIRST, WIDE_THEN_NARROW, "operands must discriminate");

    let mut a: ActionInterpolator<f32> = ActionInterpolator::new(3).unwrap();
    a.add(&[PREV]).unwrap();
    a.get();
    a.add(&[ACTION]).unwrap();
    assert_eq!(a.get(), Some([NARROW_FIRST].as_slice()));
}

// --- the multiplier's `int` domain --------------------------------------
//
// `ActionInterpolator.__init__` stores whatever Python `int` it is handed, and
// the only thing it checks is `multiplier < 1`. A Python `int` is unbounded, so
// no fixed Rust width is the domain: `i64` truncates at `2**63`, which is a
// value `int` holds exactly. Storage, the getter, `enabled` and the control
// interval are therefore all exact at every magnitude, and the two operations
// that cannot be — building a Rust buffer, and converting to a float — say so
// explicitly rather than wrapping, clamping or going quietly infinite.

#[test]
fn a_multiplier_at_two_to_the_sixty_three_is_stored_exactly() {
    // The first value an `i64` cannot hold, and an unremarkable Python `int`.
    let big: ActionInterpolator<f32> = ActionInterpolator::new(two_pow(63)).unwrap();
    assert_eq!(*big.multiplier(), two_pow(63));
    assert_eq!(big.multiplier().to_string(), "9223372036854775808");
    assert!(big.enabled());
}

#[test]
fn a_multiplier_far_beyond_every_machine_integer_is_stored_exactly() {
    // 10**100, which no fixed-width Rust integer comes close to.
    let googol = BigInt::from(10).pow(100);
    let big: ActionInterpolator<f32> = ActionInterpolator::new(googol.clone()).unwrap();
    assert_eq!(*big.multiplier(), googol);
    assert!(big.enabled());
    // Exact, not rounded: it is still one more than 10**100 - 1.
    assert_eq!(big.multiplier() - BigInt::from(1), googol - BigInt::from(1));
}

#[test]
fn a_negative_multiplier_far_below_i64_min_is_rejected_carrying_its_exact_value() {
    let very_negative = -two_pow(200);
    let err = ActionInterpolator::<f32>::new(very_negative.clone()).unwrap_err();
    assert_eq!(
        err,
        InterpolatorError::InvalidMultiplier(very_negative.clone())
    );
    // Upstream's f-string interpolates the `int` itself, digits and all.
    assert_eq!(
        err.to_string(),
        format!("multiplier must be >= 1, got {very_negative}")
    );
}

#[test]
fn control_interval_is_exact_for_a_multiplier_past_i64() {
    let big: ActionInterpolator<f32> = ActionInterpolator::new(two_pow(100)).unwrap();
    assert_eq!(
        big.get_control_interval(30.0).unwrap(),
        1.0 / (30.0 * 2f64.powi(100))
    );
}

#[test]
fn control_interval_fails_exactly_where_pythons_int_to_float_conversion_does() {
    // CPython evaluates `fps * self.multiplier` by converting the `int` to a
    // double first, which raises `OverflowError: int too large to convert to
    // float` once the value no longer fits. `2**1023` is the largest power of
    // two that does; `2**1024` is the first that does not.
    let ok: ActionInterpolator<f32> = ActionInterpolator::new(two_pow(1023)).unwrap();
    assert_eq!(
        ok.get_control_interval(30.0).unwrap(),
        1.0 / (30.0 * 2f64.powi(1023))
    );

    let over: ActionInterpolator<f32> = ActionInterpolator::new(two_pow(1024)).unwrap();
    let err = over.get_control_interval(30.0).unwrap_err();
    assert_eq!(
        err,
        InterpolatorError::MultiplierNotFloatRepresentable(two_pow(1024))
    );
    assert_eq!(err.to_string(), "int too large to convert to float");
    // The multiplier itself is untouched by the failed conversion.
    assert_eq!(*over.multiplier(), two_pow(1024));
}

#[test]
fn a_huge_multiplier_still_passes_the_first_action_through() {
    // The first `add` takes upstream's `else` branch, which never touches the
    // multiplier, so it succeeds no matter how large the multiplier is.
    let mut big: ActionInterpolator<f32> = ActionInterpolator::new(two_pow(200)).unwrap();
    big.add(&[1.0, 2.0]).unwrap();
    assert_eq!(big.get(), Some([1.0f32, 2.0].as_slice()));
    assert_eq!(big.get(), None);
}

#[test]
fn add_reports_a_multiplier_whose_buffer_cannot_be_allocated() {
    // `2**60` interpolated steps is past what any allocator will hand out, so
    // the buffer cannot be built. Upstream would grind through a `list` of that
    // length and end in `MemoryError`; the port refuses up front instead of
    // truncating the step count to something it can index.
    let mut big: ActionInterpolator<f32> = ActionInterpolator::new(two_pow(60)).unwrap();
    big.add(&[0.0]).unwrap();
    big.get();
    assert_eq!(
        big.add(&[1.0]).unwrap_err(),
        InterpolatorError::BufferNotAllocatable {
            multiplier: two_pow(60)
        }
    );
    // Same post-failure state as every other failed `add`: buffer cleared,
    // previous action untouched.
    assert_eq!(big.get(), None);
    assert!(big.needs_new_action());
}

#[test]
fn add_reports_a_multiplier_that_no_machine_word_can_even_count_to() {
    // `2**128` cannot be converted to a `usize` on any target Rust has, so the
    // step count is rejected at the conversion rather than wrapping.
    let mut big: ActionInterpolator<f32> = ActionInterpolator::new(two_pow(128)).unwrap();
    big.add(&[0.0]).unwrap();
    big.get();
    assert_eq!(
        big.add(&[1.0]).unwrap_err(),
        InterpolatorError::BufferNotAllocatable {
            multiplier: two_pow(128)
        }
    );
}

#[test]
fn a_non_broadcastable_action_is_reported_before_the_allocation_is_attempted() {
    // Upstream raises the tensor `RuntimeError` on the *first* loop iteration,
    // before the list has grown, so the shape error wins over any capacity
    // problem the rest of the loop would have had.
    let mut big: ActionInterpolator<f32> = ActionInterpolator::new(two_pow(60)).unwrap();
    big.add(&[0.0, 0.0, 0.0]).unwrap();
    big.get();
    assert_eq!(
        big.add(&[1.0, 2.0]).unwrap_err(),
        InterpolatorError::NotBroadcastable {
            prev_len: 3,
            action_len: 2
        }
    );
}

#[test]
fn negative_and_fractional_actions_interpolate_linearly() {
    let mut i: ActionInterpolator<f64> = ActionInterpolator::new(4).unwrap();
    i.add(&[-1.0, 0.5]).unwrap();
    i.get();
    i.add(&[1.0, -0.5]).unwrap();
    assert_eq!(i.get(), Some([-0.5f64, 0.25].as_slice()));
    assert_eq!(i.get(), Some([0.0f64, 0.0].as_slice()));
    assert_eq!(i.get(), Some([0.5f64, -0.25].as_slice()));
    assert_eq!(i.get(), Some([1.0f64, -0.5].as_slice()));
}

#[test]
fn adding_before_draining_discards_the_unconsumed_tail() {
    let mut i = interp(3);
    i.add(&[0.0]).unwrap();
    i.get();
    i.add(&[3.0]).unwrap();
    assert_eq!(i.get(), Some([1.0f32].as_slice()));
    // Upstream `add` unconditionally rebuilds the buffer and resets the index.
    i.add(&[6.0]).unwrap();
    assert_eq!(i.get(), Some([4.0f32].as_slice()));
}

#[test]
fn add_rejects_a_reasonable_multiplier_that_would_materialize_too_many_values() {
    // A steps-only guard is not enough: 1,000 steps of a 17,000-wide action
    // still asks for more than sixteen million scalar values and many separate
    // heap allocations. The resource check must cover the whole output grid.
    let mut wide: ActionInterpolator<f32> = ActionInterpolator::new(1_000).unwrap();
    let previous = vec![0.0f32; 17_000];
    wide.add(&previous).unwrap();
    wide.get();
    assert_eq!(
        wide.add(&previous).unwrap_err(),
        InterpolatorError::BufferNotAllocatable {
            multiplier: BigInt::from(1_000)
        }
    );
}
