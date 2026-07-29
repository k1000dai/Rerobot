//! Behaviour parity tests for `RolloutRingBuffer`, derived from upstream
//! `src/lerobot/rollout/ring_buffer.py` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1 and verified against the
//! equivalent pure-Python execution.

use rerobot_core::ring_buffer::{
    estimate_frame_bytes, Frame, FrameValue, RingBufferError, RolloutRingBuffer,
};

fn frame(pairs: &[(&str, FrameValue)]) -> Frame {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

#[test]
fn empty_frame_costs_one_byte() {
    assert_eq!(estimate_frame_bytes(&Frame::new()), 1);
}

#[test]
fn unrecognised_value_costs_one_byte() {
    assert_eq!(estimate_frame_bytes(&frame(&[("n", FrameValue::Other)])), 1);
}

#[test]
fn tensor_costs_numel_times_element_size() {
    let f = frame(&[(
        "t",
        FrameValue::Tensor {
            numel: 4,
            element_size: 4,
        },
    )]);
    assert_eq!(estimate_frame_bytes(&f), 16);
}

#[test]
fn ndarray_costs_nbytes() {
    assert_eq!(
        estimate_frame_bytes(&frame(&[("a", FrameValue::NBytes(3))])),
        3
    );
}

#[test]
fn int_and_float_cost_eight_bytes_each() {
    assert_eq!(
        estimate_frame_bytes(&frame(&[("i", FrameValue::Int(1))])),
        8
    );
    assert_eq!(
        estimate_frame_bytes(&frame(&[("f", FrameValue::Float(1.5))])),
        8
    );
    assert_eq!(
        estimate_frame_bytes(&frame(&[
            ("i", FrameValue::Int(1)),
            ("f", FrameValue::Float(0.0))
        ])),
        16
    );
}

#[test]
fn str_costs_code_points_not_utf8_bytes() {
    // Python `len("héllo")` is 5 even though it encodes to 6 UTF-8 bytes.
    let f = frame(&[("s", FrameValue::Str("héllo".to_string()))]);
    assert_eq!(estimate_frame_bytes(&f), 5);
}

#[test]
fn bytes_cost_their_length() {
    let f = frame(&[("b", FrameValue::Bytes(vec![0u8; 7]))]);
    assert_eq!(estimate_frame_bytes(&f), 7);
}

#[test]
fn mixed_frame_sums_every_recognised_value() {
    let f = frame(&[
        (
            "t",
            FrameValue::Tensor {
                numel: 2,
                element_size: 8,
            },
        ),
        ("a", FrameValue::NBytes(5)),
        ("i", FrameValue::Int(-3)),
        ("s", FrameValue::Str("ab".to_string())),
        ("skip", FrameValue::Other),
    ]);
    assert_eq!(estimate_frame_bytes(&f), 16 + 5 + 8 + 2);
}

#[test]
fn defaults_match_upstream() {
    let b = RolloutRingBuffer::with_defaults();
    assert_eq!(b.max_frames(), 900);
    assert_eq!(b.max_bytes(), 2048 * 1024 * 1024);
    assert_eq!(b.len(), 0);
    assert!(b.is_empty());
    assert_eq!(b.estimated_bytes(), 0);
}

#[test]
fn max_frames_truncates_toward_zero() {
    assert_eq!(RolloutRingBuffer::new(0.9, 1, 1.0).unwrap().max_frames(), 0);
    assert_eq!(
        RolloutRingBuffer::new(2.0 / 30.0, 1, 30.0)
            .unwrap()
            .max_frames(),
        2
    );
    assert_eq!(
        RolloutRingBuffer::new(1.999, 1, 1.0).unwrap().max_frames(),
        1
    );
}

#[test]
fn negative_frame_capacity_is_rejected() {
    // `deque(maxlen=-30)` -> ValueError("maxlen must be non-negative").
    assert_eq!(
        RolloutRingBuffer::new(-1.0, 1, 30.0).unwrap_err(),
        RingBufferError::NegativeMaxLen(-30.0)
    );
}

#[test]
fn nan_frame_capacity_is_rejected_like_pythons_int() {
    // `int(float("nan"))` -> ValueError("cannot convert float NaN to integer").
    assert_eq!(
        RolloutRingBuffer::new(f64::NAN, 1, 30.0).unwrap_err(),
        RingBufferError::NanMaxLen
    );
    assert_eq!(
        RolloutRingBuffer::new(f64::NAN, 1, 30.0)
            .unwrap_err()
            .to_string(),
        "cannot convert float NaN to integer"
    );
}

#[test]
fn infinite_frame_capacity_is_rejected_in_both_directions() {
    // `int(float("inf"))` and `int(float("-inf"))` both raise
    // OverflowError("cannot convert float infinity to integer").
    for seconds in [f64::INFINITY, f64::NEG_INFINITY] {
        let err = RolloutRingBuffer::new(seconds, 1, 30.0).unwrap_err();
        assert_eq!(err, RingBufferError::InfiniteMaxLen);
        assert_eq!(err.to_string(), "cannot convert float infinity to integer");
    }
}

#[test]
fn frame_capacity_beyond_py_ssize_t_is_an_overflow_not_a_wrapped_length() {
    // CPython converts `maxlen` with `PyLong_AsSsize_t`, so anything above
    // `sys.maxsize` raises OverflowError before the non-negative check.
    let err = RolloutRingBuffer::new(1e30, 1, 1.0).unwrap_err();
    assert_eq!(err, RingBufferError::MaxLenNotRepresentable(1e30));
    assert!(
        err.to_string()
            .starts_with("Python int too large to convert to C ssize_t"),
        "{err}"
    );
}

#[test]
fn frame_capacity_below_py_ssize_t_min_is_an_overflow_not_a_value_error() {
    // `deque(maxlen=-(sys.maxsize + 2))` raises OverflowError, not the
    // ValueError that a small negative maxlen raises.
    let err = RolloutRingBuffer::new(-1e30, 1, 1.0).unwrap_err();
    assert_eq!(err, RingBufferError::MaxLenNotRepresentable(-1e30));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn the_largest_representable_frame_capacity_is_accepted() {
    // 9223372036854774784 is the largest integral f64 strictly below 2^63, so
    // it is the largest frame cap `int(max_seconds * fps)` can hand to a deque.
    let ok = RolloutRingBuffer::new(9223372036854774784.0, 1, 1.0).unwrap();
    assert_eq!(ok.max_frames(), 9223372036854774784);

    // 2^63 itself is `sys.maxsize + 1` -> OverflowError.
    assert_eq!(
        RolloutRingBuffer::new(9223372036854775808.0, 1, 1.0).unwrap_err(),
        RingBufferError::MaxLenNotRepresentable(9223372036854775808.0)
    );
}

#[test]
fn append_then_len_and_bytes_track_the_frames() {
    let mut b = RolloutRingBuffer::new(10.0, 1, 30.0).unwrap();
    b.append(frame(&[("i", FrameValue::Int(0))]));
    b.append(frame(&[("i", FrameValue::Int(1))]));
    assert_eq!(b.len(), 2);
    assert_eq!(b.estimated_bytes(), 16);
    assert!(!b.is_empty());
}

#[test]
fn drain_returns_frames_in_insertion_order_and_empties_the_buffer() {
    let mut b = RolloutRingBuffer::new(10.0, 1, 30.0).unwrap();
    for i in 0..3 {
        b.append(frame(&[("i", FrameValue::Int(i))]));
    }
    let drained = b.drain();
    assert_eq!(drained.len(), 3);
    for (i, f) in drained.iter().enumerate() {
        assert_eq!(f["i"], FrameValue::Int(i as i64));
    }
    assert_eq!(b.len(), 0);
    assert_eq!(b.estimated_bytes(), 0);
    assert!(b.drain().is_empty());
}

#[test]
fn clear_discards_everything() {
    let mut b = RolloutRingBuffer::new(10.0, 1, 30.0).unwrap();
    b.append(frame(&[("i", FrameValue::Int(0))]));
    b.clear();
    assert_eq!(b.len(), 0);
    assert_eq!(b.estimated_bytes(), 0);
}

#[test]
fn byte_cap_evicts_the_oldest_frames_first() {
    // 1 MiB byte cap; two 409608-byte frames fit, the third evicts the first.
    let mut b = RolloutRingBuffer::new(100.0, 1, 1.0).unwrap();
    let big = 400 * 1024;
    for i in 0..3 {
        let mut f = frame(&[("a", FrameValue::NBytes(big))]);
        f.insert("i".to_string(), FrameValue::Int(i));
        b.append(f);
    }
    assert_eq!(b.len(), 2);
    assert_eq!(b.drain()[0]["i"], FrameValue::Int(1));
}

#[test]
fn frame_count_eviction_keeps_only_the_newest_frames() {
    let mut b = RolloutRingBuffer::new(2.0 / 30.0, 1024, 30.0).unwrap();
    for i in 0..4 {
        b.append(frame(&[("i", FrameValue::Int(i))]));
    }
    assert_eq!(b.len(), 2);
    let drained = b.drain();
    assert_eq!(drained[0]["i"], FrameValue::Int(2));
    assert_eq!(drained[1]["i"], FrameValue::Int(3));
}

#[test]
fn frame_count_eviction_does_not_decrement_the_byte_accounting() {
    // Upstream quirk: `deque(maxlen=...)` drops the oldest frame without going
    // through the eviction branch, so `_current_bytes` keeps growing.
    let mut b = RolloutRingBuffer::new(2.0 / 30.0, 1024, 30.0).unwrap();
    for i in 0..4 {
        b.append(frame(&[("i", FrameValue::Int(i))]));
    }
    assert_eq!(b.len(), 2);
    assert_eq!(b.estimated_bytes(), 32);
}

#[test]
fn zero_frame_capacity_discards_appends_but_still_accrues_bytes() {
    // Upstream quirk verified against `deque(maxlen=0)`.
    let mut b = RolloutRingBuffer::new(0.0, 1, 30.0).unwrap();
    b.append(frame(&[("x", FrameValue::Int(1))]));
    assert_eq!(b.len(), 0);
    assert_eq!(b.estimated_bytes(), 8);
    assert!(b.drain().is_empty());
}

#[test]
fn zero_byte_cap_still_admits_one_frame_at_a_time() {
    let mut b = RolloutRingBuffer::new(10.0, 0, 30.0).unwrap();
    b.append(frame(&[("x", FrameValue::Int(1))]));
    b.append(frame(&[("y", FrameValue::Int(2))]));
    assert_eq!(b.len(), 1);
    assert_eq!(b.estimated_bytes(), 8);
    assert_eq!(b.drain()[0]["y"], FrameValue::Int(2));
}

#[test]
fn oversized_frame_evicts_everything_and_is_still_stored() {
    let mut b = RolloutRingBuffer::new(100.0, 1, 1.0).unwrap();
    b.append(frame(&[("i", FrameValue::Int(0))]));
    b.append(frame(&[("a", FrameValue::NBytes(4 * 1024 * 1024))]));
    assert_eq!(b.len(), 1);
    assert_eq!(b.estimated_bytes(), 4 * 1024 * 1024);
}

// --- numeric-domain boundaries ------------------------------------------
//
// Upstream accounts bytes in Python `int`s, which are unbounded. Rust has to
// pick a width; these tests pin that the port neither wraps, panics, nor
// undercounts anywhere a Python run would have produced an exact answer that
// Rust can represent. `340282366920938463426481119284349108225` is
// `(2**64 - 1) ** 2`, the largest single-value estimate the API can express.

/// The largest frame a caller can describe: one maximal tensor.
#[cfg(target_pointer_width = "64")]
fn maximal_frame() -> Frame {
    frame(&[(
        "t",
        FrameValue::Tensor {
            numel: usize::MAX,
            element_size: usize::MAX,
        },
    )])
}

#[test]
#[cfg(target_pointer_width = "64")]
fn a_tensor_larger_than_i64_bytes_is_costed_exactly() {
    let f = frame(&[(
        "t",
        FrameValue::Tensor {
            numel: usize::MAX,
            element_size: 8,
        },
    )]);
    assert_eq!(estimate_frame_bytes(&f), (usize::MAX as u128) * 8);
    assert!(estimate_frame_bytes(&f) > i64::MAX as u128);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn the_maximal_tensor_estimate_is_exact() {
    assert_eq!(
        estimate_frame_bytes(&maximal_frame()),
        340_282_366_920_938_463_426_481_119_284_349_108_225
    );
}

#[test]
fn nbytes_beyond_i64_are_costed_exactly() {
    let f = frame(&[
        ("a", FrameValue::NBytes(usize::MAX)),
        ("b", FrameValue::NBytes(usize::MAX)),
    ]);
    assert_eq!(estimate_frame_bytes(&f), (usize::MAX as u128) * 2);
}

#[test]
fn the_byte_cap_is_exact_for_the_largest_megabyte_count() {
    // Python: `int(2**63 - 1) * 1024 * 1024` == 9671406556917033396600832,
    // which does not fit in an i64 and must not be clamped to one.
    let b = RolloutRingBuffer::new(1.0, i64::MAX, 1.0).unwrap();
    assert_eq!(b.max_bytes(), 9_671_406_556_917_033_396_600_832);
    assert_eq!(b.max_bytes(), (i64::MAX as i128) * 1024 * 1024);
}

#[test]
fn a_negative_byte_cap_evicts_before_every_append() {
    // Python allows a negative `_max_bytes`; the while-condition is then always
    // true, so each append first drains the buffer and then stores its frame.
    let mut b = RolloutRingBuffer::new(10.0, -1, 1.0).unwrap();
    assert_eq!(b.max_bytes(), -1_048_576);
    b.append(frame(&[("x", FrameValue::Int(1))]));
    assert_eq!(b.len(), 1);
    assert_eq!(b.estimated_bytes(), 8);
    b.append(frame(&[("y", FrameValue::Int(2))]));
    assert_eq!(b.len(), 1);
    assert_eq!(b.estimated_bytes(), 8);
    assert_eq!(b.drain()[0]["y"], FrameValue::Int(2));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn a_frame_far_larger_than_the_cap_is_accounted_without_wrapping() {
    let mut b = RolloutRingBuffer::new(10.0, i64::MAX, 1.0).unwrap();
    b.append(maximal_frame());
    assert_eq!(b.len(), 1);
    assert_eq!(
        b.estimated_bytes(),
        340_282_366_920_938_463_426_481_119_284_349_108_225
    );
    // Appending a second one evicts the first, so the total does not grow.
    b.append(maximal_frame());
    assert_eq!(b.len(), 1);
    assert_eq!(
        b.estimated_bytes(),
        340_282_366_920_938_463_426_481_119_284_349_108_225
    );
}

// --- past the 128-bit boundary ------------------------------------------
//
// Everything below describes a frame or a running total whose exact Python
// value is larger than `u128::MAX` (340282366920938463463374607431768211455).
// Every expected decimal literal was produced by CPython 3.12 evaluating the
// same expression `_estimate_frame_bytes` evaluates, so these are upstream's
// answers, not this port's.

/// `n` maximal tensors in one frame, distinct keys so none of them collide.
#[cfg(target_pointer_width = "64")]
fn maximal_tensors(n: usize) -> Frame {
    (0..n)
        .map(|i| {
            (
                format!("t{i}"),
                FrameValue::Tensor {
                    numel: usize::MAX,
                    element_size: usize::MAX,
                },
            )
        })
        .collect()
}

#[test]
#[cfg(target_pointer_width = "64")]
fn two_maximal_tensors_in_one_frame_are_costed_exactly() {
    // Python: 2 * (2**64 - 1)**2. One value below `u128::MAX`, their sum above
    // it — the smallest frame a fixed 128-bit total gets wrong.
    assert_eq!(
        estimate_frame_bytes(&maximal_tensors(2)).to_string(),
        "680564733841876926852962238568698216450"
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn four_maximal_tensors_in_one_frame_are_costed_exactly() {
    // Python: 4 * (2**64 - 1)**2.
    assert_eq!(
        estimate_frame_bytes(&maximal_tensors(4)).to_string(),
        "1361129467683753853705924477137396432900"
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn a_frame_mixing_maximal_tensors_and_maximal_nbytes_is_costed_exactly() {
    // Python: 3 * (2**64 - 1)**2 + 2 * (2**64 - 1) + 8.
    let mut f = maximal_tensors(3);
    f.insert("a".to_string(), FrameValue::NBytes(usize::MAX));
    f.insert("b".to_string(), FrameValue::NBytes(usize::MAX));
    f.insert("i".to_string(), FrameValue::Int(-1));
    assert_eq!(
        estimate_frame_bytes(&f).to_string(),
        "1020847100762815390316336846000466427913"
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn accrual_across_frames_past_the_128_bit_boundary_is_exact() {
    // With a zero-length frame cap nothing is ever stored, so the eviction
    // branch can never fire and `_current_bytes` grows without bound — the one
    // path on which a Python run has no upper limit at all. Two maximal frames
    // already exceed `u128::MAX`; the total has to follow Python exactly, not
    // clamp.
    let mut b = RolloutRingBuffer::new(0.0, i64::MAX, 1.0).unwrap();
    b.append(maximal_frame());
    assert_eq!(
        b.estimated_bytes().to_string(),
        "340282366920938463426481119284349108225"
    );
    b.append(maximal_frame());
    assert_eq!(b.len(), 0);
    // Python: 2 * (2**64 - 1)**2.
    assert_eq!(
        b.estimated_bytes().to_string(),
        "680564733841876926852962238568698216450"
    );
    b.append(maximal_frame());
    b.append(maximal_frame());
    // Python: 4 * (2**64 - 1)**2.
    assert_eq!(
        b.estimated_bytes().to_string(),
        "1361129467683753853705924477137396432900"
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn a_thousand_maximal_frames_accrue_exactly() {
    // Python: 1000 * (2**64 - 1)**2 — three decimal orders past `u128::MAX`,
    // so nothing about this total is representable in a fixed 128-bit width.
    let mut b = RolloutRingBuffer::new(0.0, i64::MAX, 1.0).unwrap();
    for _ in 0..1000 {
        b.append(maximal_frame());
    }
    assert_eq!(b.len(), 0);
    assert_eq!(
        b.estimated_bytes().to_string(),
        "340282366920938463426481119284349108225000"
    );
}

#[test]
fn repeated_appends_under_a_zero_frame_cap_accrue_exactly() {
    // The same unbounded-accrual path, at sizes Python and Rust agree on.
    let mut b = RolloutRingBuffer::new(0.0, 1, 1.0).unwrap();
    for _ in 0..1000 {
        b.append(frame(&[("x", FrameValue::Int(1))]));
    }
    assert_eq!(b.len(), 0);
    assert_eq!(b.estimated_bytes(), 8 * 1000);
}
