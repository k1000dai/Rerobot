//! Direct tests for the exact byte-count arithmetic.
//!
//! `tests/ring_buffer.rs` exercises this type through the buffer, which is what
//! the parity claim is actually about. These tests pin the primitive itself,
//! because a bignum that is subtly wrong at a carry boundary would make the
//! buffer's exactness claim wrong in a way the frame-level tests would not
//! localise. Every expected decimal literal below came out of CPython 3.12
//! evaluating the same expression.

use num_bigint::BigUint;
use rerobot_core::byte_count::ByteCount;

/// `2**64 - 1`, the largest `usize` on the targets this workspace supports.
#[cfg(target_pointer_width = "64")]
const M: usize = usize::MAX;

/// A deterministic 64-bit xorshift, so a differential failure below reproduces
/// exactly rather than only sometimes. Seeded from a fixed constant; there is
/// no randomness in the run.
struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A `usize` operand, and the same value as an independent `BigUint`.
    fn operand(&mut self) -> (usize, BigUint) {
        let value = self.next() as usize;
        (value, BigUint::from(value))
    }
}

#[test]
fn zero_is_zero_in_every_representation() {
    let zero = ByteCount::zero();
    assert!(zero.is_zero());
    assert_eq!(zero, 0u128);
    assert_eq!(zero.to_string(), "0");
    assert_eq!(zero.to_u128(), Some(0));
    assert_eq!(zero, ByteCount::default());
    assert_eq!(ByteCount::from(0u64), zero);
}

#[test]
fn small_values_round_trip_through_u128() {
    for value in [1u128, 8, 255, u128::from(u64::MAX), u128::MAX] {
        let count = ByteCount::from(value);
        assert_eq!(count.to_u128(), Some(value));
        assert_eq!(count, value);
        assert_eq!(count.to_string(), value.to_string());
    }
}

#[test]
fn addition_carries_across_the_64_and_128_bit_boundaries() {
    // 2**64, the first carry out of one limb.
    let limb_boundary = ByteCount::from(u64::MAX) + ByteCount::from(1u64);
    assert_eq!(limb_boundary, 1u128 << 64);

    // 2**128, the first value that no longer fits `u128` at all.
    let past_u128 = ByteCount::from(u128::MAX) + ByteCount::from(1u64);
    assert_eq!(past_u128.to_u128(), None);
    assert_eq!(
        past_u128.to_string(),
        "340282366920938463463374607431768211456"
    );
    assert!(past_u128 > u128::MAX);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn the_maximal_usize_product_is_exact() {
    // Python: (2**64 - 1) ** 2.
    assert_eq!(
        ByteCount::product(M, M).to_string(),
        "340282366920938463426481119284349108225"
    );
    assert_eq!(
        ByteCount::product(M, M).to_u128(),
        Some((M as u128) * (M as u128))
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn products_accumulate_exactly_well_past_128_bits() {
    let mut total = ByteCount::zero();
    for _ in 0..3 {
        total += ByteCount::product(M, M);
    }
    // Python: 3 * (2**64 - 1)**2.
    assert_eq!(
        total.to_string(),
        "1020847100762815390279443357853047324675"
    );
    // And it keeps going: 1000 times over.
    for _ in 3..1000 {
        total += ByteCount::product(M, M);
    }
    assert_eq!(
        total.to_string(),
        "340282366920938463426481119284349108225000"
    );
}

#[test]
fn product_is_zero_when_either_side_is_zero() {
    assert!(ByteCount::product(0, usize::MAX).is_zero());
    assert!(ByteCount::product(usize::MAX, 0).is_zero());
    assert_eq!(ByteCount::product(3, 5), 15u128);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn subtraction_borrows_back_down_across_a_limb_boundary() {
    // Python: (2**64 - 1)**2 - (2**64 - 1).
    let difference = ByteCount::product(M, M).saturating_sub(&ByteCount::from(M));
    assert_eq!(
        difference.to_string(),
        "340282366920938463408034375210639556610"
    );
    // Adding it back recovers the exact original.
    assert_eq!(
        (difference + ByteCount::from(M)).to_string(),
        "340282366920938463426481119284349108225"
    );
}

#[test]
fn subtraction_clamps_at_zero_instead_of_going_negative() {
    assert!(ByteCount::from(3u64)
        .saturating_sub(&ByteCount::from(4u64))
        .is_zero());
    assert!(ByteCount::from(4u64)
        .saturating_sub(&ByteCount::from(4u64))
        .is_zero());
    assert!(ByteCount::zero()
        .saturating_sub(&ByteCount::from(u128::MAX))
        .is_zero());
    assert_eq!(
        ByteCount::from(u128::MAX).saturating_sub(&ByteCount::from(1u64)),
        u128::MAX - 1
    );
}

#[test]
fn ordering_is_by_magnitude_not_by_limb_count_alone() {
    let mut values = [
        ByteCount::from(u128::MAX),
        ByteCount::zero(),
        ByteCount::from(u128::MAX) + ByteCount::from(1u64),
        ByteCount::from(1u64),
        ByteCount::from(u64::MAX),
    ];
    values.sort();
    let rendered: Vec<String> = values.iter().map(ByteCount::to_string).collect();
    assert_eq!(
        rendered,
        vec![
            "0".to_string(),
            "1".to_string(),
            "18446744073709551615".to_string(),
            "340282366920938463463374607431768211455".to_string(),
            "340282366920938463463374607431768211456".to_string(),
        ]
    );
}

#[test]
fn decimal_rendering_keeps_the_zeros_inside_a_number() {
    // 10**38 is exactly two 19-digit chunks of zero above a leading 1, so a
    // renderer that forgot to zero-pad later chunks would print "1".
    let mut ten_pow_38 = ByteCount::from(1u64);
    for _ in 0..38 {
        // Ten additions rather than a multiply: this type has no general `Mul`,
        // and building the value out of the operations it does have is the
        // point — the renderer has to cope with whatever they produce.
        let unit = ten_pow_38.clone();
        let mut tenfold = ByteCount::zero();
        for _ in 0..10 {
            tenfold += &unit;
        }
        ten_pow_38 = tenfold;
    }
    assert_eq!(
        ten_pow_38.to_string(),
        "100000000000000000000000000000000000000"
    );
    assert_eq!(
        (ten_pow_38 + ByteCount::from(7u64)).to_string(),
        "100000000000000000000000000000000000007"
    );
}

// --- differential tests against an independent bignum -------------------
//
// The hand-checked cases above pin the values CPython produces for the frame
// estimates that matter. These pin the *arithmetic*, against `num_bigint`'s
// `BigUint` computed separately from the type under test: every operation
// `ByteCount` exposes, over a deterministic sweep wide enough to cross limb
// boundaries in both directions. They are the reason the accounting's
// exactness claim does not rest on a hand-audit of carry and borrow code.

#[test]
fn product_agrees_with_biguint_over_a_deterministic_sweep() {
    let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
    for _ in 0..2_000 {
        let (a, big_a) = rng.operand();
        let (b, big_b) = rng.operand();
        assert_eq!(
            ByteCount::product(a, b).to_string(),
            (big_a * big_b).to_string(),
            "product({a}, {b})"
        );
    }
}

#[test]
fn a_running_total_agrees_with_biguint_over_a_deterministic_sweep() {
    // Exactly the accounting shape: accrue frame estimates, then evict some of
    // them back out again, and compare against the same sum kept in `BigUint`.
    let mut rng = Xorshift(0x9e37_79b9_7f4a_7c15);
    let mut total = ByteCount::zero();
    let mut oracle = BigUint::ZERO;
    let mut evictable = Vec::new();

    for step in 0..1_000 {
        let (a, big_a) = rng.operand();
        let (b, big_b) = rng.operand();
        let frame = ByteCount::product(a, b);
        let frame_oracle = big_a * big_b;

        total += &frame;
        oracle += &frame_oracle;
        evictable.push((frame, frame_oracle));

        // Evict the oldest frame every third step, the way the memory cap does.
        if step % 3 == 2 {
            let (frame, frame_oracle) = evictable.remove(0);
            total = total.saturating_sub(&frame);
            oracle -= frame_oracle;
        }
        assert_eq!(total.to_string(), oracle.to_string(), "after step {step}");
    }
    assert!(!total.is_zero());
}

#[test]
fn comparison_and_u128_narrowing_agree_with_biguint() {
    // One product of two `usize`s never needs more than 128 bits, so the
    // sweep compares each product against a *running total*, which passes the
    // 128-bit boundary within the first few steps and keeps going.
    let mut rng = Xorshift(0xdead_beef_0bad_f00d);
    let ceiling = BigUint::from(u128::MAX);
    let mut total = ByteCount::zero();
    let mut total_oracle = BigUint::ZERO;
    let mut sampled_wider_than_u128 = false;
    let mut sampled_narrower_than_u128 = false;

    for _ in 0..1_000 {
        let (a, big_a) = rng.operand();
        let (b, big_b) = rng.operand();
        let term = ByteCount::product(a, b);
        let term_oracle = big_a * big_b;
        total += &term;
        total_oracle += &term_oracle;

        assert_eq!(
            term.cmp(&total),
            term_oracle.cmp(&total_oracle),
            "cmp({term_oracle}, {total_oracle})"
        );
        assert_eq!(term == total, term_oracle == total_oracle);

        // `to_u128` must be `None` exactly when the value needs more bits, and
        // the value itself otherwise.
        for (value, oracle) in [(&term, &term_oracle), (&total, &total_oracle)] {
            let expected = u128::try_from(oracle).ok();
            assert_eq!(value.to_u128(), expected, "to_u128 of {oracle}");
            assert_eq!(value > &u128::MAX, oracle > &ceiling);
            sampled_wider_than_u128 |= expected.is_none();
            sampled_narrower_than_u128 |= expected.is_some();
        }
    }
    assert!(
        sampled_wider_than_u128 && sampled_narrower_than_u128,
        "the sweep stayed on one side of the u128 boundary, so it proved nothing"
    );
}

#[test]
fn saturating_sub_agrees_with_biguint_where_biguint_can_subtract_at_all() {
    // Both directions every step, so the clamp is exercised by whichever
    // operand is the smaller one rather than by luck.
    let mut rng = Xorshift(0x0123_4567_89ab_cdef);
    let mut subtracted = 0usize;
    let mut clamped = 0usize;

    for _ in 0..1_000 {
        let (a, big_a) = rng.operand();
        let (b, big_b) = rng.operand();
        let left = ByteCount::product(a, b);
        let right = ByteCount::from(a) + ByteCount::from(b);
        let left_oracle = big_a.clone() * big_b.clone();
        let right_oracle = big_a + big_b;

        for (x, x_oracle, y, y_oracle) in [
            (&left, &left_oracle, &right, &right_oracle),
            (&right, &right_oracle, &left, &left_oracle),
        ] {
            if x_oracle >= y_oracle {
                assert_eq!(
                    x.saturating_sub(y).to_string(),
                    (x_oracle - y_oracle).to_string(),
                    "{x_oracle} - {y_oracle}"
                );
                subtracted += 1;
            } else {
                // `BigUint` cannot represent the result at all, which is
                // exactly the case the clamp exists for.
                assert!(x.saturating_sub(y).is_zero(), "{x_oracle} - {y_oracle}");
                clamped += 1;
            }
        }
    }
    assert!(subtracted > 0 && clamped > 0, "one branch was never taken");
}

#[test]
fn debug_and_display_agree_and_read_as_a_number() {
    let value = ByteCount::from(u128::MAX) + ByteCount::from(1u64);
    assert_eq!(format!("{value:?}"), value.to_string());
    assert_eq!(
        format!("{value}"),
        "340282366920938463463374607431768211456"
    );
}
