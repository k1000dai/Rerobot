//! Behaviour tests for `rerobot_core::random`, the workspace's own
//! deterministic pseudo-random generator.
//!
//! This module has **no upstream counterpart**: upstream seeds Python's
//! `random`, NumPy's global `RandomState` and PyTorch's Mersenne generator, and
//! Rerobot reproduces none of those streams. What it does have to be is a
//! documented, portable, exactly reproducible generator, and that is what these
//! tests pin — including the published SplitMix64 reference vectors, so a
//! reviewer can check the stream against the algorithm rather than against this
//! implementation.

use rerobot_core::random::{mix64, shuffled_permutation, SplitMix64, GAMMA};

// ---------------------------------------------------------------------------
// The generator is SplitMix64, not "some" PRNG
// ---------------------------------------------------------------------------

#[test]
fn the_stream_matches_the_published_splitmix64_vectors_for_seed_zero() {
    let mut rng = SplitMix64::new(0);
    assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
    assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
    assert_eq!(rng.next_u64(), 0xf88b_b8a8_724c_81ec);
}

#[test]
fn the_state_advances_by_gamma_per_draw_and_round_trips() {
    let mut rng = SplitMix64::new(7);
    assert_eq!(rng.state(), 7);
    rng.next_u64();
    assert_eq!(rng.state(), 7u64.wrapping_add(GAMMA));

    // The state is the whole generator: restoring it resumes the same stream.
    let saved = rng.state();
    let expected: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
    let mut resumed = SplitMix64::from_state(saved);
    let replayed: Vec<u64> = (0..4).map(|_| resumed.next_u64()).collect();
    assert_eq!(replayed, expected);
}

#[test]
fn mix64_is_the_finalizer_so_the_nth_output_is_available_in_constant_time() {
    // `mix64(seed + GAMMA * n)` is the n-th output of `SplitMix64::new(seed)`
    // for n >= 1. The sampler relies on this to derive an epoch seed without
    // stepping the generator epoch-many times.
    let seed = 1000u64;
    let mut rng = SplitMix64::new(seed);
    for n in 1..=16u64 {
        let stepped = rng.next_u64();
        let jumped = mix64(seed.wrapping_add(GAMMA.wrapping_mul(n)));
        assert_eq!(stepped, jumped, "output {n} disagrees");
    }
}

// ---------------------------------------------------------------------------
// Derived distributions
// ---------------------------------------------------------------------------

#[test]
fn uniform_draws_stay_in_the_unit_interval_and_are_reproducible() {
    let mut rng = SplitMix64::new(42);
    let first: Vec<f64> = (0..256).map(|_| rng.next_f64()).collect();
    assert!(
        first.iter().all(|value| (0.0..1.0).contains(value)),
        "a draw escaped [0, 1)"
    );

    let mut again = SplitMix64::new(42);
    let second: Vec<f64> = (0..256).map(|_| again.next_f64()).collect();
    assert_eq!(first, second);
}

#[test]
fn bounded_draws_are_inside_the_bound_and_a_bound_of_one_consumes_nothing() {
    let mut rng = SplitMix64::new(3);
    for bound in [1u64, 2, 3, 5, 17, 1024] {
        for _ in 0..64 {
            assert!(rng.bounded(bound) < bound, "escaped bound {bound}");
        }
    }

    // A single-valued range is answered without touching the stream, so the
    // permutation of a one-element slice cannot perturb later draws.
    let mut untouched = SplitMix64::new(99);
    let state_before = untouched.state();
    assert_eq!(untouched.bounded(1), 0);
    assert_eq!(untouched.state(), state_before);
}

#[test]
fn a_zero_bound_is_rejected_rather_than_wrapping() {
    let mut rng = SplitMix64::new(0);
    assert_eq!(rng.checked_bounded(0), None);
    assert_eq!(rng.checked_bounded(4).map(|value| value < 4), Some(true));
}

#[test]
fn standard_normal_draws_are_finite_reproducible_and_roughly_standard() {
    let mut rng = SplitMix64::new(2024);
    let draws: Vec<f64> = (0..20_000).map(|_| rng.standard_normal()).collect();
    assert!(draws.iter().all(|value| value.is_finite()));

    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    let variance = draws
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / draws.len() as f64;
    // Loose bounds: this asserts the transform is a standard normal at all, not
    // that it is a good one. 20k draws put the sample mean's standard error at
    // ~0.007, so 0.05 is a ~7-sigma envelope.
    assert!(mean.abs() < 0.05, "sample mean {mean} is not near zero");
    assert!(
        (variance - 1.0).abs() < 0.05,
        "sample variance {variance} is not near one"
    );

    let mut again = SplitMix64::new(2024);
    let replay: Vec<f64> = (0..20_000).map(|_| again.standard_normal()).collect();
    assert_eq!(draws, replay);
}

#[test]
fn standard_normal_consumes_a_fixed_number_of_words_so_the_state_stays_scalar() {
    // Box-Muller here discards the second variate on purpose: caching it would
    // make the generator's state a (u64, Option<f64>) pair, and the checkpoint
    // writes a single 64-bit word.
    let mut rng = SplitMix64::new(11);
    let before = rng.state();
    rng.standard_normal();
    assert_eq!(rng.state(), before.wrapping_add(GAMMA.wrapping_mul(2)));
}

// ---------------------------------------------------------------------------
// Permutations
// ---------------------------------------------------------------------------

#[test]
fn a_permutation_is_a_permutation() {
    for n in [0usize, 1, 2, 3, 4, 5, 16, 129] {
        let permutation = shuffled_permutation(n, 1234);
        assert_eq!(permutation.len(), n);
        let mut sorted = permutation.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }
}

#[test]
fn permutations_depend_on_the_seed_and_repeat_for_the_same_seed() {
    assert_eq!(shuffled_permutation(64, 5), shuffled_permutation(64, 5));
    assert_ne!(shuffled_permutation(64, 5), shuffled_permutation(64, 6));
}

#[test]
fn a_permutation_of_four_elements_is_pinned_so_the_order_cannot_drift_silently() {
    // The exact order is not upstream's — see the module docs — but it is a
    // published property of Rerobot: a checkpoint's data order is only
    // reproducible if this value never changes by accident.
    assert_eq!(shuffled_permutation(4, 1000), vec![3, 1, 2, 0]);
    assert_eq!(shuffled_permutation(8, 1000), vec![6, 0, 3, 2, 7, 4, 5, 1]);
}
