//! Rerobot's own deterministic pseudo-random generator.
//!
//! This module has **no upstream counterpart**, and that is a compatibility
//! statement rather than an omission. `lerobot.utils.random_utils.set_seed`
//! seeds three generators — Python's `random`, NumPy's global `RandomState` and
//! PyTorch's Mersenne Twister — and every random choice upstream makes
//! (parameter initialization, dropout masks, the VAE's reparameterization noise,
//! the sampler's per-epoch permutation) draws from one of those streams.
//! Reproducing them bit for bit would mean porting three RNGs and PyTorch's
//! per-operator sampling algorithms; Rerobot does not, so its random *values*
//! differ from upstream's for the same seed.
//!
//! What Rerobot does promise is stronger than "some RNG":
//!
//! * the generator is [SplitMix64], published and checkable against its own
//!   reference vectors rather than against this implementation;
//! * its entire state is one 64-bit word, so a checkpoint's `rng_state` is a
//!   single scalar that restores the stream exactly;
//! * every distribution derived from it ([`SplitMix64::next_f64`],
//!   [`SplitMix64::bounded`], [`SplitMix64::standard_normal`]) is a documented
//!   transform of that stream, so the same seed gives the same numbers on every
//!   platform and target width.
//!
//! [SplitMix64]: https://dl.acm.org/doi/10.1145/2714064.2660195

/// The odd increment SplitMix64 adds to its state per draw (⌊2⁶⁴/φ⌋).
pub const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

const MIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;

/// SplitMix64's finalizer: the bijection applied to the state to produce output.
///
/// Exposed because `mix64(seed + GAMMA * n)` is the `n`-th output of
/// `SplitMix64::new(seed)` for `n >= 1`, which lets a caller derive an
/// independent sub-stream in constant time instead of stepping the generator.
///
/// ```
/// use rerobot_core::random::{mix64, SplitMix64, GAMMA};
///
/// let mut rng = SplitMix64::new(1000);
/// assert_eq!(rng.next_u64(), mix64(1000u64.wrapping_add(GAMMA)));
/// ```
pub fn mix64(value: u64) -> u64 {
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(MIX_MULTIPLIER_1);
    z = (z ^ (z >> 27)).wrapping_mul(MIX_MULTIPLIER_2);
    z ^ (z >> 31)
}

/// A SplitMix64 generator: 64 bits of state, one addition and one mix per draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// A generator seeded with `seed`, whose first output is `mix64(seed + GAMMA)`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// A generator resumed from a state previously read with [`Self::state`].
    pub fn from_state(state: u64) -> Self {
        Self { state }
    }

    /// The whole state of the generator.
    pub fn state(&self) -> u64 {
        self.state
    }

    /// The next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GAMMA);
        mix64(self.state)
    }

    /// The next `f64` in `[0, 1)`, using the top 53 bits (one per mantissa bit).
    pub fn next_f64(&mut self) -> f64 {
        // 2^-53 exactly; the shift leaves 53 bits, so the result is in [0, 1).
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A uniform integer in `[0, bound)`, or `None` when `bound` is zero.
    ///
    /// Uses Lemire's multiply-shift method with rejection, so the result is
    /// exactly uniform rather than modulo-biased. A `bound` of 1 is answered
    /// without consuming a word.
    pub fn checked_bounded(&mut self, bound: u64) -> Option<u64> {
        if bound == 0 {
            return None;
        }
        if bound == 1 {
            return Some(0);
        }
        let mut product = u128::from(self.next_u64()) * u128::from(bound);
        let mut low = product as u64;
        if low < bound {
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                product = u128::from(self.next_u64()) * u128::from(bound);
                low = product as u64;
            }
        }
        Some((product >> 64) as u64)
    }

    /// [`Self::checked_bounded`], panicking on a zero bound.
    ///
    /// # Panics
    ///
    /// If `bound` is zero, which has no valid answer.
    pub fn bounded(&mut self, bound: u64) -> u64 {
        self.checked_bounded(bound)
            .expect("a bound of zero has no uniform value")
    }

    /// A draw from the standard normal distribution.
    ///
    /// Box–Muller, taking the cosine variate and discarding the sine one. The
    /// discard is deliberate: caching the spare would make the generator's state
    /// a `(u64, Option<f64>)` pair, and the state written to a checkpoint is one
    /// 64-bit word. Every call therefore consumes exactly two words.
    pub fn standard_normal(&mut self) -> f64 {
        // Shift the first uniform off zero so `ln` stays finite: 2^-53 is the
        // smallest value `next_f64` can return above zero.
        const TINY: f64 = 1.0 / (1u64 << 53) as f64;
        let u1 = self.next_f64().max(TINY);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// A uniformly random permutation of `0..n`, derived from `seed`.
///
/// Fisher–Yates (Durstenfeld), descending, with unbiased bounded draws. This is
/// Rerobot's stand-in for `torch.randperm`; it is *not* the same sequence. See
/// the module documentation and `docs/compatibility.md`.
///
/// ```
/// use rerobot_core::random::shuffled_permutation;
///
/// let permutation = shuffled_permutation(4, 1000);
/// assert_eq!(permutation, vec![3, 1, 2, 0]);
/// assert_eq!(shuffled_permutation(4, 1000), permutation);
/// ```
pub fn shuffled_permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = SplitMix64::new(seed);
    let mut permutation: Vec<usize> = (0..n).collect();
    for index in (1..n).rev() {
        let swap_with = rng.bounded(index as u64 + 1) as usize;
        permutation.swap(index, swap_with);
    }
    permutation
}
