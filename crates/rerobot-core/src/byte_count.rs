//! An exact unsigned integer, sized to whatever it is asked to hold.
//!
//! [`crate::ring_buffer`] accounts bytes the way upstream does, in Python
//! `int`s. Those are unbounded, and the values are not hypothetical: one
//! [`crate::ring_buffer::FrameValue::Tensor`] can cost `usize::MAX *
//! usize::MAX`, two of them in one frame already exceed `u128::MAX`, and a
//! running total under upstream's zero-length-cap quirk grows without any
//! limit at all. Any fixed width therefore has a frame it silently
//! under-counts, which is a wrong answer rather than a narrower one.
//!
//! [`ByteCount`] is a newtype over [`num_bigint::BigUint`] exposing only the
//! operations the accounting performs — add, subtract-on-eviction, compare
//! against the cap, and the one product `numel * element_size` comes from. It
//! is deliberately not a general bignum: there is no division, no
//! multiplication of two arbitrary counts, and no signed counterpart. The cap
//! stays an `i128` because `int(max_memory_mb * 1024 * 1024)` for any `i64`
//! megabyte count fits one exactly, negative caps included.
//!
//! The arithmetic is `BigUint`'s rather than hand-written, so its exactness is
//! not a property of carry and borrow code reviewed once in this repository.
//! `tests/byte_count.rs` additionally checks every operation below against
//! `BigUint` computed independently of this type, over a deterministic sweep
//! that crosses the 64- and 128-bit boundaries in both directions.
//!
//! ```
//! use rerobot_core::byte_count::ByteCount;
//!
//! // Two maximal tensors, the smallest total a 128-bit accumulator gets wrong.
//! let one = ByteCount::product(usize::MAX, usize::MAX);
//! let two = &one + &one;
//! assert!(two > u128::MAX);
//! assert_eq!(two.to_u128(), None);
//!
//! // Small values stay ordinary to use.
//! assert_eq!(ByteCount::from(8u64) + ByteCount::from(8u64), 16u128);
//! ```

use num_bigint::BigUint;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign};

/// An exact, arbitrary-precision unsigned integer.
///
/// Comparison, [`Display`](fmt::Display) and the arithmetic below are exact for
/// every value that can be built, with no saturating and no wrapping anywhere.
/// Values that happen to fit 128 bits interoperate with `u128` directly:
///
/// ```
/// use rerobot_core::byte_count::ByteCount;
///
/// assert_eq!(ByteCount::from(7u64), 7u128);
/// assert!(ByteCount::from(u128::MAX) + ByteCount::from(1u64) > u128::MAX);
/// ```
#[derive(Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ByteCount(BigUint);

impl ByteCount {
    /// The additive identity.
    pub const fn zero() -> Self {
        Self(BigUint::ZERO)
    }

    /// Whether this is [`ByteCount::zero`].
    pub fn is_zero(&self) -> bool {
        self.0 == BigUint::ZERO
    }

    /// `a * b`, exactly, for any two `usize` values on any target.
    ///
    /// This is `nelement() * element_size()` from upstream's
    /// `_estimate_frame_bytes`. It is the only multiplication this type has,
    /// and it is exact regardless of how wide a `usize` is.
    pub fn product(a: usize, b: usize) -> Self {
        Self(BigUint::from(a) * BigUint::from(b))
    }

    /// This value as a `u128`, or `None` when it needs more than 128 bits.
    pub fn to_u128(&self) -> Option<u128> {
        u128::try_from(&self.0).ok()
    }

    /// `self - rhs`, clamped at zero.
    ///
    /// The subtraction the ring buffer performs is always of a frame that was
    /// added first and is removed at most once, so the clamp is unreachable
    /// there; it exists so that no caller can construct a negative byte count.
    pub fn saturating_sub(&self, rhs: &Self) -> Self {
        if self <= rhs {
            return Self::zero();
        }
        Self(&self.0 - &rhs.0)
    }
}

impl From<u128> for ByteCount {
    fn from(value: u128) -> Self {
        Self(BigUint::from(value))
    }
}

impl From<u64> for ByteCount {
    fn from(value: u64) -> Self {
        Self(BigUint::from(value))
    }
}

impl From<usize> for ByteCount {
    fn from(value: usize) -> Self {
        Self(BigUint::from(value))
    }
}

impl Add<&ByteCount> for &ByteCount {
    type Output = ByteCount;

    fn add(self, rhs: &ByteCount) -> ByteCount {
        ByteCount(&self.0 + &rhs.0)
    }
}

impl Add for ByteCount {
    type Output = ByteCount;

    fn add(self, rhs: ByteCount) -> ByteCount {
        ByteCount(self.0 + rhs.0)
    }
}

impl AddAssign<&ByteCount> for ByteCount {
    fn add_assign(&mut self, rhs: &ByteCount) {
        self.0 += &rhs.0;
    }
}

impl AddAssign for ByteCount {
    fn add_assign(&mut self, rhs: ByteCount) {
        self.0 += rhs.0;
    }
}

// `u128` interoperation, so that the overwhelmingly common small case reads
// like ordinary integer code at call sites and in tests.
impl PartialEq<u128> for ByteCount {
    fn eq(&self, other: &u128) -> bool {
        // A value that needs more than 128 bits equals no `u128`.
        self.to_u128() == Some(*other)
    }
}

impl PartialEq<ByteCount> for u128 {
    fn eq(&self, other: &ByteCount) -> bool {
        other.to_u128() == Some(*self)
    }
}

impl PartialOrd<u128> for ByteCount {
    fn partial_cmp(&self, other: &u128) -> Option<Ordering> {
        Some(match self.to_u128() {
            Some(value) => value.cmp(other),
            // Wider than 128 bits, so larger than every `u128`.
            None => Ordering::Greater,
        })
    }
}

impl PartialOrd<ByteCount> for u128 {
    fn partial_cmp(&self, other: &ByteCount) -> Option<Ordering> {
        other.partial_cmp(self).map(Ordering::reverse)
    }
}

impl fmt::Display for ByteCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl fmt::Debug for ByteCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The decimal value, not the newtype wrapper: this is what shows up in
        // a failing `assert_eq!` against a byte total.
        fmt::Display::fmt(&self.0, f)
    }
}
