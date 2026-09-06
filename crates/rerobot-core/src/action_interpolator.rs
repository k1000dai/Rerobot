//! Port of `lerobot.utils.action_interpolator.ActionInterpolator`.
//!
//! Upstream operates on `torch.Tensor` actions; this port operates on flat
//! slices, which is what the control loop actually feeds it (a 1-D action
//! vector per step). The arithmetic, buffering, and index bookkeeping are a
//! statement-for-statement port.

use num_bigint::BigInt;
use std::fmt;

/// Hard upper bound for the number of interpolated action vectors retained at
/// once. The upstream Python list can grow until `MemoryError`; a Rust
/// `Vec<Vec<T>>` must refuse a hostile multiplier before the allocator is asked
/// for a potentially enormous virtual allocation.
pub const MAX_INTERPOLATION_STEPS: usize = 1_000_000;

/// Hard upper bound for scalar elements in one materialized interpolation
/// buffer. This complements [`MAX_INTERPOLATION_STEPS`]: a modest step count
/// combined with an exceptionally wide action must not create an unbounded
/// number of per-step vectors.
pub const MAX_INTERPOLATION_ELEMENTS: usize = 1 << 24;

/// Scalar element type an [`ActionInterpolator`] can operate on.
///
/// Implemented for `f32` (matching a `torch.float32` action tensor, the default
/// upstream dtype) and `f64` (`torch.float64`). The interpolation weight
/// `t = i / multiplier` is computed in `f64` and then narrowed to `Self`, which
/// is what PyTorch does when a Python `float` scalar meets a typed tensor.
pub trait Scalar: Copy + PartialEq + fmt::Debug + sealed::Sealed {
    /// Narrow an `f64` interpolation weight to this scalar width.
    fn from_f64(v: f64) -> Self;
    /// `a + t * (b - a)`, evaluated at this scalar width.
    fn lerp(a: Self, t: Self, b: Self) -> Self;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

impl Scalar for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn lerp(a: Self, t: Self, b: Self) -> Self {
        a + t * (b - a)
    }
}

impl Scalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn lerp(a: Self, t: Self, b: Self) -> Self {
        a + t * (b - a)
    }
}

/// Errors raised by [`ActionInterpolator`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpolatorError {
    /// `multiplier < 1`; mirrors upstream `ValueError`.
    InvalidMultiplier(BigInt),
    /// The new action cannot be broadcast against the previous one.
    ///
    /// Upstream this is the `RuntimeError` raised by the tensor subtraction
    /// `action - self._prev` inside `add`. Two 1-D lengths broadcast when they
    /// are equal or when one of them is exactly `1`; anything else fails.
    NotBroadcastable {
        /// Length of the previously buffered action (`tensor b` upstream).
        prev_len: usize,
        /// Length of the action just handed to `add` (`tensor a` upstream).
        action_len: usize,
    },
    /// [`ActionInterpolator::add`] cannot build a buffer of `multiplier`
    /// interpolated actions.
    ///
    /// Returned when the multiplier does not fit a `usize`, or when reserving
    /// that many buffer slots fails. Upstream has no equivalent check: CPython
    /// appends to a `list` until it runs out of memory and raises
    /// `MemoryError`. This is the one place where a Rust buffer is a narrower
    /// thing than a Python one, and the port names the boundary instead of
    /// truncating the step count to something it can index.
    BufferNotAllocatable {
        /// The multiplier whose buffer could not be built, exactly as stored.
        multiplier: BigInt,
    },
    /// [`ActionInterpolator::get_control_interval`] cannot convert `multiplier`
    /// to an `f64`.
    ///
    /// Upstream evaluates `fps * self.multiplier`, and CPython converts the
    /// `int` operand to a double first, raising `OverflowError: int too large
    /// to convert to float` when the value is outside the `f64` range. This
    /// carries that failure rather than letting the interval silently become
    /// `1.0 / inf == 0.0`.
    MultiplierNotFloatRepresentable(BigInt),
}

impl fmt::Display for InterpolatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Wording copied from upstream's `ValueError` f-string.
            Self::InvalidMultiplier(m) => write!(f, "multiplier must be >= 1, got {m}"),
            // Wording copied from the PyTorch `RuntimeError`; `a` is the
            // right-hand operand of `action - self._prev`, i.e. the action.
            Self::NotBroadcastable {
                prev_len,
                action_len,
            } => write!(
                f,
                "The size of tensor a ({action_len}) must match the size of tensor b \
                 ({prev_len}) at non-singleton dimension 0"
            ),
            Self::BufferNotAllocatable { multiplier } => write!(
                f,
                "cannot allocate a buffer of {multiplier} interpolated actions"
            ),
            // Wording copied from CPython's `OverflowError`.
            Self::MultiplierNotFloatRepresentable(_) => {
                f.write_str("int too large to convert to float")
            }
        }
    }
}

impl std::error::Error for InterpolatorError {}

/// Length of the 1-D broadcast of `a` and `b`, or `None` if they do not
/// broadcast.
///
/// PyTorch's rule for a single dimension: the sizes must be equal, or one of
/// them must be `1`. Note that `0` broadcasts against `1` (to `0`) but not
/// against any other length.
fn broadcast_len(a: usize, b: usize) -> Option<usize> {
    if a == b {
        Some(a)
    } else if a == 1 {
        Some(b)
    } else if b == 1 {
        Some(a)
    } else {
        None
    }
}

/// Interpolates between consecutive actions for smoother control.
///
/// With `multiplier = N`, every action handed to [`add`](ActionInterpolator::add)
/// yields `N` actions from [`get`](ActionInterpolator::get) — except the very
/// first one after construction or [`reset`](ActionInterpolator::reset), which
/// passes through unchanged because there is no previous action to blend from.
///
/// ```
/// use rerobot_core::action_interpolator::ActionInterpolator;
///
/// let mut interp: ActionInterpolator<f64> = ActionInterpolator::new(3).unwrap();
/// interp.add(&[0.0]).unwrap();
/// assert_eq!(interp.get(), Some([0.0].as_slice())); // first action passes through
///
/// interp.add(&[3.0]).unwrap();
/// assert_eq!(interp.get(), Some([1.0].as_slice()));
/// assert_eq!(interp.get(), Some([2.0].as_slice()));
/// assert_eq!(interp.get(), Some([3.0].as_slice()));
/// assert_eq!(interp.get(), None);
/// ```
#[derive(Debug, Clone)]
pub struct ActionInterpolator<T: Scalar = f32> {
    /// The Python `int` upstream stores, at its own domain rather than at any
    /// machine width: `__init__` checks only `multiplier < 1`, and an `int` is
    /// unbounded, so `2**63` and `10**100` are values it holds exactly.
    /// Storage, [`multiplier`](ActionInterpolator::multiplier), and
    /// [`enabled`](ActionInterpolator::enabled) are exact at every magnitude.
    /// The two operations that cannot cover that full domain — allocating the
    /// buffer and converting to a float in
    /// [`get_control_interval`](ActionInterpolator::get_control_interval) —
    /// return an error naming the value instead of narrowing it.
    multiplier: BigInt,
    prev: Option<Vec<T>>,
    buffer: Vec<Vec<T>>,
    idx: usize,
}

impl<T: Scalar> ActionInterpolator<T> {
    /// Create an interpolator. `multiplier` must be `>= 1`.
    ///
    /// Any `>= 1` value is accepted, matching upstream, which validates the
    /// multiplier and nothing else. A multiplier too large to buffer is
    /// therefore constructed successfully and fails in [`add`](Self::add),
    /// where a Python `list` of that length would also have failed.
    ///
    /// Every Rust integer converts into a [`BigInt`], so `new(3)` still reads
    /// the way it did; larger values can be passed as a [`BigInt`] directly.
    pub fn new(multiplier: impl Into<BigInt>) -> Result<Self, InterpolatorError> {
        let multiplier = multiplier.into();
        if multiplier < BigInt::from(1) {
            return Err(InterpolatorError::InvalidMultiplier(multiplier));
        }
        Ok(Self {
            multiplier,
            prev: None,
            buffer: Vec::new(),
            idx: 0,
        })
    }

    /// Configured control-rate multiplier, exactly as it was given.
    pub fn multiplier(&self) -> &BigInt {
        &self.multiplier
    }

    /// Whether interpolation is active (`multiplier > 1`).
    pub fn enabled(&self) -> bool {
        self.multiplier > BigInt::from(1)
    }

    /// Reset interpolation state (call between episodes).
    pub fn reset(&mut self) {
        self.prev = None;
        self.buffer.clear();
        self.idx = 0;
    }

    /// Whether a new action is needed from the queue.
    pub fn needs_new_action(&self) -> bool {
        self.idx >= self.buffer.len()
    }

    /// Add a new action and compute the interpolated sequence.
    ///
    /// Any not-yet-consumed steps from a previous `add` are discarded, matching
    /// upstream's unconditional buffer rebuild.
    ///
    /// The action broadcasts against the previous one the way 1-D tensors do:
    /// equal lengths, or either side of length `1`. On a non-broadcastable pair
    /// this returns [`InterpolatorError::NotBroadcastable`] *after* clearing the
    /// buffer and *without* touching the previous action or the read index —
    /// upstream assigns `self._buffer = []` before the loop that raises and
    /// reassigns `self._prev` only after it.
    ///
    /// This is the one operation with a domain narrower than upstream's, and
    /// the boundary is exact: the interpolated sequence is `multiplier`
    /// elements of a Rust `Vec`, so a multiplier that does not fit a `usize`,
    /// or whose slots cannot be reserved, returns
    /// [`InterpolatorError::BufferNotAllocatable`] rather than being truncated
    /// to a step count that does fit. The check happens *after* the broadcast
    /// check, because upstream's `RuntimeError` comes from the first loop
    /// iteration, before the list has grown. Below that boundary the buffer is
    /// genuinely built, and an allocator that fails part-way aborts the process
    /// the way any Rust allocation failure does, where CPython would raise
    /// `MemoryError`.
    pub fn add(&mut self, action: &[T]) -> Result<(), InterpolatorError> {
        match (&self.prev, self.enabled()) {
            (Some(prev), true) => {
                // `self._buffer = []` happens before the arithmetic upstream, so
                // it is observable even when the arithmetic raises.
                self.buffer.clear();
                let Some(width) = broadcast_len(prev.len(), action.len()) else {
                    return Err(InterpolatorError::NotBroadcastable {
                        prev_len: prev.len(),
                        action_len: action.len(),
                    });
                };
                let steps = usize::try_from(&self.multiplier)
                    .ok()
                    .filter(|steps| *steps <= MAX_INTERPOLATION_STEPS)
                    .filter(|steps| {
                        steps
                            .checked_mul(width)
                            .is_some_and(|elements| elements <= MAX_INTERPOLATION_ELEMENTS)
                    })
                    .filter(|steps| self.buffer.try_reserve(*steps).is_ok())
                    .ok_or_else(|| InterpolatorError::BufferNotAllocatable {
                        multiplier: self.multiplier.clone(),
                    })?;
                // A length-1 operand is reused for every output element, which
                // is exactly what a stride-0 broadcast does.
                let prev_stride = usize::from(prev.len() != 1);
                let action_stride = usize::from(action.len() != 1);
                for i in 1..=steps {
                    let t = T::from_f64(i as f64 / steps as f64);
                    let step = (0..width)
                        .map(|k| T::lerp(prev[k * prev_stride], t, action[k * action_stride]))
                        .collect();
                    self.buffer.push(step);
                }
            }
            _ => {
                // First step: no previous action yet, so run at base FPS.
                self.buffer.clear();
                self.buffer.push(action.to_vec());
            }
        }
        self.prev = Some(action.to_vec());
        self.idx = 0;
        Ok(())
    }

    /// Next interpolated action, or `None` when the buffer is exhausted.
    pub fn get(&mut self) -> Option<&[T]> {
        if self.idx >= self.buffer.len() {
            return None;
        }
        let i = self.idx;
        self.idx += 1;
        Some(&self.buffer[i])
    }

    /// Control interval in seconds for a base rate of `fps`.
    ///
    /// Upstream computes `1.0 / (fps * self.multiplier)`. CPython converts the
    /// `int` multiplier to a double to do that, and raises `OverflowError: int
    /// too large to convert to float` when it will not fit; this returns
    /// [`InterpolatorError::MultiplierNotFloatRepresentable`] at exactly that
    /// point rather than dividing by an infinity and reporting an interval of
    /// zero. Below the boundary the conversion is the nearest `f64` to the
    /// multiplier: it goes through the decimal digits and Rust's
    /// correctly-rounded float parser, so nothing is truncated on the way.
    pub fn get_control_interval(&self, fps: f64) -> Result<f64, InterpolatorError> {
        let multiplier: f64 = self
            .multiplier
            .to_string()
            .parse()
            .expect("a decimal integer is always a parseable float literal");
        if !multiplier.is_finite() {
            return Err(InterpolatorError::MultiplierNotFloatRepresentable(
                self.multiplier.clone(),
            ));
        }
        Ok(1.0 / (fps * multiplier))
    }
}
