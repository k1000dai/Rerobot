//! Port of `lerobot.rollout.ring_buffer` (`RolloutRingBuffer`, `_estimate_frame_bytes`).
//!
//! Two upstream quirks are reproduced deliberately, because callers observe
//! them through `len()` / `estimated_bytes()`:
//!
//! * frames dropped by the `deque(maxlen=...)` length cap do **not** decrement
//!   the byte accounting — only the explicit memory-cap eviction branch does;
//! * a zero-length cap (`int(max_seconds * fps) == 0`) silently discards every
//!   appended frame while still accruing its bytes.
//!
//! # Numeric domain
//!
//! Upstream counts bytes in Python `int`s, which are unbounded, so this port
//! counts them in [`ByteCount`], which is also unbounded. Every frame estimate
//! and every running total is exact: there is no width at which the accounting
//! saturates, wraps, or panics, in debug or release, for any input a caller can
//! construct. That includes the cases a fixed 128-bit total gets wrong — two
//! `usize::MAX * usize::MAX` tensors in one frame, and the unbounded accrual
//! the zero-length-cap quirk above makes possible.
//!
//! The byte *cap* is the one quantity that stays fixed-width, at `i128`. That
//! is not an approximation: the cap is `int(max_memory_mb * 1024 * 1024)` and
//! `max_memory_mb` is an `i64`, so `i64::MAX * 2^20` and `i64::MIN * 2^20` both
//! fit `i128` exactly, negative caps included.

use crate::byte_count::ByteCount;
use indexmap::IndexMap;
use std::collections::VecDeque;
use std::fmt;

/// One value inside a telemetry frame, modelling the Python types that
/// `_estimate_frame_bytes` discriminates on.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameValue {
    /// `torch.Tensor`: costed as `nelement() * element_size()`.
    Tensor {
        /// Number of elements.
        numel: usize,
        /// Bytes per element.
        element_size: usize,
    },
    /// `numpy.ndarray`, or any object exposing `nbytes`.
    NBytes(usize),
    /// Python `int` (and `bool`, an `int` subclass): 8 bytes.
    Int(i64),
    /// Python `float`: 8 bytes.
    Float(f64),
    /// Python `str`: costed as `len(v)`, i.e. code points, not UTF-8 bytes.
    Str(String),
    /// Python `bytes`: costed as `len(v)`.
    Bytes(Vec<u8>),
    /// Anything else: contributes nothing.
    Other,
}

/// An ordered telemetry frame (`dict[str, Any]` upstream).
pub type Frame = IndexMap<String, FrameValue>;

/// Errors raised while constructing a [`RolloutRingBuffer`].
///
/// The four variants are the four ways upstream's
/// `deque(maxlen=int(max_seconds * fps))` can fail, in the order CPython
/// reaches them. The payloads carry the exact truncated frame capacity as an
/// `f64` because a value that fails these checks need not fit any integer type
/// — `{:.0}` prints it with the same digits Python's `int()` would.
#[derive(Debug, Clone, PartialEq)]
pub enum RingBufferError {
    /// `max_seconds * fps` is NaN. Python: `int()` raises
    /// `ValueError: cannot convert float NaN to integer`.
    NanMaxLen,
    /// `max_seconds * fps` is infinite, in either direction. Python: `int()`
    /// raises `OverflowError: cannot convert float infinity to integer`.
    InfiniteMaxLen,
    /// The frame capacity is outside `Py_ssize_t`. CPython converts `maxlen`
    /// with `PyLong_AsSsize_t`, which raises
    /// `OverflowError: Python int too large to convert to C ssize_t` — for large
    /// magnitudes of *either* sign, and before the non-negative check below.
    MaxLenNotRepresentable(f64),
    /// The frame capacity is negative but representable; `collections.deque`
    /// raises `ValueError: maxlen must be non-negative`.
    NegativeMaxLen(f64),
}

impl fmt::Display for RingBufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Wording copied from CPython's own exception messages.
            Self::NanMaxLen => f.write_str("cannot convert float NaN to integer"),
            Self::InfiniteMaxLen => f.write_str("cannot convert float infinity to integer"),
            Self::MaxLenNotRepresentable(n) => write!(
                f,
                "Python int too large to convert to C ssize_t: maxlen would be {n:.0}"
            ),
            Self::NegativeMaxLen(n) => write!(f, "maxlen must be non-negative, got {n:.0}"),
        }
    }
}

impl std::error::Error for RingBufferError {}

/// Rough byte estimate for a single frame, port of `_estimate_frame_bytes`.
///
/// Never returns less than `1`, matching upstream's `max(total, 1)`. The result
/// is exact for every frame, at any magnitude; see the module docs.
pub fn estimate_frame_bytes(frame: &Frame) -> ByteCount {
    let mut total = ByteCount::zero();
    for value in frame.values() {
        let cost = match value {
            FrameValue::Tensor {
                numel,
                element_size,
            } => ByteCount::product(*numel, *element_size),
            FrameValue::NBytes(n) => ByteCount::from(*n),
            // Python `int`/`float` (and `bool`, an `int` subclass) are flat 8s.
            FrameValue::Int(_) | FrameValue::Float(_) => ByteCount::from(8u64),
            // `len(str)` counts code points, not UTF-8 bytes.
            FrameValue::Str(s) => ByteCount::from(s.chars().count()),
            FrameValue::Bytes(b) => ByteCount::from(b.len()),
            FrameValue::Other => ByteCount::zero(),
        };
        total += cost;
    }
    total.max(ByteCount::from(1u64))
}

/// Fixed-capacity circular buffer for observation/action frames.
///
/// Bounded by both a frame count (`int(max_seconds * fps)`) and a byte budget
/// (`int(max_memory_mb * 1024 * 1024)`).
///
/// ```
/// use rerobot_core::ring_buffer::{Frame, FrameValue, RolloutRingBuffer};
///
/// let mut buffer = RolloutRingBuffer::new(2.0 / 30.0, 1024, 30.0).unwrap();
/// assert_eq!(buffer.max_frames(), 2);
///
/// for i in 0..3 {
///     let mut frame = Frame::new();
///     frame.insert("step".to_string(), FrameValue::Int(i));
///     buffer.append(frame);
/// }
///
/// let drained = buffer.drain();
/// assert_eq!(drained.len(), 2);
/// assert_eq!(drained[0]["step"], FrameValue::Int(1));
/// ```
#[derive(Debug, Clone)]
pub struct RolloutRingBuffer {
    max_frames: usize,
    max_bytes: i128,
    buffer: VecDeque<Frame>,
    current_bytes: ByteCount,
}

impl RolloutRingBuffer {
    /// Construct with the upstream defaults: 30 s, 2048 MiB, 30 fps.
    pub fn with_defaults() -> Self {
        Self::new(30.0, 2048, 30.0).expect("upstream defaults are valid")
    }

    /// Construct a buffer bounded by `max_seconds * fps` frames and `max_memory_mb` MiB.
    ///
    /// The frame cap follows `deque(maxlen=int(max_seconds * fps))` exactly,
    /// including which of Python's four errors each rejected value produces; see
    /// [`RingBufferError`].
    pub fn new(max_seconds: f64, max_memory_mb: i64, fps: f64) -> Result<Self, RingBufferError> {
        let frames = max_seconds * fps;
        // `int()` rejects NaN with ValueError and infinity with OverflowError.
        if frames.is_nan() {
            return Err(RingBufferError::NanMaxLen);
        }
        if frames.is_infinite() {
            return Err(RingBufferError::InfiniteMaxLen);
        }
        // Python `int()` truncates toward zero.
        let frames = frames.trunc();
        // `PyLong_AsSsize_t` runs before the non-negative check, so an
        // out-of-range magnitude is an OverflowError whichever sign it has.
        // `PY_SSIZE_T_MAX + 1` is exact in f64, so `>=` is the right boundary.
        let ssize_max_plus_one = (isize::MAX as u128 + 1) as f64;
        if frames >= ssize_max_plus_one || frames < isize::MIN as f64 {
            return Err(RingBufferError::MaxLenNotRepresentable(frames));
        }
        if frames < 0.0 {
            return Err(RingBufferError::NegativeMaxLen(frames));
        }
        Ok(Self {
            // In range by the checks above, so this conversion is lossless.
            max_frames: frames as usize,
            // Exact for every `i64` input: `i64::MAX * 2^20` fits in `i128`.
            max_bytes: (max_memory_mb as i128) * 1024 * 1024,
            buffer: VecDeque::new(),
            current_bytes: ByteCount::zero(),
        })
    }

    /// Frame-count cap (`int(max_seconds * fps)`, truncated toward zero).
    pub fn max_frames(&self) -> usize {
        self.max_frames
    }

    /// Byte cap (`int(max_memory_mb * 1024 * 1024)`).
    ///
    /// Signed, because Python accepts a negative `max_memory_mb` and the
    /// resulting cap makes every `append` drain the buffer first.
    pub fn max_bytes(&self) -> i128 {
        self.max_bytes
    }

    /// Whether admitting `frame_bytes` more would breach the byte cap, i.e.
    /// upstream's `self._current_bytes + frame_bytes > self._max_bytes`.
    fn over_cap(&self, frame_bytes: &ByteCount) -> bool {
        // A negative cap is always breached; a byte count cannot express one.
        let Ok(max_bytes) = u128::try_from(self.max_bytes) else {
            return true;
        };
        &self.current_bytes + frame_bytes > max_bytes
    }

    /// Add a frame, evicting the oldest frames until under the memory cap.
    pub fn append(&mut self, frame: Frame) {
        let frame_bytes = estimate_frame_bytes(&frame);

        while self.over_cap(&frame_bytes) && !self.buffer.is_empty() {
            let evicted = self.buffer.pop_front().expect("buffer is non-empty");
            // Every subtracted frame was added first and is popped at most once,
            // so the running total never goes below zero; `saturating_sub` just
            // makes that invariant unfalsifiable.
            self.current_bytes = self
                .current_bytes
                .saturating_sub(&estimate_frame_bytes(&evicted));
        }

        // `deque(maxlen=0)` discards everything appended to it, and a `deque` at
        // capacity drops its oldest element without going through the eviction
        // branch above. Upstream adjusts `_current_bytes` for neither, so the
        // increment below is unconditional.
        if self.max_frames > 0 {
            if self.buffer.len() == self.max_frames {
                self.buffer.pop_front();
            }
            self.buffer.push_back(frame);
        }
        // Exact at any magnitude, like the Python `int` it stands in for.
        self.current_bytes += &frame_bytes;
    }

    /// Return all buffered frames and clear the buffer.
    pub fn drain(&mut self) -> Vec<Frame> {
        let frames = self.buffer.drain(..).collect();
        self.current_bytes = ByteCount::zero();
        frames
    }

    /// Discard all buffered frames.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.current_bytes = ByteCount::zero();
    }

    /// Number of buffered frames.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer holds no frames.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Estimated total byte size of all buffered frames.
    ///
    /// Upstream's `_current_bytes`, quirks included: it is not decremented when
    /// the frame-count cap drops a frame, so it is an upper bound on what is
    /// actually buffered rather than an exact measurement.
    pub fn estimated_bytes(&self) -> ByteCount {
        self.current_bytes.clone()
    }
}
