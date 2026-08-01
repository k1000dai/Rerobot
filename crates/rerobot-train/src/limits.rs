//! The resource budget the training slice enforces, in one auditable place.
//!
//! Every number this module bounds arrives from outside the process — a command
//! line, a `meta/info.json`, an episode table, a parquet footer. Two of this
//! crate's dependencies then act on those numbers: candle allocates tensors and
//! Arrow decodes columns, both in code with a large unsafe surface. `forbid(unsafe_code)`
//! on this crate says nothing about either.
//!
//! So the budget is enforced *before* a value reaches them. Three failure modes are
//! being prevented, and they are different from each other:
//!
//! * **allocation aborts.** `Vec::with_capacity(n)` on a hostile `n` aborts the
//!   process. An abort is not a refusal: no message, no exit code a caller can
//!   interpret, and a partially written checkpoint left behind.
//! * **wrapped arithmetic.** A product of two shape dimensions that overflows
//!   panics in a checked build and *wraps* in release. The release behaviour is the
//!   dangerous one: a wrapped width becomes a small number, the allocation succeeds,
//!   and the reader then walks off the end of what it allocated. Every product of
//!   untrusted operands here goes through [`checked_product`].
//! * **unbounded work.** A file that declares ten billion rows costs nothing to
//!   write and unbounded time to read.
//!
//! # How the numbers were chosen
//!
//! Each limit is at least an order of magnitude above what upstream's own defaults
//! and documented file-size caps require, and far enough below a machine's limits
//! that exceeding one is a diagnosable error rather than a resource event.
//! `crates/rerobot-train/tests/limits.rs` asserts both directions: an over-budget
//! value is refused by name, and a realistic ACT configuration still runs. Raising
//! one is a one-line change here, which is the point of having them in one file.

use crate::error::{Result, TrainError};
use num_bigint::BigInt;

/// `dim_model`. Upstream's default is 512; this allows sixteen times that.
pub const MAX_DIM_MODEL: usize = 8_192;

/// `n_heads`. Upstream's default is 8, and a head count above `dim_model` cannot
/// divide it anyway.
pub const MAX_HEADS: usize = 1_024;

/// `dim_feedforward`. Upstream's default is 3200.
pub const MAX_DIM_FEEDFORWARD: usize = 65_536;

/// Any one of the three layer counts. Upstream's defaults are 4, 1 and 4.
pub const MAX_LAYERS: usize = 128;

/// `latent_dim`. Upstream's default is 32.
pub const MAX_LATENT_DIM: usize = 8_192;

/// `chunk_size`, which is also the decoder's sequence length and the action window's
/// width. Upstream's default is 100.
pub const MAX_CHUNK_SIZE: usize = 8_192;

/// Scalars in one frame of one feature. A 512×512×3 image is 786 432, so a camera
/// frame five times that size still fits.
pub const MAX_FEATURE_WIDTH: usize = 1 << 22;

/// Either spatial extent of one camera image.
///
/// Upstream's ACT datasets are 96×96 to 640×480; this allows more than an order of
/// magnitude above the largest of them. The bound is per axis as well as on the
/// product ([`MAX_FEATURE_WIDTH`]) because a 1×4194304 image passes the product and
/// is still a shape no backbone can pool.
pub const MAX_IMAGE_EXTENT: usize = 8_192;

/// Cameras one policy may consume.
///
/// Every camera adds a full ResNet evaluation to each forward pass and `h * w` tokens
/// to the transformer encoder's sequence, so the count is the multiplier on both. The
/// largest upstream ACT configuration uses four.
pub const MAX_CAMERAS: usize = 64;

/// `batch_size`. Upstream's default is 8.
pub const MAX_BATCH_SIZE: usize = 65_536;

/// `steps`. Upstream's default is 100 000.
pub const MAX_STEPS: u64 = 1_000_000_000;

/// Bytes in one parquet file. Upstream caps a data file at
/// `DEFAULT_DATA_FILE_SIZE_IN_MB` = 100 MB and a video file at 200 MB.
pub const MAX_PARQUET_FILE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Parquet files one dataset may present, across `data/` and `meta/episodes/`.
pub const MAX_PARQUET_FILES: usize = 100_000;

/// Rows one dataset may hold in total, across every file.
///
/// The predecessor of this limit was per file and applied *after* Arrow had already
/// decoded a batch, so a dataset of many files was unbounded and a single file's
/// decode was unbounded too.
pub const MAX_DATASET_ROWS: usize = 50_000_000;

/// Decoded scalars one dataset may materialize, summed over every column of every
/// file.
///
/// Rows alone do not bound the work: one row of a very wide feature costs as much as
/// many rows of a narrow one. At `f32` this is a gibibyte.
pub const MAX_DECODED_VALUES: usize = 1 << 28;

/// UTF-8 bytes one string column may hold in total.
pub const MAX_STRING_BYTES: usize = 64 * 1024 * 1024;

/// Elements one list column may hold in total, summed across its rows.
pub const MAX_LIST_ELEMENTS: usize = 1 << 26;

/// Episodes one dataset may declare.
pub const MAX_EPISODES: usize = 1_000_000;

/// Bytes one single tensor may occupy.
///
/// `dim_feedforward` at 65 536 by `dim_model` at 8 192 is one 2 GiB `f32` weight, and
/// every individual field is inside its own limit while that happens. A per-tensor
/// budget is what makes one absurd pair refusable without narrowing either field.
///
/// 512 MiB is 134 million `f32`s — comfortably above upstream's largest ACT tensor
/// (`dim_feedforward` 3200 by `dim_model` 512 is 6.5 MB) and far below a figure that
/// could exhaust a machine unnoticed.
pub const MAX_TENSOR_BYTES: usize = 512 * 1024 * 1024;

/// Bytes every tensor of one model may occupy in total.
///
/// The per-tensor budget does not bound the model: the allowed layer counts multiply
/// the largest permitted tensor 128 times over for the encoder and again for the
/// decoder, which is how a configuration whose every field is legal reached roughly a
/// tebibyte of feed-forward weights alone.
///
/// 4 GiB is about twenty times upstream's own default ACT (51.6 M parameters, ~207 MB
/// at `f32`), so a stock model and a considerably larger one both fit.
pub const MAX_MODEL_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Cells — `rows * columns` — one parquet file may declare.
///
/// Rows, columns and compressed bytes bounded separately do not bound a decode: a
/// thousand rows of a thousand wide columns is inside all three and still a gibibyte
/// of cells. The product is what the work actually costs, and the footer states both
/// factors, so it can be refused before Arrow allocates anything.
pub const MAX_PARQUET_CELLS: usize = 1 << 30;

/// Uncompressed bytes one parquet file may declare across its columns.
///
/// Cells do not bound bytes either: one cell of a wide list column costs far more than
/// one cell of an `int64`. Every parquet footer records each column's uncompressed
/// size, which is the closest thing to the decode's true cost available before
/// decoding.
pub const MAX_DECODED_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Parquet files the `meta/episodes/` tree may contain.
///
/// Separate from [`MAX_PARQUET_FILES`], which bounds the `data/` files the episode
/// table names: the metadata tree is discovered by walking the directory, so its size
/// is bounded by what is on disk rather than by anything already validated.
pub const MAX_EPISODE_FILES: usize = 10_000;

/// Columns one parquet file may declare.
///
/// Upstream's episode table has seven columns plus ten statistics per feature, so a
/// dataset with a hundred features is still comfortably inside this.
pub const MAX_PARQUET_COLUMNS: usize = 65_536;

/// The product of `dimensions`, or an error naming `what` if it overflows.
///
/// Returns 1 for an empty slice, matching the mathematical convention and Python's
/// `math.prod(())`.
///
/// ```
/// use rerobot_train::limits::checked_product;
///
/// assert_eq!(checked_product(&[2, 3, 4], "a shape"), Ok(24));
/// assert!(checked_product(&[usize::MAX, 2], "a shape").is_err());
/// ```
pub fn checked_product(dimensions: &[usize], what: &str) -> Result<usize> {
    let mut total: usize = 1;
    for dimension in dimensions {
        total = total.checked_mul(*dimension).ok_or_else(|| {
            TrainError::Metadata(format!(
                "{what} is too large: the product of {dimensions:?} overflows a {}-bit integer",
                usize::BITS
            ))
        })?;
    }
    Ok(total)
}

/// `left * right`, or an error naming `what`.
pub fn checked_mul(left: usize, right: usize, what: &str) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        TrainError::Metadata(format!(
            "{what} is too large: {left} * {right} overflows a {}-bit integer",
            usize::BITS
        ))
    })
}

/// `left + right`, or an error naming `what`.
pub fn checked_add(left: usize, right: usize, what: &str) -> Result<usize> {
    left.checked_add(right).ok_or_else(|| {
        TrainError::Metadata(format!(
            "{what} is too large: {left} + {right} overflows a {}-bit integer",
            usize::BITS
        ))
    })
}

/// An arbitrary-precision integer narrowed to a `usize` no larger than `limit`.
///
/// One function for both halves on purpose: a value that does not fit `usize` and a
/// value that fits but is absurd are the same problem, and reporting them the same
/// way means a caller cannot handle one and forget the other.
pub fn bounded_usize(value: &BigInt, name: &str, limit: usize) -> Result<usize> {
    let narrowed = usize::try_from(value).map_err(|_| out_of_range(name, value, limit))?;
    if narrowed > limit {
        return Err(out_of_range(name, value, limit));
    }
    Ok(narrowed)
}

/// [`bounded_usize`] for a value that must also be at least one.
pub fn bounded_positive_usize(value: &BigInt, name: &str, limit: usize) -> Result<usize> {
    let narrowed = bounded_usize(value, name, limit)?;
    if narrowed == 0 {
        return Err(TrainError::Metadata(format!("{name} must be positive")));
    }
    Ok(narrowed)
}

/// Check a plain machine integer against a limit, reporting it the same way.
pub fn within(value: usize, name: &str, limit: usize) -> Result<usize> {
    if value > limit {
        return Err(out_of_range(name, &BigInt::from(value), limit));
    }
    Ok(value)
}

/// Check a `u64` against a `u64` limit.
pub fn within_u64(value: u64, name: &str, limit: u64) -> Result<u64> {
    if value > limit {
        return Err(TrainError::Metadata(format!(
            "{name} = {value} exceeds the limit of {limit}; \
             raise MAX_* in rerobot-train's `limits` module if this is genuinely needed"
        )));
    }
    Ok(value)
}

fn out_of_range(name: &str, value: &BigInt, limit: usize) -> TrainError {
    TrainError::Metadata(format!(
        "{name} = {value} is outside the supported range 0..={limit}; \
         raise MAX_* in rerobot-train's `limits` module if this is genuinely needed"
    ))
}
