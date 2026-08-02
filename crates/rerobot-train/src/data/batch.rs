//! Collating frames into tensors, and the normalization step that follows.
//!
//! Upstream's order in `lerobot_train.train` is: the `DataLoader` collates, then
//! `preprocessor(batch)` normalizes. That order is preserved here —
//! [`crate::data::batch::collate`] produces raw tensors and
//! [`crate::data::batch::Batch::normalized`] applies
//! [`rerobot_core::policy::normalize::Normalizer`] — so that the two steps stay
//! separately testable and the normalization arithmetic stays in the pure,
//! upstream-pinned core crate rather than being reimplemented on tensors.
//!
//! Normalization is elementwise, so applying it per row and then stacking is the
//! same computation as applying it to the stacked tensor. That is what makes the
//! reuse sound rather than convenient.

use crate::data::dataset::Frame;
use crate::data::image::{camera_tensor, CameraNormalization};
use crate::error::{Result, TrainError};
use candle_core::{DType, Device, Tensor};
use indexmap::IndexMap;
use rerobot_core::policy::normalize::Normalizer;

/// A collated batch of frames.
#[derive(Debug)]
pub struct Batch {
    /// Per feature key, a `[batch, window, width]` tensor for windowed keys and a
    /// `[batch, width]` tensor for the rest.
    pub features: IndexMap<String, Tensor>,
    /// Per camera key, a `[batch, channels, height, width]` `f32` tensor.
    ///
    /// Separate from [`Self::features`] rather than mixed into it, because the two
    /// are handled differently at every step that touches them: the collator builds
    /// features out of the dataset's flat parquet rows and cannot build these at all,
    /// and [`Self::normalized`] resolves one statistic per scalar for a feature and
    /// one per *channel* for a camera. Upstream keeps them in one dict only because
    /// torch broadcasts the difference away.
    ///
    /// Attach them with [`Self::with_images`], which is where the contract in
    /// [`crate::data::image`] is enforced.
    pub images: IndexMap<String, Tensor>,
    /// Per windowed feature key, its `[batch, window]` `u8` padding mask.
    ///
    /// `u8` rather than a boolean dtype because candle has no `bool`; `1` is
    /// padded, matching `torch.BoolTensor`'s truth value.
    pub padding: IndexMap<String, Tensor>,
    /// The task string of each frame, in batch order.
    pub tasks: Vec<String>,
    /// The dataset-absolute index of each frame, in batch order.
    pub indices: Vec<i64>,
}

impl Batch {
    /// How many frames the batch holds.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// One feature's tensor.
    pub fn feature(&self, key: &str) -> Result<&Tensor> {
        self.features.get(key).ok_or_else(|| {
            TrainError::Metadata(format!(
                "the batch has no {key:?}; it has {:?}",
                self.features.keys().collect::<Vec<_>>()
            ))
        })
    }

    /// One camera's tensor, `[batch, channels, height, width]`.
    pub fn image(&self, key: &str) -> Result<&Tensor> {
        self.images.get(key).ok_or_else(|| {
            TrainError::Metadata(format!(
                "the batch has no camera {key:?}; it has {:?}. Attach camera tensors with \
                 Batch::with_images",
                self.images.keys().collect::<Vec<_>>()
            ))
        })
    }

    /// One windowed feature's padding mask.
    pub fn padding_mask(&self, key: &str) -> Result<&Tensor> {
        self.padding
            .get(key)
            .ok_or_else(|| TrainError::Metadata(format!("the batch has no {key}_is_pad")))
    }

    /// Attach raw camera tensors, checking each against the contract in
    /// [`crate::data::image`] and applying `normalization` to it.
    ///
    /// `images` holds one entry per camera key, each `[batch, channels, height,
    /// width]` — or `[batch, 1, channels, height, width]`, which is squeezed —  of
    /// `f32` with every element in `[0, 1]`. Insertion order is preserved, and the
    /// model consumes the cameras in the order its own config declares them rather
    /// than in this one, so a batch cannot silently reorder them.
    ///
    /// Normalization happens here rather than in [`Self::normalized`] because it is
    /// the raw `[0, 1]` tensor that the contract can be checked against: once the
    /// statistics have been subtracted there is no range left to verify. The two
    /// steps compose in either order — both are elementwise — so this is the same
    /// computation upstream's collate-then-normalize performs.
    ///
    /// # Errors
    ///
    /// [`TrainError::Metadata`] naming the camera and what was wrong with it, or
    /// when a key is attached twice.
    pub fn with_images(
        mut self,
        images: &IndexMap<String, Tensor>,
        normalization: &CameraNormalization,
    ) -> Result<Self> {
        crate::limits::within(
            images.len(),
            "the number of cameras",
            crate::limits::MAX_CAMERAS,
        )?;
        let batch_size = self.len();
        for (key, tensor) in images {
            if self.images.contains_key(key) {
                return Err(TrainError::Metadata(format!(
                    "camera {key:?} is already attached to this batch"
                )));
            }
            let checked = camera_tensor(key, tensor, batch_size)?;
            self.images
                .insert(key.clone(), normalization.apply(key, &checked)?);
        }
        Ok(self)
    }

    /// This batch with every feature the normalizer knows about transformed.
    ///
    /// Camera tensors ride through untouched: they were normalized when
    /// [`Self::with_images`] attached them.
    pub fn normalized(&self, normalizer: &Normalizer) -> Result<Self> {
        let mut features = IndexMap::with_capacity(self.features.len());
        for (key, tensor) in &self.features {
            features.insert(key.clone(), normalize_tensor(normalizer, key, tensor)?);
        }
        Ok(Self {
            features,
            images: self.images.clone(),
            padding: self.padding.clone(),
            tasks: self.tasks.clone(),
            indices: self.indices.clone(),
        })
    }
}

/// Apply the normalizer along the last axis of a rank-2 or rank-3 tensor.
fn normalize_tensor(normalizer: &Normalizer, key: &str, tensor: &Tensor) -> Result<Tensor> {
    if normalizer.mode(key).is_none() {
        return Ok(tensor.clone());
    }
    let shape = tensor.dims().to_vec();
    let width = *shape.last().ok_or_else(|| {
        TrainError::Metadata(format!(
            "{key:?} is a scalar tensor and cannot be normalized"
        ))
    })?;
    // `chunks(0)` panics. The width comes from a tensor built out of `info.json`, so
    // this is a refusal rather than an assertion: the caller gets a message naming the
    // feature instead of an abort.
    if width == 0 {
        return Err(TrainError::Metadata(format!(
            "{key:?} has width 0, so it carries no scalars to normalize; a feature must \
             declare at least one"
        )));
    }
    let flat = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(flat.len());
    for row in flat.chunks(width) {
        out.extend(normalizer.normalize(key, row)?);
    }
    Ok(Tensor::from_vec(out, shape, tensor.device())?)
}

/// Stack `frames` into tensors on `device`.
///
/// # Errors
///
/// When `frames` is empty, or when two frames disagree about a feature's width or
/// window length — either would be a silently ragged batch.
pub fn collate(frames: &[Frame], device: &Device) -> Result<Batch> {
    let Some(first) = frames.first() else {
        return Err(TrainError::Metadata(
            "cannot collate an empty batch".to_owned(),
        ));
    };
    // `collate` is a public entry point, so it carries its own bound rather than
    // trusting that a `TrainConfig` was validated before it.
    crate::limits::within(
        frames.len(),
        "the batch size",
        crate::limits::MAX_BATCH_SIZE,
    )?;

    let mut features = IndexMap::with_capacity(first.windows.len());
    for (key, first_window) in &first.windows {
        let window_length = first_window.len();
        let width = first_window.first().map_or(0, Vec::len);
        // An empty window or an empty row is refused here rather than turned into a
        // degenerate tensor: every consumer of a batch divides the flat buffer by one
        // of these two, and `slice::chunks` panics on a zero divisor.
        if window_length == 0 || width == 0 {
            return Err(TrainError::Metadata(format!(
                "{key:?} collates to {window_length} rows of width {width}; a feature with \
                 no scalars cannot be batched"
            )));
        }
        // Checked, not `a * b * c`. All three operands come from outside the process:
        // the batch size from the command line, the window length from `chunk_size`,
        // and the width from `info.json`. An overflowing product panics in a checked
        // build and wraps in release, and a wrapped reservation is the worse of the
        // two — the allocation succeeds at the wrong size and the extend then grows it
        // silently, or worse.
        let reservation = crate::limits::checked_product(
            &[frames.len(), window_length, width],
            &format!("the collated size of {key:?}"),
        )?;
        crate::limits::within(
            reservation,
            &format!("the collated size of {key:?}"),
            crate::limits::MAX_DECODED_VALUES,
        )?;
        let mut values = Vec::with_capacity(reservation);
        for frame in frames {
            let window = frame.windows.get(key).ok_or_else(|| {
                TrainError::Metadata(format!("frame {} has no {key:?}", frame.index))
            })?;
            if window.len() != window_length {
                return Err(TrainError::Metadata(format!(
                    "frame {} has {} rows of {key:?} but the batch expects {window_length}",
                    frame.index,
                    window.len()
                )));
            }
            for row in window {
                if row.len() != width {
                    return Err(TrainError::Metadata(format!(
                        "frame {} has width {} for {key:?} but the batch expects {width}",
                        frame.index,
                        row.len()
                    )));
                }
                values.extend_from_slice(row);
            }
        }
        // A key with no delta window contributes one row per frame, and upstream
        // hands the policy a `(batch, width)` tensor for it -- not
        // `(batch, 1, width)`. The squeeze is what keeps the shapes upstream's.
        let shape: Vec<usize> = if window_length == 1 && !first.padding.contains_key(key) {
            vec![frames.len(), width]
        } else {
            vec![frames.len(), window_length, width]
        };
        features.insert(key.clone(), Tensor::from_vec(values, shape, device)?);
    }

    let mut padding = IndexMap::with_capacity(first.padding.len());
    for (key, first_flags) in &first.padding {
        let window_length = first_flags.len();
        let reservation = crate::limits::checked_mul(
            frames.len(),
            window_length,
            &format!("the collated padding mask of {key:?}"),
        )?;
        let mut flags = Vec::with_capacity(reservation);
        for frame in frames {
            let frame_flags = frame.padding.get(key).ok_or_else(|| {
                TrainError::Metadata(format!("frame {} has no {key}_is_pad", frame.index))
            })?;
            if frame_flags.len() != window_length {
                return Err(TrainError::Metadata(format!(
                    "frame {} has {} padding flags for {key:?} but the batch expects \
                     {window_length}",
                    frame.index,
                    frame_flags.len()
                )));
            }
            flags.extend(frame_flags.iter().map(|flag| u8::from(*flag)));
        }
        padding.insert(
            key.clone(),
            Tensor::from_vec(flags, (frames.len(), window_length), device)?.to_dtype(DType::U8)?,
        );
    }

    Ok(Batch {
        features,
        // Cameras are attached separately, by `Batch::with_images`, which is the one
        // place the contract in `crate::data::image` is enforced and the per-channel
        // statistics are applied. `collate_images` produces what it takes.
        images: IndexMap::new(),
        padding,
        tasks: frames.iter().map(|frame| frame.task.clone()).collect(),
        indices: frames.iter().map(|frame| frame.index).collect(),
    })
}

/// Stack the frames' decoded embedded cameras into `[batch, channels, height, width]`
/// tensors, one per camera key.
///
/// The counterpart of [`collate`] for [`crate::data::dataset::Frame::images`]. What it
/// returns is what [`Batch::with_images`] takes, deliberately: the range check, the
/// camera count bound and the normalization all live there, and going through it is
/// what keeps a dataset-decoded camera and a caller-supplied one the same computation.
///
/// The map is empty for a dataset with no embedded camera, which is the state-only
/// case, and `with_images` on an empty map is a no-op.
///
/// # Errors
///
/// When `frames` is empty, when two frames disagree about which cameras they carry or
/// about a camera's extent, or when the stacked size is past [`crate::limits`].
pub fn collate_images(frames: &[Frame], device: &Device) -> Result<IndexMap<String, Tensor>> {
    let Some(first) = frames.first() else {
        return Err(TrainError::Metadata(
            "cannot collate an empty batch".to_owned(),
        ));
    };
    crate::limits::within(
        first.images.len(),
        "the number of cameras",
        crate::limits::MAX_CAMERAS,
    )?;

    let mut out = IndexMap::with_capacity(first.images.len());
    for (key, first_image) in &first.images {
        let extent = (first_image.channels, first_image.height, first_image.width);
        let per_frame = crate::limits::checked_product(
            &[extent.0, extent.1, extent.2],
            &format!("the size of one frame of camera {key:?}"),
        )?;
        let reservation = crate::limits::checked_mul(
            per_frame,
            frames.len(),
            &format!("the collated size of camera {key:?}"),
        )?;
        crate::limits::within(
            reservation,
            &format!("the collated size of camera {key:?}"),
            crate::limits::MAX_DECODED_VALUES,
        )?;
        let mut values = Vec::with_capacity(reservation);
        for frame in frames {
            let image = frame.images.get(key).ok_or_else(|| {
                TrainError::Metadata(format!("frame {} has no camera {key:?}", frame.index))
            })?;
            // A ragged batch is the failure this prevents: `Tensor::from_vec` would
            // accept the flat buffer against the first frame's shape and silently
            // reinterpret every later frame's pixels.
            if (image.channels, image.height, image.width) != extent {
                return Err(TrainError::Metadata(format!(
                    "frame {} carries camera {key:?} as {}x{}x{} but the batch expects \
                     {}x{}x{}",
                    frame.index,
                    image.channels,
                    image.height,
                    image.width,
                    extent.0,
                    extent.1,
                    extent.2
                )));
            }
            values.extend_from_slice(&image.pixels);
        }
        out.insert(
            key.clone(),
            Tensor::from_vec(values, (frames.len(), extent.0, extent.1, extent.2), device)?,
        );
    }
    Ok(out)
}
