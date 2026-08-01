//! The camera-image contract: what a camera tensor must be, and the per-channel
//! normalization applied to it on the way into a [`crate::data::batch::Batch`].
//!
//! # The one supported input form
//!
//! A camera arrives as a **candle tensor already in memory**:
//!
//! * dtype `f32` — the dtype `LeRobotDataset` produces and the dtype the model's
//!   parameters carry. Nothing is converted on the way in; an integer tensor is a
//!   different representation of an image and is refused rather than divided by 255
//!   on a guess.
//! * shape `[batch, channels, height, width]`, or `[batch, 1, channels, height,
//!   width]` with the singleton axis being the observation step. ACT fixes
//!   `n_obs_steps` at 1, so that axis can only ever be one frame deep, and it is
//!   squeezed away rather than silently averaged or truncated.
//! * every element in `[0, 1]` — the range `torchvision`'s decoder produces once
//!   the `uint8` frame has been divided by 255, and the range
//!   `dataset.return_uint8 = false` promises.
//!
//! # What is *not* supported, and why it is an error rather than a gap
//!
//! Neither on-disk camera form of a LeRobot v3.0 dataset can be read here:
//!
//! * `dtype: "video"` features live in `videos/<key>/chunk-XXX/file-XXX.mp4`. Reading
//!   one frame means an AV1 or H.264 decoder; upstream shells out to `torchcodec` or
//!   `pyav`, and neither has a pure-Rust equivalent in this workspace.
//! * `dtype: "image"` features live as PNG or JPEG files under
//!   `images/<key>/episode-XXXXXX/`, which needs an image codec that is likewise not
//!   ported.
//!
//! [`crate::data::meta::DatasetMetadata::load`] refuses a dataset declaring either,
//! naming the feature and both formats. It does not drop the feature and read the
//! rest: a policy trained on a silently state-only view of a dataset that has
//! cameras is not the policy that was asked for.

use crate::error::{Result, TrainError};
use candle_core::{DType, Tensor};
use rerobot_core::policy::normalize::NORMALIZATION_EPS;

/// `IMAGENET_STATS["mean"]`, the per-channel mean `LeRobotDataset` attaches to every
/// camera feature when `dataset.use_imagenet_stats` is true — which is its default,
/// and what `train_config.json` records.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// `IMAGENET_STATS["std"]`.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The per-channel `MEAN_STD` transform applied to camera tensors.
///
/// Separate from [`rerobot_core::policy::normalize::Normalizer`] because the two
/// have different shapes of statistic. That normalizer resolves one statistic per
/// *scalar* of a feature, which is what a state vector needs; an image's statistics
/// are one per *channel*, broadcast over height and width. Upstream gets both from
/// the same `NormalizerProcessorStep` only because torch broadcasts for it.
///
/// The arithmetic is the same one: `(value - mean) / (std + eps)` with upstream's
/// `eps` of [`NORMALIZATION_EPS`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CameraNormalization {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl CameraNormalization {
    /// Leave camera tensors exactly as they arrive.
    ///
    /// This is upstream's behaviour for a feature with no statistics entry — `key
    /// not in self._tensor_stats`, and the tensor is returned unchanged.
    pub fn identity() -> Self {
        Self::default()
    }

    /// The ImageNet statistics, which is what a dataset with
    /// `use_imagenet_stats = true` carries for every camera.
    pub fn imagenet() -> Self {
        Self {
            mean: IMAGENET_MEAN.to_vec(),
            std: IMAGENET_STD.to_vec(),
        }
    }

    /// Per-channel statistics of a caller's own.
    ///
    /// # Errors
    ///
    /// When the two have different lengths, when either holds a non-finite value, or
    /// when a standard deviation is negative — each of which would produce a
    /// normalized tensor that no longer describes the image.
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Result<Self> {
        if mean.is_empty() {
            return Err(TrainError::Metadata(
                "camera statistics must cover at least one channel; use \
                 CameraNormalization::identity to leave images untouched"
                    .to_owned(),
            ));
        }
        if mean.len() != std.len() {
            return Err(TrainError::Metadata(format!(
                "the camera mean has {} channels but the standard deviation has {}",
                mean.len(),
                std.len()
            )));
        }
        for (name, values) in [("mean", &mean), ("std", &std)] {
            if let Some((channel, value)) = values
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(TrainError::Metadata(format!(
                    "the camera {name} is {value} in channel {channel}, which is not finite"
                )));
            }
        }
        if let Some((channel, value)) = std.iter().enumerate().find(|(_, value)| **value < 0.0) {
            return Err(TrainError::Metadata(format!(
                "the camera standard deviation is {value} in channel {channel}; a negative \
                 spread would flip the sign of every pixel it scales"
            )));
        }
        Ok(Self { mean, std })
    }

    /// How many channels the statistics cover, or `None` for the identity.
    pub fn channels(&self) -> Option<usize> {
        if self.mean.is_empty() {
            None
        } else {
            Some(self.mean.len())
        }
    }

    /// Apply to a `[batch, channels, height, width]` tensor.
    ///
    /// # Errors
    ///
    /// When the tensor's channel count differs from the statistics'.
    pub fn apply(&self, key: &str, images: &Tensor) -> Result<Tensor> {
        let Some(channels) = self.channels() else {
            return Ok(images.clone());
        };
        let (_, found, _, _) = images.dims4()?;
        if found != channels {
            return Err(TrainError::Metadata(format!(
                "camera {key:?} has {found} channels but its statistics cover {channels}"
            )));
        }
        let device = images.device();
        let shape = (1, channels, 1, 1);
        let mean = Tensor::from_vec(self.mean.clone(), shape, device)?;
        let spread = Tensor::from_vec(
            self.std
                .iter()
                .map(|value| value + NORMALIZATION_EPS as f32)
                .collect::<Vec<f32>>(),
            shape,
            device,
        )?;
        Ok(images.broadcast_sub(&mean)?.broadcast_div(&spread)?)
    }
}

/// Check a *raw* camera tensor against the contract at the top of this module and
/// return it as `[batch, channels, height, width]`.
///
/// [`camera_view`] plus the range check, which is the half that only holds before
/// normalization.
///
/// # Errors
///
/// Everything [`camera_view`] refuses, plus a value outside `[0, 1]`.
pub fn camera_tensor(key: &str, images: &Tensor, batch_size: usize) -> Result<Tensor> {
    let squeezed = camera_view(key, images, batch_size)?;
    // The range, every element of it. A tensor that is already normalized, or one
    // still in 0..255, produces a plausible-looking loss and a policy trained on the
    // wrong input scale; there is no later point at which that can be detected. A
    // NaN fails this check too, `contains` being false for it.
    let values = squeezed.flatten_all()?.to_vec1::<f32>()?;
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !(0.0..=1.0).contains(*value))
    {
        return Err(TrainError::Metadata(format!(
            "camera {key:?} element {index} is {value}, outside [0, 1]; a camera tensor holds \
             the decoded frame divided by 255, not raw 0..255 samples and not an \
             already-normalized one"
        )));
    }
    Ok(squeezed)
}

/// Refuse a camera tensor holding a value the forward pass cannot use.
///
/// The counterpart of the range check for a tensor that has already been normalized
/// and therefore has no range left to check: one NaN pixel makes every action the
/// policy predicts NaN, and the run's own tripwire then fires a whole training step
/// away from the tensor that caused it.
pub fn require_finite(key: &str, images: &Tensor) -> Result<()> {
    let values = images.flatten_all()?.to_vec1::<f32>()?;
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(TrainError::Metadata(format!(
            "camera {key:?} element {index} is {value}, which is not finite"
        )));
    }
    Ok(())
}

/// The `[batch, channels, height, width]` view of a camera tensor, shape and dtype
/// checked and bounded.
///
/// The `[batch, 1, channels, height, width]` spelling is accepted and squeezed; every
/// other rank is refused.
///
/// # Errors
///
/// [`TrainError::Metadata`] naming `key` and what was wrong: the dtype, the rank, the
/// batch size, an empty extent, or a size past [`crate::limits`].
pub fn camera_view(key: &str, images: &Tensor, batch_size: usize) -> Result<Tensor> {
    if images.dtype() != DType::F32 {
        return Err(TrainError::Metadata(format!(
            "camera {key:?} has dtype {:?}; a camera tensor is f32 with every element in \
             [0, 1], and nothing is converted on the way in",
            images.dtype()
        )));
    }
    let squeezed = match images.rank() {
        4 => images.clone(),
        5 => {
            let steps = images.dims()[1];
            if steps != 1 {
                return Err(TrainError::Metadata(format!(
                    "camera {key:?} has shape {:?}, whose observation axis is {steps} steps \
                     deep; ACT fixes n_obs_steps at 1, so only a singleton axis can be \
                     squeezed away",
                    images.dims()
                )));
            }
            images.squeeze(1)?
        }
        other => {
            return Err(TrainError::Metadata(format!(
                "camera {key:?} has shape {:?} of rank {other}; a camera tensor is \
                 [batch, channels, height, width], or [batch, 1, channels, height, width] \
                 with the singleton observation axis",
                images.dims()
            )))
        }
    };

    let (batch, channels, height, width) = squeezed.dims4()?;
    if batch != batch_size {
        return Err(TrainError::Metadata(format!(
            "camera {key:?} carries {batch} images but the batch holds {batch_size} frames"
        )));
    }
    if channels == 0 || height == 0 || width == 0 {
        return Err(TrainError::Metadata(format!(
            "camera {key:?} has extent {channels}x{height}x{width}; an image with an empty \
             axis carries no pixels"
        )));
    }
    crate::limits::within(
        height,
        &format!("the height of camera {key:?}"),
        crate::limits::MAX_IMAGE_EXTENT,
    )?;
    crate::limits::within(
        width,
        &format!("the width of camera {key:?}"),
        crate::limits::MAX_IMAGE_EXTENT,
    )?;
    let pixels = crate::limits::checked_product(
        &[channels, height, width],
        &format!("the size of one frame of camera {key:?}"),
    )?;
    crate::limits::within(
        pixels,
        &format!("the size of one frame of camera {key:?}"),
        crate::limits::MAX_FEATURE_WIDTH,
    )?;
    let total = crate::limits::checked_mul(
        pixels,
        batch,
        &format!("the collated size of camera {key:?}"),
    )?;
    crate::limits::within(
        total,
        &format!("the collated size of camera {key:?}"),
        crate::limits::MAX_DECODED_VALUES,
    )?;
    Ok(squeezed)
}
