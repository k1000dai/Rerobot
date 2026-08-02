//! The camera-image contract: what a camera tensor must be, and the per-channel
//! normalization applied to it on the way into a [`crate::data::batch::Batch`].
//!
//! # The in-memory input form
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
//! # The other supported input form: an embedded `image` column
//!
//! A LeRobot v3.0 dataset whose `info.json` declares `dtype: "image"` stores one
//! encoded frame per row *inside* the parquet file, as
//! `struct<bytes: binary, path: string>`. [`crate::data::image::DecodedImage::from_encoded`] decodes those
//! bytes natively — PNG and JPEG, through the `image` crate with exactly those two
//! codecs compiled in — and produces the same `[0, 1]` CHW `f32` values the in-memory
//! contract above describes. Nothing shells out: no Python, no PIL, no ffmpeg.
//!
//! Every decode is bounded before it starts. The encoded cell is capped at
//! [`crate::limits::MAX_EMBEDDED_IMAGE_BYTES`], the format must be one of the two
//! compiled in, the decoded extent is checked against the shape `info.json` declares
//! *before* any pixel buffer is allocated, and the decoder is additionally given the
//! same limits of its own. A cell that disagrees with `info.json` about its size is an
//! error rather than a silently resized frame.
//!
//! # What is *not* supported, and why it is an error rather than a gap
//!
//! * `dtype: "video"` features live in `videos/<key>/chunk-XXX/file-XXX.mp4`. Reading
//!   one frame means an AV1 or H.264 decoder; upstream shells out to `torchcodec` or
//!   `pyav`, and neither has a pure-Rust equivalent in this workspace.
//!   [`crate::data::meta::DatasetMetadata::load`] refuses a dataset declaring one,
//!   naming the feature. It does not drop the feature and read the rest: a policy
//!   trained on a silently state-only view of a dataset that has cameras is not the
//!   policy that was asked for.
//! * Any encoded format other than PNG and JPEG. The codec set is deliberately narrow
//!   and is named in the error, so a dataset carrying WebP says so rather than
//!   producing an opaque decoder failure.

use crate::error::{Result, TrainError};
use candle_core::{DType, Device, Tensor};
use rerobot_core::policy::normalize::NORMALIZATION_EPS;
use std::io::Cursor;
// Qualified at every use site as `::image::` too, because this module is itself
// named `image`; the import is what brings the decoder trait's methods into scope.
use ::image::ImageDecoder as _;

/// The encoded formats an embedded camera cell may carry.
///
/// The `image` dependency is compiled with exactly these two codecs, so this constant
/// and the manifest's feature list are the same fact stated twice; `tests/image.rs`
/// asserts a third format is refused by name rather than by decoder failure.
pub const SUPPORTED_IMAGE_FORMATS: [&str; 2] = ["PNG", "JPEG"];

/// A decoded LeRobot v3.0 embedded camera frame, in RGB channel-first layout.
///
/// `pixels` holds `channels * height * width` values in CHW order, each the encoded
/// sample divided by 255 — the same `[0, 1]` range `torchvision`'s decoder produces
/// and the range the rest of this module's contract requires. The `path` field that
/// travels beside the bytes in the parquet struct is kept as provenance; the encoded
/// bytes are not retained after the decode.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedImage {
    /// The `path` field carried beside the embedded bytes, when the cell has one.
    pub path: Option<String>,
    /// Channels. Always three: an embedded ACT camera decodes as RGB.
    pub channels: usize,
    /// Height in pixels.
    pub height: usize,
    /// Width in pixels.
    pub width: usize,
    /// RGB pixels in CHW order, each in `[0, 1]`.
    pub pixels: Vec<f32>,
}

impl DecodedImage {
    /// Decode one embedded cell, against the shape `info.json` declared for it.
    ///
    /// `declared_shape` is `(channels, height, width)`, already validated to be a
    /// three-channel shape inside [`crate::limits`] by the dataset reader. It is
    /// checked against the encoded header *before* the pixel buffer is allocated, so a
    /// cell that does not match costs one header parse rather than a full decode.
    ///
    /// # Errors
    ///
    /// [`TrainError::Metadata`] naming `key` and which contract was broken: an empty
    /// or oversized cell, an unidentifiable or unsupported format, an extent past
    /// [`crate::limits::MAX_IMAGE_EXTENT`], a decoded size past
    /// [`crate::limits::MAX_FEATURE_WIDTH`], a shape disagreeing with `info.json`, or
    /// a decoder failure on truncated or corrupt bytes.
    pub fn from_encoded(
        key: &str,
        bytes: &[u8],
        path: Option<String>,
        declared_shape: (usize, usize, usize),
    ) -> Result<Self> {
        // Both ends first, and before a decoder exists: an empty cell has no header to
        // parse, and an oversized one must not reach a decoder at all.
        if bytes.is_empty() {
            return Err(TrainError::Metadata(format!(
                "embedded image {key:?} has no encoded bytes"
            )));
        }
        if bytes.len() > crate::limits::MAX_EMBEDDED_IMAGE_BYTES {
            return Err(TrainError::Metadata(format!(
                "embedded image {key:?} carries {} encoded bytes, above the {} the reader will \
                 hand to an image decoder",
                bytes.len(),
                crate::limits::MAX_EMBEDDED_IMAGE_BYTES
            )));
        }

        let reader = ::image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|error| {
                TrainError::Metadata(format!("embedded image {key:?} could not be read: {error}"))
            })?;
        // The format allow-list is checked by name rather than left to the decoder.
        // `image` is compiled with two codecs, so a WebP cell would otherwise fail as
        // "unsupported by this build", which reads like a build fault rather than the
        // deliberate boundary it is.
        match reader.format() {
            Some(::image::ImageFormat::Png | ::image::ImageFormat::Jpeg) => {}
            Some(other) => {
                return Err(TrainError::unsupported(format!(
                    "embedded image {key:?} is {other:?}; this reader decodes {} only",
                    SUPPORTED_IMAGE_FORMATS.join(" and ")
                )))
            }
            None => {
                return Err(TrainError::Metadata(format!(
                    "embedded image {key:?} does not begin with a {} header",
                    SUPPORTED_IMAGE_FORMATS.join(" or ")
                )))
            }
        }
        let mut decoder = reader.into_decoder().map_err(|error| {
            TrainError::Metadata(format!(
                "embedded image {key:?} could not be decoded as {}: {error}",
                SUPPORTED_IMAGE_FORMATS.join(" or ")
            ))
        })?;

        let (width, height) = decoder.dimensions();
        let width = usize::try_from(width).map_err(|_| {
            TrainError::Metadata(format!("embedded image {key:?} has a width past usize"))
        })?;
        let height = usize::try_from(height).map_err(|_| {
            TrainError::Metadata(format!("embedded image {key:?} has a height past usize"))
        })?;
        for (name, value) in [("height", height), ("width", width)] {
            crate::limits::within(
                value,
                &format!("the {name} of embedded image {key:?}"),
                crate::limits::MAX_IMAGE_EXTENT,
            )?;
        }
        let pixels = crate::limits::checked_product(
            &[CAMERA_CHANNELS, height, width],
            &format!("the decoded size of embedded image {key:?}"),
        )?;
        crate::limits::within(
            pixels,
            &format!("the decoded size of embedded image {key:?}"),
            crate::limits::MAX_FEATURE_WIDTH,
        )?;
        if (CAMERA_CHANNELS, height, width) != declared_shape {
            let (declared_channels, declared_height, declared_width) = declared_shape;
            return Err(TrainError::Metadata(format!(
                "embedded image {key:?} decodes as [{CAMERA_CHANNELS}, {height}, {width}] but \
                 info.json declares [{declared_channels}, {declared_height}, {declared_width}]; \
                 the frame is not resized to fit a declaration it contradicts"
            )));
        }

        // The decoder's own budget, on top of the checks above, because a malformed
        // header can declare one size and the stream then demand another.
        let mut limits = ::image::Limits::no_limits();
        let extent = u32::try_from(crate::limits::MAX_IMAGE_EXTENT).unwrap_or(u32::MAX);
        limits.max_image_width = Some(extent);
        limits.max_image_height = Some(extent);
        limits.max_alloc = Some(crate::limits::MAX_EMBEDDED_IMAGE_BYTES as u64);
        decoder.set_limits(limits).map_err(|error| {
            TrainError::Metadata(format!(
                "embedded image {key:?} is past the decoder's allocation budget: {error}"
            ))
        })?;
        let decoded = ::image::DynamicImage::from_decoder(decoder).map_err(|error| {
            TrainError::Metadata(format!(
                "embedded image {key:?} could not be decoded as {}: {error}",
                SUPPORTED_IMAGE_FORMATS.join(" or ")
            ))
        })?;

        // `into_rgb8` is `PIL.Image.convert("RGB")`: a greyscale or palette frame
        // becomes three channels, and a 16-bit one is narrowed to 8. The declared
        // shape is three-channel either way, so the conversion is what makes the
        // dataset's own declaration true rather than an assumption about the file.
        let rgb = decoded.into_rgb8();
        let interleaved = rgb.as_raw();
        let plane = height * width;
        let mut chw = vec![0.0f32; pixels];
        for channel in 0..CAMERA_CHANNELS {
            for offset in 0..plane {
                chw[channel * plane + offset] =
                    f32::from(interleaved[offset * CAMERA_CHANNELS + channel]) / 255.0;
            }
        }
        Ok(Self {
            path,
            channels: CAMERA_CHANNELS,
            height,
            width,
            pixels: chw,
        })
    }

    /// The `[channels, height, width]` tensor of this frame, on `device`.
    ///
    /// One frame, not a batch: [`crate::data::batch::collate`] stacks these into the
    /// `[batch, channels, height, width]` tensor [`crate::data::batch::Batch::with_images`]
    /// takes.
    pub fn tensor(&self, device: &Device) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            self.pixels.clone(),
            (self.channels, self.height, self.width),
            device,
        )?)
    }
}

/// Channels an embedded camera frame decodes to.
///
/// Three, because torchvision's ResNet stem convolves three input planes and nothing
/// in this slice reshapes a frame to fit a different count.
pub const CAMERA_CHANNELS: usize = 3;

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
