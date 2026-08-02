//! The image backbone `ACT.__init__` builds when the policy has camera features:
//! a torchvision ResNet wrapped in `IntermediateLayerGetter` so that only
//! `layer4`'s feature map comes out.
//!
//! ```text
//! backbone_model = getattr(torchvision.models, config.vision_backbone)(
//!     replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
//!     weights=config.pretrained_backbone_weights,
//!     norm_layer=FrozenBatchNorm2d,
//! )
//! self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
//! ```
//!
//! What this module reproduces, and what it refuses, is deliberate on both sides.
//!
//! # Reproduced
//!
//! * The **architecture** of the `BasicBlock` ResNets — `resnet18` and `resnet34` —
//!   including the 7×7 stride-2 stem, the 3×3 stride-2 max-pool, the four stages and
//!   their 1×1 projection shortcuts. `layer4`'s output has 512 channels, which is
//!   the `backbone_model.fc.in_features` that sizes `encoder_img_feat_input_proj`.
//! * The **parameter names**, `model.backbone.` prefix included, exactly as
//!   `IntermediateLayerGetter` preserves them: `conv1.weight`, `bn1.running_var`,
//!   `layer2.0.downsample.0.weight`, and the rest. A checkpoint written here names
//!   its backbone tensors what upstream's names them.
//! * **`FrozenBatchNorm2d`**, so the normalization statistics are buffers rather
//!   than parameters. This is what makes `get_optim_params`' backbone group hold
//!   convolution weights and nothing else.
//! * The **initialization distributions**: `kaiming_normal_(mode="fan_out",
//!   nonlinearity="relu")` for every convolution, and ones/zeros/zeros/ones for
//!   every frozen normalization layer. The *stream* is Rerobot's `SplitMix64`, not
//!   torch's, exactly as it is for the transformer — see
//!   [`crate::model::params`].
//!
//! # Refused
//!
//! * **Pretrained weights.** `pretrained_backbone_weights` names a torchvision
//!   checkpoint downloaded from `download.pytorch.org`. Nothing in this repository
//!   ships or fetches one, so a config asking for them is refused by
//!   [`crate::model::act::ActModel::new`] rather than silently trained from a random
//!   initialization while `config.json` claims ImageNet weights. Random
//!   initialization is the one supported mode, and it has to be requested by setting
//!   `pretrained_backbone_weights` to null.
//! * **The `Bottleneck` ResNets** — `resnet50` and larger. They are a different
//!   block with a 2048-channel output, and building a `BasicBlock` stack under one
//!   of their names would produce a model that is not what was asked for.
//! * **`replace_final_stride_with_dilation`.** Upstream cannot honour it on a
//!   `BasicBlock` ResNet either: torchvision's `BasicBlock.__init__` raises
//!   `NotImplementedError("Dilation > 1 not supported in BasicBlock")`, and
//!   `_make_layer` hands `dilation=2` to every block of `layer4` after the first.
//!   The refusal here quotes that message.

use crate::error::{Result, TrainError};
use crate::model::ops::{max_pool2d, Conv2d, FrozenBatchNorm2d};
use crate::model::params::{Initializer, ParameterStore};
use candle_core::Tensor;

/// Channels the stem produces, which is also `layer1`'s width.
const STEM_CHANNELS: usize = 64;

/// The widths of the four stages, before the block expansion.
const STAGE_CHANNELS: [usize; 4] = [64, 128, 256, 512];

/// How many `BasicBlock`s each stage of a named ResNet holds.
///
/// # Errors
///
/// [`TrainError::Unsupported`] for the `Bottleneck` variants and for any other
/// name. `ActConfig::validate` has already refused everything that does not begin
/// `resnet`, so the remaining cases are the ResNets this port does not build.
pub fn stage_blocks(name: &str) -> Result<[usize; 4]> {
    match name {
        "resnet18" => Ok([2, 2, 2, 2]),
        "resnet34" => Ok([3, 4, 6, 3]),
        "resnet50" | "resnet101" | "resnet152" => Err(TrainError::unsupported(format!(
            "vision_backbone = {name:?} is a Bottleneck ResNet, whose blocks are a different \
             shape and whose layer4 emits 2048 channels rather than 512; only the BasicBlock \
             variants resnet18 and resnet34 are ported"
        ))),
        other => Err(TrainError::unsupported(format!(
            "vision_backbone = {other:?} is not a ResNet this port builds; the ported variants \
             are resnet18 and resnet34"
        ))),
    }
}

/// The channel count `layer4` emits, i.e. `backbone_model.fc.in_features`.
///
/// Constant across the `BasicBlock` variants because their block expansion is 1.
pub const FEATURE_CHANNELS: usize = 512;

/// One `torchvision.models.resnet.BasicBlock`.
#[derive(Debug)]
struct BasicBlock {
    conv1: Conv2d,
    bn1: FrozenBatchNorm2d,
    conv2: Conv2d,
    bn2: FrozenBatchNorm2d,
    /// `downsample`, an `nn.Sequential(conv1x1, norm_layer)` present only when the
    /// block changes the shape of its input.
    downsample: Option<(Conv2d, FrozenBatchNorm2d)>,
}

impl BasicBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.bn1.forward(&self.conv1.forward(input)?)?.relu()?;
        let hidden = self.bn2.forward(&self.conv2.forward(&hidden)?)?;
        let identity = match &self.downsample {
            Some((convolution, norm)) => norm.forward(&convolution.forward(input)?)?,
            None => input.clone(),
        };
        Ok((hidden + identity)?.relu()?)
    }
}

/// A `BasicBlock` ResNet truncated at `layer4`, which is what
/// `IntermediateLayerGetter(..., {"layer4": "feature_map"})` leaves.
#[derive(Debug)]
pub struct ResNetBackbone {
    conv1: Conv2d,
    bn1: FrozenBatchNorm2d,
    stages: Vec<Vec<BasicBlock>>,
}

impl ResNetBackbone {
    /// Build the named ResNet, registering every tensor under `prefix`.
    ///
    /// `prefix` is `model.backbone` for an ACT policy; it is a parameter so that the
    /// naming stays with the caller that owns the `state_dict` contract.
    pub fn new(
        name: &str,
        in_channels: usize,
        store: &mut ParameterStore,
        init: &mut Initializer<'_>,
        prefix: &str,
    ) -> Result<Self> {
        let blocks = stage_blocks(name)?;

        let conv1 = convolution(
            store,
            init,
            &format!("{prefix}.conv1"),
            STEM_CHANNELS,
            in_channels,
            7,
            2,
            3,
        )?;
        let bn1 = frozen_batch_norm(store, init, &format!("{prefix}.bn1"), STEM_CHANNELS)?;

        let mut stages = Vec::with_capacity(STAGE_CHANNELS.len());
        let mut inplanes = STEM_CHANNELS;
        for (stage, (planes, count)) in STAGE_CHANNELS.iter().zip(blocks).enumerate() {
            // `_make_layer(block, planes, blocks, stride=1)` for `layer1` and
            // `stride=2` for the rest.
            let stride = if stage == 0 { 1 } else { 2 };
            let mut layer = Vec::with_capacity(count);
            for index in 0..count {
                let block_stride = if index == 0 { stride } else { 1 };
                let block_prefix = format!("{prefix}.layer{}.{index}", stage + 1);
                layer.push(basic_block(
                    store,
                    init,
                    &block_prefix,
                    inplanes,
                    *planes,
                    block_stride,
                )?);
                inplanes = *planes;
            }
            stages.push(layer);
        }
        Ok(Self { conv1, bn1, stages })
    }

    /// `backbone(image)["feature_map"]`.
    ///
    /// Takes `[batch, channels, height, width]` and returns
    /// `[batch, 512, height / 32, width / 32]`, rounded the way the strides round.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.bn1.forward(&self.conv1.forward(input)?)?.relu()?;
        let mut hidden = max_pool2d(&hidden, 3, 2, 1)?;
        for stage in &self.stages {
            for block in stage {
                hidden = block.forward(&hidden)?;
            }
        }
        Ok(hidden)
    }
}

fn basic_block(
    store: &mut ParameterStore,
    init: &mut Initializer<'_>,
    prefix: &str,
    inplanes: usize,
    planes: usize,
    stride: usize,
) -> Result<BasicBlock> {
    // torchvision registers the block's own children first and the downsample
    // branch last, and `named_parameters()` follows that order.
    let conv1 = convolution(
        store,
        init,
        &format!("{prefix}.conv1"),
        planes,
        inplanes,
        3,
        stride,
        1,
    )?;
    let bn1 = frozen_batch_norm(store, init, &format!("{prefix}.bn1"), planes)?;
    let conv2 = convolution(
        store,
        init,
        &format!("{prefix}.conv2"),
        planes,
        planes,
        3,
        1,
        1,
    )?;
    let bn2 = frozen_batch_norm(store, init, &format!("{prefix}.bn2"), planes)?;
    // `if stride != 1 or self.inplanes != planes * block.expansion`, with an
    // expansion of 1.
    let downsample = if stride != 1 || inplanes != planes {
        let convolution = convolution(
            store,
            init,
            &format!("{prefix}.downsample.0"),
            planes,
            inplanes,
            1,
            stride,
            0,
        )?;
        let norm = frozen_batch_norm(store, init, &format!("{prefix}.downsample.1"), planes)?;
        Some((convolution, norm))
    } else {
        None
    };
    Ok(BasicBlock {
        conv1,
        bn1,
        conv2,
        bn2,
        downsample,
    })
}

/// A ResNet convolution: no bias, and `kaiming_normal_(mode="fan_out")`.
#[allow(clippy::too_many_arguments)]
fn convolution(
    store: &mut ParameterStore,
    init: &mut Initializer<'_>,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<Conv2d> {
    // `fan_out = out_channels * kernel_h * kernel_w`, and the relu gain is sqrt(2),
    // so the standard deviation is `sqrt(2 / fan_out)`.
    let fan_out = crate::limits::checked_product(
        &[out_channels, kernel, kernel],
        &format!("the fan-out of {prefix:?}"),
    )?;
    let standard_deviation = (2.0 / fan_out as f64).sqrt();
    let weight = init.normal(
        &[out_channels, in_channels, kernel, kernel],
        standard_deviation,
    )?;
    Ok(Conv2d {
        weight: store.parameter(format!("{prefix}.weight"), weight)?,
        // `bias=False` on every convolution torchvision's ResNet builds.
        bias: None,
        stride,
        padding,
    })
}

/// A frozen normalization layer, whose four tensors are buffers rather than
/// parameters and are therefore never updated by the optimizer.
fn frozen_batch_norm(
    store: &mut ParameterStore,
    init: &mut Initializer<'_>,
    prefix: &str,
    channels: usize,
) -> Result<FrozenBatchNorm2d> {
    let weight = init.ones(&[channels])?;
    let bias = init.zeros(&[channels])?;
    let running_mean = init.zeros(&[channels])?;
    let running_var = init.ones(&[channels])?;
    Ok(FrozenBatchNorm2d {
        weight: store.buffer(format!("{prefix}.weight"), weight)?,
        bias: store.buffer(format!("{prefix}.bias"), bias)?,
        running_mean: store.buffer(format!("{prefix}.running_mean"), running_mean)?,
        running_var: store.buffer(format!("{prefix}.running_var"), running_var)?,
    })
}
