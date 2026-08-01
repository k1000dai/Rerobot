//! The PyTorch operators the ACT model needs, reimplemented on candle with
//! upstream's parameter layout and upstream's defaults.
//!
//! Each of these exists because the layout, not just the arithmetic, has to match:
//! a checkpoint written here has to carry the same tensor names and shapes
//! `torch.nn` would have written, or it is not an ACT checkpoint.

use crate::error::{Result, TrainError};
use candle_core::{DType, Tensor};
use rerobot_core::random::SplitMix64;

/// `torch.nn.LayerNorm`'s default epsilon.
pub const LAYER_NORM_EPS: f64 = 1e-5;

/// `torchvision.ops.misc.FrozenBatchNorm2d`'s default epsilon.
pub const FROZEN_BATCH_NORM_EPS: f64 = 1e-5;

/// `torch.nn.Linear`: `y = x @ weight.T + bias`.
#[derive(Debug, Clone)]
pub struct Linear {
    /// `weight`, shaped `[out_features, in_features]` as `torch.nn.Linear` stores it.
    pub weight: Tensor,
    /// `bias`, shaped `[out_features]`.
    pub bias: Tensor,
}

impl Linear {
    /// Apply to a tensor whose last axis is `in_features`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight_t = self.weight.t()?;
        let output = match input.rank() {
            2 => input.matmul(&weight_t)?,
            // candle's `matmul` needs matching batch ranks, so a rank-3 input is
            // flattened rather than broadcast. Flatten/reshape is a view here.
            3 => {
                let (batch, sequence, features) = input.dims3()?;
                input
                    .reshape((batch * sequence, features))?
                    .matmul(&weight_t)?
                    .reshape((batch, sequence, self.out_features()))?
            }
            other => {
                return Err(TrainError::Tensor(format!(
                    "Linear expects a rank-2 or rank-3 input, got rank {other}"
                )))
            }
        };
        Ok(output.broadcast_add(&self.bias)?)
    }

    /// `out_features`.
    pub fn out_features(&self) -> usize {
        self.weight.dims()[0]
    }

    /// `in_features`.
    pub fn in_features(&self) -> usize {
        self.weight.dims()[1]
    }
}

/// `torch.nn.LayerNorm` over the last axis, with learned scale and shift.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// `weight`, shaped `[normalized_shape]`.
    pub weight: Tensor,
    /// `bias`, shaped `[normalized_shape]`.
    pub bias: Tensor,
}

impl LayerNorm {
    /// Apply to the last axis.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Written out rather than delegated so the epsilon placement is visible:
        // torch divides by `sqrt(var + eps)` with the *biased* variance.
        let mean = input.mean_keepdim(input.rank() - 1)?;
        let centred = input.broadcast_sub(&mean)?;
        let variance = centred.sqr()?.mean_keepdim(input.rank() - 1)?;
        let normalized = centred.broadcast_div(&(variance + LAYER_NORM_EPS)?.sqrt()?)?;
        Ok(normalized
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?)
    }
}

/// `torch.nn.MultiheadAttention` with `batch_first=False` upstream, evaluated here
/// on `[batch, sequence, dim]` tensors.
///
/// The transposition is safe: torch's implementation permutes to batch-first
/// internally, so the arithmetic on a transposed input is the same. What is *not*
/// negotiable is the parameter layout, which is why the projections are one packed
/// `in_proj_weight` of shape `[3 * dim, dim]` sliced into query, key and value
/// blocks in that order, exactly as torch packs them.
#[derive(Debug, Clone)]
pub struct MultiheadAttention {
    /// `in_proj_weight`, shaped `[3 * embed_dim, embed_dim]`.
    pub in_proj_weight: Tensor,
    /// `in_proj_bias`, shaped `[3 * embed_dim]`.
    pub in_proj_bias: Tensor,
    /// `out_proj`.
    pub out_proj: Linear,
    /// `num_heads`.
    pub num_heads: usize,
}

impl MultiheadAttention {
    /// Attention over `[batch, sequence, dim]` inputs.
    ///
    /// `key_padding_mask` is `[batch, key_sequence]` with `1` marking a position
    /// to ignore, matching `torch.BoolTensor`'s `True`.
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        key_padding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, query_length, embed_dim) = query.dims3()?;
        let key_length = key.dims3()?.1;
        if embed_dim % self.num_heads != 0 {
            return Err(TrainError::Tensor(format!(
                "embed_dim {embed_dim} is not divisible by num_heads {}",
                self.num_heads
            )));
        }
        let head_dim = embed_dim / self.num_heads;

        let projected_query = self.project(query, 0, embed_dim)?;
        let projected_key = self.project(key, 1, embed_dim)?;
        let projected_value = self.project(value, 2, embed_dim)?;

        // [batch, heads, sequence, head_dim]
        let split = |tensor: &Tensor, length: usize| -> Result<Tensor> {
            Ok(tensor
                .reshape((batch, length, self.num_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = split(&projected_query, query_length)?;
        let k = split(&projected_key, key_length)?;
        let v = split(&projected_value, key_length)?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = key_padding_mask {
            // torch fills masked logits with -inf before the softmax. Doing that
            // literally produces NaN for a row that is entirely masked; torch has
            // the same behaviour, and no such row can occur here because the cls
            // and state tokens are never padded. A large finite negative would
            // hide that, so -inf it is.
            // candle's `where_cond` takes its condition as an integer dtype, so the
            // mask stays `u8` here rather than being cast to the score dtype.
            let mask = mask
                .to_dtype(DType::U8)?
                .reshape((batch, 1, 1, key_length))?
                .broadcast_as(scores.shape())?
                .contiguous()?;
            let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?;
            scores = mask.where_cond(&neg_inf, &scores)?;
        }

        // `candle_nn::ops::softmax` and not `softmax_last_dim`. The latter is a
        // fused custom op whose backward pass does not propagate into its input, so
        // using it would leave every attention *logit* without a gradient: the
        // position embeddings would never train and each projection would learn
        // only through its value path. The composed version is differentiable.
        // `tests/model.rs::attention_logits_receive_gradients` pins this.
        let weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        let attended =
            weights
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((batch, query_length, embed_dim))?;
        self.out_proj.forward(&attended)
    }

    /// One of the three packed projections.
    fn project(&self, input: &Tensor, block: usize, embed_dim: usize) -> Result<Tensor> {
        let weight = self
            .in_proj_weight
            .narrow(0, block * embed_dim, embed_dim)?
            .contiguous()?;
        let bias = self.in_proj_bias.narrow(0, block * embed_dim, embed_dim)?;
        Linear { weight, bias }.forward(input)
    }
}

/// `create_sinusoidal_pos_embedding`: the fixed 1-D table the VAE encoder uses.
///
/// Computed in `f64` and cast down, because upstream builds it in NumPy (`float64`)
/// and only then calls `.float()`. Doing it in `f32` throughout would change the
/// low bits of every entry.
pub fn sinusoidal_position_embedding(
    num_positions: usize,
    dimension: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut table = Vec::with_capacity(num_positions * dimension);
    for position in 0..num_positions {
        for index in 0..dimension {
            let exponent = 2 * (index / 2);
            let angle = position as f64 / 10_000f64.powf(exponent as f64 / dimension as f64);
            table.push(if index % 2 == 0 {
                angle.sin() as f32
            } else {
                angle.cos() as f32
            });
        }
    }
    Ok(Tensor::from_vec(table, (num_positions, dimension), device)?)
}

/// `torch.nn.Conv2d` with `groups = 1` and `dilation = 1`.
///
/// Only that case exists in ACT: the ResNet backbone's convolutions and the 1×1
/// `encoder_img_feat_input_proj` are all ungrouped and undilated. Dilation is not
/// offered here because `replace_final_stride_with_dilation` cannot be honoured on
/// a `BasicBlock` ResNet at all — see [`crate::model::backbone`].
#[derive(Debug, Clone)]
pub struct Conv2d {
    /// `weight`, shaped `[out_channels, in_channels, kernel_h, kernel_w]`.
    pub weight: Tensor,
    /// `bias`, shaped `[out_channels]`. Absent for every ResNet convolution, which
    /// torchvision builds with `bias=False` because a normalization layer follows.
    pub bias: Option<Tensor>,
    /// `stride`, the same in both spatial axes as torchvision uses it.
    pub stride: usize,
    /// `padding`, the same in both spatial axes.
    pub padding: usize,
}

impl Conv2d {
    /// Apply to a `[batch, in_channels, height, width]` tensor.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = input.conv2d(&self.weight, self.padding, self.stride, 1, 1)?;
        match &self.bias {
            // `[out_channels]` reshaped to `[1, out_channels, 1, 1]`, which is the
            // axis torch adds it on.
            Some(bias) => {
                let channels = bias.dims1()?;
                Ok(output.broadcast_add(&bias.reshape((1, channels, 1, 1))?)?)
            }
            None => Ok(output),
        }
    }

    /// `out_channels`.
    pub fn out_channels(&self) -> usize {
        self.weight.dims()[0]
    }

    /// `in_channels`.
    pub fn in_channels(&self) -> usize {
        self.weight.dims()[1]
    }
}

/// `torchvision.ops.misc.FrozenBatchNorm2d`: batch normalization with the four
/// statistics held as *buffers* and never updated.
///
/// Upstream builds every ResNet normalization layer as this, which is why the
/// backbone has no trainable normalization parameters and why a backbone gradient
/// flows only through the convolutions.
#[derive(Debug, Clone)]
pub struct FrozenBatchNorm2d {
    /// `weight`, shaped `[num_features]`.
    pub weight: Tensor,
    /// `bias`, shaped `[num_features]`.
    pub bias: Tensor,
    /// `running_mean`, shaped `[num_features]`.
    pub running_mean: Tensor,
    /// `running_var`, shaped `[num_features]`.
    pub running_var: Tensor,
}

impl FrozenBatchNorm2d {
    /// Apply to a `[batch, num_features, height, width]` tensor.
    ///
    /// Written in torchvision's own order — `scale = weight * rsqrt(var + eps)` and
    /// `shift = bias - mean * scale` — because the epsilon sits inside the square
    /// root there, and folding it in afterwards would move the low bits.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let channels = self.weight.dims1()?;
        let shape = (1, channels, 1, 1);
        let scale = (&self.weight / (&self.running_var + FROZEN_BATCH_NORM_EPS)?.sqrt()?)?;
        let shift = (&self.bias - (&self.running_mean * &scale)?)?;
        Ok(input
            .broadcast_mul(&scale.reshape(shape)?)?
            .broadcast_add(&shift.reshape(shape)?)?)
    }
}

/// `torch.nn.MaxPool2d` with a square kernel, stride and zero-padding.
///
/// Written out of `index_select` and `maximum` rather than delegated to candle's
/// fused `max_pool2d`, for two reasons that are both about the *backward* pass:
/// candle refuses to differentiate a max-pool whose kernel differs from its stride,
/// and ResNet's stem pools 3×3 with stride 2, and its fused op takes no padding at
/// all. Both facts are structural, not incidental, so the operator is composed from
/// differentiable primitives here instead.
///
/// The padding is `-inf`, which is what torch pools with, so a padded cell can never
/// win a window. One divergence is worth naming: when several cells in a window hold
/// the identical maximum, torch routes the whole gradient to the first of them and
/// candle's `maximum` splits it evenly. The forward values are the same either way.
pub fn max_pool2d(input: &Tensor, kernel: usize, stride: usize, padding: usize) -> Result<Tensor> {
    let (_, _, height, width) = input.dims4()?;
    if kernel == 0 || stride == 0 {
        return Err(TrainError::Tensor(format!(
            "max_pool2d needs a positive kernel and stride, got {kernel} and {stride}"
        )));
    }
    let padded_height = height + 2 * padding;
    let padded_width = width + 2 * padding;
    if padded_height < kernel || padded_width < kernel {
        return Err(TrainError::Tensor(format!(
            "a {kernel}x{kernel} max-pool does not fit a {padded_height}x{padded_width} \
             padded input; the image is too small for this backbone"
        )));
    }
    let out_height = (padded_height - kernel) / stride + 1;
    let out_width = (padded_width - kernel) / stride + 1;

    let padded = if padding == 0 {
        input.clone()
    } else {
        pad_with(
            &pad_with(input, 2, padding, f32::NEG_INFINITY)?,
            3,
            padding,
            f32::NEG_INFINITY,
        )?
    };

    let device = input.device();
    let mut pooled: Option<Tensor> = None;
    for row_offset in 0..kernel {
        let rows: Vec<u32> = (0..out_height)
            .map(|index| (index * stride + row_offset) as u32)
            .collect();
        let rows = Tensor::from_vec(rows, out_height, device)?;
        let selected_rows = padded.index_select(&rows, 2)?;
        for column_offset in 0..kernel {
            let columns: Vec<u32> = (0..out_width)
                .map(|index| (index * stride + column_offset) as u32)
                .collect();
            let columns = Tensor::from_vec(columns, out_width, device)?;
            let window = selected_rows.index_select(&columns, 3)?;
            pooled = Some(match pooled {
                None => window,
                Some(best) => best.maximum(&window)?,
            });
        }
    }
    Ok(pooled.expect("a positive kernel produces at least one window"))
}

/// `dimension` extended by `width` cells of `value` on both sides.
fn pad_with(input: &Tensor, dimension: usize, width: usize, value: f32) -> Result<Tensor> {
    let mut shape = input.dims().to_vec();
    shape[dimension] = width;
    let block = Tensor::full(value, shape, input.device())?.to_dtype(input.dtype())?;
    Ok(Tensor::cat(&[&block, input, &block], dimension)?)
}

/// `ACTSinusoidalPositionEmbedding2d`: the fixed 2-D table added to a camera feature
/// map's tokens.
///
/// `dimension` is `dim_model / 2`; the y half and the x half are concatenated, so the
/// result carries `2 * dimension` channels. Both halves interleave sine and cosine,
/// which needs `dimension` itself to be even — hence `dim_model` divisible by four,
/// checked by the caller.
///
/// Two details are upstream's rather than the textbook's, and are reproduced
/// deliberately: the position indices run `1..=height` rather than `0..height`, and
/// the normalization divides by `height + 1e-6` rather than by `height`.
///
/// Returns `[1, 2 * dimension, height, width]`, which broadcasts over the batch.
pub fn sinusoidal_position_embedding_2d(
    height: usize,
    width: usize,
    dimension: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    if dimension == 0 || dimension % 2 != 0 {
        return Err(TrainError::Tensor(format!(
            "the 2-D camera position embedding needs an even positive dimension, got {dimension}"
        )));
    }
    if height == 0 || width == 0 {
        return Err(TrainError::Tensor(
            "the 2-D camera position embedding needs a non-empty feature map".to_owned(),
        ));
    }
    const TWO_PI: f32 = 2.0 * std::f32::consts::PI;
    const EPS: f32 = 1e-6;
    const TEMPERATURE: f32 = 10_000.0;

    let inverse_frequency: Vec<f32> = (0..dimension)
        .map(|index| TEMPERATURE.powf(2.0 * (index / 2) as f32 / dimension as f32))
        .collect();

    let channels = 2 * dimension;
    let mut table = vec![0f32; channels * height * width];
    for row in 0..height {
        // `cumsum` over a tensor of ones, so the first row is 1 rather than 0.
        let y_range = (row + 1) as f32 / (height as f32 + EPS) * TWO_PI;
        for column in 0..width {
            let x_range = (column + 1) as f32 / (width as f32 + EPS) * TWO_PI;
            for index in 0..dimension {
                let y = y_range / inverse_frequency[index];
                let x = x_range / inverse_frequency[index];
                let (y, x) = if index % 2 == 0 {
                    (y.sin(), x.sin())
                } else {
                    (y.cos(), x.cos())
                };
                // `cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)`: the
                // y half occupies the first `dimension` channels.
                table[(index * height + row) * width + column] = y;
                table[((dimension + index) * height + row) * width + column] = x;
            }
        }
    }
    Ok(Tensor::from_vec(
        table,
        (1, channels, height, width),
        device,
    )?)
}

/// `get_activation_fn`: the three activations upstream accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `F.relu`.
    Relu,
    /// `F.gelu`, the exact (erf) form torch defaults to.
    Gelu,
    /// `F.glu`, which halves the last axis.
    Glu,
}

impl Activation {
    /// Resolve upstream's `feedforward_activation` string.
    ///
    /// The message is upstream's `RuntimeError` text verbatim, including its
    /// trailing full stop, because that is what a user comparing the two sees.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "relu" => Ok(Self::Relu),
            "gelu" => Ok(Self::Gelu),
            "glu" => Ok(Self::Glu),
            other => Err(TrainError::Metadata(format!(
                "activation should be relu/gelu/glu, not {other}."
            ))),
        }
    }

    /// Apply.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Self::Relu => input.relu()?,
            // candle's `gelu` is the tanh approximation and `gelu_erf` is the
            // exact one; `F.gelu` defaults to exact.
            Self::Gelu => input.gelu_erf()?,
            Self::Glu => {
                let axis = input.rank() - 1;
                let width = input.dims()[axis];
                if width % 2 != 0 {
                    return Err(TrainError::Tensor(format!(
                        "glu needs an even last axis, got {width}"
                    )));
                }
                let first = input.narrow(axis, 0, width / 2)?;
                let second = input.narrow(axis, width / 2, width / 2)?;
                (first * candle_nn::ops::sigmoid(&second)?)?
            }
        })
    }
}

/// Draw a Bernoulli keep-mask scaled by `1 / (1 - p)`, i.e. `torch.nn.Dropout` in
/// training mode.
pub fn dropout_mask(
    rng: &mut SplitMix64,
    shape: &[usize],
    probability: f64,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let count: usize = shape.iter().product();
    let scale = 1.0 / (1.0 - probability);
    let values: Vec<f32> = (0..count)
        .map(|_| {
            if rng.next_f64() < probability {
                0.0
            } else {
                scale as f32
            }
        })
        .collect();
    Ok(Tensor::from_vec(values, shape.to_vec(), device)?)
}
