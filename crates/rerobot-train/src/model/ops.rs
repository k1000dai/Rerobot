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
