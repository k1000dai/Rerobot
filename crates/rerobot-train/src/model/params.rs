//! The named parameter store, its initialization, and its safetensors form.
//!
//! Names are upstream's `state_dict` keys, `model.` prefix included, so that
//! `model.safetensors` written here is the file `ACTPolicy.save_pretrained` would
//! have written. That is the whole reason this module exists rather than using
//! candle's `VarBuilder` naming.
//!
//! # What the initializer does and does not promise
//!
//! The *distributions* are torch's:
//!
//! * `nn.Linear` — weight and bias both `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`,
//!   which is what `kaiming_uniform_(a=sqrt(5))` reduces to;
//! * `nn.Embedding` — `N(0, 1)`;
//! * `nn.LayerNorm` — weight 1, bias 0;
//! * then ACT's `_reset_parameters`, which overwrites every parameter of the
//!   transformer encoder and decoder with `dim() > 1` using
//!   `xavier_uniform_`: `U(-a, a)` with `a = sqrt(6 / (fan_in + fan_out))`.
//!
//! The *stream* is not torch's. Rerobot draws from
//! [`rerobot_core::random::SplitMix64`], so a run seeded 1000 here and a run
//! seeded 1000 upstream start from different weights. This is why the golden
//! comparison against PyTorch loads explicitly exported weights instead of
//! initializing both sides and comparing.

use crate::error::{Result, TrainError};
use candle_core::{DType, Device, Tensor, Var};
use rerobot_core::random::SplitMix64;
use std::collections::BTreeMap;
use std::path::Path;

/// A trainable parameter and the `state_dict` key it is saved under.
#[derive(Debug, Clone)]
pub struct NamedParameter {
    /// `state_dict` key, e.g. `model.encoder.layers.0.self_attn.in_proj_weight`.
    pub name: String,
    /// The variable itself.
    pub value: Var,
}

/// Every parameter and buffer of a model, in insertion order.
#[derive(Debug, Default)]
pub struct ParameterStore {
    parameters: Vec<NamedParameter>,
    buffers: Vec<(String, Tensor)>,
    total_bytes: usize,
}

impl ParameterStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a trainable parameter.
    pub fn parameter(&mut self, name: impl Into<String>, value: Tensor) -> Result<Tensor> {
        let name = name.into();
        self.account(&name, &value)?;
        let variable = Var::from_tensor(&value)?;
        let handle = variable.as_tensor().clone();
        self.parameters.push(NamedParameter {
            name,
            value: variable,
        });
        Ok(handle)
    }

    /// Register a non-trainable buffer, e.g. ACT's `vae_encoder_pos_enc`.
    pub fn buffer(&mut self, name: impl Into<String>, value: Tensor) -> Result<Tensor> {
        let name = name.into();
        self.account(&name, &value)?;
        self.buffers.push((name, value.clone()));
        Ok(value)
    }

    /// Add a tensor to the running byte total, refusing it if the model would exceed
    /// its budget.
    ///
    /// Per-field limits do not bound a model: the permitted layer counts multiply the
    /// largest permitted tensor 128 times over for the encoder and again for the
    /// decoder, which is how a configuration whose every field was legal reached
    /// roughly a tebibyte of feed-forward weights. The total is accumulated here, as
    /// each tensor is registered, so the refusal happens on the first tensor that
    /// crosses the line rather than after the whole model has been allocated.
    fn account(&mut self, name: &str, value: &Tensor) -> Result<()> {
        let bytes = crate::limits::checked_mul(
            value.elem_count(),
            value.dtype().size_in_bytes(),
            &format!("the size of {name:?} in bytes"),
        )?;
        crate::limits::within(
            bytes,
            &format!("the size of tensor {name:?} in bytes"),
            crate::limits::MAX_TENSOR_BYTES,
        )?;
        self.total_bytes =
            crate::limits::checked_add(self.total_bytes, bytes, "the model's total size in bytes")?;
        crate::limits::within(
            self.total_bytes,
            "the model's total size in bytes",
            crate::limits::MAX_MODEL_BYTES,
        )?;
        Ok(())
    }

    /// Trainable parameters, in registration order.
    pub fn parameters(&self) -> &[NamedParameter] {
        &self.parameters
    }

    /// Bytes every registered parameter and buffer occupies in total.
    ///
    /// Checked as the store grows, in [`Self::parameter`] and [`Self::buffer`], so a
    /// configuration whose tensors are individually legal but jointly enormous is
    /// refused while it is being built rather than after it has been allocated.
    pub fn bytes(&self) -> usize {
        self.total_bytes
    }

    /// How many scalars the trainable parameters hold in total.
    pub fn numel(&self) -> usize {
        self.parameters
            .iter()
            .map(|parameter| parameter.value.elem_count())
            .sum()
    }

    /// The full `state_dict`: parameters and buffers, sorted by key.
    ///
    /// Sorted because safetensors is a map and `safetensors.torch.save_file`
    /// writes whatever order it is handed; a deterministic order makes two
    /// checkpoints of the same model byte-comparable.
    ///
    /// Every tensor is a **detached deep copy**, and that is load-bearing rather
    /// than defensive. `Var::set` writes through to the variable's existing
    /// storage, so a map of `Var::as_tensor().clone()` handles would alias the
    /// live parameters: a caller that snapshotted the weights, took an optimizer
    /// step and then compared would find no difference, because the "before"
    /// snapshot had been mutated underneath it. Copying is what makes
    /// before/after comparison — and therefore the "did this step train
    /// anything?" question — answerable at all.
    pub fn state_dict(&self) -> Result<BTreeMap<String, Tensor>> {
        let mut out = BTreeMap::new();
        for parameter in &self.parameters {
            out.insert(
                parameter.name.clone(),
                parameter.value.as_tensor().copy()?.detach(),
            );
        }
        for (name, tensor) in &self.buffers {
            out.insert(name.clone(), tensor.copy()?.detach());
        }
        Ok(out)
    }

    /// Write the `state_dict` to `path` as safetensors.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| TrainError::io(parent, &error))?;
        }
        let tensors: std::collections::HashMap<String, Tensor> = self
            .state_dict()?
            .into_iter()
            .map(|(name, tensor)| Ok((name, tensor.contiguous()?)))
            .collect::<Result<_>>()?;
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    /// Overwrite every parameter and buffer from a safetensors file.
    ///
    /// Every key the model expects must be present with the exact shape it
    /// expects, and the file must carry no extra keys: a checkpoint that does not
    /// describe *this* architecture is refused rather than partially loaded.
    pub fn load(&mut self, path: &Path, device: &Device) -> Result<()> {
        let loaded = candle_core::safetensors::load(path, device)?;
        let expected = self.state_dict()?;
        for (name, tensor) in &expected {
            let found = loaded
                .get(name)
                .ok_or_else(|| TrainError::checkpoint(path, format!("no tensor named {name:?}")))?;
            if found.dims() != tensor.dims() {
                return Err(TrainError::checkpoint(
                    path,
                    format!(
                        "tensor {name:?} has shape {:?} but the model expects {:?}",
                        found.dims(),
                        tensor.dims()
                    ),
                ));
            }
            // The dtype is checked rather than coerced. `to_dtype(F32)` accepted an
            // `f64` checkpoint with a quiet precision change and an integer one as a
            // lattice of whole numbers; neither is the model that was saved, and
            // neither said so. Upstream writes `f32`, so anything else is a file this
            // reader should not pretend to understand.
            if found.dtype() != DType::F32 {
                return Err(TrainError::checkpoint(
                    path,
                    format!(
                        "tensor {name:?} has dtype {:?} but the model is f32; a checkpoint is \
                         not converted on load",
                        found.dtype()
                    ),
                ));
            }
            // And the *values*, not only the shape and the dtype. A NaN or infinite
            // weight loads into a model that produces NaN for every input, and the
            // run's own non-finite tripwire then fires on the first step -- so the
            // failure surfaces as a training divergence, far from the corrupt file
            // that caused it. This is the last point at which the real reason can be
            // reported.
            require_finite(path, name, found)?;
        }
        let unexpected: Vec<&String> = loaded
            .keys()
            .filter(|name| !expected.contains_key(*name))
            .collect();
        if !unexpected.is_empty() {
            let mut names: Vec<&str> = unexpected.iter().map(|name| name.as_str()).collect();
            names.sort_unstable();
            return Err(TrainError::checkpoint(
                path,
                format!(
                    "holds tensors this model does not have: {}",
                    names.join(", ")
                ),
            ));
        }
        for parameter in &self.parameters {
            // Already known to be `f32` of the right shape by the loop above.
            let value = loaded.get(&parameter.name).expect("checked above");
            parameter.value.set(value)?;
        }
        // Buffers are recomputed by the constructor and are a pure function of the
        // config, so a checkpoint's copy is verified above and then ignored: if it
        // disagreed in value the config would have to disagree too, and the config
        // is loaded separately.
        Ok(())
    }
}

/// The element count of `shape`, checked and budgeted.
///
/// `shape.iter().product()` was unchecked here. Both the shape and the count come from
/// the policy config, so an overflowing product panics in a checked build and *wraps*
/// in release — and a wrapped count is the worse case: it allocates a small vector
/// which is then handed to `Tensor::from_vec` with the original enormous shape.
///
/// The per-tensor byte budget is here rather than only at the config layer because
/// this is the function that actually allocates.
fn element_count(shape: &[usize]) -> Result<usize> {
    let count = crate::limits::checked_product(shape, "a parameter shape")?;
    let bytes = crate::limits::checked_mul(
        count,
        std::mem::size_of::<f32>(),
        "a parameter's size in bytes",
    )?;
    crate::limits::within(
        bytes,
        "a single parameter tensor's size in bytes",
        crate::limits::MAX_TENSOR_BYTES,
    )?;
    Ok(count)
}

/// Refuse a tensor holding a value training cannot proceed from.
///
/// Every element, not a sample: one NaN weight makes the whole forward pass NaN.
fn require_finite(path: &Path, name: &str, tensor: &Tensor) -> Result<()> {
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(TrainError::checkpoint(
            path,
            format!(
                "tensor {name:?} element {index} is {value}, which is not finite; a model \
                 holding it cannot produce a finite prediction"
            ),
        ));
    }
    Ok(())
}

/// Draws torch's default initializations from Rerobot's own stream.
pub struct Initializer<'a> {
    rng: &'a mut SplitMix64,
    device: Device,
}

impl<'a> Initializer<'a> {
    /// An initializer drawing from `rng` onto `device`.
    pub fn new(rng: &'a mut SplitMix64, device: Device) -> Self {
        Self { rng, device }
    }

    /// The device parameters are created on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `U(-bound, bound)` of the given shape.
    pub fn uniform(&mut self, shape: &[usize], bound: f64) -> Result<Tensor> {
        let count = element_count(shape)?;
        let values: Vec<f32> = (0..count)
            .map(|_| ((self.rng.next_f64() * 2.0 - 1.0) * bound) as f32)
            .collect();
        Ok(Tensor::from_vec(values, shape.to_vec(), &self.device)?)
    }

    /// `N(0, 1)` of the given shape.
    pub fn standard_normal(&mut self, shape: &[usize]) -> Result<Tensor> {
        let count = element_count(shape)?;
        let values: Vec<f32> = (0..count)
            .map(|_| self.rng.standard_normal() as f32)
            .collect();
        Ok(Tensor::from_vec(values, shape.to_vec(), &self.device)?)
    }

    /// `nn.Linear.reset_parameters`, both tensors.
    pub fn linear(&mut self, out_features: usize, in_features: usize) -> Result<(Tensor, Tensor)> {
        let bound = if in_features == 0 {
            0.0
        } else {
            1.0 / (in_features as f64).sqrt()
        };
        let weight = self.uniform(&[out_features, in_features], bound)?;
        let bias = self.uniform(&[out_features], bound)?;
        Ok((weight, bias))
    }

    /// `nn.init.xavier_uniform_` for a rank-2 tensor.
    pub fn xavier_uniform(&mut self, out_features: usize, in_features: usize) -> Result<Tensor> {
        let bound = (6.0 / (in_features as f64 + out_features as f64)).sqrt();
        self.uniform(&[out_features, in_features], bound)
    }

    /// A zero tensor.
    pub fn zeros(&self, shape: &[usize]) -> Result<Tensor> {
        Ok(Tensor::zeros(shape, DType::F32, &self.device)?)
    }

    /// A one tensor.
    pub fn ones(&self, shape: &[usize]) -> Result<Tensor> {
        Ok(Tensor::ones(shape, DType::F32, &self.device)?)
    }
}
