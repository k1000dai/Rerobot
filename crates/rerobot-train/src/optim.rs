//! `torch.optim.AdamW` and `torch.nn.utils.clip_grad_norm_`, and the
//! `optimizer_state.safetensors` / `optimizer_param_groups.json` pair upstream
//! writes them out as.
//!
//! Both are written out rather than delegated to `candle_nn::AdamW`, for three
//! reasons that all matter to the checkpoint: the update has to be torch's
//! decoupled-weight-decay one in torch's order, the per-parameter state has to be
//! `step` / `exp_avg` / `exp_avg_sq` under those names, and the parameter groups
//! have to survive a save and reload.

use crate::error::{Result, TrainError};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor};
use indexmap::IndexMap;
use rerobot_core::dataset::json::{JsonLike, JsonObject};
use rerobot_core::policy::act::AdamWConfig;
use std::collections::{BTreeMap, HashMap};

/// The hyper-parameters of one `torch.optim.AdamW` parameter group.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupSettings {
    /// `lr`.
    pub lr: f64,
    /// `betas`.
    pub betas: [f64; 2],
    /// `eps`.
    pub eps: f64,
    /// `weight_decay`.
    pub weight_decay: f64,
}

impl GroupSettings {
    /// The settings ACT's optimizer preset produces for the main group.
    pub fn from_preset(preset: &AdamWConfig) -> Self {
        Self {
            lr: preset.lr,
            betas: preset.betas,
            eps: preset.eps,
            weight_decay: preset.weight_decay,
        }
    }

    /// The same settings with a different learning rate, which is how upstream's
    /// `get_optim_params` describes the backbone group.
    pub fn with_lr(&self, lr: f64) -> Self {
        Self { lr, ..self.clone() }
    }
}

/// One parameter group: which parameters, under which hyper-parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterGroup {
    /// Indices into the model's parameter list, in order.
    pub params: Vec<usize>,
    /// The group's hyper-parameters.
    pub settings: GroupSettings,
}

/// Per-parameter AdamW state, under torch's own names.
#[derive(Debug, Clone)]
struct MomentState {
    step: f64,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
}

/// `torch.optim.AdamW` over a fixed parameter list.
#[derive(Debug)]
pub struct AdamW {
    groups: Vec<ParameterGroup>,
    /// Keyed by parameter index, present only once that parameter has a gradient.
    state: BTreeMap<usize, MomentState>,
}

impl AdamW {
    /// An optimizer over `groups`.
    ///
    /// # Errors
    ///
    /// When a parameter index appears in two groups or in none, which torch would
    /// either double-update or silently freeze.
    pub fn new(groups: Vec<ParameterGroup>, parameter_count: usize) -> Result<Self> {
        let mut seen = vec![false; parameter_count];
        for group in &groups {
            for index in &group.params {
                let slot = seen.get_mut(*index).ok_or_else(|| {
                    TrainError::Metadata(format!(
                        "parameter group names index {index}, but the model has \
                         {parameter_count} parameters"
                    ))
                })?;
                if *slot {
                    return Err(TrainError::Metadata(format!(
                        "parameter index {index} appears in more than one group"
                    )));
                }
                *slot = true;
            }
        }
        if let Some(missing) = seen.iter().position(|covered| !covered) {
            return Err(TrainError::Metadata(format!(
                "parameter index {missing} is in no group, so it would never be updated"
            )));
        }
        Ok(Self {
            groups,
            state: BTreeMap::new(),
        })
    }

    /// The parameter groups, in order.
    pub fn groups(&self) -> &[ParameterGroup] {
        &self.groups
    }

    /// One AdamW step over the parameters that have gradients.
    ///
    /// Follows `torch.optim.adamw._single_tensor_adamw` exactly: decoupled weight
    /// decay first, then the moment updates, then the bias-corrected step.
    pub fn step(
        &mut self,
        parameters: &[crate::model::params::NamedParameter],
        gradients: &GradStore,
    ) -> Result<()> {
        for group in &self.groups {
            let GroupSettings {
                lr,
                betas: [beta1, beta2],
                eps,
                weight_decay,
            } = group.settings;
            for index in &group.params {
                let parameter = &parameters[*index];
                let Some(gradient) = gradients.get(parameter.value.as_tensor()) else {
                    // A parameter the forward pass did not touch has no gradient.
                    // torch skips it too (`p.grad is None`), so the moment state
                    // deliberately does not advance either.
                    continue;
                };
                let entry = match self.state.get(index) {
                    Some(state) => state.clone(),
                    None => MomentState {
                        step: 0.0,
                        exp_avg: Tensor::zeros(gradient.shape(), DType::F32, gradient.device())?,
                        exp_avg_sq: Tensor::zeros(gradient.shape(), DType::F32, gradient.device())?,
                    },
                };
                let step = entry.step + 1.0;

                // p.mul_(1 - lr * weight_decay)
                let mut value = parameter.value.as_tensor().clone();
                if weight_decay != 0.0 {
                    value = (value * (1.0 - lr * weight_decay))?;
                }

                let exp_avg = ((entry.exp_avg * beta1)? + (gradient * (1.0 - beta1))?)?;
                let exp_avg_sq =
                    ((entry.exp_avg_sq * beta2)? + (gradient.sqr()? * (1.0 - beta2))?)?;

                let bias_correction1 = 1.0 - beta1.powf(step);
                let bias_correction2 = 1.0 - beta2.powf(step);
                let step_size = lr / bias_correction1;
                let denominator = ((exp_avg_sq.sqrt()? / bias_correction2.sqrt())? + eps)?;
                let update = ((&exp_avg / denominator)? * step_size)?;
                parameter.value.set(&(value - update)?)?;

                // Detached, and it is not an optimization: a gradient candle hands
                // back still carries the `BackpropOp` chain it was produced from, so a
                // moment computed from one keeps that step's entire forward graph —
                // every activation of every ResNet and transformer layer — reachable.
                // Storing it undetached links each step's graph to the next through
                // `exp_avg`, and the live set then grows by one full step per step
                // until the device runs out of memory. `detach` shares the same
                // storage and drops only the history, which is exactly what
                // `torch.optim` gets for free from operating under `no_grad`.
                self.state.insert(
                    *index,
                    MomentState {
                        step,
                        exp_avg: exp_avg.detach(),
                        exp_avg_sq: exp_avg_sq.detach(),
                    },
                );
            }
        }
        Ok(())
    }

    /// `optimizer_state.safetensors`: `optimizer.state_dict()["state"]`, flattened
    /// with `/` the way `lerobot.utils.utils.flatten_dict` does.
    pub fn state_tensors(&self, device: &Device) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        for (index, state) in &self.state {
            // torch stores `step` as a zero-dimensional float tensor.
            out.insert(
                format!("state/{index}/step"),
                Tensor::new(state.step as f32, device)?,
            );
            out.insert(
                format!("state/{index}/exp_avg"),
                state.exp_avg.contiguous()?,
            );
            out.insert(
                format!("state/{index}/exp_avg_sq"),
                state.exp_avg_sq.contiguous()?,
            );
        }
        Ok(out)
    }

    /// Restore the state written by [`Self::state_tensors`].
    ///
    /// Validated against `parameters`, which is what makes this a load rather than a
    /// hope. A checkpoint is data from outside the process, and the previous version
    /// accepted whatever it could parse: it skipped keys it did not recognize, never
    /// checked that a parameter index existed, never checked a moment's shape or
    /// dtype, and summed a step tensor of any shape. Each of those turns a checkpoint
    /// for a *different model* into a silently wrong resume — the real parameters keep
    /// zero moments while the optimizer reports that it restored state.
    ///
    /// What is required, and why:
    ///
    /// * **every key understood.** An unrecognized key means the file was written by
    ///   something else; skipping it hides that.
    /// * **every index in range.** `state/<n>/` for `n` past the parameter list is a
    ///   checkpoint for a different architecture.
    /// * **every entry complete.** All three of `step`, `exp_avg` and `exp_avg_sq`, or
    ///   AdamW would resume with a moment it never saved.
    /// * **shapes and dtypes exact.** A moment must match its parameter's shape and be
    ///   `f32`; a mismatch is a different model, and a conversion would hide it.
    /// * **the step count finite and non-negative.** It is a divisor: the bias
    ///   corrections are `1 - beta^step`, so a NaN or negative step poisons every
    ///   subsequent update.
    pub fn load_state_tensors(
        &mut self,
        parameters: &[crate::model::params::NamedParameter],
        tensors: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // Parse every key first, so an unrecognized one is reported before any state is
        // installed and a rejected file leaves the optimizer untouched.
        let slots = ["step", "exp_avg", "exp_avg_sq"];
        let mut seen: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
        for key in tensors.keys() {
            let parts: Vec<&str> = key.split('/').collect();
            let malformed = || {
                TrainError::Metadata(format!(
                    "the optimizer state holds {key:?}, which is not a \
                     state/<parameter>/<step|exp_avg|exp_avg_sq> key"
                ))
            };
            if parts.len() != 3 || parts[0] != "state" {
                return Err(malformed());
            }
            let index: usize = parts[1].parse().map_err(|_| malformed())?;
            if !slots.contains(&parts[2]) {
                return Err(malformed());
            }
            if index >= parameters.len() {
                return Err(TrainError::Metadata(format!(
                    "the optimizer state names parameter {index}, but the model has {} \
                     parameters; this state belongs to a different model",
                    parameters.len()
                )));
            }
            seen.entry(index).or_default().push(parts[2]);
        }

        let mut restored: BTreeMap<usize, MomentState> = BTreeMap::new();
        for (index, present) in seen {
            for slot in slots {
                if !present.contains(&slot) {
                    return Err(TrainError::Metadata(format!(
                        "the optimizer state has no state/{index}/{slot}; an entry must carry \
                         all three of step, exp_avg and exp_avg_sq"
                    )));
                }
            }
            let expected = parameters[index].value.as_tensor().shape().clone();
            let fetch = |name: &str| -> Result<Tensor> {
                Ok(tensors
                    .get(&format!("state/{index}/{name}"))
                    .expect("presence checked above")
                    .clone())
            };

            let step_tensor = fetch("step")?;
            if step_tensor.dtype() != DType::F32 {
                return Err(TrainError::Metadata(format!(
                    "state/{index}/step has dtype {:?} but torch stores it as f32",
                    step_tensor.dtype()
                )));
            }
            // Exactly a scalar. `torch.optim.AdamW` stores `step` as a
            // zero-dimensional tensor; a `[1]` tensor holds the same number but is not
            // the same value, and accepting it means guessing at the format rather than
            // reading it. `elem_count() == 1` accepted both.
            if !step_tensor.dims().is_empty() {
                return Err(TrainError::Metadata(format!(
                    "state/{index}/step has shape {:?} but torch stores a step count as a \
                     zero-dimensional scalar tensor ([])",
                    step_tensor.dims()
                )));
            }
            let step = f64::from(step_tensor.to_scalar::<f32>()?);
            if !step.is_finite() || step < 0.0 {
                return Err(TrainError::Metadata(format!(
                    "state/{index}/step is {step}, which is not a finite non-negative step \
                     count; the bias corrections 1 - beta^step would poison every update"
                )));
            }
            // And a whole number. The bias corrections are `1 - beta^step`, so a
            // fractional step is not a count of anything and torch cannot have written
            // one: it would silently rescale every subsequent update.
            if step.fract() != 0.0 {
                return Err(TrainError::Metadata(format!(
                    "state/{index}/step is {step}, which is not a whole number of steps; \
                     an integral count is what the bias corrections 1 - beta^step assume"
                )));
            }

            let mut moments = Vec::with_capacity(2);
            for slot in ["exp_avg", "exp_avg_sq"] {
                let tensor = fetch(slot)?;
                if tensor.dtype() != DType::F32 {
                    return Err(TrainError::Metadata(format!(
                        "state/{index}/{slot} has dtype {:?} but the moments are f32",
                        tensor.dtype()
                    )));
                }
                if tensor.shape() != &expected {
                    return Err(TrainError::Metadata(format!(
                        "state/{index}/{slot} has shape {:?} but parameter {:?} has shape {:?}",
                        tensor.dims(),
                        parameters[index].name,
                        expected.dims()
                    )));
                }
                // The values too. A NaN moment is worse than a NaN weight: AdamW
                // divides by `sqrt(exp_avg_sq)`, so one poisoned moment turns its
                // parameter to NaN on the next step and every step after.
                let values = tensor.flatten_all()?.to_vec1::<f32>()?;
                if let Some((position, value)) = values
                    .iter()
                    .enumerate()
                    .find(|(_, value)| !value.is_finite())
                {
                    return Err(TrainError::Metadata(format!(
                        "state/{index}/{slot} element {position} is {value}, which is not \
                         finite; AdamW divides by sqrt(exp_avg_sq), so this would poison \
                         parameter {:?} on every subsequent step",
                        parameters[index].name
                    )));
                }
                moments.push(tensor);
            }

            restored.insert(
                index,
                MomentState {
                    step,
                    exp_avg_sq: moments.pop().expect("two moments were pushed"),
                    exp_avg: moments.pop().expect("two moments were pushed"),
                },
            );
        }

        // Complete, or empty. A *partial* state is the dangerous middle: every missing
        // parameter keeps zero moments while the optimizer reports a restored state, so
        // a resume trains those parameters as if from scratch, with the bias-corrected
        // step counts of a run that had already progressed. Empty is the legitimate
        // case -- `optimizer.state_dict()` of a fresh optimizer has no entries, and
        // restoring that is a no-op.
        if !restored.is_empty() && restored.len() != parameters.len() {
            let missing: Vec<String> = (0..parameters.len())
                .filter(|index| !restored.contains_key(index))
                .map(|index| format!("{index} ({})", parameters[index].name))
                .collect();
            let shown = missing.len().min(4);
            return Err(TrainError::Metadata(format!(
                "the optimizer state covers {} of {} parameters; it must be complete or \
                 empty. Missing: {}{}",
                restored.len(),
                parameters.len(),
                missing[..shown].join(", "),
                if missing.len() > shown {
                    format!(" and {} more", missing.len() - shown)
                } else {
                    String::new()
                }
            )));
        }

        // Installed only once every entry has been accepted.
        self.state = restored;
        Ok(())
    }

    /// `optimizer_param_groups.json`: `optimizer.state_dict()["param_groups"]`.
    ///
    /// The keys and their order are `torch.optim.AdamW`'s own, so that a checkpoint
    /// written here reloads into a real `AdamW`. The flags this port does not
    /// implement are written with the values torch defaults them to, and
    /// `docs/compatibility.md` lists them.
    pub fn param_groups_json(&self) -> JsonLike {
        let groups = self
            .groups
            .iter()
            .map(|group| {
                let mut object = JsonObject::new();
                object.insert("lr".into(), JsonLike::Float(group.settings.lr));
                object.insert(
                    "betas".into(),
                    JsonLike::Array(vec![
                        JsonLike::Float(group.settings.betas[0]),
                        JsonLike::Float(group.settings.betas[1]),
                    ]),
                );
                object.insert("eps".into(), JsonLike::Float(group.settings.eps));
                object.insert(
                    "weight_decay".into(),
                    JsonLike::Float(group.settings.weight_decay),
                );
                object.insert("amsgrad".into(), JsonLike::Bool(false));
                object.insert("maximize".into(), JsonLike::Bool(false));
                object.insert("foreach".into(), JsonLike::Null);
                object.insert("capturable".into(), JsonLike::Bool(false));
                object.insert("differentiable".into(), JsonLike::Bool(false));
                object.insert("fused".into(), JsonLike::Null);
                // `torch.optim.AdamW.__init__` sets `decoupled_weight_decay=True`
                // and records it in every group. `Optimizer.load_state_dict`
                // compares key *sets*, so a checkpoint missing this key is refused
                // by upstream with `ValueError: Dictionary keys do not match.` --
                // which is exactly what happened before this line existed.
                object.insert("decoupled_weight_decay".into(), JsonLike::Bool(true));
                object.insert(
                    "params".into(),
                    JsonLike::Array(
                        group
                            .params
                            .iter()
                            .map(|index| JsonLike::Int(num_bigint::BigInt::from(*index)))
                            .collect(),
                    ),
                );
                JsonLike::Object(object)
            })
            .collect();
        JsonLike::Array(groups)
    }

    /// The learning rate of the first group, which is what upstream logs as `lr`.
    pub fn first_lr(&self) -> f64 {
        self.groups
            .first()
            .map(|group| group.settings.lr)
            .unwrap_or_default()
    }
}

/// `torch.nn.utils.clip_grad_norm_` with `norm_type=2`.
///
/// Returns the total norm *before* clipping, which is what torch returns and what
/// upstream logs as `grad_norm`. Gradients are scaled in place in `gradients`.
///
/// A `max_norm` of infinity is the `grad_clip_norm <= 0` path in
/// `lerobot_train.update_policy`: the norm is measured and reported, and nothing
/// is scaled.
pub fn clip_grad_norm(
    parameters: &[crate::model::params::NamedParameter],
    gradients: &mut GradStore,
    max_norm: f64,
) -> Result<f64> {
    let mut squared = 0.0f64;
    for parameter in parameters {
        if let Some(gradient) = gradients.get(parameter.value.as_tensor()) {
            let norm = gradient.sqr()?.sum_all()?.to_scalar::<f32>()?;
            squared += f64::from(norm);
        }
    }
    let total_norm = squared.sqrt();
    if !max_norm.is_finite() {
        return Ok(total_norm);
    }
    // torch: `clip_coef = max_norm / (total_norm + 1e-6)`, then
    // `clamp(clip_coef, max=1.0)`, so a gradient already inside the ball is
    // untouched rather than scaled up.
    let coefficient = max_norm / (total_norm + 1e-6);
    if coefficient < 1.0 {
        let scaled: Vec<(Tensor, Tensor)> = parameters
            .iter()
            .filter_map(|parameter| {
                gradients
                    .get(parameter.value.as_tensor())
                    .map(|gradient| (parameter.value.as_tensor().clone(), gradient.clone()))
            })
            .collect();
        for (handle, gradient) in scaled {
            gradients.insert(&handle, (gradient * coefficient)?);
        }
    }
    Ok(total_norm)
}

/// The two parameter groups `ACTPolicy.get_optim_params` describes.
pub fn act_parameter_groups(
    indices: [Vec<usize>; 2],
    preset: &AdamWConfig,
    lr_backbone: f64,
) -> Vec<ParameterGroup> {
    let [main, backbone] = indices;
    let settings = GroupSettings::from_preset(preset);
    vec![
        ParameterGroup {
            params: main,
            settings: settings.clone(),
        },
        ParameterGroup {
            params: backbone,
            settings: settings.with_lr(lr_backbone),
        },
    ]
}

/// The sum of squares of every parameter, used to prove a step changed the model.
pub fn parameter_l2(parameters: &[crate::model::params::NamedParameter]) -> Result<f64> {
    let mut total = 0.0f64;
    for parameter in parameters {
        total += f64::from(
            parameter
                .value
                .as_tensor()
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?,
        );
    }
    Ok(total.sqrt())
}

/// The L2 distance between two `state_dict`s, keyed the same way.
pub fn state_dict_distance(
    left: &BTreeMap<String, Tensor>,
    right: &BTreeMap<String, Tensor>,
) -> Result<f64> {
    let mut total = 0.0f64;
    for (name, tensor) in left {
        let other = right
            .get(name)
            .ok_or_else(|| TrainError::Metadata(format!("the other state dict has no {name:?}")))?;
        let difference = (tensor - other)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        total += f64::from(difference);
    }
    Ok(total.sqrt())
}

/// A named view of the per-parameter moment state, for tests and diagnostics.
pub fn moment_summary(optimizer: &AdamW) -> IndexMap<usize, f64> {
    optimizer
        .state
        .iter()
        .map(|(index, state)| (*index, state.step))
        .collect()
}

/// Whether any stored moment still carries the graph it was computed from.
///
/// This is the observable side of the one thing about [`AdamW::step`] that is not
/// visible in the numbers: a moment computed from a candle gradient inherits that
/// gradient's `BackpropOp`, and holding one across a step pins the whole forward
/// pass it came from — activations included. The moments are the only state that
/// outlives a step, so this is the only place it can happen, and every training run
/// is long enough that the difference is the difference between finishing and
/// exhausting the device.
///
/// Always `false` for an optimizer built by this crate. Exposed because `track_op`
/// is on the tensor and the moments are private.
pub fn any_moment_tracks_its_graph(optimizer: &AdamW) -> bool {
    optimizer
        .state
        .values()
        .any(|state| state.exp_avg.track_op() || state.exp_avg_sq.track_op())
}
