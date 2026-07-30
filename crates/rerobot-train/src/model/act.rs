//! Port of `lerobot.policies.act.modeling_act`: the `ACT` module and the loss
//! `ACTPolicy.forward` computes.
//!
//! Ported in full for the state-only case: the BERT-style VAE encoder over
//! `[cls, robot_state, *actions]` with its fixed sinusoidal positions and its
//! `action_is_pad` key-padding mask, the reparameterization trick, the
//! transformer encoder over `[latent, robot_state, env_state]`, the DETR-style
//! decoder with learned object queries, the action head, and the masked L1 plus
//! KL objective.
//!
//! **Not** ported: the ResNet backbone, the 2-D sinusoidal camera embedding, and
//! the temporal ensembler. The first two only exist when the config has image
//! features, which [`crate::data::meta::DatasetMetadata::load`] refuses; the third
//! only runs at inference. [`crate::model::act::ActModel::new`] refuses a config that would need any
//! of them rather than quietly building a smaller model.

use crate::data::batch::Batch;
use crate::data::meta::{ACTION, OBS_ENV_STATE, OBS_STATE};
use crate::error::{Result, TrainError};
use crate::model::ops::{
    dropout_mask, sinusoidal_position_embedding, Activation, LayerNorm, Linear, MultiheadAttention,
};
use crate::model::params::{Initializer, NamedParameter, ParameterStore};
use candle_core::{DType, Device, Tensor};
use rerobot_core::policy::act::ActConfig;
use rerobot_core::random::SplitMix64;
use std::collections::BTreeMap;
use std::path::Path;

/// The prefix `ACTPolicy.state_dict()` puts on the `ACT` module's parameters.
pub const MODEL_PREFIX: &str = "model";

/// Where the latent noise and the dropout masks come from on a training pass.
pub enum Randomness<'a> {
    /// Draw both from a seeded generator. This is what a real run uses.
    Seeded(&'a mut SplitMix64),
    /// Use `latent_noise` verbatim as the standard-normal draw, and treat every
    /// dropout layer as the identity.
    ///
    /// This is the oracle mode: PyTorch's dropout mask and `randn_like` come from
    /// a Mersenne stream Rerobot does not reproduce, so the differential test
    /// against upstream supplies the noise and runs at `dropout = 0.0`, where both
    /// sides agree that dropout is a no-op.
    Fixed(Tensor),
}

/// Which pass to run.
pub enum Pass<'a> {
    /// `policy.train()`: the VAE encoder runs and dropout applies.
    Train(Randomness<'a>),
    /// `policy.eval()`: the latent is zeros and dropout is the identity.
    Eval,
}

impl Pass<'_> {
    fn is_training(&self) -> bool {
        matches!(self, Self::Train(_))
    }
}

/// What a forward pass returns: `(actions, (mu, log_sigma_x2))`.
#[derive(Debug)]
pub struct ActOutput {
    /// `[batch, chunk_size, action_dim]`.
    pub actions: Tensor,
    /// `[batch, latent_dim]`, present only on a VAE training pass.
    pub mu: Option<Tensor>,
    /// `[batch, latent_dim]`, present only on a VAE training pass.
    pub log_sigma_x2: Option<Tensor>,
}

/// The loss and the scalars `ACTPolicy.forward` reports alongside it.
#[derive(Debug)]
pub struct ActLoss {
    /// The differentiable total: `l1 + kl_weight * mean_kld` when the VAE is on.
    pub loss: Tensor,
    /// `loss_dict["l1_loss"]`.
    pub l1_loss: f64,
    /// `loss_dict["kld_loss"]`, when the VAE is on.
    pub kld_loss: Option<f64>,
    /// `loss.item()`.
    pub total: f64,
}

/// The config fields the model needs, narrowed from arbitrary-precision integers
/// to machine sizes exactly once, at construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActShape {
    /// `dim_model`.
    pub dim_model: usize,
    /// `n_heads`.
    pub n_heads: usize,
    /// `dim_feedforward`.
    pub dim_feedforward: usize,
    /// `n_encoder_layers`.
    pub n_encoder_layers: usize,
    /// `n_decoder_layers`.
    pub n_decoder_layers: usize,
    /// `n_vae_encoder_layers`.
    pub n_vae_encoder_layers: usize,
    /// `latent_dim`.
    pub latent_dim: usize,
    /// `chunk_size`.
    pub chunk_size: usize,
    /// `n_action_steps`.
    pub n_action_steps: usize,
    /// Width of `observation.state`, when the config has one.
    pub robot_state_dim: Option<usize>,
    /// Width of `observation.environment_state`, when the config has one.
    pub env_state_dim: Option<usize>,
    /// Width of `action`.
    pub action_dim: usize,
}

#[derive(Debug)]
struct EncoderLayer {
    self_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

#[derive(Debug)]
struct DecoderLayer {
    self_attn: MultiheadAttention,
    multihead_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

/// The Action Chunking Transformer.
#[derive(Debug)]
pub struct ActModel {
    store: ParameterStore,
    shape: ActShape,
    dropout: f64,
    kl_weight: f64,
    use_vae: bool,
    pre_norm: bool,
    activation: Activation,
    device: Device,

    vae_encoder: Option<Vec<EncoderLayer>>,
    vae_encoder_norm: Option<LayerNorm>,
    vae_encoder_cls_embed: Option<Tensor>,
    vae_encoder_robot_state_input_proj: Option<Linear>,
    vae_encoder_action_input_proj: Option<Linear>,
    vae_encoder_latent_output_proj: Option<Linear>,
    vae_encoder_pos_enc: Option<Tensor>,

    encoder: Vec<EncoderLayer>,
    encoder_norm: Option<LayerNorm>,
    decoder: Vec<DecoderLayer>,
    decoder_norm: LayerNorm,

    encoder_robot_state_input_proj: Option<Linear>,
    encoder_env_state_input_proj: Option<Linear>,
    encoder_latent_input_proj: Linear,
    encoder_1d_feature_pos_embed: Tensor,
    decoder_pos_embed: Tensor,
    action_head: Linear,
}

impl ActModel {
    /// Build a freshly initialized model from `config`, drawing from `rng`.
    pub fn new(config: &ActConfig, device: &Device, rng: &mut SplitMix64) -> Result<Self> {
        let shape = resolve_shape(config)?;
        let mut store = ParameterStore::new();
        let mut init = Initializer::new(rng, device.clone());
        let activation = Activation::parse(&config.feedforward_activation)?;
        let dimension = shape.dim_model;

        let (vae_encoder, vae_encoder_norm) = if config.use_vae {
            let layers = (0..shape.n_vae_encoder_layers)
                .map(|index| {
                    encoder_layer(
                        &mut store,
                        &mut init,
                        &format!("{MODEL_PREFIX}.vae_encoder.layers.{index}"),
                        &shape,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let norm = if config.pre_norm {
                Some(layer_norm(
                    &mut store,
                    &mut init,
                    &format!("{MODEL_PREFIX}.vae_encoder.norm"),
                    dimension,
                )?)
            } else {
                None
            };
            (Some(layers), norm)
        } else {
            (None, None)
        };

        let (
            vae_encoder_cls_embed,
            vae_encoder_robot_state_input_proj,
            vae_encoder_action_input_proj,
            vae_encoder_latent_output_proj,
            vae_encoder_pos_enc,
        ) = if config.use_vae {
            let cls = init.standard_normal(&[1, dimension])?;
            let cls =
                store.parameter(format!("{MODEL_PREFIX}.vae_encoder_cls_embed.weight"), cls)?;
            let robot_state = match shape.robot_state_dim {
                Some(width) => Some(linear(
                    &mut store,
                    &mut init,
                    &format!("{MODEL_PREFIX}.vae_encoder_robot_state_input_proj"),
                    dimension,
                    width,
                )?),
                None => None,
            };
            let action = linear(
                &mut store,
                &mut init,
                &format!("{MODEL_PREFIX}.vae_encoder_action_input_proj"),
                dimension,
                shape.action_dim,
            )?;
            let latent = linear(
                &mut store,
                &mut init,
                &format!("{MODEL_PREFIX}.vae_encoder_latent_output_proj"),
                shape.latent_dim * 2,
                dimension,
            )?;
            let tokens = 1 + shape.chunk_size + usize::from(shape.robot_state_dim.is_some());
            let table = sinusoidal_position_embedding(tokens, dimension, device)?
                .reshape((1, tokens, dimension))?;
            let table = store.buffer(format!("{MODEL_PREFIX}.vae_encoder_pos_enc"), table)?;
            (
                Some(cls),
                robot_state,
                Some(action),
                Some(latent),
                Some(table),
            )
        } else {
            (None, None, None, None, None)
        };

        let encoder = (0..shape.n_encoder_layers)
            .map(|index| {
                encoder_layer(
                    &mut store,
                    &mut init,
                    &format!("{MODEL_PREFIX}.encoder.layers.{index}"),
                    &shape,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let encoder_norm = if config.pre_norm {
            Some(layer_norm(
                &mut store,
                &mut init,
                &format!("{MODEL_PREFIX}.encoder.norm"),
                dimension,
            )?)
        } else {
            None
        };
        let decoder = (0..shape.n_decoder_layers)
            .map(|index| {
                decoder_layer(
                    &mut store,
                    &mut init,
                    &format!("{MODEL_PREFIX}.decoder.layers.{index}"),
                    &shape,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let decoder_norm = layer_norm(
            &mut store,
            &mut init,
            &format!("{MODEL_PREFIX}.decoder.norm"),
            dimension,
        )?;

        let encoder_robot_state_input_proj = match shape.robot_state_dim {
            Some(width) => Some(linear(
                &mut store,
                &mut init,
                &format!("{MODEL_PREFIX}.encoder_robot_state_input_proj"),
                dimension,
                width,
            )?),
            None => None,
        };
        let encoder_env_state_input_proj = match shape.env_state_dim {
            Some(width) => Some(linear(
                &mut store,
                &mut init,
                &format!("{MODEL_PREFIX}.encoder_env_state_input_proj"),
                dimension,
                width,
            )?),
            None => None,
        };
        let encoder_latent_input_proj = linear(
            &mut store,
            &mut init,
            &format!("{MODEL_PREFIX}.encoder_latent_input_proj"),
            dimension,
            shape.latent_dim,
        )?;

        let one_d_tokens = 1
            + usize::from(shape.robot_state_dim.is_some())
            + usize::from(shape.env_state_dim.is_some());
        let pos_embed = init.standard_normal(&[one_d_tokens, dimension])?;
        let encoder_1d_feature_pos_embed = store.parameter(
            format!("{MODEL_PREFIX}.encoder_1d_feature_pos_embed.weight"),
            pos_embed,
        )?;
        let queries = init.standard_normal(&[shape.chunk_size, dimension])?;
        let decoder_pos_embed =
            store.parameter(format!("{MODEL_PREFIX}.decoder_pos_embed.weight"), queries)?;
        let action_head = linear(
            &mut store,
            &mut init,
            &format!("{MODEL_PREFIX}.action_head"),
            shape.action_dim,
            dimension,
        )?;

        let model = Self {
            store,
            shape,
            dropout: config.dropout,
            kl_weight: config.kl_weight,
            use_vae: config.use_vae,
            pre_norm: config.pre_norm,
            activation,
            device: device.clone(),
            vae_encoder,
            vae_encoder_norm,
            vae_encoder_cls_embed,
            vae_encoder_robot_state_input_proj,
            vae_encoder_action_input_proj,
            vae_encoder_latent_output_proj,
            vae_encoder_pos_enc,
            encoder,
            encoder_norm,
            decoder,
            decoder_norm,
            encoder_robot_state_input_proj,
            encoder_env_state_input_proj,
            encoder_latent_input_proj,
            encoder_1d_feature_pos_embed,
            decoder_pos_embed,
            action_head,
        };
        model.reset_transformer_parameters()?;
        Ok(model)
    }

    /// `ACT._reset_parameters`: xavier-uniform over every encoder and decoder
    /// parameter with more than one dimension, applied after construction exactly
    /// as upstream applies it after `__init__` has run.
    ///
    /// The initializer's stream has already advanced past the default
    /// initializations by this point, which is what makes the overwrite observable
    /// rather than a no-op.
    fn reset_transformer_parameters(&self) -> Result<()> {
        let prefix_encoder = format!("{MODEL_PREFIX}.encoder.");
        let prefix_decoder = format!("{MODEL_PREFIX}.decoder.");
        let mut rng = SplitMix64::new(XAVIER_SUBSTREAM_SEED);
        let mut init = Initializer::new(&mut rng, self.device.clone());
        for parameter in self.store.parameters() {
            let in_transformer = parameter.name.starts_with(&prefix_encoder)
                || parameter.name.starts_with(&prefix_decoder);
            if !in_transformer || parameter.value.rank() <= 1 {
                continue;
            }
            let dims = parameter.value.dims();
            let value = init.xavier_uniform(dims[0], dims[1])?;
            parameter.value.set(&value)?;
        }
        Ok(())
    }

    /// The narrowed config shape in force.
    pub fn shape(&self) -> &ActShape {
        &self.shape
    }

    /// The device the parameters live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Trainable parameters, in registration order.
    pub fn parameters(&self) -> &[NamedParameter] {
        self.store.parameters()
    }

    /// How many trainable scalars the model holds.
    pub fn num_parameters(&self) -> usize {
        self.store.numel()
    }

    /// The `state_dict` written to `model.safetensors`, as detached copies.
    ///
    /// See [`ParameterStore::state_dict`] for why the copies matter.
    pub fn state_dict(&self) -> Result<BTreeMap<String, Tensor>> {
        self.store.state_dict()
    }

    /// Write `model.safetensors`.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.store.save(path)
    }

    /// Overwrite every parameter from `model.safetensors`.
    pub fn load(&mut self, path: &Path) -> Result<()> {
        let device = self.device.clone();
        self.store.load(path, &device)
    }

    /// `ACTPolicy.get_optim_params`: the non-backbone group, then the backbone one.
    ///
    /// The backbone group is always empty here, because a state-only config has no
    /// image features and therefore no backbone. It is still reported, because
    /// upstream reports it and `optimizer_param_groups.json` records both.
    pub fn optimizer_parameter_groups(&self) -> [Vec<usize>; 2] {
        let backbone_prefix = format!("{MODEL_PREFIX}.backbone");
        let mut main = Vec::new();
        let mut backbone = Vec::new();
        for (index, parameter) in self.store.parameters().iter().enumerate() {
            if parameter.name.starts_with(&backbone_prefix) {
                backbone.push(index);
            } else {
                main.push(index);
            }
        }
        [main, backbone]
    }

    /// A forward pass.
    pub fn forward(&self, batch: &Batch, mut pass: Pass<'_>) -> Result<ActOutput> {
        let batch_size = batch.len();
        let dimension = self.shape.dim_model;

        let robot_state = match self.shape.robot_state_dim {
            Some(_) => Some(batch.feature(OBS_STATE)?),
            None => None,
        };
        let env_state = match self.shape.env_state_dim {
            Some(_) => Some(batch.feature(OBS_ENV_STATE)?),
            None => None,
        };

        let use_vae_branch =
            self.use_vae && pass.is_training() && batch.features.contains_key(ACTION);
        if self.use_vae && pass.is_training() && !use_vae_branch {
            return Err(TrainError::Metadata(
                "actions must be provided when using the variational objective in training mode."
                    .to_owned(),
            ));
        }

        let (latent_sample, mu, log_sigma_x2) = if use_vae_branch {
            let actions = batch.feature(ACTION)?;
            let is_pad = batch.padding_mask(ACTION)?;
            let cls = self
                .vae_encoder_cls_embed
                .as_ref()
                .expect("the VAE branch implies the cls embedding")
                .reshape((1, 1, dimension))?
                .broadcast_as((batch_size, 1, dimension))?
                .contiguous()?;
            let action_embed = self
                .vae_encoder_action_input_proj
                .as_ref()
                .expect("the VAE branch implies the action projection")
                .forward(actions)?;

            let mut tokens = vec![cls];
            let mut leading = 1usize;
            if let (Some(projection), Some(state)) =
                (&self.vae_encoder_robot_state_input_proj, robot_state)
            {
                tokens.push(
                    projection
                        .forward(state)?
                        .reshape((batch_size, 1, dimension))?,
                );
                leading += 1;
            }
            tokens.push(action_embed);
            let vae_input = Tensor::cat(&tokens, 1)?;

            let positions = self
                .vae_encoder_pos_enc
                .as_ref()
                .expect("the VAE branch implies the position table")
                .broadcast_as(vae_input.shape())?
                .contiguous()?;

            // `[cls_joint_is_pad, action_is_pad]`, where the leading tokens are
            // never padding.
            let leading_mask = Tensor::zeros((batch_size, leading), DType::U8, &self.device)?;
            let key_padding_mask = Tensor::cat(&[&leading_mask, is_pad], 1)?;

            let encoded = self.run_encoder(
                self.vae_encoder
                    .as_ref()
                    .expect("the VAE branch implies the encoder"),
                self.vae_encoder_norm.as_ref(),
                &vae_input,
                Some(&positions),
                Some(&key_padding_mask),
                &mut pass,
            )?;
            let cls_out = encoded.narrow(1, 0, 1)?.reshape((batch_size, dimension))?;
            let pdf = self
                .vae_encoder_latent_output_proj
                .as_ref()
                .expect("the VAE branch implies the latent projection")
                .forward(&cls_out)?;
            let mu = pdf.narrow(1, 0, self.shape.latent_dim)?.contiguous()?;
            let log_sigma_x2 = pdf
                .narrow(1, self.shape.latent_dim, self.shape.latent_dim)?
                .contiguous()?;
            let noise = match &mut pass {
                Pass::Train(Randomness::Seeded(rng)) => {
                    let count = batch_size * self.shape.latent_dim;
                    let values: Vec<f32> =
                        (0..count).map(|_| rng.standard_normal() as f32).collect();
                    Tensor::from_vec(values, (batch_size, self.shape.latent_dim), &self.device)?
                }
                Pass::Train(Randomness::Fixed(noise)) => {
                    if noise.dims() != [batch_size, self.shape.latent_dim] {
                        return Err(TrainError::Tensor(format!(
                            "the supplied latent noise has shape {:?} but the batch needs {:?}",
                            noise.dims(),
                            [batch_size, self.shape.latent_dim]
                        )));
                    }
                    noise.clone()
                }
                Pass::Eval => unreachable!("the VAE branch is training-only"),
            };
            let sample = (&mu + ((&log_sigma_x2 / 2.0)?.exp()? * noise)?)?;
            (sample, Some(mu), Some(log_sigma_x2))
        } else {
            (
                Tensor::zeros(
                    (batch_size, self.shape.latent_dim),
                    DType::F32,
                    &self.device,
                )?,
                None,
                None,
            )
        };

        // `[latent, (robot_state), (env_state)]`.
        let mut encoder_tokens = vec![self
            .encoder_latent_input_proj
            .forward(&latent_sample)?
            .reshape((batch_size, 1, dimension))?];
        if let (Some(projection), Some(state)) = (&self.encoder_robot_state_input_proj, robot_state)
        {
            encoder_tokens.push(
                projection
                    .forward(state)?
                    .reshape((batch_size, 1, dimension))?,
            );
        }
        if let (Some(projection), Some(state)) = (&self.encoder_env_state_input_proj, env_state) {
            encoder_tokens.push(
                projection
                    .forward(state)?
                    .reshape((batch_size, 1, dimension))?,
            );
        }
        let encoder_input = Tensor::cat(&encoder_tokens, 1)?;
        let token_count = encoder_tokens.len();
        let encoder_positions = self
            .encoder_1d_feature_pos_embed
            .reshape((1, token_count, dimension))?
            .broadcast_as((batch_size, token_count, dimension))?
            .contiguous()?;

        let encoder_out = self.run_encoder(
            &self.encoder,
            self.encoder_norm.as_ref(),
            &encoder_input,
            Some(&encoder_positions),
            None,
            &mut pass,
        )?;

        let decoder_input = Tensor::zeros(
            (batch_size, self.shape.chunk_size, dimension),
            DType::F32,
            &self.device,
        )?;
        let decoder_positions = self
            .decoder_pos_embed
            .reshape((1, self.shape.chunk_size, dimension))?
            .broadcast_as((batch_size, self.shape.chunk_size, dimension))?
            .contiguous()?;
        let decoder_out = self.run_decoder(
            &decoder_input,
            &encoder_out,
            &encoder_positions,
            &decoder_positions,
            &mut pass,
        )?;

        Ok(ActOutput {
            actions: self.action_head.forward(&decoder_out)?,
            mu,
            log_sigma_x2,
        })
    }

    /// `ACTPolicy.predict_action_chunk` followed by the `n_action_steps` slice
    /// `select_action` applies: an eval pass, latent zeroed, dropout off.
    pub fn predict_action_steps(&self, batch: &Batch) -> Result<Tensor> {
        let actions = self.forward(batch, Pass::Eval)?.actions;
        Ok(actions
            .narrow(1, 0, self.shape.n_action_steps)?
            .contiguous()?)
    }

    /// `ACTPolicy.forward`'s loss: masked L1, plus the KL term when the VAE is on.
    pub fn loss(&self, batch: &Batch, output: &ActOutput) -> Result<ActLoss> {
        let target = batch.feature(ACTION)?;
        let is_pad = batch.padding_mask(ACTION)?;

        let absolute_error = (target - &output.actions)?.abs()?;
        // `valid_mask = ~action_is_pad`, unsqueezed on the last axis so it
        // broadcasts over the action dimension.
        let (batch_size, window) = is_pad.dims2()?;
        let valid = (1.0 - is_pad.to_dtype(DType::F32)?)?.reshape((batch_size, window, 1))?;
        let masked = absolute_error.broadcast_mul(&valid)?;
        let valid_count = valid.sum_all()?.to_scalar::<f32>()? as f64;
        let action_dim = *absolute_error
            .dims()
            .last()
            .expect("the error has the action axis") as f64;
        // `num_valid.clamp_min(1)`: an all-padding batch divides by one rather
        // than by zero.
        let denominator = (valid_count * action_dim).max(1.0);
        let l1 = (masked.sum_all()? / denominator)?;
        let l1_value = l1.to_scalar::<f32>()? as f64;

        match (&output.mu, &output.log_sigma_x2) {
            (Some(mu), Some(log_sigma_x2)) if self.use_vae => {
                // -0.5 * sum(1 + log_sigma_x2 - mu^2 - exp(log_sigma_x2)), summed
                // over the latent axis and averaged over the batch.
                let term = ((log_sigma_x2 + 1.0)? - mu.sqr()?)? - log_sigma_x2.exp()?;
                let mean_kld = (term?.sum(1)? * -0.5)?.mean_all()?;
                let kld_value = mean_kld.to_scalar::<f32>()? as f64;
                let loss = (&l1 + (mean_kld * self.kl_weight)?)?;
                let total = loss.to_scalar::<f32>()? as f64;
                Ok(ActLoss {
                    loss,
                    l1_loss: l1_value,
                    kld_loss: Some(kld_value),
                    total,
                })
            }
            _ => Ok(ActLoss {
                loss: l1,
                l1_loss: l1_value,
                kld_loss: None,
                total: l1_value,
            }),
        }
    }

    fn maybe_dropout(&self, input: &Tensor, pass: &mut Pass<'_>) -> Result<Tensor> {
        match pass {
            Pass::Train(Randomness::Seeded(rng)) if self.dropout > 0.0 => {
                let mask = dropout_mask(rng, input.dims(), self.dropout, &self.device)?;
                Ok((input * mask)?)
            }
            // `Fixed` is the oracle mode and `Eval` is `policy.eval()`; both treat
            // dropout as the identity, and so does a config with `dropout = 0`.
            _ => Ok(input.clone()),
        }
    }

    fn run_encoder(
        &self,
        layers: &[EncoderLayer],
        norm: Option<&LayerNorm>,
        input: &Tensor,
        positions: Option<&Tensor>,
        key_padding_mask: Option<&Tensor>,
        pass: &mut Pass<'_>,
    ) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in layers {
            x = self.encoder_layer_forward(layer, &x, positions, key_padding_mask, pass)?;
        }
        // `nn.Identity()` when `pre_norm` is false.
        match norm {
            Some(norm) => norm.forward(&x),
            None => Ok(x),
        }
    }

    fn encoder_layer_forward(
        &self,
        layer: &EncoderLayer,
        input: &Tensor,
        positions: Option<&Tensor>,
        key_padding_mask: Option<&Tensor>,
        pass: &mut Pass<'_>,
    ) -> Result<Tensor> {
        let mut skip = input.clone();
        let mut x = if self.pre_norm {
            layer.norm1.forward(input)?
        } else {
            input.clone()
        };
        let query = match positions {
            Some(positions) => (&x + positions)?,
            None => x.clone(),
        };
        let attended = layer
            .self_attn
            .forward(&query, &query, &x, key_padding_mask)?;
        x = (&skip + self.maybe_dropout(&attended, pass)?)?;
        if self.pre_norm {
            skip = x.clone();
            x = layer.norm2.forward(&x)?;
        } else {
            x = layer.norm1.forward(&x)?;
            skip = x.clone();
        }
        let hidden = layer.linear1.forward(&x)?;
        let hidden = self.activation.forward(&hidden)?;
        let hidden = self.maybe_dropout(&hidden, pass)?;
        x = layer.linear2.forward(&hidden)?;
        x = (&skip + self.maybe_dropout(&x, pass)?)?;
        if !self.pre_norm {
            x = layer.norm2.forward(&x)?;
        }
        Ok(x)
    }

    fn run_decoder(
        &self,
        input: &Tensor,
        encoder_out: &Tensor,
        encoder_positions: &Tensor,
        decoder_positions: &Tensor,
        pass: &mut Pass<'_>,
    ) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &self.decoder {
            x = self.decoder_layer_forward(
                layer,
                &x,
                encoder_out,
                encoder_positions,
                decoder_positions,
                pass,
            )?;
        }
        self.decoder_norm.forward(&x)
    }

    fn decoder_layer_forward(
        &self,
        layer: &DecoderLayer,
        input: &Tensor,
        encoder_out: &Tensor,
        encoder_positions: &Tensor,
        decoder_positions: &Tensor,
        pass: &mut Pass<'_>,
    ) -> Result<Tensor> {
        let mut skip = input.clone();
        let mut x = if self.pre_norm {
            layer.norm1.forward(input)?
        } else {
            input.clone()
        };
        let query = (&x + decoder_positions)?;
        let attended = layer.self_attn.forward(&query, &query, &x, None)?;
        x = (&skip + self.maybe_dropout(&attended, pass)?)?;
        if self.pre_norm {
            skip = x.clone();
            x = layer.norm2.forward(&x)?;
        } else {
            x = layer.norm1.forward(&x)?;
            skip = x.clone();
        }
        let cross = layer.multihead_attn.forward(
            &(&x + decoder_positions)?,
            &(encoder_out + encoder_positions)?,
            encoder_out,
            None,
        )?;
        x = (&skip + self.maybe_dropout(&cross, pass)?)?;
        if self.pre_norm {
            skip = x.clone();
            x = layer.norm3.forward(&x)?;
        } else {
            x = layer.norm2.forward(&x)?;
            skip = x.clone();
        }
        let hidden = layer.linear1.forward(&x)?;
        let hidden = self.activation.forward(&hidden)?;
        let hidden = self.maybe_dropout(&hidden, pass)?;
        x = layer.linear2.forward(&hidden)?;
        x = (&skip + self.maybe_dropout(&x, pass)?)?;
        if !self.pre_norm {
            x = layer.norm3.forward(&x)?;
        }
        Ok(x)
    }
}

/// Seed of the sub-stream `_reset_parameters` draws its xavier values from.
///
/// A fixed sub-stream rather than a continuation of the model's own: upstream
/// applies `_reset_parameters` after `__init__`, and pinning it here means the
/// transformer weights of a given config are the same regardless of how many
/// default initializations ran before them. `docs/compatibility.md` records that
/// the values are Rerobot's, not torch's, either way.
const XAVIER_SUBSTREAM_SEED: u64 = 0x4143_5420_5841_5649;

fn resolve_shape(config: &ActConfig) -> Result<ActShape> {
    // Bounded, not merely narrowed. `usize::try_from` alone lets a 10^18 `dim_model`
    // through on a 64-bit target, where it becomes a tensor allocation request that
    // aborts the process rather than failing. `TrainConfig::validate` checks the same
    // fields, and this repeats the check because a library caller can reach here
    // without one.
    let field = |value: &num_bigint::BigInt, name: &str, limit: usize| -> Result<usize> {
        crate::limits::bounded_usize(value, name, limit)
    };

    let inputs = config.input_features.clone().unwrap_or_default();
    let outputs = config.output_features.clone().unwrap_or_default();

    let visual: Vec<&String> = inputs
        .iter()
        .filter(|(_, feature)| feature.r#type == rerobot_core::types::FeatureType::Visual)
        .map(|(key, _)| key)
        .collect();
    if !visual.is_empty() {
        let mut names: Vec<&str> = visual.iter().map(|key| key.as_str()).collect();
        names.sort_unstable();
        return Err(TrainError::unsupported(format!(
            "the policy config has image features ({}); ACT needs a torchvision ResNet backbone \
             and the 2-D camera position embedding for those, and neither is ported",
            names.join(", ")
        )));
    }
    if config.temporal_ensemble_coeff.is_some() {
        return Err(TrainError::unsupported(
            "temporal_ensemble_coeff is set; the temporal ensembler is an inference-time \
             component and is not ported"
                .to_owned(),
        ));
    }

    let width = |features: &indexmap::IndexMap<String, rerobot_core::types::PolicyFeature>,
                 key: &str|
     -> Result<Option<usize>> {
        match features.get(key) {
            None => Ok(None),
            Some(feature) => {
                // Every dimension bounded, then a *checked* product. The previous
                // `saturating_mul` was the worse of the two available bugs: it turned
                // an overflowing shape into `usize::MAX` silently, which is an
                // allocation request rather than an error.
                let mut dimensions = Vec::with_capacity(feature.shape.len());
                for dimension in &feature.shape {
                    dimensions.push(crate::limits::bounded_usize(
                        dimension,
                        &format!("feature {key:?} dimension"),
                        crate::limits::MAX_FEATURE_WIDTH,
                    )?);
                }
                let product = crate::limits::checked_product(
                    &dimensions,
                    &format!("the shape of feature {key:?}"),
                )?;
                crate::limits::within(
                    product,
                    &format!("the width of feature {key:?}"),
                    crate::limits::MAX_FEATURE_WIDTH,
                )?;
                Ok(Some(product))
            }
        }
    };

    let robot_state_dim = width(&inputs, OBS_STATE)?;
    let env_state_dim = width(&inputs, OBS_ENV_STATE)?;
    let action_dim = width(&outputs, ACTION)?.ok_or_else(|| {
        TrainError::Metadata(
            "the policy config has no `action` output feature, so there is nothing to predict"
                .to_owned(),
        )
    })?;
    if robot_state_dim.is_none() && env_state_dim.is_none() {
        return Err(TrainError::Metadata(
            "the policy config has neither observation.state nor \
             observation.environment_state, so the transformer encoder would have only the \
             latent token"
                .to_owned(),
        ));
    }

    let dim_model = field(&config.dim_model, "dim_model", crate::limits::MAX_DIM_MODEL)?;
    let n_heads = field(&config.n_heads, "n_heads", crate::limits::MAX_HEADS)?;
    if n_heads == 0 || dim_model % n_heads != 0 {
        return Err(TrainError::Metadata(format!(
            "dim_model {dim_model} must be a positive multiple of n_heads {n_heads}"
        )));
    }

    Ok(ActShape {
        dim_model,
        n_heads,
        dim_feedforward: field(
            &config.dim_feedforward,
            "dim_feedforward",
            crate::limits::MAX_DIM_FEEDFORWARD,
        )?,
        n_encoder_layers: field(
            &config.n_encoder_layers,
            "n_encoder_layers",
            crate::limits::MAX_LAYERS,
        )?,
        n_decoder_layers: field(
            &config.n_decoder_layers,
            "n_decoder_layers",
            crate::limits::MAX_LAYERS,
        )?,
        n_vae_encoder_layers: field(
            &config.n_vae_encoder_layers,
            "n_vae_encoder_layers",
            crate::limits::MAX_LAYERS,
        )?,
        latent_dim: field(
            &config.latent_dim,
            "latent_dim",
            crate::limits::MAX_LATENT_DIM,
        )?,
        chunk_size: field(
            &config.chunk_size,
            "chunk_size",
            crate::limits::MAX_CHUNK_SIZE,
        )?,
        n_action_steps: field(
            &config.n_action_steps,
            "n_action_steps",
            crate::limits::MAX_CHUNK_SIZE,
        )?,
        robot_state_dim,
        env_state_dim,
        action_dim,
    })
}

fn linear(
    store: &mut ParameterStore,
    init: &mut Initializer<'_>,
    prefix: &str,
    out_features: usize,
    in_features: usize,
) -> Result<Linear> {
    let (weight, bias) = init.linear(out_features, in_features)?;
    Ok(Linear {
        weight: store.parameter(format!("{prefix}.weight"), weight)?,
        bias: store.parameter(format!("{prefix}.bias"), bias)?,
    })
}

fn layer_norm(
    store: &mut ParameterStore,
    init: &mut Initializer<'_>,
    prefix: &str,
    dimension: usize,
) -> Result<LayerNorm> {
    let weight = init.ones(&[dimension])?;
    let bias = init.zeros(&[dimension])?;
    Ok(LayerNorm {
        weight: store.parameter(format!("{prefix}.weight"), weight)?,
        bias: store.parameter(format!("{prefix}.bias"), bias)?,
    })
}

fn attention(
    store: &mut ParameterStore,
    init: &mut Initializer<'_>,
    prefix: &str,
    shape: &ActShape,
) -> Result<MultiheadAttention> {
    // `nn.MultiheadAttention` initializes `in_proj_weight` with
    // `xavier_uniform_` over the packed `[3 * dim, dim]` shape and zeroes
    // `in_proj_bias`, which is why this is not `Initializer::linear`.
    let packed = init.xavier_uniform(3 * shape.dim_model, shape.dim_model)?;
    let in_proj_bias = init.zeros(&[3 * shape.dim_model])?;
    let out_weight = init.xavier_uniform(shape.dim_model, shape.dim_model)?;
    let out_bias = init.zeros(&[shape.dim_model])?;
    Ok(MultiheadAttention {
        in_proj_weight: store.parameter(format!("{prefix}.in_proj_weight"), packed)?,
        in_proj_bias: store.parameter(format!("{prefix}.in_proj_bias"), in_proj_bias)?,
        out_proj: Linear {
            weight: store.parameter(format!("{prefix}.out_proj.weight"), out_weight)?,
            bias: store.parameter(format!("{prefix}.out_proj.bias"), out_bias)?,
        },
        num_heads: shape.n_heads,
    })
}

fn encoder_layer(
    store: &mut ParameterStore,
    init: &mut Initializer<'_>,
    prefix: &str,
    shape: &ActShape,
) -> Result<EncoderLayer> {
    Ok(EncoderLayer {
        self_attn: attention(store, init, &format!("{prefix}.self_attn"), shape)?,
        linear1: linear(
            store,
            init,
            &format!("{prefix}.linear1"),
            shape.dim_feedforward,
            shape.dim_model,
        )?,
        linear2: linear(
            store,
            init,
            &format!("{prefix}.linear2"),
            shape.dim_model,
            shape.dim_feedforward,
        )?,
        norm1: layer_norm(store, init, &format!("{prefix}.norm1"), shape.dim_model)?,
        norm2: layer_norm(store, init, &format!("{prefix}.norm2"), shape.dim_model)?,
    })
}

fn decoder_layer(
    store: &mut ParameterStore,
    init: &mut Initializer<'_>,
    prefix: &str,
    shape: &ActShape,
) -> Result<DecoderLayer> {
    Ok(DecoderLayer {
        self_attn: attention(store, init, &format!("{prefix}.self_attn"), shape)?,
        multihead_attn: attention(store, init, &format!("{prefix}.multihead_attn"), shape)?,
        linear1: linear(
            store,
            init,
            &format!("{prefix}.linear1"),
            shape.dim_feedforward,
            shape.dim_model,
        )?,
        linear2: linear(
            store,
            init,
            &format!("{prefix}.linear2"),
            shape.dim_model,
            shape.dim_feedforward,
        )?,
        norm1: layer_norm(store, init, &format!("{prefix}.norm1"), shape.dim_model)?,
        norm2: layer_norm(store, init, &format!("{prefix}.norm2"), shape.dim_model)?,
        norm3: layer_norm(store, init, &format!("{prefix}.norm3"), shape.dim_model)?,
    })
}
