//! Port of `lerobot.policies.act.modeling_act`: the `ACT` module and the loss
//! `ACTPolicy.forward` computes.
//!
//! Ported: the BERT-style VAE encoder over `[cls, robot_state, *actions]` with its
//! fixed sinusoidal positions and its `action_is_pad` key-padding mask, the
//! reparameterization trick, the ResNet image backbone with its 1×1 token
//! projection and 2-D sinusoidal camera embedding, the transformer encoder over
//! `[latent, robot_state, env_state, *camera_tokens]`, the DETR-style decoder with
//! learned object queries, the action head, and the masked L1 plus KL objective.
//!
//! Cameras arrive as tensors already in memory — see [`crate::data::image`] for the
//! exact contract and for why neither on-disk camera form of a LeRobot v3.0 dataset
//! can be read here.
//!
//! **Not** ported: the temporal ensembler, which only runs at inference, and
//! pretrained torchvision backbone weights, which are a download rather than a
//! computation. [`crate::model::act::ActModel::new`] refuses a config needing either rather than
//! quietly building a different model.

use crate::data::batch::Batch;
use crate::data::image::{camera_view, require_finite};
use crate::data::meta::{ACTION, OBS_ENV_STATE, OBS_STATE};
use crate::error::{Result, TrainError};
use crate::model::backbone::{stage_blocks, ResNetBackbone, FEATURE_CHANNELS};
use crate::model::ops::{
    dropout_mask, sinusoidal_position_embedding, sinusoidal_position_embedding_2d, Activation,
    Conv2d, LayerNorm, Linear, MultiheadAttention,
};
use crate::model::params::{Initializer, NamedParameter, ParameterStore};
use candle_core::{DType, Device, Tensor};
use rerobot_core::policy::act::{ActConfig, PythonIntBool};
use rerobot_core::random::SplitMix64;
use rerobot_core::types::FeatureType;
use std::collections::BTreeMap;
use std::path::Path;

/// Channels ACT's backbone takes, which is what torchvision's ResNet stem is built
/// for: `conv1` is `nn.Conv2d(3, 64, kernel_size=7, ...)` and nothing reshapes it.
pub const CAMERA_CHANNELS: usize = 3;

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
    /// The camera features, in `input_features` order.
    ///
    /// That order is `config.image_features`', which is what upstream iterates when
    /// it builds `batch[OBS_IMAGES]`, so it is also the order the encoder's camera
    /// tokens appear in. Preserving it is what makes a multi-camera forward pass
    /// deterministic rather than dependent on a hash order.
    pub cameras: Vec<CameraSpec>,
}

/// One camera input feature, narrowed to machine sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraSpec {
    /// The `input_features` key, e.g. `observation.images.top`.
    pub key: String,
    /// Channels the feature declares. Always [`CAMERA_CHANNELS`].
    pub channels: usize,
    /// Height the feature declares.
    pub height: usize,
    /// Width the feature declares.
    pub width: usize,
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

    backbone: Option<ResNetBackbone>,
    encoder_img_feat_input_proj: Option<Conv2d>,

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

        // `if self.config.image_features: self.backbone = IntermediateLayerGetter(...)`,
        // registered here so that `named_parameters()` — and therefore the optimizer's
        // parameter indices — run in upstream's order: the VAE encoder, the backbone,
        // then the transformer.
        let backbone = if shape.cameras.is_empty() {
            None
        } else {
            Some(ResNetBackbone::new(
                &config.vision_backbone,
                CAMERA_CHANNELS,
                &mut store,
                &mut init,
                &format!("{MODEL_PREFIX}.backbone"),
            )?)
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
        // `nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)`,
        // which turns one feature-map cell into one encoder token.
        let encoder_img_feat_input_proj = if shape.cameras.is_empty() {
            None
        } else {
            Some(conv_1x1(
                &mut store,
                &mut init,
                &format!("{MODEL_PREFIX}.encoder_img_feat_input_proj"),
                dimension,
                FEATURE_CHANNELS,
            )?)
        };

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
            backbone,
            encoder_img_feat_input_proj,
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
    /// The split is by name, `model.backbone` against everything else, which is what
    /// upstream's `n.startswith("model.backbone")` does. A state-only config has no
    /// backbone and so an empty second group; it is still reported, because upstream
    /// reports it and `optimizer_param_groups.json` records both.
    ///
    /// `encoder_img_feat_input_proj` is deliberately *not* in the backbone group:
    /// upstream matches on the prefix alone, and that projection does not carry it.
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
        // `encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))`,
        // one `[1, dim]` row per 1-D token, before the camera embeddings extend it.
        let one_d_tokens = encoder_tokens.len();
        let mut encoder_position_parts =
            vec![self
                .encoder_1d_feature_pos_embed
                .reshape((1, one_d_tokens, dimension))?];

        // `for img in batch[OBS_IMAGES]`, in `config.image_features` order.
        if let (Some(backbone), Some(projection)) =
            (&self.backbone, &self.encoder_img_feat_input_proj)
        {
            for camera in &self.shape.cameras {
                let image = self.camera_input(camera, batch, batch_size)?;
                let features = backbone.forward(&image)?;
                let (_, _, height, width) = features.dims4()?;
                // `rearrange(cam_features, "b c h w -> (h w) b c")`, which is
                // `[batch, h * w, dim]` in this crate's batch-first layout.
                let tokens = projection
                    .forward(&features)?
                    .flatten_from(2)?
                    .transpose(1, 2)?
                    .contiguous()?;
                // The embedding is a pure function of the feature map's extent and is
                // the same for every frame in the batch, so it stays `[1, h * w, dim]`
                // and broadcasts, exactly as upstream's does.
                let positions =
                    sinusoidal_position_embedding_2d(height, width, dimension / 2, &self.device)?
                        .flatten_from(2)?
                        .transpose(1, 2)?
                        .contiguous()?;
                encoder_tokens.push(tokens);
                encoder_position_parts.push(positions);
            }
        }

        let encoder_input = Tensor::cat(&encoder_tokens, 1)?;
        let token_count = encoder_input.dims()[1];
        let encoder_positions = Tensor::cat(&encoder_position_parts, 1)?
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

    /// One camera's tensor from `batch`, checked against what the config declared.
    ///
    /// The batch's own contract ([`crate::data::image`]) is about the tensor in
    /// isolation — dtype, rank, batch size, bounds. This is the other half: the
    /// extent has to be the one `input_features` declares, because that declaration
    /// is what the checkpoint records and what a reader of `config.json` will size
    /// its own inputs from. A batch of a different extent would train and predict
    /// perfectly well and disagree with the file that describes it.
    fn camera_input(
        &self,
        camera: &CameraSpec,
        batch: &Batch,
        batch_size: usize,
    ) -> Result<Tensor> {
        let image = camera_view(&camera.key, batch.image(&camera.key)?, batch_size)?;
        let (_, channels, height, width) = image.dims4()?;
        if (channels, height, width) != (camera.channels, camera.height, camera.width) {
            return Err(TrainError::Metadata(format!(
                "camera {:?} carries {channels}x{height}x{width} images but the policy config \
                 declares {}x{}x{}",
                camera.key, camera.channels, camera.height, camera.width
            )));
        }
        require_finite(&camera.key, &image)?;
        Ok(image)
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

    let cameras = resolve_cameras(config, &inputs)?;
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
                // Zero would build a projection with no input columns and reach
                // `chunks(0)` in the collator; see `FeatureSpec::width`.
                if product == 0 {
                    return Err(TrainError::Metadata(format!(
                        "feature {key:?} declares an empty shape, so it carries no scalars; \
                         a policy cannot consume or produce it"
                    )));
                }
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
    if robot_state_dim.is_none() && env_state_dim.is_none() && cameras.is_empty() {
        return Err(TrainError::Metadata(
            "the policy config has no camera, no observation.state and no \
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
    // `ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)` interleaves a sine and
    // a cosine term across each half of its output, so `dim_model / 2` has to be even
    // as well. Upstream's `torch.stack` would raise on the mismatched halves instead.
    if !cameras.is_empty() && dim_model % 4 != 0 {
        return Err(TrainError::Metadata(format!(
            "dim_model {dim_model} is not a multiple of four; the 2-D camera position \
             embedding splits it into a y half and an x half and interleaves sine and cosine \
             terms within each, which needs dim_model / 2 to be even"
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
        cameras,
    })
}

/// `config.image_features`, narrowed and checked against what this port can build.
///
/// Returns an empty vector for a state-only config, which is what leaves the backbone
/// and the image projection unbuilt.
fn resolve_cameras(
    config: &ActConfig,
    inputs: &indexmap::IndexMap<String, rerobot_core::types::PolicyFeature>,
) -> Result<Vec<CameraSpec>> {
    let visual: Vec<(&String, &rerobot_core::types::PolicyFeature)> = inputs
        .iter()
        .filter(|(_, feature)| feature.r#type == FeatureType::Visual)
        .collect();
    if visual.is_empty() {
        return Ok(Vec::new());
    }
    crate::limits::within(
        visual.len(),
        "the number of camera features",
        crate::limits::MAX_CAMERAS,
    )?;

    // The backbone the config names has to be one this port builds, and it has to be
    // buildable from nothing. Both are checked before any shape is narrowed so that
    // the first failure a user sees is the structural one.
    stage_blocks(&config.vision_backbone)?;
    if let Some(weights) = &config.pretrained_backbone_weights {
        return Err(TrainError::unsupported(format!(
            "pretrained_backbone_weights = {weights:?}; that names a torchvision checkpoint \
             downloaded from download.pytorch.org, and nothing in this workspace ships or \
             fetches one. The supported initialization is torchvision's own \
             kaiming_normal_(mode=\"fan_out\") drawn from Rerobot's stream; ask for it by \
             setting pretrained_backbone_weights to null rather than training a randomly \
             initialized backbone under a config that claims ImageNet weights"
        )));
    }
    let dilated = match &config.replace_final_stride_with_dilation {
        PythonIntBool::Bool(value) => *value,
        PythonIntBool::Int(value) => value != &num_bigint::BigInt::from(0_u8),
    };
    if dilated {
        return Err(TrainError::unsupported(
            "replace_final_stride_with_dilation is set; upstream cannot honour it either on \
             the BasicBlock ResNets, because torchvision's BasicBlock.__init__ raises \
             \"Dilation > 1 not supported in BasicBlock\" and _make_layer hands dilation=2 to \
             every block of layer4 after the first"
                .to_owned(),
        ));
    }

    let mut cameras = Vec::with_capacity(visual.len());
    for (key, feature) in visual {
        if feature.shape.len() != 3 {
            return Err(TrainError::Metadata(format!(
                "camera feature {key:?} declares shape {:?}; a camera is \
                 [channels, height, width]",
                feature.shape
            )));
        }
        let extent = |index: usize, name: &str, limit: usize| -> Result<usize> {
            let value = crate::limits::bounded_usize(
                &feature.shape[index],
                &format!("the {name} of camera feature {key:?}"),
                limit,
            )?;
            if value == 0 {
                return Err(TrainError::Metadata(format!(
                    "camera feature {key:?} declares a {name} of zero, so it carries no pixels"
                )));
            }
            Ok(value)
        };
        let channels = extent(0, "channel count", crate::limits::MAX_IMAGE_EXTENT)?;
        if channels != CAMERA_CHANNELS {
            return Err(TrainError::unsupported(format!(
                "camera feature {key:?} declares {channels} channels; torchvision's ResNet stem \
                 is nn.Conv2d({CAMERA_CHANNELS}, 64, kernel_size=7), so ACT takes \
                 {CAMERA_CHANNELS}-channel images and nothing reshapes them"
            )));
        }
        let height = extent(1, "height", crate::limits::MAX_IMAGE_EXTENT)?;
        let width = extent(2, "width", crate::limits::MAX_IMAGE_EXTENT)?;
        crate::limits::within(
            crate::limits::checked_product(
                &[channels, height, width],
                &format!("the size of one frame of camera feature {key:?}"),
            )?,
            &format!("the size of one frame of camera feature {key:?}"),
            crate::limits::MAX_FEATURE_WIDTH,
        )?;
        cameras.push(CameraSpec {
            key: key.clone(),
            channels,
            height,
            width,
        });
    }
    Ok(cameras)
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

/// `nn.Conv2d(in_channels, out_channels, kernel_size=1)`, initialized the way
/// `nn.Conv2d.reset_parameters` does: both tensors `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`
/// with `fan_in = in_channels * 1 * 1`, which is the same bound `nn.Linear` uses and
/// the same helper can therefore produce.
fn conv_1x1(
    store: &mut ParameterStore,
    init: &mut Initializer<'_>,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
) -> Result<Conv2d> {
    let (weight, bias) = init.linear(out_channels, in_channels)?;
    let weight = weight.reshape((out_channels, in_channels, 1, 1))?;
    Ok(Conv2d {
        weight: store.parameter(format!("{prefix}.weight"), weight)?,
        bias: Some(store.parameter(format!("{prefix}.bias"), bias)?),
        stride: 1,
        padding: 0,
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
