//! Behaviour tests for the ACT tensor model, derived from
//! `lerobot/policies/act/modeling_act.py` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1.
//!
//! Two kinds of claim are made here. The first is *structural*: the parameter
//! names and shapes are the ones `ACTPolicy.state_dict()` produces, the tensor
//! shapes through the forward pass are upstream's, and the loss is masked the way
//! upstream masks it. Those are checkable against upstream's source, and they are
//! what makes a checkpoint written here an ACT checkpoint.
//!
//! The second is *behavioural at a fixed latent*: with the reparameterization
//! noise supplied rather than drawn and dropout off, the forward pass is a pure
//! function of the weights, so it is deterministic and reproducible. That is the
//! configuration `tools/goldens/make_act_goldens.py` compares against PyTorch.

mod common;

use candle_core::{DType, Device, Tensor};
use common::{fixture_dataset, reduced_config, TempDir};
use rerobot_core::random::SplitMix64;
use rerobot_core::BigInt;
use rerobot_train::data::batch::collate;
use rerobot_train::data::dataset::StateOnlyDataset;
use rerobot_train::error::TrainError;
use rerobot_train::model::act::{ActModel, Pass, Randomness};
use rerobot_train::model::ops::{sinusoidal_position_embedding, Activation};

fn config() -> rerobot_core::policy::act::ActConfig {
    let dir = TempDir::new("model-config");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    let metadata = rerobot_train::data::meta::DatasetMetadata::load(&fixture_dataset()).unwrap();
    let (inputs, outputs) = metadata.policy_feature_split();
    config.policy.input_features = Some(inputs);
    config.policy.output_features = Some(outputs);
    config.policy
}

fn model() -> ActModel {
    let mut rng = SplitMix64::new(1234);
    ActModel::new(&config(), &Device::Cpu, &mut rng).expect("the reduced config builds")
}

fn batch(size: usize) -> rerobot_train::data::batch::Batch {
    let windows = indexmap::IndexMap::from([(
        "action".to_owned(),
        rerobot_core::dataset::delta::action_delta_timestamps(2, 10),
    )]);
    let dataset = StateOnlyDataset::load(&fixture_dataset(), &windows, 1e-4).unwrap();
    let frames: Vec<_> = (0..size).map(|index| dataset.get(index).unwrap()).collect();
    collate(&frames, &Device::Cpu).unwrap()
}

fn fixed_noise(batch_size: usize, latent_dim: usize) -> Tensor {
    // A deterministic, non-degenerate stand-in for `torch.randn_like(mu)`.
    let values: Vec<f32> = (0..batch_size * latent_dim)
        .map(|index| ((index % 7) as f32 - 3.0) / 4.0)
        .collect();
    Tensor::from_vec(values, (batch_size, latent_dim), &Device::Cpu).unwrap()
}

// ---------------------------------------------------------------------------
// Structure: the parameter set is upstream's
// ---------------------------------------------------------------------------

#[test]
fn the_state_dict_keys_are_the_ones_upstream_writes() {
    let model = model();
    let keys: Vec<String> = model.state_dict().unwrap().keys().cloned().collect();
    let expected = [
        "model.action_head.bias",
        "model.action_head.weight",
        "model.decoder.layers.0.linear1.bias",
        "model.decoder.layers.0.linear1.weight",
        "model.decoder.layers.0.linear2.bias",
        "model.decoder.layers.0.linear2.weight",
        "model.decoder.layers.0.multihead_attn.in_proj_bias",
        "model.decoder.layers.0.multihead_attn.in_proj_weight",
        "model.decoder.layers.0.multihead_attn.out_proj.bias",
        "model.decoder.layers.0.multihead_attn.out_proj.weight",
        "model.decoder.layers.0.norm1.bias",
        "model.decoder.layers.0.norm1.weight",
        "model.decoder.layers.0.norm2.bias",
        "model.decoder.layers.0.norm2.weight",
        "model.decoder.layers.0.norm3.bias",
        "model.decoder.layers.0.norm3.weight",
        "model.decoder.layers.0.self_attn.in_proj_bias",
        "model.decoder.layers.0.self_attn.in_proj_weight",
        "model.decoder.layers.0.self_attn.out_proj.bias",
        "model.decoder.layers.0.self_attn.out_proj.weight",
        "model.decoder.norm.bias",
        "model.decoder.norm.weight",
        "model.decoder_pos_embed.weight",
        "model.encoder.layers.0.linear1.bias",
        "model.encoder.layers.0.linear1.weight",
        "model.encoder.layers.0.linear2.bias",
        "model.encoder.layers.0.linear2.weight",
        "model.encoder.layers.0.norm1.bias",
        "model.encoder.layers.0.norm1.weight",
        "model.encoder.layers.0.norm2.bias",
        "model.encoder.layers.0.norm2.weight",
        "model.encoder.layers.0.self_attn.in_proj_bias",
        "model.encoder.layers.0.self_attn.in_proj_weight",
        "model.encoder.layers.0.self_attn.out_proj.bias",
        "model.encoder.layers.0.self_attn.out_proj.weight",
        "model.encoder_1d_feature_pos_embed.weight",
        "model.encoder_env_state_input_proj.bias",
        "model.encoder_env_state_input_proj.weight",
        "model.encoder_latent_input_proj.bias",
        "model.encoder_latent_input_proj.weight",
        "model.encoder_robot_state_input_proj.bias",
        "model.encoder_robot_state_input_proj.weight",
        "model.vae_encoder.layers.0.linear1.bias",
        "model.vae_encoder.layers.0.linear1.weight",
        "model.vae_encoder.layers.0.linear2.bias",
        "model.vae_encoder.layers.0.linear2.weight",
        "model.vae_encoder.layers.0.norm1.bias",
        "model.vae_encoder.layers.0.norm1.weight",
        "model.vae_encoder.layers.0.norm2.bias",
        "model.vae_encoder.layers.0.norm2.weight",
        "model.vae_encoder.layers.0.self_attn.in_proj_bias",
        "model.vae_encoder.layers.0.self_attn.in_proj_weight",
        "model.vae_encoder.layers.0.self_attn.out_proj.bias",
        "model.vae_encoder.layers.0.self_attn.out_proj.weight",
        "model.vae_encoder_action_input_proj.bias",
        "model.vae_encoder_action_input_proj.weight",
        "model.vae_encoder_cls_embed.weight",
        "model.vae_encoder_latent_output_proj.bias",
        "model.vae_encoder_latent_output_proj.weight",
        "model.vae_encoder_pos_enc",
        "model.vae_encoder_robot_state_input_proj.bias",
        "model.vae_encoder_robot_state_input_proj.weight",
    ];
    assert_eq!(keys, expected);
}

#[test]
fn every_parameter_has_the_shape_torch_would_give_it() {
    let model = model();
    let state = model.state_dict().unwrap();
    let shape = |name: &str| state[name].dims().to_vec();
    // dim_model 32, dim_feedforward 64, latent_dim 8, chunk_size 2, widths 2.
    assert_eq!(shape("model.vae_encoder_cls_embed.weight"), vec![1, 32]);
    assert_eq!(
        shape("model.vae_encoder_robot_state_input_proj.weight"),
        vec![32, 2]
    );
    assert_eq!(
        shape("model.vae_encoder_action_input_proj.weight"),
        vec![32, 2]
    );
    // `latent_dim * 2`: the block holds the mean and 2log(sigma) side by side.
    assert_eq!(
        shape("model.vae_encoder_latent_output_proj.weight"),
        vec![16, 32]
    );
    // `1 + chunk_size + 1` tokens, with the leading batch axis upstream unsqueezes.
    assert_eq!(shape("model.vae_encoder_pos_enc"), vec![1, 4, 32]);
    // `nn.MultiheadAttention` packs q, k and v into one `[3 * dim, dim]` tensor.
    assert_eq!(
        shape("model.encoder.layers.0.self_attn.in_proj_weight"),
        vec![96, 32]
    );
    assert_eq!(
        shape("model.encoder.layers.0.self_attn.in_proj_bias"),
        vec![96]
    );
    assert_eq!(shape("model.encoder.layers.0.linear1.weight"), vec![64, 32]);
    assert_eq!(shape("model.encoder.layers.0.linear2.weight"), vec![32, 64]);
    assert_eq!(shape("model.encoder_latent_input_proj.weight"), vec![32, 8]);
    // Three 1-D tokens: latent, robot state, env state.
    assert_eq!(
        shape("model.encoder_1d_feature_pos_embed.weight"),
        vec![3, 32]
    );
    // One learned object query per predicted action.
    assert_eq!(shape("model.decoder_pos_embed.weight"), vec![2, 32]);
    assert_eq!(shape("model.action_head.weight"), vec![2, 32]);
}

#[test]
fn there_is_no_backbone_because_the_config_has_no_images() {
    let model = model();
    assert!(
        model
            .state_dict()
            .unwrap()
            .keys()
            .all(|name| !name.contains("backbone")),
        "a state-only config must not build a ResNet"
    );
    let [main, backbone] = model.optimizer_parameter_groups();
    assert_eq!(main.len(), model.parameters().len());
    assert!(
        backbone.is_empty(),
        "the backbone group exists but is empty, as upstream's get_optim_params reports it"
    );
}

#[test]
fn the_narrowed_shape_reports_the_configured_dimensions() {
    let model = model();
    let shape = model.shape();
    assert_eq!(shape.dim_model, 32);
    assert_eq!(shape.n_heads, 4);
    assert_eq!(shape.dim_feedforward, 64);
    assert_eq!(shape.latent_dim, 8);
    assert_eq!(shape.chunk_size, 2);
    assert_eq!(shape.robot_state_dim, Some(2));
    assert_eq!(shape.env_state_dim, Some(2));
    assert_eq!(shape.action_dim, 2);
}

#[test]
fn the_parameter_count_is_the_sum_of_the_parameter_shapes() {
    let model = model();
    let counted: usize = model
        .parameters()
        .iter()
        .map(|parameter| parameter.value.elem_count())
        .sum();
    assert_eq!(model.num_parameters(), counted);
    assert!(counted > 0);
}

// ---------------------------------------------------------------------------
// The forward pass
// ---------------------------------------------------------------------------

#[test]
fn a_training_pass_returns_the_action_chunk_and_the_latent_parameters() {
    let model = model();
    let batch = batch(2);
    let noise = fixed_noise(2, model.shape().latent_dim);
    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(noise)))
        .expect("the forward pass runs");
    assert_eq!(output.actions.dims(), &[2, 2, 2]);
    assert_eq!(output.mu.as_ref().unwrap().dims(), &[2, 8]);
    assert_eq!(output.log_sigma_x2.as_ref().unwrap().dims(), &[2, 8]);
    let values = output
        .actions
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "the predicted actions are not all finite: {values:?}"
    );
}

#[test]
fn an_eval_pass_zeroes_the_latent_and_reports_no_distribution() {
    let model = model();
    let batch = batch(2);
    let output = model.forward(&batch, Pass::Eval).expect("eval runs");
    assert_eq!(output.actions.dims(), &[2, 2, 2]);
    assert!(output.mu.is_none());
    assert!(output.log_sigma_x2.is_none());
}

#[test]
fn a_fixed_latent_makes_the_forward_pass_deterministic() {
    let model = model();
    let batch = batch(2);
    let noise = fixed_noise(2, 8);
    let first = model
        .forward(&batch, Pass::Train(Randomness::Fixed(noise.clone())))
        .unwrap();
    let second = model
        .forward(&batch, Pass::Train(Randomness::Fixed(noise)))
        .unwrap();
    let left = first
        .actions
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let right = second
        .actions
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(left, right);
}

#[test]
fn the_latent_sample_is_mu_plus_sigma_times_the_supplied_noise() {
    // The reparameterization trick, checked directly: a zero noise tensor must
    // collapse the sample onto the mean, so a run with zero noise and a run with
    // non-zero noise must differ.
    let model = model();
    let batch = batch(2);
    let zeros = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();
    let with_zero = model
        .forward(&batch, Pass::Train(Randomness::Fixed(zeros)))
        .unwrap();
    let with_noise = model
        .forward(&batch, Pass::Train(Randomness::Fixed(fixed_noise(2, 8))))
        .unwrap();
    let left = with_zero
        .actions
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let right = with_noise
        .actions
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_ne!(left, right, "the supplied noise did not reach the latent");
    // ... and the distribution parameters themselves do not depend on the noise.
    let mu_left = with_zero
        .mu
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mu_right = with_noise
        .mu
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(mu_left, mu_right);
}

#[test]
fn a_seeded_pass_is_reproducible_and_a_different_seed_is_not() {
    let model = model();
    let batch = batch(2);
    let run = |seed: u64| {
        let mut rng = SplitMix64::new(seed);
        model
            .forward(&batch, Pass::Train(Randomness::Seeded(&mut rng)))
            .unwrap()
            .actions
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5), run(6));
}

#[test]
fn predicting_action_steps_slices_the_chunk_to_n_action_steps() {
    let dir = TempDir::new("n-action-steps");
    let mut train_config = reduced_config(fixture_dataset(), dir.child("out"));
    train_config.policy.n_action_steps = BigInt::from(1);
    let metadata = rerobot_train::data::meta::DatasetMetadata::load(&fixture_dataset()).unwrap();
    let (inputs, outputs) = metadata.policy_feature_split();
    train_config.policy.input_features = Some(inputs);
    train_config.policy.output_features = Some(outputs);

    let mut rng = SplitMix64::new(1);
    let model = ActModel::new(&train_config.policy, &Device::Cpu, &mut rng).unwrap();
    let steps = model.predict_action_steps(&batch(2)).unwrap();
    assert_eq!(steps.dims(), &[2, 1, 2]);
}

#[test]
fn a_training_pass_without_actions_is_the_upstream_assertion() {
    let model = model();
    let mut batch = batch(2);
    batch.features.shift_remove("action");
    let error = model
        .forward(&batch, Pass::Train(Randomness::Fixed(fixed_noise(2, 8))))
        .unwrap_err();
    assert!(
        error.to_string().contains(
            "actions must be provided when using the variational objective in training mode."
        ),
        "message drifted from upstream: {error}"
    );
}

#[test]
fn latent_noise_of_the_wrong_shape_is_refused() {
    let model = model();
    let batch = batch(2);
    let wrong = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();
    let error = model
        .forward(&batch, Pass::Train(Randomness::Fixed(wrong)))
        .unwrap_err();
    assert!(
        error.to_string().contains("latent noise"),
        "unexpected error: {error}"
    );
}

// ---------------------------------------------------------------------------
// The loss
// ---------------------------------------------------------------------------

#[test]
fn the_loss_is_the_masked_l1_plus_the_weighted_kl_divergence() {
    let model = model();
    let batch = batch(2);
    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(fixed_noise(2, 8))))
        .unwrap();
    let loss = model.loss(&batch, &output).unwrap();
    assert!(loss.l1_loss.is_finite() && loss.l1_loss >= 0.0);
    let kld = loss.kld_loss.expect("the VAE is on, so there is a KL term");
    assert!(kld.is_finite());
    // kl_weight defaults to 10.0.
    let expected = loss.l1_loss + 10.0 * kld;
    assert!(
        (loss.total - expected).abs() < 1e-5,
        "total {} is not l1 {} + 10 * kld {}",
        loss.total,
        loss.l1_loss,
        kld
    );
    assert_eq!(
        loss.loss.dims().len(),
        0,
        "the loss must be a scalar to be backpropagated"
    );
}

#[test]
fn padded_actions_are_excluded_from_the_l1_average() {
    // Frame 3's chunk is `[real, padded]`. Averaging over the padded entry too
    // would change the number, so the two must differ.
    let model = model();
    let windows = indexmap::IndexMap::from([(
        "action".to_owned(),
        rerobot_core::dataset::delta::action_delta_timestamps(2, 10),
    )]);
    let dataset = StateOnlyDataset::load(&fixture_dataset(), &windows, 1e-4).unwrap();
    let padded_frame = dataset.get(3).unwrap();
    assert_eq!(padded_frame.is_pad("action"), Some(&[false, true][..]));

    let batch = collate(&[padded_frame], &Device::Cpu).unwrap();
    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(fixed_noise(1, 8))))
        .unwrap();
    let loss = model.loss(&batch, &output).unwrap();

    // Recompute by hand over the unpadded entry only.
    let actions = output
        .actions
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let target = batch
        .feature("action")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let manual: f32 = (0..2)
        .map(|axis| (target[axis] - actions[axis]).abs())
        .sum::<f32>()
        / 2.0;
    assert!(
        (loss.l1_loss - f64::from(manual)).abs() < 1e-5,
        "l1 {} is not the mean over the two unpadded scalars {manual}",
        loss.l1_loss
    );
}

#[test]
fn an_entirely_padded_chunk_divides_by_one_rather_than_by_zero() {
    let model = model();
    let mut batch = batch(1);
    let ones = Tensor::ones((1, 2), DType::U8, &Device::Cpu).unwrap();
    batch.padding.insert("action".to_owned(), ones);
    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(fixed_noise(1, 8))))
        .unwrap();
    let loss = model.loss(&batch, &output).unwrap();
    assert_eq!(
        loss.l1_loss, 0.0,
        "every entry is masked, so the sum is zero"
    );
    assert!(loss.total.is_finite());
}

#[test]
fn an_eval_pass_has_no_kl_term() {
    let model = model();
    let batch = batch(2);
    let output = model.forward(&batch, Pass::Eval).unwrap();
    let loss = model.loss(&batch, &output).unwrap();
    assert!(loss.kld_loss.is_none());
    assert_eq!(loss.total, loss.l1_loss);
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn an_image_feature_in_the_policy_config_is_refused_with_a_reason() {
    let mut policy = config();
    policy.input_features.as_mut().unwrap().insert(
        "observation.images.top".to_owned(),
        rerobot_core::types::PolicyFeature::new(
            rerobot_core::types::FeatureType::Visual,
            [BigInt::from(3), BigInt::from(96), BigInt::from(96)],
        ),
    );
    let mut rng = SplitMix64::new(0);
    let error = ActModel::new(&policy, &Device::Cpu, &mut rng).unwrap_err();
    assert!(matches!(error, TrainError::Unsupported(_)));
    assert!(
        error.to_string().contains("ResNet backbone"),
        "the refusal does not say what is missing: {error}"
    );
}

#[test]
fn temporal_ensembling_is_refused_rather_than_ignored() {
    let mut policy = config();
    policy.temporal_ensemble_coeff = Some(0.01);
    policy.n_action_steps = BigInt::from(1);
    let mut rng = SplitMix64::new(0);
    let error = ActModel::new(&policy, &Device::Cpu, &mut rng).unwrap_err();
    assert!(matches!(error, TrainError::Unsupported(_)));
    assert!(error.to_string().contains("temporal ensembler"));
}

#[test]
fn a_dim_model_that_is_not_a_multiple_of_the_head_count_is_refused() {
    let mut policy = config();
    policy.dim_model = BigInt::from(33);
    let mut rng = SplitMix64::new(0);
    let error = ActModel::new(&policy, &Device::Cpu, &mut rng).unwrap_err();
    assert!(
        error.to_string().contains("multiple of n_heads"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_config_with_no_action_output_is_refused() {
    let mut policy = config();
    policy.output_features = Some(indexmap::IndexMap::new());
    let mut rng = SplitMix64::new(0);
    let error = ActModel::new(&policy, &Device::Cpu, &mut rng).unwrap_err();
    assert!(
        error.to_string().contains("nothing to predict"),
        "unexpected error: {error}"
    );
}

// ---------------------------------------------------------------------------
// The operators
// ---------------------------------------------------------------------------

#[test]
fn the_sinusoidal_table_is_the_attention_is_all_you_need_one() {
    let table = sinusoidal_position_embedding(3, 4, &Device::Cpu).unwrap();
    assert_eq!(table.dims(), &[3, 4]);
    let values = table.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // Position 0: sin(0) = 0 on even indices, cos(0) = 1 on odd ones.
    assert_eq!(&values[0..4], &[0.0, 1.0, 0.0, 1.0]);
    // Position 1, index 0: sin(1 / 10000^0) = sin(1).
    assert!((values[4] - 1.0f32.sin()).abs() < 1e-6);
    // Position 1, index 1: cos(1 / 10000^0) = cos(1).
    assert!((values[5] - 1.0f32.cos()).abs() < 1e-6);
    // Position 1, index 2: sin(1 / 10000^(2/4)) = sin(0.01).
    assert!((values[6] - 0.01f32.sin()).abs() < 1e-6);
}

#[test]
fn the_three_upstream_activations_are_accepted_and_others_are_not() {
    for name in ["relu", "gelu", "glu"] {
        assert!(Activation::parse(name).is_ok(), "{name} should be accepted");
    }
    let error = Activation::parse("silu").unwrap_err();
    assert_eq!(
        error.to_string(),
        "activation should be relu/gelu/glu, not silu."
    );
}

#[test]
fn glu_halves_the_last_axis_like_torch() {
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu).unwrap();
    let output = Activation::Glu.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 2]);
    let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    assert!((values[0] - 1.0 * sigmoid(3.0)).abs() < 1e-6);
    assert!((values[1] - 2.0 * sigmoid(4.0)).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// safetensors round trip
// ---------------------------------------------------------------------------

#[test]
fn a_saved_model_reloads_bit_for_bit() {
    let dir = TempDir::new("roundtrip");
    let path = dir.child("model.safetensors");
    let model = model();
    model.save(&path).unwrap();

    let mut reloaded = {
        let mut rng = SplitMix64::new(999);
        ActModel::new(&config(), &Device::Cpu, &mut rng).unwrap()
    };
    // Different seed, so the two disagree before the load.
    let distance_before = rerobot_train::optim::state_dict_distance(
        &model.state_dict().unwrap(),
        &reloaded.state_dict().unwrap(),
    )
    .unwrap();
    assert!(
        distance_before > 0.0,
        "the two models must differ before the load for this test to mean anything"
    );

    reloaded.load(&path).unwrap();
    let distance_after = rerobot_train::optim::state_dict_distance(
        &model.state_dict().unwrap(),
        &reloaded.state_dict().unwrap(),
    )
    .unwrap();
    assert_eq!(distance_after, 0.0);
}

#[test]
fn a_reloaded_model_predicts_exactly_what_the_saved_one_did() {
    let dir = TempDir::new("roundtrip-forward");
    let path = dir.child("model.safetensors");
    let model = model();
    model.save(&path).unwrap();
    let batch = batch(2);
    let expected = model
        .predict_action_steps(&batch)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut reloaded = {
        let mut rng = SplitMix64::new(4321);
        ActModel::new(&config(), &Device::Cpu, &mut rng).unwrap()
    };
    reloaded.load(&path).unwrap();
    let actual = reloaded
        .predict_action_steps(&batch)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn a_checkpoint_missing_a_tensor_is_refused_by_name() {
    let dir = TempDir::new("missing-tensor");
    let path = dir.child("partial.safetensors");
    let model = model();
    let mut tensors: std::collections::HashMap<String, Tensor> = model
        .state_dict()
        .unwrap()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.contiguous().unwrap()))
        .collect();
    tensors.remove("model.action_head.weight");
    candle_core::safetensors::save(&tensors, &path).unwrap();

    let mut target = model;
    let error = target.load(&path).unwrap_err();
    assert!(
        error.to_string().contains("model.action_head.weight"),
        "the refusal does not name the missing tensor: {error}"
    );
}

#[test]
fn a_checkpoint_with_an_extra_tensor_is_refused_rather_than_partly_loaded() {
    let dir = TempDir::new("extra-tensor");
    let path = dir.child("extra.safetensors");
    let model = model();
    let mut tensors: std::collections::HashMap<String, Tensor> = model
        .state_dict()
        .unwrap()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.contiguous().unwrap()))
        .collect();
    tensors.insert(
        "model.backbone.conv1.weight".to_owned(),
        Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&tensors, &path).unwrap();

    let mut target = model;
    let error = target.load(&path).unwrap_err();
    assert!(
        error.to_string().contains("model.backbone.conv1.weight"),
        "the refusal does not name the extra tensor: {error}"
    );
}

#[test]
fn a_checkpoint_with_a_mismatched_shape_is_refused_with_both_shapes() {
    let dir = TempDir::new("bad-shape");
    let path = dir.child("bad.safetensors");
    let model = model();
    let mut tensors: std::collections::HashMap<String, Tensor> = model
        .state_dict()
        .unwrap()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.contiguous().unwrap()))
        .collect();
    tensors.insert(
        "model.action_head.weight".to_owned(),
        Tensor::zeros((9, 9), DType::F32, &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&tensors, &path).unwrap();

    let mut target = model;
    let error = target.load(&path).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("[9, 9]"), "unexpected error: {message}");
    assert!(message.contains("[2, 32]"), "unexpected error: {message}");
}

// ---------------------------------------------------------------------------
// Differentiability
// ---------------------------------------------------------------------------

#[test]
fn attention_logits_receive_gradients() {
    // Regression test for a real defect. `candle_nn::ops::softmax_last_dim` is a
    // fused custom op whose backward pass does not reach its input, so building
    // attention on it leaves every logit without a gradient. The forward pass and
    // the loss look perfectly healthy; what breaks is that the query and key
    // projections and both position embeddings never train, because their only
    // path to the loss is through the softmax.
    //
    // The two position embeddings are the sharpest probe: they feed *only* the
    // logits, never a value, so if the softmax does not propagate they get no
    // gradient at all rather than a diminished one.
    let model = model();
    let batch = batch(2);
    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(fixed_noise(2, 8))))
        .unwrap();
    let loss = model.loss(&batch, &output).unwrap();
    let gradients = loss.loss.backward().unwrap();

    for name in [
        "model.encoder_1d_feature_pos_embed.weight",
        "model.decoder_pos_embed.weight",
    ] {
        let parameter = model
            .parameters()
            .iter()
            .find(|parameter| parameter.name == name)
            .expect("the parameter exists");
        let gradient = gradients
            .get(parameter.value.as_tensor())
            .unwrap_or_else(|| panic!("{name} received no gradient at all"));
        let magnitude = gradient
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            magnitude > 0.0,
            "{name} received an all-zero gradient, so it cannot train"
        );
    }
}

#[test]
fn every_parameter_of_the_model_is_reachable_from_the_loss() {
    let model = model();
    let batch = batch(2);
    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(fixed_noise(2, 8))))
        .unwrap();
    let loss = model.loss(&batch, &output).unwrap();
    let gradients = loss.loss.backward().unwrap();
    let missing: Vec<&str> = model
        .parameters()
        .iter()
        .filter(|parameter| gradients.get(parameter.value.as_tensor()).is_none())
        .map(|parameter| parameter.name.as_str())
        .collect();
    assert!(
        missing.is_empty(),
        "these parameters are not connected to the loss: {missing:?}"
    );
}
