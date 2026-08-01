//! Behaviour tests for ACT's camera path, derived from
//! `lerobot/policies/act/modeling_act.py` and `torchvision/models/resnet.py` at the
//! pinned upstream commit.
//!
//! The claims here are structural and behavioural, not numeric: there is no PyTorch
//! oracle for this path the way `tests/goldens.rs` has one for the state-only model,
//! because upstream's backbone is initialized from a torchvision download that this
//! workspace does not ship. What *is* pinned is the architecture — the parameter
//! names and shapes `ACTPolicy.state_dict()` produces, the token count the encoder
//! sees, the order the cameras are consumed in — and the properties a training run
//! depends on: the backbone receives a gradient, the optimizer moves it, and a
//! checkpoint of it reloads into the same predictions.
//!
//! The images are 32×32, which `resnet18` reduces to a 1×1 feature map, and the
//! transformer is the reduced configuration the rest of the suite uses. The backbone
//! itself is not reducible — `resnet18` is a fixed architecture — so these tests
//! carry its 11 M parameters and are the slowest in the crate.

mod common;

use candle_core::{DType, Device, Tensor};
use common::{fixture_dataset, reduced_config, TempDir};
use indexmap::IndexMap;
use rerobot_core::policy::act::ActConfig;
use rerobot_core::random::SplitMix64;
use rerobot_core::types::{FeatureType, PolicyFeature};
use rerobot_core::BigInt;
use rerobot_train::data::batch::{collate, Batch};
use rerobot_train::data::dataset::StateOnlyDataset;
use rerobot_train::data::image::CameraNormalization;
use rerobot_train::data::meta::DatasetMetadata;
use rerobot_train::error::TrainError;
use rerobot_train::model::act::{ActModel, Pass, Randomness};
use rerobot_train::optim::clip_grad_norm;
use rerobot_train::run::TrainSession;

/// The side of every test image. `resnet18` divides by 32, so this is the smallest
/// extent that still runs the whole stem and all four stages.
const EXTENT: usize = 32;

const TOP: &str = "observation.images.top";
const WRIST: &str = "observation.images.wrist";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The reduced ACT config with `cameras` added as visual input features.
fn policy(cameras: &[&str]) -> ActConfig {
    let dir = TempDir::new("image-config");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    let metadata = DatasetMetadata::load(&fixture_dataset()).unwrap();
    let (mut inputs, outputs) = metadata.policy_feature_split();
    for key in cameras {
        inputs.insert(
            (*key).to_owned(),
            PolicyFeature::new(
                FeatureType::Visual,
                [
                    BigInt::from(3),
                    BigInt::from(EXTENT as i64),
                    BigInt::from(EXTENT as i64),
                ],
            ),
        );
    }
    config.policy.input_features = Some(inputs);
    config.policy.output_features = Some(outputs);
    config.policy
}

fn model(cameras: &[&str]) -> ActModel {
    let mut rng = SplitMix64::new(1234);
    ActModel::new(&policy(cameras), &Device::Cpu, &mut rng).expect("the camera config builds")
}

/// The state and action half of a batch, straight from the committed fixture.
fn state_batch(size: usize) -> Batch {
    let windows = IndexMap::from([(
        "action".to_owned(),
        rerobot_core::dataset::delta::action_delta_timestamps(2, 10),
    )]);
    let dataset = StateOnlyDataset::load(&fixture_dataset(), &windows, 1e-4).unwrap();
    let frames: Vec<_> = (0..size).map(|index| dataset.get(index).unwrap()).collect();
    collate(&frames, &Device::Cpu).unwrap()
}

/// A deterministic `[0, 1]` ramp, distinct per camera and per frame so that a test
/// asserting "this camera changed the answer" cannot be satisfied by a constant.
fn image(key: &str, size: usize, extent: usize) -> Tensor {
    let salt = key.bytes().map(usize::from).sum::<usize>() % 17;
    let count = size * 3 * extent * extent;
    let values: Vec<f32> = (0..count)
        .map(|index| ((index * 7 + salt * 13) % 251) as f32 / 250.0)
        .collect();
    Tensor::from_vec(values, (size, 3, extent, extent), &Device::Cpu).unwrap()
}

fn camera_map(keys: &[&str], size: usize, extent: usize) -> IndexMap<String, Tensor> {
    keys.iter()
        .map(|key| ((*key).to_owned(), image(key, size, extent)))
        .collect()
}

/// A full batch: fixture states and actions, plus one image per camera.
fn batch(cameras: &[&str], size: usize) -> Batch {
    state_batch(size)
        .with_images(
            &camera_map(cameras, size, EXTENT),
            &CameraNormalization::imagenet(),
        )
        .expect("the camera tensors satisfy the contract")
}

// ---------------------------------------------------------------------------
// Structure: the backbone is upstream's, under upstream's names
// ---------------------------------------------------------------------------

#[test]
fn a_camera_config_builds_the_resnet_under_upstreams_state_dict_names() {
    let state = model(&[TOP]).state_dict().unwrap();
    let shape = |name: &str| state[name].dims().to_vec();

    // The stem: `nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)`
    // followed by a FrozenBatchNorm2d whose four tensors are buffers.
    assert_eq!(shape("model.backbone.conv1.weight"), vec![64, 3, 7, 7]);
    for statistic in ["weight", "bias", "running_mean", "running_var"] {
        assert_eq!(
            shape(&format!("model.backbone.bn1.{statistic}")),
            vec![64],
            "the frozen normalization must carry {statistic} as a buffer"
        );
    }
    assert!(
        !state.contains_key("model.backbone.bn1.num_batches_tracked"),
        "FrozenBatchNorm2d has no num_batches_tracked; a real BatchNorm2d does"
    );

    // resnet18 is [2, 2, 2, 2] BasicBlocks over widths 64, 128, 256, 512, and only
    // the stages that change shape carry a 1x1 downsample.
    assert_eq!(
        shape("model.backbone.layer1.1.conv2.weight"),
        vec![64, 64, 3, 3]
    );
    assert_eq!(
        shape("model.backbone.layer2.0.conv1.weight"),
        vec![128, 64, 3, 3]
    );
    assert_eq!(
        shape("model.backbone.layer2.0.downsample.0.weight"),
        vec![128, 64, 1, 1]
    );
    assert_eq!(
        shape("model.backbone.layer2.0.downsample.1.weight"),
        vec![128]
    );
    assert!(
        !state.contains_key("model.backbone.layer1.0.downsample.0.weight"),
        "layer1 keeps its shape, so torchvision builds it without a downsample"
    );
    assert_eq!(
        shape("model.backbone.layer4.1.conv2.weight"),
        vec![512, 512, 3, 3]
    );
    assert!(
        !state.contains_key("model.backbone.layer5.0.conv1.weight"),
        "a BasicBlock ResNet has four stages"
    );
    // `IntermediateLayerGetter(..., {"layer4": "feature_map"})` drops everything
    // after layer4, so the classifier head is not part of the model at all.
    for dropped in ["model.backbone.fc.weight", "model.backbone.fc.bias"] {
        assert!(
            !state.contains_key(dropped),
            "the intermediate-layer getter drops {dropped}"
        );
    }

    // `nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)`.
    assert_eq!(
        shape("model.encoder_img_feat_input_proj.weight"),
        vec![32, 512, 1, 1]
    );
    assert_eq!(shape("model.encoder_img_feat_input_proj.bias"), vec![32]);

    // The 2-D camera embedding is `ACTSinusoidalPositionEmbedding2d`, which upstream
    // registers with neither parameters nor buffers.
    assert!(
        state
            .keys()
            .all(|name| !name.contains("encoder_cam_feat_pos_embed")),
        "the 2-D camera embedding is computed, not stored"
    );
    // And the 1-D table still covers only the latent, the state and the env state:
    // the cameras bring their own positions.
    assert_eq!(
        shape("model.encoder_1d_feature_pos_embed.weight"),
        vec![3, 32]
    );
}

#[test]
fn the_backbone_optimizer_group_holds_the_convolutions_and_nothing_else() {
    let model = model(&[TOP]);
    let parameters = model.parameters();
    let [main, backbone] = model.optimizer_parameter_groups();

    assert!(!backbone.is_empty(), "a camera config has a backbone");
    assert_eq!(main.len() + backbone.len(), parameters.len());
    for index in &backbone {
        let name = &parameters[*index].name;
        assert!(
            name.starts_with("model.backbone"),
            "{name} is not a backbone parameter"
        );
        assert!(
            name.ends_with("conv1.weight")
                || name.ends_with("conv2.weight")
                || name.ends_with("downsample.0.weight"),
            "{name} is trainable, but FrozenBatchNorm2d holds buffers only"
        );
    }
    // The image projection is outside the backbone group: upstream splits on the
    // `model.backbone` prefix alone, and this does not carry it.
    assert!(
        main.iter()
            .any(|index| parameters[*index].name == "model.encoder_img_feat_input_proj.weight"),
        "the image projection belongs to the main group"
    );
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

#[test]
fn one_camera_predicts_the_same_action_shape_as_the_state_only_model() {
    let model = model(&[TOP]);
    let batch = batch(&[TOP], 2);
    let noise = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();

    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(noise)))
        .unwrap();
    // `[batch, chunk_size, action_dim]`, exactly as without cameras.
    assert_eq!(output.actions.dims(), &[2, 2, 2]);
    assert_eq!(output.mu.as_ref().unwrap().dims(), &[2, 8]);

    // `[batch, n_action_steps, action_dim]`, the `select_action` slice of the chunk.
    let predicted = model.predict_action_steps(&batch).unwrap();
    assert_eq!(predicted.dims(), &[2, 2, 2]);
    assert!(predicted
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
}

#[test]
fn a_second_camera_adds_its_own_tokens_and_changes_the_prediction() {
    // Same seed, so the two models differ only by the camera the config declares.
    let one = model(&[TOP]);
    let two = model(&[TOP, WRIST]);

    assert_eq!(one.shape().cameras.len(), 1);
    assert_eq!(
        two.shape()
            .cameras
            .iter()
            .map(|camera| camera.key.as_str())
            .collect::<Vec<_>>(),
        vec![TOP, WRIST],
        "the cameras keep their input_features order"
    );

    let single = one
        .predict_action_steps(&batch(&[TOP], 2))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let double = two
        .predict_action_steps(&batch(&[TOP, WRIST], 2))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(single.len(), double.len());
    assert!(
        single
            .iter()
            .zip(&double)
            .any(|(left, right)| (left - right).abs() > 1e-6),
        "a second camera contributes tokens the decoder attends to, so it cannot leave \
         the prediction identical"
    );
}

#[test]
fn a_multi_camera_forward_pass_is_deterministic() {
    let model = model(&[TOP, WRIST]);
    let first = model
        .predict_action_steps(&batch(&[TOP, WRIST], 2))
        .unwrap();
    let second = model
        .predict_action_steps(&batch(&[TOP, WRIST], 2))
        .unwrap();
    let difference = (&first - &second)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(difference, 0.0, "an eval pass draws nothing random");
}

#[test]
fn the_cameras_are_consumed_in_config_order_rather_than_batch_order() {
    let model = model(&[TOP, WRIST]);
    let size = 2;
    let declared = model
        .predict_action_steps(&batch(&[TOP, WRIST], size))
        .unwrap();

    // The same two tensors, attached to the batch in the opposite order.
    let reversed_map: IndexMap<String, Tensor> = camera_map(&[WRIST, TOP], size, EXTENT);
    let reversed = state_batch(size)
        .with_images(&reversed_map, &CameraNormalization::imagenet())
        .unwrap();
    let reversed = model.predict_action_steps(&reversed).unwrap();

    let difference = (&declared - &reversed)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(
        difference, 0.0,
        "the model iterates config.image_features, so the batch's insertion order \
         cannot reorder the encoder's camera tokens"
    );
}

// ---------------------------------------------------------------------------
// The camera tensor contract
// ---------------------------------------------------------------------------

#[test]
fn an_observation_axis_of_one_step_is_squeezed_rather_than_refused() {
    let size = 2;
    let flat = image(TOP, size, EXTENT);
    let stepped = flat.reshape((size, 1, 3, EXTENT, EXTENT)).unwrap();
    let batch = state_batch(size)
        .with_images(
            &IndexMap::from([(TOP.to_owned(), stepped)]),
            &CameraNormalization::identity(),
        )
        .unwrap();
    assert_eq!(batch.image(TOP).unwrap().dims(), &[size, 3, EXTENT, EXTENT]);
}

#[test]
fn a_camera_tensor_of_the_wrong_dtype_rank_range_or_size_is_refused_by_name() {
    let size = 2;
    let attach = |tensor: Tensor| -> TrainError {
        state_batch(size)
            .with_images(
                &IndexMap::from([(TOP.to_owned(), tensor)]),
                &CameraNormalization::identity(),
            )
            .expect_err("the tensor must be refused")
    };

    let integral = Tensor::zeros((size, 3, EXTENT, EXTENT), DType::U8, &Device::Cpu).unwrap();
    let error = attach(integral);
    assert!(
        error.to_string().contains("dtype") && error.to_string().contains(TOP),
        "unexpected error: {error}"
    );

    let rank3 = Tensor::zeros((size, 3, EXTENT), DType::F32, &Device::Cpu).unwrap();
    let error = attach(rank3);
    assert!(
        error.to_string().contains("rank 3"),
        "unexpected error: {error}"
    );

    // A five-axis tensor whose observation axis is deeper than the one step ACT
    // fixes `n_obs_steps` at.
    let history = Tensor::zeros((size, 2, 3, EXTENT, EXTENT), DType::F32, &Device::Cpu).unwrap();
    let error = attach(history);
    assert!(
        error.to_string().contains("n_obs_steps"),
        "unexpected error: {error}"
    );

    let wrong_batch = image(TOP, size + 1, EXTENT);
    let error = attach(wrong_batch);
    assert!(
        error.to_string().contains("3 images") && error.to_string().contains("2 frames"),
        "unexpected error: {error}"
    );

    // Still in 0..255 rather than divided by it.
    let raw = (image(TOP, size, EXTENT) * 255.0).unwrap();
    let error = attach(raw);
    assert!(
        error.to_string().contains("outside [0, 1]"),
        "unexpected error: {error}"
    );

    // And a NaN, which the same range check catches.
    let poisoned = (image(TOP, size, EXTENT) * f64::NAN).unwrap();
    let error = attach(poisoned);
    assert!(
        error.to_string().contains("outside [0, 1]"),
        "unexpected error: {error}"
    );
}

#[test]
fn an_image_whose_extent_disagrees_with_the_config_is_refused_by_the_forward_pass() {
    let model = model(&[TOP]);
    let size = 2;
    let wrong = state_batch(size)
        .with_images(
            &camera_map(&[TOP], size, EXTENT * 2),
            &CameraNormalization::imagenet(),
        )
        .expect("a 64x64 image satisfies the batch's own contract");
    let error = model.predict_action_steps(&wrong).unwrap_err();
    assert!(
        error.to_string().contains("3x64x64") && error.to_string().contains("3x32x32"),
        "the refusal does not name both extents: {error}"
    );
}

#[test]
fn a_missing_camera_is_refused_rather_than_zero_filled() {
    let model = model(&[TOP]);
    let error = model.predict_action_steps(&state_batch(2)).unwrap_err();
    assert!(
        error.to_string().contains(TOP) && error.to_string().contains("with_images"),
        "the refusal does not say how to supply the camera: {error}"
    );
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

#[test]
fn pretrained_backbone_weights_are_refused_rather_than_silently_randomized() {
    let mut config = policy(&[TOP]);
    config.pretrained_backbone_weights = Some("ResNet18_Weights.IMAGENET1K_V1".to_owned());
    let mut rng = SplitMix64::new(0);
    let error = ActModel::new(&config, &Device::Cpu, &mut rng).unwrap_err();
    assert!(matches!(error, TrainError::Unsupported(_)));
    assert!(
        error.to_string().contains("download.pytorch.org")
            && error.to_string().contains("kaiming_normal_"),
        "the refusal does not name the missing artifact or the supported mode: {error}"
    );
}

#[test]
fn the_bottleneck_resnets_are_refused_by_name() {
    let mut config = policy(&[TOP]);
    config.vision_backbone = "resnet50".to_owned();
    let mut rng = SplitMix64::new(0);
    let error = ActModel::new(&config, &Device::Cpu, &mut rng).unwrap_err();
    assert!(matches!(error, TrainError::Unsupported(_)));
    assert!(
        error.to_string().contains("Bottleneck") && error.to_string().contains("2048"),
        "unexpected error: {error}"
    );

    // resnet34 is the other BasicBlock variant and does build.
    let mut config = policy(&[TOP]);
    config.vision_backbone = "resnet34".to_owned();
    let mut rng = SplitMix64::new(0);
    let deeper = ActModel::new(&config, &Device::Cpu, &mut rng).expect("resnet34 is ported");
    assert!(
        deeper
            .state_dict()
            .unwrap()
            .contains_key("model.backbone.layer3.5.conv1.weight"),
        "resnet34's layer3 holds six blocks"
    );
}

#[test]
fn dilation_is_refused_with_the_reason_upstream_would_raise() {
    let mut config = policy(&[TOP]);
    config.replace_final_stride_with_dilation =
        rerobot_core::policy::act::PythonIntBool::Bool(true);
    let mut rng = SplitMix64::new(0);
    let error = ActModel::new(&config, &Device::Cpu, &mut rng).unwrap_err();
    assert!(matches!(error, TrainError::Unsupported(_)));
    assert!(
        error
            .to_string()
            .contains("Dilation > 1 not supported in BasicBlock"),
        "the refusal does not quote torchvision's own message: {error}"
    );
}

#[test]
fn a_camera_that_is_not_three_channels_is_refused() {
    let dir = TempDir::new("channels");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    let metadata = DatasetMetadata::load(&fixture_dataset()).unwrap();
    let (mut inputs, outputs) = metadata.policy_feature_split();
    inputs.insert(
        TOP.to_owned(),
        PolicyFeature::new(
            FeatureType::Visual,
            [BigInt::from(1), BigInt::from(32), BigInt::from(32)],
        ),
    );
    config.policy.input_features = Some(inputs);
    config.policy.output_features = Some(outputs);

    let mut rng = SplitMix64::new(0);
    let error = ActModel::new(&config.policy, &Device::Cpu, &mut rng).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("nn.Conv2d(3, 64, kernel_size=7)"),
        "the refusal does not say why three channels: {error}"
    );
}

#[test]
fn a_dim_model_that_is_not_a_multiple_of_four_is_refused_when_a_camera_is_present() {
    let mut config = policy(&[TOP]);
    // Still a multiple of n_heads, so this can only fail on the embedding's own rule.
    config.dim_model = BigInt::from(30);
    config.n_heads = BigInt::from(2);
    let mut rng = SplitMix64::new(0);
    let error = ActModel::new(&config, &Device::Cpu, &mut rng).unwrap_err();
    assert!(
        error.to_string().contains("multiple of four"),
        "unexpected error: {error}"
    );

    // The same dimensions are fine without a camera, because nothing then needs the
    // 2-D embedding: the state-only model is unaffected by this rule.
    let mut state_only = policy(&[]);
    state_only.dim_model = BigInt::from(30);
    state_only.n_heads = BigInt::from(2);
    let mut rng = SplitMix64::new(0);
    ActModel::new(&state_only, &Device::Cpu, &mut rng)
        .expect("a state-only model has no 2-D position embedding");
}

// ---------------------------------------------------------------------------
// Training and checkpoints
// ---------------------------------------------------------------------------

#[test]
fn the_backbone_receives_a_gradient_from_the_loss() {
    let model = model(&[TOP]);
    let batch = batch(&[TOP], 2);
    let noise = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();

    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(noise)))
        .unwrap();
    let loss = model.loss(&batch, &output).unwrap();
    let mut gradients = loss.loss.backward().unwrap();

    let named: Vec<&str> = model
        .parameters()
        .iter()
        .map(|parameter| parameter.name.as_str())
        .collect();
    for wanted in [
        "model.backbone.conv1.weight",
        "model.backbone.layer4.1.conv2.weight",
        "model.encoder_img_feat_input_proj.weight",
    ] {
        let index = named.iter().position(|name| *name == wanted).unwrap();
        let parameter = &model.parameters()[index];
        let gradient = gradients
            .get(parameter.value.as_tensor())
            .unwrap_or_else(|| panic!("{wanted} has no gradient at all"));
        let magnitude = gradient
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            magnitude > 0.0 && magnitude.is_finite(),
            "{wanted} has a gradient of {magnitude}; the loss does not reach it"
        );
    }

    // And the whole gradient survives clipping as a finite norm, which is the
    // quantity the training loop refuses to continue past.
    let norm = clip_grad_norm(model.parameters(), &mut gradients, 10.0).unwrap();
    assert!(norm.is_finite() && norm > 0.0, "the total norm is {norm}");
}

#[test]
fn a_training_step_through_the_session_moves_the_backbone() {
    let dir = TempDir::new("image-train");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    let mut inputs = config.policy.input_features.clone().unwrap_or_default();
    inputs.insert(
        TOP.to_owned(),
        PolicyFeature::new(
            FeatureType::Visual,
            [
                BigInt::from(3),
                BigInt::from(EXTENT as i64),
                BigInt::from(EXTENT as i64),
            ],
        ),
    );
    config.policy.input_features = Some(inputs);
    config.validate().unwrap();

    let mut session = TrainSession::new(&config).unwrap();
    assert_eq!(
        session
            .model
            .shape()
            .cameras
            .iter()
            .map(|camera| camera.key.as_str())
            .collect::<Vec<_>>(),
        vec![TOP],
        "a camera declared on the policy config survives into the model"
    );

    let before = backbone_snapshot(&session);
    let raw = session
        .next_batch()
        .unwrap()
        .with_images(
            &camera_map(&[TOP], config.batch_size, EXTENT),
            &CameraNormalization::imagenet(),
        )
        .unwrap();
    let metrics = session.step_on(1, &raw).unwrap();

    assert!(metrics.loss.is_finite() && metrics.grad_norm > 0.0);
    assert!(
        metrics.parameter_delta > 0.0,
        "the step reported no parameter movement at all"
    );
    let after = backbone_snapshot(&session);
    let moved = before
        .iter()
        .zip(&after)
        .filter(|(left, right)| (*left - *right).abs() > 0.0)
        .count();
    assert!(
        moved > 0,
        "AdamW ran but no backbone parameter changed; the backbone group is not being \
         optimized"
    );
}

/// The sum of squares of every backbone parameter, one entry per tensor.
fn backbone_snapshot(session: &TrainSession) -> Vec<f64> {
    session
        .model
        .parameters()
        .iter()
        .filter(|parameter| parameter.name.starts_with("model.backbone"))
        .map(|parameter| {
            f64::from(
                parameter
                    .value
                    .as_tensor()
                    .sqr()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap(),
            )
        })
        .collect()
}

#[test]
fn a_camera_checkpoint_reloads_into_the_same_predictions() {
    let dir = TempDir::new("image-checkpoint");
    let path = dir.child("model.safetensors");

    let trained = model(&[TOP, WRIST]);
    let batch = batch(&[TOP, WRIST], 2);
    let expected = trained
        .predict_action_steps(&batch)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    trained.save(&path).unwrap();

    // A different draw, so the reload has something to overwrite.
    let mut rng = SplitMix64::new(99);
    let mut reloaded = ActModel::new(&policy(&[TOP, WRIST]), &Device::Cpu, &mut rng).unwrap();
    let before = reloaded
        .predict_action_steps(&batch)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(
        before
            .iter()
            .zip(&expected)
            .any(|(left, right)| (left - right).abs() > 1e-6),
        "the two models must start out different for this test to mean anything"
    );

    reloaded.load(&path).unwrap();
    let after = reloaded
        .predict_action_steps(&batch)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (index, (left, right)) in expected.iter().zip(&after).enumerate() {
        assert_eq!(
            left, right,
            "action {index} differs after a save and reload"
        );
    }
}

#[test]
fn a_state_only_checkpoint_does_not_load_into_a_camera_model() {
    let dir = TempDir::new("image-mismatch");
    let path = dir.child("state-only.safetensors");
    let mut rng = SplitMix64::new(7);
    ActModel::new(&policy(&[]), &Device::Cpu, &mut rng)
        .unwrap()
        .save(&path)
        .unwrap();

    let mut camera = model(&[TOP]);
    let error = camera.load(&path).unwrap_err();
    assert!(
        error.to_string().contains("model.backbone."),
        "the refusal does not name a backbone tensor the file lacks: {error}"
    );
}

/// The regression the whole change has to preserve: a config with no camera builds,
/// runs and checkpoints exactly the model it did before the camera path existed.
#[test]
fn a_state_only_config_is_untouched_by_the_camera_path() {
    let model = model(&[]);
    assert!(model.shape().cameras.is_empty());
    assert!(
        model
            .state_dict()
            .unwrap()
            .keys()
            .all(|name| !name.contains("backbone") && !name.contains("img_feat")),
        "a state-only config must build neither a backbone nor an image projection"
    );
    let [_, backbone] = model.optimizer_parameter_groups();
    assert!(backbone.is_empty());

    let batch = state_batch(2);
    assert!(batch.images.is_empty());
    assert_eq!(
        model.predict_action_steps(&batch).unwrap().dims(),
        &[2, 2, 2],
        "the state-only forward pass still runs without any camera"
    );
}
