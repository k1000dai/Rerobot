//! The differential oracle: Rerobot's ACT path against PyTorch's, at fixed
//! weights, fixed inputs and a fixed latent draw.
//!
//! Every expected number in this file was produced by upstream `lerobot` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1 running on PyTorch, by
//! `tools/goldens/make_act_goldens.py`. Nothing here runs Python: the generator
//! ran once and committed three files, and this test reads them.
//!
//! # Why the oracle is shaped this way
//!
//! Three degrees of freedom have to be removed before two implementations of ACT
//! can be compared at all, and each is removed by a mechanism that exists in the
//! port for exactly this purpose:
//!
//! * **the weights** — Rerobot draws torch's initialization *distributions* from
//!   its own generator, so two same-seeded runs do not agree on a single weight.
//!   The oracle exports `ACTPolicy.state_dict()` and loads it, which is also a
//!   real test of the checkpoint format: if the tensor names or shapes were not
//!   upstream's, the load would fail rather than compare.
//! * **the latent draw** — `mu + exp(log_sigma_x2 / 2) * randn_like(mu)` is
//!   random. The generator replaces `torch.randn_like`; Rerobot takes
//!   [`Randomness::Fixed`]. Both get the same tensor.
//! * **dropout** — active in training mode, which is the only mode the VAE branch
//!   runs in. The oracle configuration sets `dropout = 0.0`, where the mask is the
//!   identity on both sides.
//!
//! What remains is the architecture and the arithmetic, which is what is being
//! checked.
//!
//! # Why the tolerances are what they are
//!
//! Both sides compute in `f32`, and they do not reduce in the same order: candle
//! and PyTorch pick different matmul kernels, and this port scales the attention
//! scores after the product where torch scales the query before it. So the numbers
//! agree to `f32` round-off accumulated over the network's depth, not bit for bit.
//!
//! The tolerances come from measurement, not from guessing. On the machine that
//! generated the fixtures, the largest disagreements were:
//!
//! | Quantity | Worst absolute error, as a fraction of the tensor's own scale | Worst relative error among entries at or above 1% of that scale |
//! | --- | --- | --- |
//! | predicted actions | 3.2e-7 | 4.1e-6 |
//! | `mu`, `log_sigma_x2` | 2.2e-7 | 7.0e-6 |
//! | the eleven gradients | 9.4e-7 | 1.6e-5 |
//! | parameters after the AdamW step | 0 (bit-identical) | 0 |
//!
//! and the three loss scalars plus the gradient norm agreed to better than 1e-7
//! relative. That is `f32` epsilon (1.2e-7) times a small constant, which is what
//! two correct implementations should look like. The tolerances below sit about two
//! orders of magnitude above those figures, which leaves room for a different
//! platform's SIMD reduction order without leaving room for a real defect.
//!
//! # The tolerances are not vacuous
//!
//! Checked by deliberately breaking the port and confirming this file fails:
//!
//! | Injected defect | Result |
//! | --- | --- |
//! | the packed `q` projection reads the `k` block | 5 of 12 tests fail; the first predicted action is -0.429 against the oracle's 0.055 |
//! | `LayerNorm`'s epsilon dropped | 4 of 12 tests fail |
//! | `softmax_last_dim` restored in place of the differentiable `softmax` | 3 of 12 tests fail: both position embeddings receive no gradient at all |
//!
//! The third is the defect that actually occurred while writing this port. It is
//! invisible in the forward pass and in every loss value, and this oracle is what
//! catches it.

mod common;

use candle_core::{DType, Device, Tensor};
use common::{fixture_dataset, reduced_config};
use indexmap::IndexMap;
use rerobot_core::dataset::json::{loads, JsonLike};
use rerobot_core::random::SplitMix64;
use rerobot_train::data::batch::Batch;
use rerobot_train::model::act::{ActModel, Pass, Randomness};
use rerobot_train::optim::{act_parameter_groups, clip_grad_norm, AdamW};
use std::collections::HashMap;
use std::path::PathBuf;

/// Relative tolerance for forward values and losses. Measured worst case 7.0e-6.
const FORWARD_TOLERANCE: f64 = 1e-4;

/// Relative tolerance for gradients and post-optimizer-step parameters.
///
/// Looser than [`FORWARD_TOLERANCE`] because a gradient accumulates the forward
/// pass's round-off through the whole backward pass. Measured worst case 1.6e-5.
const GRADIENT_TOLERANCE: f64 = 2e-3;

/// The absolute tolerance is this fraction of the compared tensor's own largest
/// magnitude, i.e. `atol = ABSOLUTE_SCALE * max|expected|`.
///
/// The criterion is `|a - b| <= atol + rtol * |b|`, which is the one
/// `torch.allclose` uses, with the absolute term made scale-aware instead of
/// fixed. A fixed `atol` cannot work here: a gradient tensor's entries span many
/// orders of magnitude, and the smallest of them are `f32` cancellation noise —
/// two implementations summing the same products in different orders disagree on
/// them by a large *relative* amount while agreeing perfectly on everything that
/// matters. Tying `atol` to the tensor's own scale says "an element four orders of
/// magnitude below this tensor's largest is noise", which is true and does not
/// weaken the comparison where it counts: a wrong transpose, a swapped `q`/`k`/`v`
/// block or a dropped residual moves the *large* entries by roughly the tensor's
/// whole scale, four orders of magnitude outside this allowance.
///
/// Measured worst case, across every tensor compared here: 9.4e-7.
const ABSOLUTE_SCALE: f64 = 1e-4;

/// The floor below which a scalar comparison switches from relative to absolute.
const SCALAR_FLOOR: f64 = 1e-12;

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/goldens")
}

fn oracle_metadata() -> IndexMap<String, JsonLike> {
    let path = goldens_dir().join("act_oracle.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    match loads(&text).expect("act_oracle.json is valid JSON") {
        JsonLike::Object(object) => object,
        other => panic!("act_oracle.json is a {}, not an object", other.type_name()),
    }
}

fn oracle_tensors() -> HashMap<String, Tensor> {
    let path = goldens_dir().join("act_oracle_tensors.safetensors");
    candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

fn scalar(metadata: &IndexMap<String, JsonLike>, name: &str) -> f64 {
    let JsonLike::Object(scalars) = &metadata["scalars"] else {
        panic!("`scalars` is not an object");
    };
    match &scalars[name] {
        JsonLike::Float(value) => *value,
        JsonLike::Int(value) => value.to_string().parse().expect("an integer scalar parses"),
        other => panic!("scalar {name} is a {}", other.type_name()),
    }
}

fn flat(tensor: &Tensor) -> Vec<f64> {
    tensor
        .flatten_all()
        .expect("flattening a loaded tensor")
        .to_dtype(DType::F32)
        .expect("f32")
        .to_vec1::<f32>()
        .expect("f32 values")
        .into_iter()
        .map(f64::from)
        .collect()
}

/// Compare two tensors elementwise, reporting the worst disagreement.
///
/// Panics with the index, both values and both error measures, so a failure names
/// which element drifted rather than only that something did.
fn assert_close(label: &str, actual: &Tensor, expected: &Tensor, tolerance: f64) {
    assert_eq!(
        actual.dims(),
        expected.dims(),
        "{label}: shape {:?} does not match the oracle's {:?}",
        actual.dims(),
        expected.dims()
    );
    let actual = flat(actual);
    let expected = flat(expected);
    let scale = expected
        .iter()
        .fold(0.0f64, |largest, value| largest.max(value.abs()));
    let atol = ABSOLUTE_SCALE * scale;

    // The worst element by how far it exceeds its own allowance, so the reported
    // failure is the most-wrong one rather than merely the largest.
    let mut worst = (0usize, 0.0f64, 0.0f64, 0.0f64, -1.0f64);
    for (index, (left, right)) in actual.iter().zip(&expected).enumerate() {
        let absolute = (left - right).abs();
        let allowance = atol + tolerance * right.abs();
        let excess = if allowance > 0.0 {
            absolute / allowance
        } else if absolute > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        if excess > worst.4 {
            worst = (index, *left, *right, absolute, excess);
        }
    }
    let (index, left, right, absolute, excess) = worst;
    assert!(
        excess <= 1.0,
        "{label}: element {index} is {left} but the oracle says {right}\n  \
         absolute difference {absolute:e}, allowance {:e} \
         (atol {atol:e} = {ABSOLUTE_SCALE:e} * scale {scale:e}, rtol {tolerance:e})",
        atol + tolerance * right.abs()
    );
}

fn assert_scalar_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
    let relative = if expected.abs() > SCALAR_FLOOR {
        (actual - expected).abs() / expected.abs()
    } else {
        (actual - expected).abs()
    };
    assert!(
        relative <= tolerance,
        "{label}: {actual} but the oracle says {expected} \
         (relative error {relative:e}, tolerance {tolerance:e})"
    );
}

/// Which frames of the dataset fixture the oracle batched, read from the oracle
/// rather than hardcoded so the two cannot drift apart.
fn oracle_batch_frames(metadata: &IndexMap<String, JsonLike>) -> Vec<usize> {
    let JsonLike::Array(frames) = &metadata["batch_frames"] else {
        panic!("`batch_frames` is not a list");
    };
    frames
        .iter()
        .map(|frame| match frame {
            JsonLike::Int(value) => value
                .to_string()
                .parse()
                .expect("a frame index fits in usize"),
            other => panic!("a frame index is a {}", other.type_name()),
        })
        .collect()
}

/// The policy config the oracle was generated at.
fn oracle_config() -> rerobot_core::policy::act::ActConfig {
    let mut config = reduced_config(fixture_dataset(), PathBuf::from("unused"));
    // The one deliberate difference from `reduced_config`: the oracle pins dropout
    // at zero so that the mask is the identity on both sides.
    config.policy.dropout = 0.0;
    let metadata = rerobot_train::data::meta::DatasetMetadata::load(&fixture_dataset())
        .expect("the fixture loads");
    let (inputs, outputs) = metadata.policy_feature_split();
    config.policy.input_features = Some(inputs);
    config.policy.output_features = Some(outputs);
    config.policy
}

/// The model, with upstream's exported weights loaded into it.
fn oracle_model() -> ActModel {
    // Seeded arbitrarily: every parameter is about to be overwritten from the
    // oracle's file, and the load refuses to leave any of them untouched.
    let mut rng = SplitMix64::new(0xBADC0FFEE);
    let mut model =
        ActModel::new(&oracle_config(), &Device::Cpu, &mut rng).expect("the oracle config builds");
    model
        .load(&goldens_dir().join("act_oracle_weights.safetensors"))
        .expect("upstream's exported state dict loads into Rerobot's ACT");
    model
}

/// The oracle's batch, as a [`Batch`].
fn oracle_batch(tensors: &HashMap<String, Tensor>) -> Batch {
    let mut features = IndexMap::new();
    for key in [
        "observation.state",
        "observation.environment_state",
        "action",
    ] {
        features.insert(
            key.to_owned(),
            tensors[&format!("input/{key}")]
                .to_dtype(DType::F32)
                .expect("f32"),
        );
    }
    let mut padding = IndexMap::new();
    padding.insert(
        "action".to_owned(),
        tensors["input/action_is_pad"]
            .to_dtype(DType::U8)
            .expect("u8"),
    );
    let batch_size = features["observation.state"].dims()[0];
    Batch {
        features,
        padding,
        tasks: vec!["reach the target".to_owned(); batch_size],
        indices: (0..batch_size as i64).collect(),
    }
}

// ---------------------------------------------------------------------------
// The fixtures themselves
// ---------------------------------------------------------------------------

#[test]
fn the_oracle_records_the_upstream_commit_it_was_generated_at() {
    let metadata = oracle_metadata();
    assert_eq!(
        metadata["upstream_commit"],
        JsonLike::Str("f37be3edbee60f3a09a5183788b91eb19f0c07d1".into()),
        "the oracle was generated at a different upstream commit than this port targets"
    );
    assert_eq!(
        metadata["upstream_version"],
        JsonLike::Str(rerobot_train::UPSTREAM_VERSION.into())
    );
    assert_eq!(
        metadata["generator"],
        JsonLike::Str("tools/goldens/make_act_goldens.py".into()),
        "the oracle must name the script that can regenerate it"
    );
}

#[test]
fn the_oracle_was_generated_at_the_configuration_this_test_uses() {
    // A config drift would make every comparison below meaningless while still
    // producing numbers, so it is checked rather than assumed.
    let metadata = oracle_metadata();
    let JsonLike::Object(config) = &metadata["config"] else {
        panic!("`config` is not an object");
    };
    let policy = oracle_config();
    let integer = |name: &str| -> i64 {
        match &config[name] {
            JsonLike::Int(value) => value.to_string().parse().expect("an integer"),
            other => panic!("{name} is a {}", other.type_name()),
        }
    };
    assert_eq!(integer("chunk_size"), 2);
    assert_eq!(
        policy.chunk_size,
        rerobot_core::BigInt::from(integer("chunk_size"))
    );
    assert_eq!(
        policy.dim_model,
        rerobot_core::BigInt::from(integer("dim_model"))
    );
    assert_eq!(
        policy.n_heads,
        rerobot_core::BigInt::from(integer("n_heads"))
    );
    assert_eq!(
        policy.dim_feedforward,
        rerobot_core::BigInt::from(integer("dim_feedforward"))
    );
    assert_eq!(
        policy.n_encoder_layers,
        rerobot_core::BigInt::from(integer("n_encoder_layers"))
    );
    assert_eq!(
        policy.n_decoder_layers,
        rerobot_core::BigInt::from(integer("n_decoder_layers"))
    );
    assert_eq!(
        policy.n_vae_encoder_layers,
        rerobot_core::BigInt::from(integer("n_vae_encoder_layers"))
    );
    assert_eq!(
        policy.latent_dim,
        rerobot_core::BigInt::from(integer("latent_dim"))
    );
    assert_eq!(config["use_vae"], JsonLike::Bool(true));
    assert_eq!(config["pre_norm"], JsonLike::Bool(false));
    assert_eq!(config["dropout"], JsonLike::Float(0.0));
    assert_eq!(policy.dropout, 0.0);
    assert_eq!(config["kl_weight"], JsonLike::Float(10.0));
    assert_eq!(policy.kl_weight, 10.0);
    assert_eq!(
        config["feedforward_activation"],
        JsonLike::Str("relu".into())
    );
    assert_eq!(policy.feedforward_activation, "relu");
}

#[test]
fn upstreams_state_dict_is_exactly_the_one_rerobot_builds() {
    // The load in `oracle_model` would already fail on a mismatch. This asserts it
    // directly so that the failure names the difference instead of a file path.
    let metadata = oracle_metadata();
    let JsonLike::Array(keys) = &metadata["state_dict_keys"] else {
        panic!("`state_dict_keys` is not a list");
    };
    let mut upstream: Vec<&str> = keys
        .iter()
        .map(|key| match key {
            JsonLike::Str(name) => name.as_str(),
            other => panic!("a state dict key is a {}", other.type_name()),
        })
        .collect();
    upstream.sort_unstable();

    let mut rng = SplitMix64::new(1);
    let model = ActModel::new(&oracle_config(), &Device::Cpu, &mut rng).unwrap();
    let state = model.state_dict().unwrap();
    let mut ours: Vec<&str> = state.keys().map(String::as_str).collect();
    ours.sort_unstable();

    assert_eq!(
        ours, upstream,
        "Rerobot's state dict is not the one ACTPolicy.state_dict() produces"
    );
    assert_eq!(ours.len(), 62);
}

// ---------------------------------------------------------------------------
// The inputs: the dataset path reaches the oracle's batch
// ---------------------------------------------------------------------------

#[test]
fn rerobots_own_dataset_and_normalizer_reproduce_the_oracles_batch() {
    // This is what ties the oracle to the rest of the port. Everything below
    // compares model arithmetic on inputs read out of a file; this test shows those
    // inputs are what Rerobot's reader, delta-window expansion, collate and
    // normalizer actually produce from the committed dataset fixture, so the
    // comparison is not against a batch only the oracle can make.
    let tensors = oracle_tensors();
    let policy = oracle_config();

    let windows = IndexMap::from([(
        "action".to_owned(),
        rerobot_core::dataset::delta::action_delta_timestamps(2, 10),
    )]);
    let dataset =
        rerobot_train::data::dataset::StateOnlyDataset::load(&fixture_dataset(), &windows, 1e-4)
            .expect("the fixture loads");
    let wanted = oracle_batch_frames(&oracle_metadata());
    let frames: Vec<_> = wanted
        .iter()
        .map(|index| dataset.get(*index).unwrap())
        .collect();
    let raw = rerobot_train::data::batch::collate(&frames, &Device::Cpu).unwrap();

    let normalizer = rerobot_core::policy::normalize::Normalizer::new(
        &policy
            .input_features
            .clone()
            .unwrap()
            .into_iter()
            .chain(policy.output_features.clone().unwrap())
            .collect(),
        &policy.normalization_mapping,
        &dataset.metadata().stats,
    )
    .unwrap();
    let batch = raw.normalized(&normalizer).unwrap();

    for key in [
        "observation.state",
        "observation.environment_state",
        "action",
    ] {
        assert_close(
            &format!("normalized {key}"),
            batch.feature(key).unwrap(),
            &tensors[&format!("input/{key}")],
            FORWARD_TOLERANCE,
        );
    }
    // The padding mask is exact, not approximate: it is a set of booleans.
    assert_eq!(
        flat(batch.padding_mask("action").unwrap()),
        flat(&tensors["input/action_is_pad"]),
        "the action padding mask disagrees with upstream's"
    );
    // ... and it is not vacuously all-false, or masking would go unexercised.
    assert!(
        flat(&tensors["input/action_is_pad"])
            .iter()
            .any(|flag| *flag != 0.0),
        "the oracle batch has no padded action, so the mask is untested"
    );
}

// ---------------------------------------------------------------------------
// The forward pass
// ---------------------------------------------------------------------------

#[test]
fn the_forward_pass_matches_pytorch_at_a_fixed_latent() {
    let tensors = oracle_tensors();
    let model = oracle_model();
    let batch = oracle_batch(&tensors);
    let noise = tensors["input/latent_noise"].to_dtype(DType::F32).unwrap();

    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(noise)))
        .expect("the forward pass runs");

    assert_close(
        "predicted actions",
        &output.actions,
        &tensors["output/actions"],
        FORWARD_TOLERANCE,
    );
    assert_close(
        "latent mu",
        output.mu.as_ref().expect("the VAE branch ran"),
        &tensors["output/mu"],
        FORWARD_TOLERANCE,
    );
    assert_close(
        "latent log_sigma_x2",
        output.log_sigma_x2.as_ref().expect("the VAE branch ran"),
        &tensors["output/log_sigma_x2"],
        FORWARD_TOLERANCE,
    );
}

#[test]
fn the_loss_matches_pytorch_term_by_term() {
    let metadata = oracle_metadata();
    let tensors = oracle_tensors();
    let model = oracle_model();
    let batch = oracle_batch(&tensors);
    let noise = tensors["input/latent_noise"].to_dtype(DType::F32).unwrap();

    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(noise)))
        .unwrap();
    let loss = model.loss(&batch, &output).unwrap();

    assert_scalar_close(
        "l1_loss",
        loss.l1_loss,
        scalar(&metadata, "l1_loss"),
        FORWARD_TOLERANCE,
    );
    assert_scalar_close(
        "kld_loss",
        loss.kld_loss.expect("the VAE is on"),
        scalar(&metadata, "kld_loss"),
        FORWARD_TOLERANCE,
    );
    assert_scalar_close(
        "total_loss",
        loss.total,
        scalar(&metadata, "total_loss"),
        FORWARD_TOLERANCE,
    );

    // The total is not independently informative unless it really is the weighted
    // sum, and the oracle's own numbers must satisfy that too.
    let expected_total = scalar(&metadata, "l1_loss") + 10.0 * scalar(&metadata, "kld_loss");
    assert_scalar_close(
        "the oracle's own total against its parts",
        scalar(&metadata, "total_loss"),
        expected_total,
        1e-6,
    );
}

#[test]
fn the_masked_l1_divides_by_the_number_of_unpadded_scalars_pytorch_counted() {
    // `num_valid = valid_mask.sum() * abs_err.shape[-1]`. The fixture's second
    // frame has one padded action, so the count is not simply batch * chunk * dim,
    // and getting the divisor wrong would show up here rather than only in the loss.
    let metadata = oracle_metadata();
    let tensors = oracle_tensors();
    let flags = flat(&tensors["input/action_is_pad"]);
    let unpadded = flags.iter().filter(|flag| **flag == 0.0).count();
    let action_dim = tensors["input/action"].dims()[2];
    assert_eq!(
        (unpadded * action_dim) as f64,
        scalar(&metadata, "num_valid_scalars"),
        "the oracle's divisor is not the count of unpadded scalars"
    );
    assert!(
        unpadded * action_dim < tensors["input/action"].elem_count(),
        "every action is unpadded, so the divisor is untested"
    );
}

// ---------------------------------------------------------------------------
// The backward pass
// ---------------------------------------------------------------------------

#[test]
fn the_representative_gradients_match_pytorch() {
    let metadata = oracle_metadata();
    let tensors = oracle_tensors();
    let model = oracle_model();
    let batch = oracle_batch(&tensors);
    let noise = tensors["input/latent_noise"].to_dtype(DType::F32).unwrap();

    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(noise)))
        .unwrap();
    let loss = model.loss(&batch, &output).unwrap();
    let gradients = loss.loss.backward().expect("the backward pass runs");

    let JsonLike::Array(names) = &metadata["gradient_keys"] else {
        panic!("`gradient_keys` is not a list");
    };
    assert!(
        names.len() >= 10,
        "the oracle pins only {} gradients; that is not representative",
        names.len()
    );
    for name in names {
        let JsonLike::Str(name) = name else {
            panic!("a gradient key is not a string");
        };
        let parameter = model
            .parameters()
            .iter()
            .find(|parameter| &parameter.name == name)
            .unwrap_or_else(|| panic!("{name} is not a parameter of this model"));
        let gradient = gradients
            .get(parameter.value.as_tensor())
            .unwrap_or_else(|| panic!("{name} received no gradient"));
        assert_close(
            &format!("gradient of {name}"),
            gradient,
            &tensors[&format!("grad/{name}")],
            GRADIENT_TOLERANCE,
        );
    }
}

#[test]
fn the_position_embedding_gradients_are_non_zero_in_both_implementations() {
    // The two position embeddings reach the loss only through the attention
    // *logits*. A softmax whose backward pass does not propagate into its input
    // leaves them at exactly zero while every forward number stays correct -- a
    // defect that really occurred during this port. Comparing them against
    // PyTorch's non-zero values is what makes the oracle catch it, so the
    // non-zeroness is asserted on both sides explicitly rather than relied upon.
    let tensors = oracle_tensors();
    for name in [
        "model.encoder_1d_feature_pos_embed.weight",
        "model.decoder_pos_embed.weight",
    ] {
        let magnitude: f64 = flat(&tensors[&format!("grad/{name}")])
            .iter()
            .map(|value| value.abs())
            .sum();
        assert!(
            magnitude > 1e-6,
            "PyTorch's gradient for {name} is ~zero ({magnitude:e}), so comparing against \
             it would not detect a non-propagating softmax"
        );
    }

    let model = oracle_model();
    let batch = oracle_batch(&tensors);
    let noise = tensors["input/latent_noise"].to_dtype(DType::F32).unwrap();
    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(noise)))
        .unwrap();
    let gradients = model
        .loss(&batch, &output)
        .unwrap()
        .loss
        .backward()
        .unwrap();
    for name in [
        "model.encoder_1d_feature_pos_embed.weight",
        "model.decoder_pos_embed.weight",
    ] {
        let parameter = model
            .parameters()
            .iter()
            .find(|parameter| parameter.name == name)
            .unwrap();
        let magnitude = gradients
            .get(parameter.value.as_tensor())
            .unwrap_or_else(|| panic!("{name} received no gradient"))
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            magnitude > 1e-6,
            "Rerobot's gradient for {name} is ~zero, so the attention logits are not \
             differentiable"
        );
    }
}

#[test]
fn the_clipped_gradient_norm_matches_pytorch() {
    let metadata = oracle_metadata();
    let tensors = oracle_tensors();
    let model = oracle_model();
    let batch = oracle_batch(&tensors);
    let noise = tensors["input/latent_noise"].to_dtype(DType::F32).unwrap();

    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(noise)))
        .unwrap();
    let loss = model.loss(&batch, &output).unwrap();
    let mut gradients = loss.loss.backward().unwrap();
    let norm = clip_grad_norm(model.parameters(), &mut gradients, 10.0).unwrap();

    assert_scalar_close(
        "total gradient norm before clipping",
        norm,
        scalar(&metadata, "grad_norm"),
        GRADIENT_TOLERANCE,
    );
    // The oracle's norm is far above the clip, so the scaling branch is the one
    // both sides take. If it were below, `clip_grad_norm_` would be a no-op and the
    // post-step comparison would not test the clip at all.
    assert!(
        scalar(&metadata, "grad_norm") > 10.0,
        "the oracle's gradient norm is inside the clip, so clipping is untested"
    );
}

// ---------------------------------------------------------------------------
// The optimizer step
// ---------------------------------------------------------------------------

#[test]
fn the_parameters_after_one_adamw_step_match_pytorch() {
    // The whole pipeline in one assertion: forward, loss, backward, clip, AdamW.
    // Any disagreement anywhere upstream of the optimizer reaches this comparison.
    let metadata = oracle_metadata();
    let tensors = oracle_tensors();
    let model = oracle_model();
    let batch = oracle_batch(&tensors);
    let noise = tensors["input/latent_noise"].to_dtype(DType::F32).unwrap();

    let preset = oracle_config().optimizer_preset();
    assert_eq!(preset.lr, 1e-5);
    assert_eq!(preset.weight_decay, 1e-4);
    assert_eq!(preset.betas, [0.9, 0.999]);
    assert_eq!(preset.eps, 1e-8);
    assert_eq!(preset.grad_clip_norm, 10.0);

    let mut optimizer = AdamW::new(
        act_parameter_groups(
            model.optimizer_parameter_groups(),
            &preset,
            oracle_config().optimizer_lr_backbone,
        ),
        model.parameters().len(),
    )
    .unwrap();

    let output = model
        .forward(&batch, Pass::Train(Randomness::Fixed(noise)))
        .unwrap();
    let loss = model.loss(&batch, &output).unwrap();
    let mut gradients = loss.loss.backward().unwrap();
    clip_grad_norm(model.parameters(), &mut gradients, preset.grad_clip_norm).unwrap();
    optimizer.step(model.parameters(), &gradients).unwrap();

    let JsonLike::Array(names) = &metadata["post_step_keys"] else {
        panic!("`post_step_keys` is not a list");
    };
    assert!(!names.is_empty());
    // Measured: these come out bit-identical to PyTorch's, because after one step
    // AdamW moves every element by `lr * m̂ / (sqrt(v̂) + eps)`, which is within a
    // few `f32` ulps of `lr * sign(g)` and rounds the same way on both sides. The
    // comparison is still made with a tolerance rather than for exact equality,
    // because that agreement is a property of this configuration and this
    // platform, not a promise.
    let state = model.state_dict().unwrap();
    for name in names {
        let JsonLike::Str(name) = name else {
            panic!("a post-step key is not a string");
        };
        assert_close(
            &format!("{name} after one AdamW step"),
            &state[name],
            &tensors[&format!("post_step/{name}")],
            GRADIENT_TOLERANCE,
        );
    }
}

#[test]
fn the_step_moved_the_recorded_parameters_so_the_comparison_is_not_vacuous() {
    // If the oracle's post-step values happened to equal its pre-step weights, the
    // test above would pass without the optimizer having done anything.
    let metadata = oracle_metadata();
    let tensors = oracle_tensors();
    let before = candle_core::safetensors::load(
        goldens_dir().join("act_oracle_weights.safetensors"),
        &Device::Cpu,
    )
    .unwrap();

    let JsonLike::Array(names) = &metadata["post_step_keys"] else {
        panic!("`post_step_keys` is not a list");
    };
    for name in names {
        let JsonLike::Str(name) = name else { panic!() };
        let start = flat(&before[name]);
        let end = flat(&tensors[&format!("post_step/{name}")]);
        let moved: f64 = start
            .iter()
            .zip(&end)
            .map(|(left, right)| (left - right).abs())
            .sum();
        assert!(
            moved > 0.0,
            "PyTorch's step did not move {name}, so comparing against it proves nothing"
        );
        // AdamW's first step moves each element by at most `lr`, so the total move
        // is bounded; a much larger one would mean the oracle recorded the wrong
        // tensor.
        assert!(
            moved <= 1e-5 * start.len() as f64 * 1.5,
            "PyTorch's step moved {name} by {moved}, more than one learning rate per element"
        );
    }
}
