//! End-to-end tests for the training loop: a genuine forward, loss, backward,
//! gradient clip and AdamW step over the committed dataset fixture, and the
//! upstream checkpoint layout it writes.
//!
//! Derived from `lerobot/scripts/lerobot_train.py`, `lerobot/common/train_utils.py`
//! and `lerobot/optim/optimizers.py` at commit
//! f37be3edbee60f3a09a5183788b91eb19f0c07d1.
//!
//! The claim these tests are here to defend is narrow and load-bearing: the step
//! *trains*. Not that it runs without raising, but that a loss is computed from
//! real data, that gradients flow back to every parameter, and that the weights
//! move as a result. Each of those is asserted separately, because a pipeline can
//! satisfy the first two and silently fail the third.

mod common;

use common::{
    copy_fixture_dataset, fixture_dataset, reduced_config, rewrite_episode_rows,
    rewrite_frame_episode_indices, TempDir,
};
use rerobot_core::dataset::json::{loads, JsonLike};
use rerobot_train::checkpoint::{self, LastCheckpointKind, TrainingStep};
use rerobot_train::config::TrainConfig;
use rerobot_train::error::TrainError;
use rerobot_train::model::act::ActModel;
use rerobot_train::optim::state_dict_distance;
use rerobot_train::run::{train, TrainSession};

fn train_once(label: &str) -> (TempDir, rerobot_train::run::TrainOutcome, Vec<String>) {
    let dir = TempDir::new(label);
    let config = reduced_config(fixture_dataset(), dir.child("out"));
    let mut logs = Vec::new();
    let outcome =
        train(&config, &mut |line| logs.push(line.to_owned())).expect("the one-step run completes");
    (dir, outcome, logs)
}

// ---------------------------------------------------------------------------
// One genuine step
// ---------------------------------------------------------------------------

#[test]
fn one_step_produces_a_finite_loss_from_real_data() {
    let (_dir, outcome, _) = train_once("one-step");
    assert_eq!(outcome.steps.len(), 1);
    let step = &outcome.steps[0];
    assert_eq!(step.step, 1);
    assert!(
        step.loss.is_finite() && step.loss > 0.0,
        "loss {} is not a finite positive number",
        step.loss
    );
    assert!(step.l1_loss.is_finite() && step.l1_loss >= 0.0);
    assert!(
        step.kld_loss.expect("the VAE is on").is_finite(),
        "the KL term is not finite"
    );
    assert!(
        step.grad_norm.is_finite() && step.grad_norm > 0.0,
        "grad_norm {} is not a finite positive number",
        step.grad_norm
    );
    assert_eq!(step.lr, 1e-5, "the ACT preset learning rate");
    assert_eq!(
        step.frame_indices.len(),
        2,
        "the batch size is two, so two frames were consumed"
    );
    assert!(
        step.frame_indices
            .iter()
            .all(|index| (0..4).contains(index)),
        "the sampler produced a frame outside the fixture: {:?}",
        step.frame_indices
    );
}

#[test]
fn a_training_run_consumes_only_the_configured_episodes() {
    let dir = TempDir::new("episode-filter-train");
    let dataset = dir.child("dataset");
    copy_fixture_dataset(&dataset);
    let info_path = dataset.join("meta/info.json");
    let info = std::fs::read_to_string(&info_path).unwrap();
    std::fs::write(
        info_path,
        info.replace("\"total_episodes\": 1", "\"total_episodes\": 2"),
    )
    .unwrap();
    rewrite_episode_rows(&dataset, &[(0, 0, 2, 2), (1, 2, 4, 2)]);
    rewrite_frame_episode_indices(&dataset, &[0, 0, 1, 1]);

    let mut config = reduced_config(dataset, dir.child("out"));
    config.dataset_episodes = Some(vec![1]);
    config
        .validate()
        .expect("the selected episode config validates");
    let mut logs = |_line: &str| {};
    let outcome = train(&config, &mut logs).expect("the filtered run completes");

    assert_eq!(outcome.steps.len(), 1);
    assert!(
        outcome.steps[0]
            .frame_indices
            .iter()
            .all(|index| (2..4).contains(index)),
        "the sampler consumed an unselected absolute frame: {:?}",
        outcome.steps[0].frame_indices
    );
}

#[test]
fn a_pretrained_act_path_loads_checkpoint_weights_before_training() {
    let (source_dir, source_outcome, _) = train_once("pretrained-source");
    let source_checkpoint = source_outcome
        .checkpoints
        .first()
        .expect("the source run writes its final checkpoint")
        .join("pretrained_model");

    let target_dir = TempDir::new("pretrained-target");
    let mut config = reduced_config(fixture_dataset(), target_dir.child("out"));
    config.policy.pretrained_path = Some(source_checkpoint.to_string_lossy().into_owned());
    config
        .validate()
        .expect("the pretrained path is a valid ACT config");

    let session = TrainSession::new(&config).expect("the pretrained session builds");
    let expected = candle_core::safetensors::load(
        source_checkpoint.join("model.safetensors"),
        session.device(),
    )
    .expect("the source weights load");
    let actual = session.model.state_dict().expect("the target weights load");

    assert_eq!(actual.len(), expected.len());
    for (name, tensor) in actual {
        let source = expected
            .get(&name)
            .unwrap_or_else(|| panic!("source checkpoint is missing {name}"));
        assert_eq!(
            tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            source.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "{name}"
        );
    }

    drop(source_dir);
}

#[test]
fn a_pretrained_policy_uses_checkpoint_processor_statistics_not_dataset_statistics() {
    let (source_dir, source_outcome, _) = train_once("processor-source");
    let source_checkpoint = source_outcome
        .checkpoints
        .first()
        .expect("the source run writes its final checkpoint")
        .join("pretrained_model");

    let target_dir = TempDir::new("processor-target");
    let target_dataset = target_dir.child("dataset");
    common::copy_fixture_dataset(&target_dataset);
    let stats_path = target_dataset.join("meta/stats.json");
    let stats = std::fs::read_to_string(&stats_path).expect("the copied stats load");
    let changed_stats = stats
        .replace(
            "\"mean\": [\n            0.4375,\n            0.5625",
            "\"mean\": [\n            100.0,\n            100.0",
        )
        .replace(
            "\"std\": [\n            0.36975499987602234,\n            0.36975499987602234",
            "\"std\": [\n            1.0,\n            1.0",
        );
    std::fs::write(&stats_path, changed_stats).expect("the changed stats write");

    let policy_text = std::fs::read_to_string(source_checkpoint.join("config.json"))
        .expect("the source policy config loads");
    let mut policy = rerobot_core::policy::act::ActConfig::from_checkpoint_json(&policy_text)
        .expect("the source policy config parses");
    policy.pretrained_path = Some(source_checkpoint.to_string_lossy().into_owned());
    let mut config = reduced_config(target_dataset, target_dir.child("out"));
    config.policy = policy;
    let target_session = TrainSession::new(&config).expect("the target session builds");

    let source_config = reduced_config(fixture_dataset(), source_dir.child("unused"));
    let source_session = TrainSession::new(&source_config).expect("the source session builds");
    assert_eq!(
        target_session.normalizer, source_session.normalizer,
        "pretrained training must use the checkpoint normalizer"
    );
}

#[test]
fn resuming_uses_checkpoint_processor_statistics_not_dataset_statistics() {
    let (source_dir, source_outcome, _) = train_once("resume-processor-source");
    let checkpoint = source_outcome
        .checkpoints
        .first()
        .expect("the source run writes its final checkpoint")
        .clone();

    let target_dir = TempDir::new("resume-processor-target");
    let target_dataset = target_dir.child("dataset");
    common::copy_fixture_dataset(&target_dataset);
    let stats_path = target_dataset.join("meta/stats.json");
    let stats = std::fs::read_to_string(&stats_path).expect("the copied stats load");
    let changed_stats = stats
        .replace(
            "\"mean\": [\n            0.4375,\n            0.5625",
            "\"mean\": [\n            100.0,\n            100.0",
        )
        .replace(
            "\"std\": [\n            0.36975499987602234,\n            0.36975499987602234",
            "\"std\": [\n            1.0,\n            1.0",
        );
    std::fs::write(&stats_path, changed_stats).expect("the changed stats write");

    let mut config =
        TrainConfig::from_checkpoint_dir(&checkpoint).expect("the checkpoint config reconstructs");
    config.dataset_root = target_dataset;
    config.output_dir = target_dir.child("out");
    let resumed = TrainSession::new(&config).expect("the resumed session builds");

    let source_config = reduced_config(fixture_dataset(), source_dir.child("unused"));
    let source_session = TrainSession::new(&source_config).expect("the source session builds");
    assert_eq!(
        resumed.normalizer, source_session.normalizer,
        "resume must keep the checkpoint normalizer"
    );
}

#[test]
fn resume_reconstructs_the_policy_feature_namespace_from_policy_config() {
    let (_source_dir, source_outcome, _) = train_once("resume-policy-config");
    let checkpoint = source_outcome
        .checkpoints
        .first()
        .expect("the source run writes a checkpoint");
    let train_config =
        TrainConfig::from_checkpoint_dir(checkpoint).expect("the train config reconstructs");
    let policy_path = checkpoint.join("pretrained_model/config.json");
    let policy_text = std::fs::read_to_string(&policy_path).expect("the policy config exists");
    let policy = rerobot_core::policy::act::ActConfig::from_checkpoint_json(&policy_text)
        .expect("the policy config parses");

    assert_eq!(train_config.policy.input_features, policy.input_features);
    assert_eq!(train_config.policy.output_features, policy.output_features);
    assert!(
        train_config
            .policy
            .input_features
            .as_ref()
            .is_some_and(|features| !features.is_empty()),
        "resume must not discard the model's resolved input namespace"
    );
}

#[test]
fn oversized_policy_config_is_rejected_before_unbounded_checkpoint_read() {
    let (_source_dir, source_outcome, _) = train_once("oversized-resume-policy-config");
    let checkpoint = source_outcome
        .checkpoints
        .first()
        .expect("the source run writes a checkpoint");
    let policy_path = checkpoint.join("pretrained_model/config.json");
    let oversized = vec![b' '; rerobot_train::limits::MAX_CHECKPOINT_JSON_BYTES as usize + 1];
    std::fs::write(&policy_path, oversized).expect("the fixture policy config is replaceable");

    let error = TrainConfig::from_checkpoint_dir(checkpoint)
        .expect_err("a policy config beyond the checkpoint budget must be refused");

    assert_eq!(
        error.to_string(),
        format!(
            "{}: config.json exceeds the {}-byte limit",
            policy_path.display(),
            rerobot_train::limits::MAX_CHECKPOINT_JSON_BYTES
        )
    );
}

#[test]
fn the_step_moves_the_weights_rather_than_merely_running() {
    let (_dir, outcome, _) = train_once("weights-move");
    let step = &outcome.steps[0];
    assert!(
        step.parameter_delta > 0.0,
        "the parameter norm did not change, so the optimizer did not update anything"
    );
}

/// The three tensors whose gradient is *exactly* zero on the first step of a
/// freshly initialized ACT, and why.
///
/// This is upstream's behaviour too, not a shortfall of the port, and it follows
/// from three facts that all come from `modeling_act.py` and `torch.nn`:
///
/// 1. the transformer decoder's input is `torch.zeros(...)` — ACT's object
///    queries carry all the information, so the sequence itself starts empty;
/// 2. `nn.MultiheadAttention` zero-initializes `in_proj_bias` and
///    `out_proj.bias`;
/// 3. therefore the decoder self-attention's *value* stream is identically zero,
///    so its output is constant in the attention weights and in `out_proj.weight`,
///    and the LayerNorm that follows sees an all-zero input, whose normalized form
///    is zero and so does not depend on `norm1.weight` either.
///
/// All three train from the second step onward, once the biases have moved off
/// zero. `the_decoder_self_attention_weights_train_from_the_second_step` asserts
/// exactly that, so this list cannot be used to excuse a real regression.
const ZERO_GRADIENT_AT_INITIALIZATION: [&str; 3] = [
    "model.decoder.layers.0.norm1.weight",
    "model.decoder.layers.0.self_attn.in_proj_weight",
    "model.decoder.layers.0.self_attn.out_proj.weight",
];

#[test]
fn every_parameter_is_connected_to_the_loss() {
    // "Has a gradient" and "has a non-zero gradient" are different claims, and so
    // is "was updated". A parameter absent from the gradient store is not connected
    // to the loss at all and can never train, whatever the data.
    let dir = TempDir::new("gradients");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().expect("the reduced config validates");
    let mut session = TrainSession::new(&config).expect("the session builds");

    let batch = session.next_batch().unwrap();
    let batch = batch.normalized(&session.normalizer).unwrap();
    let output = session
        .model
        .forward(
            &batch,
            rerobot_train::model::act::Pass::Train(rerobot_train::model::act::Randomness::Seeded(
                &mut session.rng,
            )),
        )
        .unwrap();
    let loss = session.model.loss(&batch, &output).unwrap();
    let gradients = loss.loss.backward().unwrap();

    let without_gradient: Vec<&str> = session
        .model
        .parameters()
        .iter()
        .filter(|parameter| gradients.get(parameter.value.as_tensor()).is_none())
        .map(|parameter| parameter.name.as_str())
        .collect();
    assert!(
        without_gradient.is_empty(),
        "these parameters received no gradient, so they can never train: {without_gradient:?}"
    );
}

#[test]
fn exactly_the_three_explained_tensors_have_a_zero_gradient_at_initialization() {
    // Pinned in both directions. A new zero-gradient tensor means something stopped
    // being differentiable; a missing one means the architecture changed. Either
    // way it should be a deliberate edit to this list, not a silent drift.
    let dir = TempDir::new("zero-gradients");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().unwrap();
    let mut session = TrainSession::new(&config).unwrap();

    let batch = session.next_batch().unwrap();
    let batch = batch.normalized(&session.normalizer).unwrap();
    let output = session
        .model
        .forward(
            &batch,
            rerobot_train::model::act::Pass::Train(rerobot_train::model::act::Randomness::Seeded(
                &mut session.rng,
            )),
        )
        .unwrap();
    let loss = session.model.loss(&batch, &output).unwrap();
    let gradients = loss.loss.backward().unwrap();

    let mut zero: Vec<&str> = session
        .model
        .parameters()
        .iter()
        .filter(|parameter| {
            let gradient = gradients
                .get(parameter.value.as_tensor())
                .expect("every parameter is connected to the loss");
            gradient
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                == 0.0
        })
        .map(|parameter| parameter.name.as_str())
        .collect();
    zero.sort_unstable();
    let mut expected = ZERO_GRADIENT_AT_INITIALIZATION;
    expected.sort_unstable();
    assert_eq!(zero, expected.to_vec());

    // The premise of the explanation, checked rather than asserted in prose: the
    // two biases that make the value stream vanish really are zero at init.
    let state = session.model.state_dict().unwrap();
    for name in [
        "model.decoder.layers.0.self_attn.in_proj_bias",
        "model.decoder.layers.0.self_attn.out_proj.bias",
    ] {
        let magnitude = state[name]
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            magnitude, 0.0,
            "{name} is not zero-initialized, so the explanation above no longer holds"
        );
    }
}

#[test]
fn every_other_parameter_moves_after_one_adamw_step() {
    let dir = TempDir::new("updates");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().unwrap();
    let mut session = TrainSession::new(&config).unwrap();

    let before = session.model.state_dict().unwrap();
    session.step(1).unwrap();
    let after = session.model.state_dict().unwrap();

    let mut unchanged = Vec::new();
    let mut moved = 0usize;
    for (name, tensor) in &after {
        let distance = (tensor - &before[name])
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        if name == "model.vae_encoder_pos_enc" {
            assert_eq!(
                distance, 0.0,
                "the sinusoidal position buffer is not trainable and must not move"
            );
            continue;
        }
        if ZERO_GRADIENT_AT_INITIALIZATION.contains(&name.as_str()) {
            assert_eq!(
                distance, 0.0,
                "{name} has a zero gradient at init, so it must not move on step 1"
            );
            continue;
        }
        if distance == 0.0 {
            unchanged.push(name.clone());
        } else {
            moved += 1;
        }
    }
    assert!(
        unchanged.is_empty(),
        "these parameters did not move after one AdamW step: {unchanged:?}"
    );
    assert!(moved > 50, "only {moved} parameters moved");
}

#[test]
fn the_decoder_self_attention_weights_train_from_the_second_step() {
    // The other half of the zero-gradient story: the three tensors are frozen only
    // at initialization, and they start training as soon as the biases feeding them
    // are non-zero. Without this test, the list above could hide a permanently
    // disconnected parameter.
    let dir = TempDir::new("second-step");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.steps = 2;
    config.validate().unwrap();
    let mut session = TrainSession::new(&config).unwrap();

    session.step(1).unwrap();
    let after_first = session.model.state_dict().unwrap();
    session.step(2).unwrap();
    let after_second = session.model.state_dict().unwrap();

    for name in ZERO_GRADIENT_AT_INITIALIZATION {
        let distance = (&after_second[name] - &after_first[name])
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            distance > 0.0,
            "{name} still has not moved after the second step, so it is disconnected \
             rather than merely zero at initialization"
        );
    }
}

#[test]
fn the_first_adamw_step_moves_each_weight_by_about_the_learning_rate() {
    // AdamW's first update is `lr * m̂ / (sqrt(v̂) + eps)`, and after one step
    // `m̂ / sqrt(v̂)` is `sign(g)` up to the epsilon, so every element with a
    // non-zero gradient moves by very nearly `lr`. That is a sharp, checkable
    // prediction, and it is the strongest evidence available that the update is
    // torch's rather than merely *an* update.
    let dir = TempDir::new("adamw-magnitude");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().unwrap();
    let mut session = TrainSession::new(&config).unwrap();
    let before = session.model.state_dict().unwrap();
    session.step(1).unwrap();
    let after = session.model.state_dict().unwrap();

    let lr = 1e-5f32;
    let mut checked = 0usize;
    for (name, tensor) in &after {
        if name == "model.vae_encoder_pos_enc"
            || ZERO_GRADIENT_AT_INITIALIZATION.contains(&name.as_str())
        {
            continue;
        }
        let deltas = (tensor - &before[name])
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let max = deltas.iter().copied().fold(0.0f32, f32::max);
        // Upper bound: AdamW cannot move an element further than one learning rate
        // on its first step, so anything larger means the update is not AdamW's.
        assert!(
            max <= lr * 1.5,
            "{name} moved by {max}, more than one learning rate; the update is not AdamW's"
        );
        // Lower bound, and the reason this test is not vacuous: with eps at 1e-8 and
        // gradients well above it, `m̂ / (sqrt(v̂) + eps)` is within a few percent of
        // one, so the largest element must move by nearly `lr`. Without this, an
        // aliased snapshot -- or an optimizer that did nothing -- would pass.
        assert!(
            max >= lr * 0.5,
            "{name} moved by only {max}, far less than one learning rate; either the \
             gradient never arrived or the snapshot aliased the live parameter"
        );
        checked += 1;
    }
    assert!(checked > 50, "only {checked} parameters were checked");
}

#[test]
fn a_non_finite_loss_stops_the_run_rather_than_reporting_it() {
    // Config validation refuses the values that *cause* this, so this is the guard
    // behind that: a loss can also go non-finite from data or from divergence, and a
    // step that produced NaN has not trained anything. Reporting `loss:NaN` and
    // exiting zero -- which is what happened before -- writes a checkpoint of NaN
    // weights that looks like a successful run.
    //
    // Poisoning a parameter with infinity is the shortest path to a non-finite
    // forward pass, and it is a state a diverging run really reaches.
    let dir = TempDir::new("non-finite-loss");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().unwrap();
    let mut session = TrainSession::new(&config).unwrap();

    let poisoned = session
        .model
        .parameters()
        .iter()
        .find(|parameter| parameter.name == "model.action_head.weight")
        .expect("the head exists");
    let infinite = candle_core::Tensor::full(
        f32::INFINITY,
        poisoned.value.as_tensor().shape(),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    poisoned.value.set(&infinite).unwrap();

    let error = session
        .step(1)
        .expect_err("a step that produces a non-finite loss must fail");
    let message = error.to_string();
    assert!(
        message.contains("not finite") || message.contains("non-finite"),
        "the error does not say the step went non-finite: {message}"
    );
    assert!(
        message.contains("loss") || message.contains("step 1"),
        "the error does not say which step or quantity: {message}"
    );
}

#[test]
fn a_non_finite_gradient_stops_the_run() {
    // The gradient norm is the other quantity a diverging step corrupts first, and it
    // is what upstream logs as `grdn`. An infinite norm must not reach the optimizer:
    // AdamW would turn every weight it touches into NaN.
    let dir = TempDir::new("non-finite-grad");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().unwrap();
    let mut session = TrainSession::new(&config).unwrap();

    // A very large but finite weight makes the forward pass finite and the gradient
    // overflow `f32`, which is exactly how a real divergence presents.
    let parameter = session
        .model
        .parameters()
        .iter()
        .find(|parameter| parameter.name == "model.vae_encoder_latent_output_proj.weight")
        .expect("the latent head exists");
    let huge = candle_core::Tensor::full(
        1e30f32,
        parameter.value.as_tensor().shape(),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    parameter.value.set(&huge).unwrap();

    let error = session
        .step(1)
        .expect_err("a step whose gradient is non-finite must fail");
    let message = error.to_string();
    assert!(
        message.contains("not finite") || message.contains("non-finite"),
        "the error does not say what went wrong: {message}"
    );
}

#[test]
fn an_ordinary_step_is_not_tripped_by_the_finiteness_guard() {
    // The guard must not have narrowed what works.
    let (_dir, outcome, _) = train_once("guard-passes");
    let step = &outcome.steps[0];
    assert!(step.loss.is_finite());
    assert!(step.grad_norm.is_finite());
    assert!(step.parameter_delta.is_finite());
}

#[test]
fn the_same_seed_reproduces_the_run_exactly_and_a_different_seed_does_not() {
    let run = |seed: u64, label: &str| {
        let dir = TempDir::new(label);
        let mut config = reduced_config(fixture_dataset(), dir.child("out"));
        config.seed = Some(seed);
        let outcome = train(&config, &mut |_| {}).unwrap();
        // Keep the directory alive until the outcome is read.
        let steps = outcome.steps.clone();
        drop(dir);
        steps
    };
    assert_eq!(run(1000, "seed-a"), run(1000, "seed-b"));
    assert_ne!(run(1000, "seed-c"), run(7, "seed-d"));
}

#[test]
fn a_two_step_run_takes_two_steps_and_keeps_training() {
    let dir = TempDir::new("two-steps");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.steps = 2;
    config.save_freq = 2.into();
    let outcome = train(&config, &mut |_| {}).unwrap();
    assert_eq!(outcome.steps.len(), 2);
    assert_eq!(outcome.steps[0].step, 1);
    assert_eq!(outcome.steps[1].step, 2);
    assert!(outcome.steps.iter().all(|step| step.parameter_delta > 0.0));
    assert!(outcome.steps.iter().all(|step| step.loss.is_finite()));
}

#[test]
fn a_saved_checkpoint_resumes_at_the_next_step_with_optimizer_rng_and_sampler_state() {
    let dir = TempDir::new("resume");
    let first_output = dir.child("first");
    let mut first_config = reduced_config(fixture_dataset(), first_output.clone());
    first_config.steps = 1;
    let first = train(&first_config, &mut |_| {}).expect("the first run completes");
    let checkpoint = first.checkpoints[0].clone();

    let mut resumed_config = reduced_config(fixture_dataset(), dir.child("resumed"));
    resumed_config.resume = true;
    resumed_config.checkpoint_path = Some(checkpoint);
    resumed_config.steps = 2;
    let resumed = train(&resumed_config, &mut |_| {}).expect("the checkpoint resumes");

    assert_eq!(resumed.steps.len(), 1, "step one must not be repeated");
    assert_eq!(resumed.steps[0].step, 2);
    assert_eq!(
        resumed.steps[0].frame_indices.len(),
        resumed_config.batch_size
    );
    assert_eq!(
        resumed.checkpoints,
        vec![dir.child("resumed/checkpoints/000002")]
    );
    let state = TrainingStep::read(&resumed.checkpoints[0].join("training_state")).unwrap();
    assert_eq!(state.step, 2);

    let tensors = candle_core::safetensors::load(
        resumed.checkpoints[0].join("training_state/optimizer_state.safetensors"),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    let step = tensors["state/0/step"]
        .to_scalar::<f32>()
        .expect("the restored optimizer took a second step");
    assert_eq!(step, 2.0);
}

#[test]
fn the_batch_cycles_through_epochs_when_the_dataset_is_smaller_than_the_run() {
    // Four frames, batch size three: the third step has to have wrapped into a
    // second epoch, and `cycle` must not stall or repeat a frame within a batch.
    let dir = TempDir::new("cycling");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.batch_size = 3;
    config.steps = 3;
    config.save_freq = 3.into();
    let outcome = train(&config, &mut |_| {}).unwrap();
    assert_eq!(outcome.steps.len(), 3);
    for step in &outcome.steps {
        assert_eq!(step.frame_indices.len(), 3);
        let mut sorted = step.frame_indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            3,
            "a batch repeated a frame: {:?}",
            step.frame_indices
        );
    }
}

#[test]
fn the_run_reports_the_dataset_and_model_size_it_used() {
    let (_dir, outcome, logs) = train_once("reporting");
    assert_eq!(outcome.num_frames, 4);
    assert_eq!(outcome.num_episodes, 1);
    assert!(outcome.num_parameters > 0);
    assert!(
        logs.iter().any(|line| line == "dataset.num_frames=4"),
        "the log does not report the frame count: {logs:?}"
    );
    assert!(
        logs.iter().any(|line| line == "dataset.num_episodes=1"),
        "the log does not report the episode count: {logs:?}"
    );
    assert!(
        logs.iter()
            .any(|line| line.starts_with("num_learnable_params=")),
        "the log does not report the parameter count: {logs:?}"
    );
    assert!(
        logs.iter().any(|line| line.starts_with("step:1 loss:")),
        "the log does not report the step: {logs:?}"
    );
    assert_eq!(
        logs.last().map(String::as_str),
        Some("End of training"),
        "the log does not end the way upstream's does"
    );
}

// ---------------------------------------------------------------------------
// The checkpoint
// ---------------------------------------------------------------------------

#[test]
fn the_checkpoint_has_upstreams_directory_layout() {
    let (_dir, outcome, _) = train_once("layout");
    assert_eq!(outcome.checkpoints.len(), 1);
    let directory = &outcome.checkpoints[0];
    assert_eq!(
        directory.file_name().unwrap().to_string_lossy(),
        "000001",
        "the step directory is zero-padded to at least six digits"
    );
    for relative in [
        "pretrained_model/config.json",
        "pretrained_model/model.safetensors",
        "pretrained_model/train_config.json",
        // `save_checkpoint` passes both processors, so their artifacts are part of
        // the layout rather than an extra. Without them the checkpoint has lost the
        // statistics the weights were trained against.
        "pretrained_model/policy_preprocessor.json",
        "pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors",
        "pretrained_model/policy_postprocessor.json",
        "pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "training_state/training_step.json",
        "training_state/rng_state.safetensors",
        "training_state/optimizer_state.safetensors",
        "training_state/optimizer_param_groups.json",
    ] {
        assert!(
            directory.join(relative).is_file(),
            "the checkpoint has no {relative}"
        );
    }
}

#[test]
fn the_saved_normalizer_state_is_the_statistics_the_run_normalized_with() {
    // The artifacts are not decoration: a deployment reads them to reproduce the
    // normalization the weights were trained under, so the values must be the
    // dataset's own rather than a default.
    let (_dir, outcome, _) = train_once("processor-state");
    let path = outcome.checkpoints[0]
        .join("pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors");
    let state = candle_core::safetensors::load(&path, &candle_core::Device::Cpu).expect("it reads");
    let read =
        |key: &str| -> Vec<f32> { state[key].flatten_all().unwrap().to_vec1::<f32>().unwrap() };
    assert_eq!(read("observation.state.mean"), vec![0.4375, 0.5625]);
    assert_eq!(read("action.mean"), vec![0.0625, -0.0625]);
    assert_eq!(read("observation.environment_state.mean"), vec![11.5, -2.5]);
}

#[test]
fn the_last_marker_resolves_to_the_checkpoint_that_was_written() {
    let (_dir, outcome, _) = train_once("last-marker");
    let directory = &outcome.checkpoints[0];
    let checkpoints = directory.parent().unwrap();
    let resolved = checkpoint::read_last_checkpoint(checkpoints).unwrap();
    assert!(
        resolved
            .join("pretrained_model/model.safetensors")
            .is_file(),
        "the last marker does not resolve to a checkpoint"
    );
}

#[test]
fn the_portable_last_marker_resolves_the_same_way_as_the_symlink() {
    // The fallback branch is what Windows takes, and it is unreachable on a Unix
    // runner unless it is forced. Forcing it is what keeps it tested.
    let (_dir, outcome, _) = train_once("portable-marker");
    let directory = &outcome.checkpoints[0];
    let checkpoints = directory.parent().unwrap();

    let kind =
        checkpoint::write_last_checkpoint(directory, LastCheckpointKind::PortableFile).unwrap();
    assert_eq!(kind, LastCheckpointKind::PortableFile);
    let marker = checkpoints.join("last");
    assert!(
        marker.is_file() && !marker.is_dir(),
        "the portable marker should be a plain file"
    );
    assert_eq!(
        std::fs::read_to_string(&marker).unwrap().trim(),
        "000001",
        "the marker should name the target directory"
    );
    assert_eq!(
        checkpoint::read_last_checkpoint(checkpoints).unwrap(),
        directory.clone()
    );
}

#[test]
fn training_step_json_records_the_step_and_the_batch_size() {
    let (_dir, outcome, _) = train_once("training-step");
    let training_state = outcome.checkpoints[0].join("training_state");
    let recorded = TrainingStep::read(&training_state).unwrap();
    assert_eq!(
        recorded,
        TrainingStep {
            step: 1,
            num_processes: 1,
            batch_size: 2,
        }
    );
}

#[test]
fn the_rng_state_round_trips_so_the_stream_resumes_exactly() {
    let (_dir, outcome, _) = train_once("rng-state");
    let training_state = outcome.checkpoints[0].join("training_state");
    let mut restored = checkpoint::read_rng_state(&training_state).unwrap();
    let mut same = rerobot_core::random::SplitMix64::from_state(restored.state());
    assert_eq!(restored.next_u64(), same.next_u64());
}

#[test]
fn the_optimizer_state_holds_a_step_and_two_moments_per_parameter() {
    let (_dir, outcome, _) = train_once("optim-state");
    let path = outcome.checkpoints[0].join("training_state/optimizer_state.safetensors");
    let tensors = candle_core::safetensors::load(&path, &candle_core::Device::Cpu).unwrap();
    let step_keys: Vec<&String> = tensors
        .keys()
        .filter(|key| key.ends_with("/step"))
        .collect();
    assert!(
        !step_keys.is_empty(),
        "no per-parameter step counters were written"
    );
    // Every parameter that took a step has all three tensors, under torch's names.
    for key in &step_keys {
        let prefix = key.trim_end_matches("step");
        assert!(
            tensors.contains_key(&format!("{prefix}exp_avg")),
            "{prefix} has no exp_avg"
        );
        assert!(
            tensors.contains_key(&format!("{prefix}exp_avg_sq")),
            "{prefix} has no exp_avg_sq"
        );
    }
    // After one step every counter is one.
    for key in &step_keys {
        let value = tensors[*key]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(value, vec![1.0], "{key} should be 1 after one step");
    }
}

#[test]
fn the_param_groups_json_is_the_two_groups_torch_would_record() {
    let (_dir, outcome, _) = train_once("param-groups");
    let path = outcome.checkpoints[0].join("training_state/optimizer_param_groups.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let JsonLike::Array(groups) = loads(&text).unwrap() else {
        panic!("param groups should be a JSON list");
    };
    assert_eq!(groups.len(), 2, "the main group and the backbone group");

    let JsonLike::Object(main) = &groups[0] else {
        panic!("a group should be a JSON object");
    };
    assert_eq!(
        main.keys().collect::<Vec<_>>(),
        vec![
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
            // `decoupled_weight_decay` is not decoration, and it goes here rather
            // than anywhere else because this is where `torch.optim.AdamW` puts it.
            // `lerobot.optim.optimizers.load_optimizer_state` reaches
            // `Optimizer.load_state_dict`, which compares the saved group's key
            // *set* against the live optimizer's and raises
            // `ValueError: Dictionary keys do not match.` when they differ. Omitting
            // it makes the checkpoint unresumable by upstream.
            "decoupled_weight_decay",
            "params",
        ],
        "the key order is torch's own"
    );
    assert_eq!(
        main["decoupled_weight_decay"],
        JsonLike::Bool(true),
        "torch.optim.AdamW defaults it to True, and AdamW *is* decoupled decay"
    );
    assert_eq!(main["lr"], JsonLike::Float(1e-5));
    assert_eq!(main["weight_decay"], JsonLike::Float(1e-4));
    assert_eq!(main["eps"], JsonLike::Float(1e-8));
    assert_eq!(
        main["betas"],
        JsonLike::Array(vec![JsonLike::Float(0.9), JsonLike::Float(0.999)])
    );
    let JsonLike::Array(params) = &main["params"] else {
        panic!("params should be a list");
    };
    assert!(!params.is_empty());

    let JsonLike::Object(backbone) = &groups[1] else {
        panic!("a group should be a JSON object");
    };
    assert_eq!(
        backbone["params"],
        JsonLike::Array(Vec::new()),
        "a state-only config has no backbone, so the second group is empty"
    );
}

#[test]
fn the_saved_policy_config_is_the_one_the_model_was_built_from() {
    let (_dir, outcome, _) = train_once("saved-config");
    let text = std::fs::read_to_string(outcome.checkpoints[0].join("pretrained_model/config.json"))
        .unwrap();
    // Byte-exactness of this file is already pinned by `rerobot-core`'s
    // `act_checkpoint.rs`; what matters here is that the *features* were resolved
    // from the dataset rather than left at their empty default.
    let reloaded = rerobot_core::policy::act::ActConfig::from_checkpoint_json(&text).unwrap();
    assert_eq!(
        reloaded
            .input_features
            .as_ref()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        vec!["observation.state", "observation.environment_state"]
    );
    assert_eq!(
        reloaded
            .output_features
            .as_ref()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        vec!["action"]
    );
    assert_eq!(reloaded.chunk_size, rerobot_core::BigInt::from(2));
    assert_eq!(reloaded.dim_model, rerobot_core::BigInt::from(32));
    assert!(text.starts_with("{\n    \"type\": \"act\","));
}

#[test]
fn the_saved_train_config_carries_upstreams_full_field_set() {
    let (_dir, outcome, _) = train_once("saved-train-config");
    let text =
        std::fs::read_to_string(outcome.checkpoints[0].join("pretrained_model/train_config.json"))
            .unwrap();
    let JsonLike::Object(root) = loads(&text).unwrap() else {
        panic!("train_config.json should be an object");
    };
    // Upstream's `TrainPipelineConfig` field order, so that a checkpoint written
    // here is one Draccus can read back.
    assert_eq!(
        root.keys().collect::<Vec<_>>(),
        vec![
            "dataset",
            "env",
            "policy",
            "reward_model",
            "output_dir",
            "job_name",
            "resume",
            "seed",
            "cudnn_deterministic",
            "num_workers",
            "batch_size",
            "prefetch_factor",
            "persistent_workers",
            "dataloader_multiprocessing_context",
            "steps",
            "env_eval_freq",
            "log_freq",
            "eval_steps",
            "max_eval_samples",
            "tolerance_s",
            "save_checkpoint",
            "save_freq",
            "use_policy_training_preset",
            "optimizer",
            "scheduler",
            "eval",
            "wandb",
            "peft",
            "job",
            "save_checkpoint_to_hub",
            "sample_weighting",
            "rename_map",
        ]
    );
    assert_eq!(root["job_name"], JsonLike::Str("act".into()));
    assert_eq!(root["steps"], JsonLike::Int(1.into()));
    assert_eq!(root["batch_size"], JsonLike::Int(2.into()));
    assert_eq!(root["num_workers"], JsonLike::Int(0.into()));
    assert_eq!(root["resume"], JsonLike::Bool(false));
    // Every field this slice does not implement is present at its upstream default.
    for field in rerobot_train::config::TrainConfig::unimplemented_fields() {
        assert!(
            root.contains_key(*field),
            "train_config.json is missing {field}, so upstream could not read it back"
        );
    }
}

#[test]
fn the_saved_weights_reload_into_a_model_that_predicts_identically() {
    // This is the round trip that matters: the file a run wrote is the file a run
    // can resume from.
    let dir = TempDir::new("reload");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().unwrap();
    let mut session = TrainSession::new(&config).unwrap();
    session.step(1).unwrap();

    let path = dir.child("model.safetensors");
    session.model.save(&path).unwrap();

    let batch = session.next_batch().unwrap();
    let expected = session
        .model
        .predict_action_steps(&batch)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut fresh = {
        let mut rng = rerobot_core::random::SplitMix64::new(31337);
        let (inputs, outputs) = session.dataset.metadata().policy_feature_split();
        let mut policy = config.policy.clone();
        policy.input_features = Some(inputs);
        policy.output_features = Some(outputs);
        ActModel::new(&policy, &candle_core::Device::Cpu, &mut rng).unwrap()
    };
    assert!(
        state_dict_distance(
            &session.model.state_dict().unwrap(),
            &fresh.state_dict().unwrap()
        )
        .unwrap()
            > 0.0,
        "the fresh model must differ before the load"
    );
    fresh.load(&path).unwrap();
    assert_eq!(
        state_dict_distance(
            &session.model.state_dict().unwrap(),
            &fresh.state_dict().unwrap()
        )
        .unwrap(),
        0.0
    );
    let actual = fresh
        .predict_action_steps(&batch)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn the_checkpoints_own_weights_reload_and_predict_identically() {
    // The same round trip, but through the checkpoint the *run* wrote rather than
    // one the test wrote.
    let dir = TempDir::new("reload-checkpoint");
    let config = reduced_config(fixture_dataset(), dir.child("out"));
    let outcome = train(&config, &mut |_| {}).unwrap();
    let weights = outcome.checkpoints[0].join("pretrained_model/model.safetensors");
    let config_json =
        std::fs::read_to_string(outcome.checkpoints[0].join("pretrained_model/config.json"))
            .unwrap();

    // Rebuild the policy from the checkpoint's own config, as a resume would.
    let policy = rerobot_core::policy::act::ActConfig::from_checkpoint_json(&config_json).unwrap();
    let mut rng = rerobot_core::random::SplitMix64::new(0);
    let mut model = ActModel::new(&policy, &candle_core::Device::Cpu, &mut rng).unwrap();
    model
        .load(&weights)
        .expect("the checkpoint the run wrote loads into a model built from its own config");

    let windows = indexmap::IndexMap::from([(
        "action".to_owned(),
        rerobot_core::dataset::delta::action_delta_timestamps(2, 10),
    )]);
    let dataset =
        rerobot_train::data::dataset::StateOnlyDataset::load(&fixture_dataset(), &windows, 1e-4)
            .unwrap();
    let frames: Vec<_> = (0..2).map(|index| dataset.get(index).unwrap()).collect();
    let batch = rerobot_train::data::batch::collate(&frames, &candle_core::Device::Cpu).unwrap();
    let predictions = model
        .predict_action_steps(&batch)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(predictions.len(), 2 * 2 * 2);
    assert!(predictions.iter().all(|value| value.is_finite()));
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn an_existing_output_directory_is_refused_without_resume() {
    let dir = TempDir::new("existing-output");
    let output = dir.child("out");
    std::fs::create_dir_all(&output).unwrap();
    let config = reduced_config(fixture_dataset(), output);
    let error = train(&config, &mut |_| {}).unwrap_err();
    assert!(
        error.to_string().contains("resume is false"),
        "unexpected error: {error}"
    );
}

/// A default build has no CUDA backend compiled in, so the run stops and says
/// how to get one. The alternative -- training on the CPU and reporting success
/// -- is the failure this whole check exists to prevent.
#[cfg(not(feature = "cuda"))]
#[test]
fn a_cuda_run_on_a_build_without_cuda_is_refused_and_names_the_rebuild() {
    let dir = TempDir::new("cuda");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.policy.device = Some("cuda".to_owned());
    let error = train(&config, &mut |_| {}).unwrap_err();
    assert!(matches!(error, TrainError::Unsupported(_)));
    let message = error.to_string();
    assert!(
        message.contains("only \"cpu\" is accepted"),
        "unexpected error: {message}"
    );
    assert!(
        message.contains("--features cuda"),
        "the refusal must name the rebuild: {message}"
    );
}

/// Not a device this port has any backend for, in either build.
#[test]
fn a_device_that_is_not_ported_at_all_is_refused_with_the_reason() {
    let dir = TempDir::new("mps");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.policy.device = Some("mps".to_owned());
    let error = train(&config, &mut |_| {}).unwrap_err();
    assert!(matches!(error, TrainError::Unsupported(_)));
    assert!(
        error.to_string().contains("\"mps\""),
        "unexpected error: {error}"
    );
}

/// `TrainConfig`'s fields are public, so a library caller can reach
/// `TrainSession::new` without going through `validate`. The session resolves
/// the device itself for exactly that reason.
#[cfg(not(feature = "cuda"))]
#[test]
fn a_session_built_by_hand_refuses_cuda_rather_than_falling_back_to_the_cpu() {
    let dir = TempDir::new("session-cuda");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.policy.device = Some("cuda".to_owned());
    // `TrainSession` is not `Debug`, so `expect_err` is not available here.
    let Err(error) = TrainSession::new(&config) else {
        panic!("a session must not silently fall back to the CPU");
    };
    assert!(matches!(error, TrainError::Unsupported(_)));
}

/// Every tensor a step touches comes from the session's device, so pinning the
/// session's device pins all of them.
#[test]
fn the_session_and_everything_it_builds_live_on_the_named_device() {
    let dir = TempDir::new("session-device");
    let config = reduced_config(fixture_dataset(), dir.child("out"));
    assert_eq!(config.policy.device.as_deref(), Some("cpu"));
    let mut session = TrainSession::new(&config).expect("the session builds");

    let device = session.device().clone();
    assert!(device.is_cpu());
    assert!(session.model.device().same_device(&device));
    for (name, parameter) in session.model.state_dict().expect("the state dict builds") {
        assert!(
            parameter.device().same_device(&device),
            "parameter {name} is not on the session device"
        );
    }
    let batch = session.next_batch().expect("a batch collates");
    for (name, tensor) in &batch.features {
        assert!(
            tensor.device().same_device(&device),
            "batch feature {name} is not on the session device"
        );
    }
    for (name, tensor) in &batch.padding {
        assert!(
            tensor.device().same_device(&device),
            "padding mask {name} is not on the session device"
        );
    }
    let normalized = batch
        .normalized(&session.normalizer)
        .expect("normalization succeeds");
    for (name, tensor) in &normalized.features {
        assert!(
            tensor.device().same_device(&device),
            "normalized feature {name} left the session device"
        );
    }
}

#[test]
fn dataloader_workers_are_refused_rather_than_ignored() {
    let dir = TempDir::new("workers");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.num_workers = 4;
    let error = train(&config, &mut |_| {}).unwrap_err();
    assert!(
        error.to_string().contains("calling thread"),
        "unexpected error: {error}"
    );
}

#[test]
fn mixed_precision_and_peft_are_refused() {
    for (label, mutate) in [
        (
            "amp",
            Box::new(|config: &mut rerobot_train::config::TrainConfig| config.policy.use_amp = true)
                as Box<dyn Fn(&mut rerobot_train::config::TrainConfig)>,
        ),
        (
            "peft",
            Box::new(|config: &mut rerobot_train::config::TrainConfig| {
                config.policy.use_peft = true
            }),
        ),
    ] {
        let dir = TempDir::new(label);
        let mut config = reduced_config(fixture_dataset(), dir.child("out"));
        mutate(&mut config);
        let error = train(&config, &mut |_| {}).unwrap_err();
        assert!(
            matches!(error, TrainError::Unsupported(_)),
            "{label} was not refused: {error}"
        );
    }
}

#[test]
fn pushing_to_the_hub_is_refused_and_a_missing_repo_id_is_upstreams_message() {
    let dir = TempDir::new("hub");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.policy.push_to_hub = true;
    let error = train(&config, &mut |_| {}).unwrap_err();
    assert_eq!(
        error.to_string(),
        "'repo_id' argument missing. Please specify it to push the model to the hub."
    );

    let dir = TempDir::new("hub-with-repo");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.policy.push_to_hub = true;
    config.policy.repo_id = Some("someone/act".to_owned());
    let error = train(&config, &mut |_| {}).unwrap_err();
    assert!(
        error.to_string().contains("no Hub client"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_hand_specified_optimizer_is_refused_because_the_registry_is_not_ported() {
    let dir = TempDir::new("no-preset");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.use_policy_training_preset = false;
    let error = train(&config, &mut |_| {}).unwrap_err();
    assert!(matches!(error, TrainError::Unsupported(_)));
    assert!(error.to_string().contains("optimizer registry"));
}

#[test]
fn nothing_is_written_when_the_configuration_is_refused() {
    let dir = TempDir::new("no-writes");
    let output = dir.child("out");
    let mut config = reduced_config(fixture_dataset(), output.clone());
    config.policy.device = Some("mps".to_owned());
    assert!(train(&config, &mut |_| {}).is_err());
    assert!(
        !output.exists(),
        "a refused run must not leave an output directory behind"
    );
}

// ---------------------------------------------------------------------------
// The checkpoint helpers, directly
// ---------------------------------------------------------------------------

#[test]
fn step_identifiers_are_padded_to_six_digits_or_the_width_of_the_total() {
    assert_eq!(checkpoint::step_identifier(1, 1), "000001");
    assert_eq!(checkpoint::step_identifier(1, 100_000), "000001");
    assert_eq!(checkpoint::step_identifier(42, 100_000), "000042");
    // Seven digits of total steps widen the identifier to seven.
    assert_eq!(checkpoint::step_identifier(42, 1_000_000), "0000042");
}

#[test]
fn the_checkpoint_directory_is_output_dir_checkpoints_step() {
    let directory =
        checkpoint::step_checkpoint_dir(std::path::Path::new("outputs/train/x"), 100_000, 5000);
    assert_eq!(
        directory,
        std::path::Path::new("outputs/train/x/checkpoints/005000")
    );
}

#[test]
fn an_absent_last_marker_is_an_error_not_an_empty_path() {
    let dir = TempDir::new("no-last");
    let error = checkpoint::read_last_checkpoint(dir.path()).unwrap_err();
    assert!(
        matches!(error, TrainError::Io { .. }),
        "unexpected: {error}"
    );
}
