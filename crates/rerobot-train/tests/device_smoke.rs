//! A whole training step, asserted to have happened *on the device that was
//! asked for*.
//!
//! The body — forward, loss, backward, gradient clip, AdamW update, checkpoint
//! write, and reload of what was written — is one function, [`one_step_on`],
//! compiled into every build. Two tests call it:
//!
//! * `the_cpu_path_...` runs everywhere, including CI, and is what keeps the
//!   shared body compiling and correct.
//! * `the_cuda_path_...` exists only when the crate's `cuda` feature is on, so a
//!   green default run says nothing whatever about a GPU. It is not `#[ignore]`d:
//!   building with `--features cuda` is a deliberate act that needs the NVIDIA
//!   toolkit, and a GPU test that is compiled and then skipped is exactly the
//!   "we tested CUDA" illusion this file is arranged to avoid.
//!
//! To run it:
//!
//! ```text
//! cargo test -p rerobot-train --features cuda --test device_smoke --locked
//! ```
//!
//! on a host with an NVIDIA GPU and the CUDA toolkit.

mod common;

use common::{fixture_dataset, reduced_config, TempDir};
use rerobot_train::checkpoint;
use rerobot_train::model::act::ActModel;
use rerobot_train::run::{save_checkpoint, TrainSession};

/// One genuine optimization step and checkpoint on the device `spec` names.
///
/// Every assertion is about something a device mix-up would break: a tensor
/// living somewhere other than the session's device, a step that ran but moved
/// no weights, or a checkpoint that cannot be read back.
fn one_step_on(spec: &str, label: &str) {
    let dir = TempDir::new(label);
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.policy.device = Some(spec.to_owned());
    config.validate().expect("the configuration validates");

    let mut session = TrainSession::new(&config).expect("the session builds on the named device");
    let device = session.device().clone();

    // 1. Everything the session built is on the one device.
    for (name, parameter) in session.model.state_dict().expect("the state dict builds") {
        assert!(
            parameter.device().same_device(&device),
            "parameter {name} is not on the {spec} device"
        );
    }

    // 2. The step: forward, loss, backward, clip, AdamW, and the finite checks
    //    around them. `TrainSession::step` is the real loop body, not a copy.
    let metrics = session.step(1).expect("the step completes");
    assert!(
        metrics.loss.is_finite() && metrics.loss > 0.0,
        "loss {} is not a finite positive number",
        metrics.loss
    );
    assert!(metrics.l1_loss.is_finite() && metrics.l1_loss >= 0.0);
    assert!(
        metrics.kld_loss.expect("the VAE is on").is_finite(),
        "the KL term is not finite"
    );
    assert!(
        metrics.grad_norm.is_finite() && metrics.grad_norm > 0.0,
        "grad_norm {} is not a finite positive number; nothing was differentiated",
        metrics.grad_norm
    );
    assert!(
        metrics.parameter_delta > 0.0,
        "the optimizer ran but the weights did not move"
    );

    // 3. The AdamW moments were allocated where the gradients were.
    for (name, tensor) in session
        .optimizer
        .state_tensors(&device)
        .expect("the optimizer state serializes")
    {
        assert!(
            tensor.device().same_device(&device),
            "optimizer state {name} is not on the {spec} device"
        );
    }

    // 4. The checkpoint, through the real writer, and read back. safetensors
    //    serialization copies to host memory at the boundary; the round trip is
    //    what proves the copy is complete and correctly shaped.
    let directory = dir.child("checkpoint");
    save_checkpoint(&config, &session, 1, &directory).expect("the checkpoint writes");

    let pretrained = directory.join(checkpoint::PRETRAINED_MODEL_DIR);
    let training_state = directory.join(checkpoint::TRAINING_STATE_DIR);
    for file in [
        pretrained.join(checkpoint::MODEL_FILE),
        training_state.join(checkpoint::OPTIMIZER_STATE),
        training_state.join(checkpoint::RNG_STATE),
    ] {
        let written =
            std::fs::metadata(&file).unwrap_or_else(|error| panic!("{}: {error}", file.display()));
        assert!(written.len() > 0, "{} is empty", file.display());
    }

    let saved = session.model.state_dict().expect("the state dict builds");
    let mut reloaded = ActModel::new(
        &config_with_features(&config, &session),
        &device,
        &mut rerobot_core::random::SplitMix64::new(7),
    )
    .expect("a second model builds on the same device");
    reloaded
        .load(&pretrained.join(checkpoint::MODEL_FILE))
        .expect("the written weights reload");
    let round_tripped = reloaded.state_dict().expect("the state dict builds");
    assert_eq!(saved.len(), round_tripped.len());
    for (name, tensor) in &round_tripped {
        assert!(
            tensor.device().same_device(&device),
            "reloaded parameter {name} did not land on the {spec} device"
        );
        let before = saved[name]
            .flatten_all()
            .and_then(|flat| flat.to_vec1::<f32>())
            .expect("the saved parameter reads");
        let after = tensor
            .flatten_all()
            .and_then(|flat| flat.to_vec1::<f32>())
            .expect("the reloaded parameter reads");
        assert_eq!(
            before, after,
            "parameter {name} did not survive the round trip"
        );
    }
}

/// The policy config the session actually built from: features resolved from the
/// dataset, which `TrainConfig` alone does not carry.
fn config_with_features(
    config: &rerobot_train::config::TrainConfig,
    session: &TrainSession,
) -> rerobot_core::policy::act::ActConfig {
    let (inputs, outputs) = session.dataset.metadata().policy_feature_split();
    let mut policy = config.policy.clone();
    policy.input_features = Some(inputs);
    policy.output_features = Some(outputs);
    policy
}

#[test]
fn the_cpu_path_runs_a_whole_step_and_writes_a_checkpoint_that_reloads() {
    one_step_on("cpu", "smoke-cpu");
}

/// Only compiled with the `cuda` feature, and it needs a real NVIDIA GPU: a
/// missing driver makes `TrainSession::new` fail rather than fall back, so this
/// test cannot pass without one.
#[cfg(feature = "cuda")]
#[test]
fn the_cuda_path_runs_a_whole_step_on_the_gpu_and_writes_a_checkpoint_that_reloads() {
    one_step_on("cuda", "smoke-cuda");
    // The same run through the `cuda:0` spelling, which upstream also accepts.
    one_step_on("cuda:0", "smoke-cuda-zero");
}

/// The device really was the GPU, not a CPU session that happened to work.
#[cfg(feature = "cuda")]
#[test]
fn a_cuda_session_puts_its_parameters_on_the_gpu() {
    let dir = TempDir::new("smoke-cuda-device");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.policy.device = Some("cuda".to_owned());
    let session = TrainSession::new(&config).expect("the CUDA session builds");
    assert!(session.device().is_cuda(), "the session is not on a GPU");
    assert!(session.model.device().is_cuda());
    for (name, parameter) in session.model.state_dict().expect("the state dict builds") {
        assert!(
            parameter.device().is_cuda(),
            "parameter {name} is on the CPU"
        );
    }
}
