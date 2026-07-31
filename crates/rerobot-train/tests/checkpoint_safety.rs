//! What the checkpoint layer refuses to do to the filesystem, and what it refuses to
//! believe about a file it reads.
//!
//! Two different hazards, both fail-closed:
//!
//! **Writing.** `checkpoints/last` is a *reserved* path, and the code that maintains
//! it used to `remove_dir_all` whatever it found there. A caller-controlled directory
//! at that path — or one substituted between the check and the removal — was
//! therefore recursively deleted. Nothing about maintaining a marker requires
//! deleting a tree, so it must not be able to.
//!
//! **Reading.** A checkpoint is data from outside the process. Loading it validated
//! names and shapes but not dtypes, accepted extra tensors in the RNG file, used only
//! the first element of whatever shape it found, and ignored malformed optimizer
//! entries entirely. Each of those turns a corrupt file into a silently wrong resume
//! rather than an error.

mod common;

use candle_core::{DType, Device, Tensor};
use common::{fixture_dataset, reduced_config, TempDir};
use rerobot_core::random::SplitMix64;
use rerobot_train::checkpoint::{self, LastCheckpointKind};
use rerobot_train::model::act::ActModel;
use rerobot_train::optim::{act_parameter_groups, AdamW};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// `checkpoints/last` must never be a recursive delete
// ---------------------------------------------------------------------------

/// A `checkpoints/` directory holding one real checkpoint directory.
fn checkpoints_with_one(dir: &TempDir) -> (PathBuf, PathBuf) {
    let checkpoints = dir.child("checkpoints");
    let step = checkpoints.join("000001");
    std::fs::create_dir_all(step.join("pretrained_model")).unwrap();
    std::fs::write(step.join("pretrained_model/config.json"), "{}").unwrap();
    (checkpoints, step)
}

#[test]
fn a_real_directory_at_the_reserved_last_path_is_refused_not_deleted() {
    let dir = TempDir::new("last-is-a-dir");
    let (checkpoints, step) = checkpoints_with_one(&dir);

    // Something valuable sitting where the marker goes.
    let squatter = checkpoints.join("last");
    std::fs::create_dir_all(squatter.join("deep/nested")).unwrap();
    std::fs::write(
        squatter.join("deep/nested/precious.txt"),
        "do not delete me",
    )
    .unwrap();

    let error = checkpoint::update_last_checkpoint(&step)
        .expect_err("a real directory at the reserved path must be refused");
    let message = error.to_string();
    assert!(
        message.contains("last"),
        "the refusal does not name the path: {message}"
    );
    assert!(
        message.contains("directory"),
        "the refusal does not say what it found: {message}"
    );

    // And nothing was removed.
    assert!(
        squatter.join("deep/nested/precious.txt").is_file(),
        "the pre-existing tree at checkpoints/last was deleted"
    );
    assert_eq!(
        std::fs::read_to_string(squatter.join("deep/nested/precious.txt")).unwrap(),
        "do not delete me"
    );
}

#[test]
fn the_forced_portable_marker_also_refuses_a_real_directory() {
    // The Windows path takes this branch, so it needs the same guarantee.
    let dir = TempDir::new("portable-is-a-dir");
    let (checkpoints, step) = checkpoints_with_one(&dir);
    let squatter = checkpoints.join("last");
    std::fs::create_dir_all(&squatter).unwrap();
    std::fs::write(squatter.join("keep.txt"), "keep").unwrap();

    let error = checkpoint::write_last_checkpoint(&step, LastCheckpointKind::PortableFile)
        .expect_err("a real directory at the reserved path must be refused");
    assert!(error.to_string().contains("directory"), "{error}");
    assert!(squatter.join("keep.txt").is_file(), "the tree was deleted");
}

#[test]
fn a_stale_symlink_marker_is_replaced_because_unlinking_one_deletes_nothing() {
    // The marker must still be maintainable: replacing a symlink removes the link,
    // never its target, so this is safe and has to keep working.
    let dir = TempDir::new("stale-symlink");
    let (checkpoints, step) = checkpoints_with_one(&dir);
    let second = checkpoints.join("000002");
    std::fs::create_dir_all(&second).unwrap();

    checkpoint::update_last_checkpoint(&step).expect("the first marker is written");
    assert_eq!(
        checkpoint::read_last_checkpoint(&checkpoints).unwrap(),
        step
    );

    checkpoint::update_last_checkpoint(&second).expect("the marker is replaced");
    assert_eq!(
        checkpoint::read_last_checkpoint(&checkpoints).unwrap(),
        second
    );
    // Both checkpoints survived.
    assert!(step.is_dir() && second.is_dir());
}

#[test]
fn a_stale_portable_marker_is_replaced() {
    let dir = TempDir::new("stale-portable");
    let (checkpoints, step) = checkpoints_with_one(&dir);
    let second = checkpoints.join("000002");
    std::fs::create_dir_all(&second).unwrap();

    checkpoint::write_last_checkpoint(&step, LastCheckpointKind::PortableFile).unwrap();
    checkpoint::write_last_checkpoint(&second, LastCheckpointKind::PortableFile).unwrap();
    assert_eq!(
        checkpoint::read_last_checkpoint(&checkpoints).unwrap(),
        second
    );
    assert!(step.is_dir());
}

#[test]
fn a_symlink_marker_pointing_at_a_directory_is_never_followed_when_removing_it() {
    // The subtle version of the same hazard: if the marker is a symlink *to* a real
    // directory, removing it must unlink the link and leave the target alone.
    let dir = TempDir::new("symlink-target");
    let (checkpoints, step) = checkpoints_with_one(&dir);
    let second = checkpoints.join("000002");
    std::fs::create_dir_all(&second).unwrap();
    std::fs::write(second.join("payload.txt"), "payload").unwrap();

    checkpoint::update_last_checkpoint(&second).unwrap();
    checkpoint::update_last_checkpoint(&step).unwrap();
    assert!(
        second.join("payload.txt").is_file(),
        "replacing the marker followed it and deleted its target"
    );
}

// ---------------------------------------------------------------------------
// Model loading: exact dtype
// ---------------------------------------------------------------------------

fn oracle_policy() -> rerobot_core::policy::act::ActConfig {
    let metadata = rerobot_train::data::meta::DatasetMetadata::load(&fixture_dataset()).unwrap();
    let mut config = reduced_config(fixture_dataset(), PathBuf::from("unused"));
    let (inputs, outputs) = metadata.policy_feature_split();
    config.policy.input_features = Some(inputs);
    config.policy.output_features = Some(outputs);
    config.policy
}

fn fresh_model() -> ActModel {
    let mut rng = SplitMix64::new(11);
    ActModel::new(&oracle_policy(), &Device::Cpu, &mut rng).unwrap()
}

/// Write a model's state dict with one tensor replaced.
fn state_with(model: &ActModel, path: &Path, name: &str, replacement: Tensor) {
    let mut tensors: HashMap<String, Tensor> = model
        .state_dict()
        .unwrap()
        .into_iter()
        .map(|(key, tensor)| (key, tensor.contiguous().unwrap()))
        .collect();
    tensors.insert(name.to_owned(), replacement);
    candle_core::safetensors::save(&tensors, path).unwrap();
}

#[test]
fn a_model_tensor_of_the_wrong_dtype_is_refused_rather_than_silently_cast() {
    // The loader used to `to_dtype(F32)` whatever it found. An `f64` checkpoint would
    // load with a quiet precision change, and an integer one would load as a lattice
    // of whole numbers -- neither is the model that was saved, and neither said so.
    let dir = TempDir::new("wrong-dtype");
    let path = dir.child("model.safetensors");
    let mut model = fresh_model();

    let shape = model
        .state_dict()
        .unwrap()
        .get("model.action_head.weight")
        .unwrap()
        .shape()
        .clone();
    for dtype in [DType::F64, DType::U8, DType::I64] {
        let replacement = Tensor::zeros(&shape, DType::F32, &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        state_with(&model, &path, "model.action_head.weight", replacement);

        let error = model
            .load(&path)
            .expect_err("a {dtype:?} tensor must be refused");
        let message = error.to_string();
        assert!(
            message.contains("model.action_head.weight"),
            "the refusal does not name the tensor: {message}"
        );
        assert!(
            message.contains("dtype") || message.contains("F32") || message.contains("f32"),
            "the refusal does not mention the dtype: {message}"
        );
    }
}

#[test]
fn a_model_of_the_right_dtype_still_loads() {
    let dir = TempDir::new("right-dtype");
    let path = dir.child("model.safetensors");
    let model = fresh_model();
    model.save(&path).unwrap();
    let mut target = fresh_model();
    target.load(&path).expect("an f32 checkpoint loads");
}

// ---------------------------------------------------------------------------
// RNG state
// ---------------------------------------------------------------------------

fn write_rng(path: &Path, tensors: HashMap<String, Tensor>) {
    candle_core::safetensors::save(&tensors, path).unwrap();
}

#[test]
fn an_rng_state_with_the_wrong_shape_dtype_or_extra_tensors_is_refused() {
    let dir = TempDir::new("rng");
    let state = dir.child("state");
    std::fs::create_dir_all(&state).unwrap();
    let path = state.join("rng_state.safetensors");
    let key = "rerobot_splitmix64_state";

    // A well-formed one, first, so the failures below are about what changed.
    checkpoint::write_rng_state(&state, &SplitMix64::from_state(0xABCD)).unwrap();
    assert_eq!(
        checkpoint::read_rng_state(&state).unwrap().state(),
        0xABCD,
        "the well-formed case must work"
    );

    // Wrong shape: the reader used to take element zero of anything non-empty, so a
    // two-element tensor silently restored half a state.
    write_rng(
        &path,
        HashMap::from([(
            key.to_owned(),
            Tensor::new(&[1i64, 2i64], &Device::Cpu).unwrap(),
        )]),
    );
    let error = checkpoint::read_rng_state(&state).unwrap_err();
    assert!(
        error.to_string().contains("shape") || error.to_string().contains("one element"),
        "a two-element state was not refused: {error}"
    );

    // Empty.
    write_rng(
        &path,
        HashMap::from([(
            key.to_owned(),
            Tensor::from_vec(Vec::<i64>::new(), 0, &Device::Cpu).unwrap(),
        )]),
    );
    assert!(checkpoint::read_rng_state(&state).is_err());

    // Wrong dtype: the state is a bit-cast `u64`, so reading it as a float would
    // lose the low bits of a large state without any complaint.
    write_rng(
        &path,
        HashMap::from([(
            key.to_owned(),
            Tensor::new(&[1.0f32], &Device::Cpu).unwrap(),
        )]),
    );
    let error = checkpoint::read_rng_state(&state).unwrap_err();
    assert!(
        error.to_string().contains("dtype") || error.to_string().contains("I64"),
        "a float state was not refused: {error}"
    );

    // An extra tensor means the file is not the one this reader understands.
    write_rng(
        &path,
        HashMap::from([
            (key.to_owned(), Tensor::new(&[7i64], &Device::Cpu).unwrap()),
            (
                "torch_random_state".to_owned(),
                Tensor::new(&[1i64], &Device::Cpu).unwrap(),
            ),
        ]),
    );
    let error = checkpoint::read_rng_state(&state).unwrap_err();
    assert!(
        error.to_string().contains("torch_random_state"),
        "an unexpected tensor was not reported: {error}"
    );
}

#[test]
fn every_rng_state_round_trips_including_the_extremes() {
    let dir = TempDir::new("rng-roundtrip");
    let state = dir.child("state");
    std::fs::create_dir_all(&state).unwrap();
    for value in [0u64, 1, u64::MAX, u64::MAX / 2, 0x8000_0000_0000_0000] {
        checkpoint::write_rng_state(&state, &SplitMix64::from_state(value)).unwrap();
        assert_eq!(
            checkpoint::read_rng_state(&state).unwrap().state(),
            value,
            "the state {value} did not round trip"
        );
    }
}

// ---------------------------------------------------------------------------
// Optimizer state
// ---------------------------------------------------------------------------

fn optimizer_for(model: &ActModel) -> AdamW {
    let preset = oracle_policy().optimizer_preset();
    AdamW::new(
        act_parameter_groups(model.optimizer_parameter_groups(), &preset, preset.lr),
        model.parameters().len(),
    )
    .unwrap()
}

#[test]
fn an_optimizer_state_naming_a_parameter_that_does_not_exist_is_refused() {
    // The loader accepted any `state/<n>/...` key. An index past the parameter list is
    // a checkpoint for a different model, and installing it would leave the real
    // parameters with no moments while reporting a successful resume.
    let model = fresh_model();
    let mut optimizer = optimizer_for(&model);
    let count = model.parameters().len();
    let tensors = HashMap::from([
        (
            format!("state/{count}/step"),
            Tensor::new(1.0f32, &Device::Cpu).unwrap(),
        ),
        (
            format!("state/{count}/exp_avg"),
            Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap(),
        ),
        (
            format!("state/{count}/exp_avg_sq"),
            Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap(),
        ),
    ]);
    let error = optimizer
        .load_state_tensors(model.parameters(), &tensors)
        .expect_err("an out-of-range parameter index must be refused");
    assert!(
        error.to_string().contains(&count.to_string()),
        "the refusal does not report the index: {error}"
    );
}

#[test]
fn an_optimizer_state_with_a_key_it_does_not_understand_is_refused() {
    let model = fresh_model();
    let mut optimizer = optimizer_for(&model);
    let mut tensors = well_formed_state(&model);
    tensors.insert(
        "param_groups/0/lr".to_owned(),
        Tensor::new(1.0f32, &Device::Cpu).unwrap(),
    );
    let error = optimizer
        .load_state_tensors(model.parameters(), &tensors)
        .expect_err("an unrecognized key must be refused");
    assert!(
        error.to_string().contains("param_groups/0/lr"),
        "the refusal does not name the key: {error}"
    );
}

#[test]
fn an_optimizer_state_with_a_mismatched_moment_shape_is_refused() {
    let model = fresh_model();
    let mut optimizer = optimizer_for(&model);
    let mut tensors = well_formed_state(&model);
    tensors.insert(
        "state/0/exp_avg".to_owned(),
        Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap(),
    );
    let error = optimizer
        .load_state_tensors(model.parameters(), &tensors)
        .expect_err("a mismatched moment shape must be refused");
    let message = error.to_string();
    assert!(message.contains("exp_avg"), "{message}");
    assert!(message.contains("shape"), "{message}");
}

#[test]
fn an_optimizer_state_with_a_non_f32_moment_is_refused() {
    let model = fresh_model();
    let mut optimizer = optimizer_for(&model);
    let mut tensors = well_formed_state(&model);
    let shape = model.parameters()[0].value.as_tensor().shape().clone();
    tensors.insert(
        "state/0/exp_avg".to_owned(),
        Tensor::zeros(&shape, DType::F64, &Device::Cpu).unwrap(),
    );
    let error = optimizer
        .load_state_tensors(model.parameters(), &tensors)
        .expect_err("a non-f32 moment must be refused");
    assert!(error.to_string().contains("dtype"), "{error}");
}

#[test]
fn an_optimizer_state_with_a_non_finite_step_is_refused() {
    let model = fresh_model();
    let mut optimizer = optimizer_for(&model);
    let mut tensors = well_formed_state(&model);
    tensors.insert(
        "state/0/step".to_owned(),
        Tensor::new(f32::NAN, &Device::Cpu).unwrap(),
    );
    let error = optimizer
        .load_state_tensors(model.parameters(), &tensors)
        .expect_err("a NaN step count must be refused");
    let message = error.to_string();
    assert!(message.contains("step"), "{message}");
    assert!(
        message.contains("finite") || message.contains("NaN"),
        "{message}"
    );
}

#[test]
fn an_optimizer_state_with_a_negative_step_is_refused() {
    let model = fresh_model();
    let mut optimizer = optimizer_for(&model);
    let mut tensors = well_formed_state(&model);
    tensors.insert(
        "state/0/step".to_owned(),
        Tensor::new(-1.0f32, &Device::Cpu).unwrap(),
    );
    assert!(optimizer
        .load_state_tensors(model.parameters(), &tensors)
        .is_err());
}

#[test]
fn an_optimizer_state_missing_a_moment_is_refused() {
    let model = fresh_model();
    let mut optimizer = optimizer_for(&model);
    let mut tensors = well_formed_state(&model);
    tensors.remove("state/0/exp_avg_sq");
    let error = optimizer
        .load_state_tensors(model.parameters(), &tensors)
        .expect_err("an incomplete entry must be refused");
    assert!(error.to_string().contains("exp_avg_sq"), "{error}");
}

#[test]
fn a_well_formed_optimizer_state_round_trips() {
    let model = fresh_model();
    let mut optimizer = optimizer_for(&model);
    let tensors = well_formed_state(&model);
    optimizer
        .load_state_tensors(model.parameters(), &tensors)
        .expect("the well-formed case must work");
    let written = optimizer.state_tensors(&Device::Cpu).unwrap();
    let mut theirs: Vec<&String> = tensors.keys().collect();
    let mut ours: Vec<&String> = written.keys().collect();
    theirs.sort();
    ours.sort();
    assert_eq!(ours, theirs);
}

/// A complete, valid optimizer state for every parameter of `model`.
fn well_formed_state(model: &ActModel) -> HashMap<String, Tensor> {
    let mut tensors = HashMap::new();
    for (index, parameter) in model.parameters().iter().enumerate() {
        let shape = parameter.value.as_tensor().shape().clone();
        tensors.insert(
            format!("state/{index}/step"),
            Tensor::new(1.0f32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("state/{index}/exp_avg"),
            Tensor::zeros(&shape, DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("state/{index}/exp_avg_sq"),
            Tensor::zeros(&shape, DType::F32, &Device::Cpu).unwrap(),
        );
    }
    tensors
}

// ---------------------------------------------------------------------------
// The round trip a real run performs still works
// ---------------------------------------------------------------------------

#[test]
fn the_optimizer_state_a_real_run_writes_reloads_into_the_same_model() {
    let dir = TempDir::new("real-roundtrip");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().unwrap();
    let mut session = rerobot_train::run::TrainSession::new(&config).unwrap();
    session.step(1).unwrap();

    let written = session.optimizer.state_tensors(&Device::Cpu).unwrap();
    let mut restored = optimizer_for(&session.model);
    restored
        .load_state_tensors(session.model.parameters(), &written)
        .expect("a state this run wrote must load back");
    let again = restored.state_tensors(&Device::Cpu).unwrap();
    assert_eq!(again.len(), written.len());
    for (key, tensor) in &written {
        let other = &again[key];
        let difference = (tensor - other)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(difference, 0.0, "{key} did not round trip");
    }
}

#[test]
fn the_checkpoint_a_real_run_writes_still_passes_every_reader_check() {
    // The guards must not have made a legitimate checkpoint unreadable.
    let dir = TempDir::new("real-checkpoint");
    let config = reduced_config(fixture_dataset(), dir.child("out"));
    let outcome = rerobot_train::run::train(&config, &mut |_| {}).unwrap();
    let checkpoint = &outcome.checkpoints[0];

    let training_state = checkpoint.join("training_state");
    checkpoint::read_rng_state(&training_state).expect("the RNG state reads");
    checkpoint::TrainingStep::read(&training_state).expect("the step reads");

    let config_text =
        std::fs::read_to_string(checkpoint.join("pretrained_model/config.json")).unwrap();
    let policy = rerobot_core::policy::act::ActConfig::from_checkpoint_json(&config_text).unwrap();
    let mut rng = SplitMix64::new(0);
    let mut model = ActModel::new(&policy, &Device::Cpu, &mut rng).unwrap();
    model
        .load(&checkpoint.join("pretrained_model/model.safetensors"))
        .expect("the weights load");

    let tensors = candle_core::safetensors::load(
        checkpoint.join("training_state/optimizer_state.safetensors"),
        &Device::Cpu,
    )
    .unwrap();
    let mut optimizer = optimizer_for(&model);
    optimizer
        .load_state_tensors(model.parameters(), &tensors)
        .expect("the optimizer state loads");

    // And the `last` marker the run wrote still resolves.
    checkpoint::read_last_checkpoint(checkpoint.parent().unwrap())
        .expect("the marker a real run wrote resolves");
}

// ---------------------------------------------------------------------------
// Values, not only shapes: a NaN checkpoint is a corrupt checkpoint
// ---------------------------------------------------------------------------

#[test]
fn a_model_tensor_holding_a_non_finite_value_is_refused() {
    // Names, shapes and dtypes were all validated; the *values* were not. A NaN or
    // infinite weight loads into a model that then produces NaN for every input, and
    // the run's own non-finite tripwire fires on the first step -- so the failure
    // surfaces far from its cause, as a training divergence rather than as the corrupt
    // file it is. A checkpoint is data from outside the process; this is the last point
    // at which the real reason can still be reported.
    let dir = TempDir::new("model-nonfinite");
    let path = dir.child("model.safetensors");
    let mut model = fresh_model();
    let shape = model
        .state_dict()
        .unwrap()
        .get("model.action_head.weight")
        .unwrap()
        .shape()
        .clone();

    for (label, poison) in [
        ("NaN", f32::NAN),
        ("inf", f32::INFINITY),
        ("-inf", f32::NEG_INFINITY),
    ] {
        let replacement = Tensor::full(poison, &shape, &Device::Cpu).unwrap();
        state_with(&model, &path, "model.action_head.weight", replacement);
        let error = model
            .load(&path)
            .expect_err("a checkpoint holding {label} must be refused");
        let message = error.to_string();
        assert!(
            message.contains("model.action_head.weight"),
            "{label}: the refusal does not name the tensor: {message}"
        );
        assert!(
            message.contains("finite"),
            "{label}: the refusal does not say why: {message}"
        );
    }
}

#[test]
fn a_single_non_finite_element_among_finite_ones_is_still_refused() {
    // The check has to be over every element, not a spot sample: one NaN weight is
    // enough to make the whole forward pass NaN.
    let dir = TempDir::new("model-one-nan");
    let path = dir.child("model.safetensors");
    let mut model = fresh_model();
    let dims = model
        .state_dict()
        .unwrap()
        .get("model.action_head.weight")
        .unwrap()
        .dims()
        .to_vec();
    let count: usize = dims.iter().product();
    let mut values = vec![0.25f32; count];
    *values.last_mut().unwrap() = f32::NAN;
    let replacement = Tensor::from_vec(values, dims, &Device::Cpu).unwrap();
    state_with(&model, &path, "model.action_head.weight", replacement);
    assert!(
        model.load(&path).is_err(),
        "a single NaN element was accepted"
    );
}

#[test]
fn a_finite_model_still_loads_and_the_buffer_is_checked_too() {
    let dir = TempDir::new("model-finite");
    let path = dir.child("model.safetensors");
    let model = fresh_model();
    model.save(&path).unwrap();
    let mut target = fresh_model();
    target.load(&path).expect("a finite checkpoint loads");

    // The sinusoidal position table is a buffer rather than a parameter, and a
    // corrupt one is just as fatal, so it is validated on the same footing.
    let shape = model
        .state_dict()
        .unwrap()
        .get("model.vae_encoder_pos_enc")
        .unwrap()
        .shape()
        .clone();
    state_with(
        &model,
        &path,
        "model.vae_encoder_pos_enc",
        Tensor::full(f32::NAN, &shape, &Device::Cpu).unwrap(),
    );
    let error = target
        .load(&path)
        .expect_err("a non-finite buffer must be refused too");
    assert!(
        error.to_string().contains("model.vae_encoder_pos_enc"),
        "unexpected: {error}"
    );
}

// ---------------------------------------------------------------------------
// Optimizer state: values, exact scalar step, integrality, completeness
// ---------------------------------------------------------------------------

#[test]
fn an_optimizer_moment_holding_a_non_finite_value_is_refused() {
    // A NaN moment is worse than a NaN weight: AdamW divides by `sqrt(exp_avg_sq)`,
    // so one poisoned moment turns its parameter to NaN on the next step and every
    // step after.
    let model = fresh_model();
    for slot in ["exp_avg", "exp_avg_sq"] {
        for poison in [f32::NAN, f32::INFINITY] {
            let mut optimizer = optimizer_for(&model);
            let mut tensors = well_formed_state(&model);
            let shape = model.parameters()[0].value.as_tensor().shape().clone();
            tensors.insert(
                format!("state/0/{slot}"),
                Tensor::full(poison, &shape, &Device::Cpu).unwrap(),
            );
            let error = optimizer
                .load_state_tensors(model.parameters(), &tensors)
                .expect_err("a non-finite moment must be refused");
            let message = error.to_string();
            assert!(message.contains(slot), "{message}");
            assert!(message.contains("finite"), "{message}");
        }
    }
}

#[test]
fn a_step_count_that_is_not_a_scalar_is_refused_even_at_one_element() {
    // `torch.optim.AdamW` stores `step` as a zero-dimensional tensor. A `[1]` tensor
    // holds the same one number but is not the same value, and accepting it means the
    // reader is guessing at the format rather than reading it.
    let model = fresh_model();
    let mut optimizer = optimizer_for(&model);
    let mut tensors = well_formed_state(&model);
    tensors.insert(
        "state/0/step".to_owned(),
        Tensor::from_vec(vec![1.0f32], 1, &Device::Cpu).unwrap(),
    );
    let error = optimizer
        .load_state_tensors(model.parameters(), &tensors)
        .expect_err("a rank-1 step must be refused");
    let message = error.to_string();
    assert!(message.contains("step"), "{message}");
    assert!(
        message.contains("scalar") || message.contains("[]"),
        "the refusal does not say a scalar is required: {message}"
    );
}

#[test]
fn a_fractional_step_count_is_refused() {
    // The bias corrections are `1 - beta^step`. A fractional step is not a count of
    // anything, and torch cannot have written one.
    let model = fresh_model();
    for value in [0.5f32, 1.5, 1e-3] {
        let mut optimizer = optimizer_for(&model);
        let mut tensors = well_formed_state(&model);
        tensors.insert(
            "state/0/step".to_owned(),
            Tensor::new(value, &Device::Cpu).unwrap(),
        );
        let error = optimizer
            .load_state_tensors(model.parameters(), &tensors)
            .expect_err("a fractional step must be refused");
        let message = error.to_string();
        assert!(message.contains("step"), "{message}");
        assert!(
            message.contains("whole") || message.contains("integral"),
            "the refusal does not say the step must be a whole number: {message}"
        );
    }
}

#[test]
fn an_optimizer_state_covering_only_some_parameters_is_refused() {
    // The reported probe: a *partial* state loaded successfully. Every parameter that
    // is missing keeps zero moments while the optimizer reports a restored state, so
    // a resume silently trains those parameters as if from scratch -- with the
    // bias-corrected step counts of a run that had already progressed.
    let model = fresh_model();
    let mut optimizer = optimizer_for(&model);
    let mut tensors = well_formed_state(&model);
    let last = model.parameters().len() - 1;
    for slot in ["step", "exp_avg", "exp_avg_sq"] {
        tensors.remove(&format!("state/{last}/{slot}"));
    }
    let error = optimizer
        .load_state_tensors(model.parameters(), &tensors)
        .expect_err("a partial state must be refused");
    let message = error.to_string();
    assert!(
        message.contains(&last.to_string()),
        "the refusal does not name the missing parameter: {message}"
    );
    assert!(
        message.contains("complete") || message.contains("every parameter"),
        "the refusal does not say the state must be complete: {message}"
    );
}

#[test]
fn an_empty_optimizer_state_is_accepted_because_a_fresh_run_has_none() {
    // Completeness must not mean "non-empty": `optimizer.state_dict()` of a fresh
    // optimizer has no entries, and restoring that is a no-op rather than an error.
    let model = fresh_model();
    let mut optimizer = optimizer_for(&model);
    optimizer
        .load_state_tensors(model.parameters(), &HashMap::new())
        .expect("an empty state is a fresh optimizer");
    assert!(optimizer.state_tensors(&Device::Cpu).unwrap().is_empty());
}

#[test]
fn the_state_a_real_run_writes_is_complete_and_reloads() {
    // The other half of the completeness rule: what this port actually writes must
    // satisfy it, or the rule would make every checkpoint unreadable.
    let dir = TempDir::new("complete-state");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().unwrap();
    let mut session = rerobot_train::run::TrainSession::new(&config).unwrap();
    session.step(1).unwrap();
    let written = session.optimizer.state_tensors(&Device::Cpu).unwrap();

    // Three tensors per parameter, for every parameter.
    assert_eq!(
        written.len(),
        session.model.parameters().len() * 3,
        "a real run's state does not cover every parameter"
    );
    let mut restored = optimizer_for(&session.model);
    restored
        .load_state_tensors(session.model.parameters(), &written)
        .expect("a real run's state satisfies the completeness rule");
}

// ---------------------------------------------------------------------------
// The portable marker must be written atomically
// ---------------------------------------------------------------------------

#[test]
fn writing_the_portable_marker_never_follows_a_symlink_planted_concurrently() {
    // The remaining race, and the only way to observe it: the marker was unlinked and
    // then `std::fs::write` opened the reserved path again. `write` *follows* a symlink,
    // so an attacker who plants one in that window has the marker's content written
    // into any file they can name, truncating it.
    //
    // A same-directory temporary plus `rename` closes it, because `rename` replaces a
    // name without following it and there is no `open` on the reserved path at all.
    //
    // This is a race, so it is exercised by racing: one thread plants the symlink in a
    // loop while the main thread writes the marker in a loop. The assertion is
    // one-sided — correct code can *never* truncate the victim, so this test cannot
    // fail spuriously; it can only fail when the window exists.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let dir = TempDir::new("marker-race");
    let (checkpoints, step) = checkpoints_with_one(&dir);
    let victim = dir.child("victim.txt");
    const PRECIOUS: &str = "PRECIOUS CONTENT THAT MUST SURVIVE";
    std::fs::write(&victim, PRECIOUS).unwrap();

    let link = checkpoints.join("last");
    let stop = Arc::new(AtomicBool::new(false));
    let planter = {
        let (stop, link, victim) = (Arc::clone(&stop), link.clone(), victim.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let _ = std::fs::remove_file(&link);
                #[cfg(unix)]
                let _ = std::os::unix::fs::symlink(&victim, &link);
                #[cfg(windows)]
                let _ = std::os::windows::fs::symlink_file(&victim, &link);
            }
        })
    };

    for _ in 0..3_000 {
        // The planter may leave a symlink where the marker goes; that is the point.
        let _ = checkpoint::write_last_checkpoint(&step, LastCheckpointKind::PortableFile);
    }
    stop.store(true, Ordering::Relaxed);
    planter.join().expect("the planting thread finished");

    assert_eq!(
        std::fs::read_to_string(&victim).unwrap(),
        PRECIOUS,
        "writing the marker followed a symlink and truncated its target"
    );
}

#[test]
fn a_symlink_at_the_reserved_path_is_replaced_rather_than_written_through() {
    // The non-racing half of the same guarantee: with a symlink already in place, the
    // marker must end up as a regular file naming the checkpoint, and the link's target
    // must be untouched.
    let dir = TempDir::new("marker-symlink");
    let (checkpoints, step) = checkpoints_with_one(&dir);
    let victim = dir.child("victim.txt");
    std::fs::write(&victim, "PRECIOUS").unwrap();
    let link = checkpoints.join("last");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&victim, &link).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(&victim, &link).unwrap();

    checkpoint::write_last_checkpoint(&step, LastCheckpointKind::PortableFile)
        .expect("the marker is written");

    assert_eq!(std::fs::read_to_string(&victim).unwrap(), "PRECIOUS");
    let metadata = std::fs::symlink_metadata(&link).unwrap();
    assert!(
        metadata.file_type().is_file(),
        "the marker is not a regular file after replacing a symlink"
    );
    assert_eq!(
        checkpoint::read_last_checkpoint(&checkpoints).unwrap(),
        step
    );
}

#[test]
fn the_portable_marker_leaves_no_temporary_file_behind() {
    // An atomic write leaves the directory holding exactly the marker and the
    // checkpoints, with no stray temporary.
    let dir = TempDir::new("no-temp-left");
    let (checkpoints, step) = checkpoints_with_one(&dir);
    checkpoint::write_last_checkpoint(&step, LastCheckpointKind::PortableFile).unwrap();
    let mut names: Vec<String> = std::fs::read_dir(&checkpoints)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, vec!["000001".to_owned(), "last".to_owned()]);
}

#[test]
fn replacing_a_portable_marker_repeatedly_is_stable() {
    let dir = TempDir::new("repeated-portable");
    let (checkpoints, step) = checkpoints_with_one(&dir);
    let second = checkpoints.join("000002");
    std::fs::create_dir_all(&second).unwrap();
    for target in [&step, &second, &step, &second] {
        checkpoint::write_last_checkpoint(target, LastCheckpointKind::PortableFile).unwrap();
        assert_eq!(
            &checkpoint::read_last_checkpoint(&checkpoints).unwrap(),
            target
        );
    }
}

// ---------------------------------------------------------------------------
// `training_step.json` is data, not a suggestion
// ---------------------------------------------------------------------------
//
// Upstream's `save_training_step` omits `num_processes` and `batch_size` when it has
// no value for them, so *absence* has to keep meaning "not recorded". A value that is
// present and malformed is a different thing: it means the file has been edited or
// truncated, and the reader used to substitute a default for it — reporting
// `num_processes=1, batch_size=0` for a file that said neither.

/// Write a `training_step.json` holding exactly `body`.
fn training_step_file(dir: &TempDir, label: &str, body: &str) -> PathBuf {
    let state = dir.child(label);
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(state.join("training_step.json"), body).unwrap();
    state
}

#[test]
fn a_training_step_field_of_the_wrong_type_is_refused_rather_than_defaulted() {
    let dir = TempDir::new("step-types");
    for (label, body, field) in [
        (
            "string-processes",
            r#"{"step": 1, "num_processes": "not a number", "batch_size": 2}"#,
            "num_processes",
        ),
        (
            "object-batch",
            r#"{"step": 1, "num_processes": 1, "batch_size": {"a": 1}}"#,
            "batch_size",
        ),
        (
            "float-batch",
            r#"{"step": 1, "num_processes": 1, "batch_size": 2.5}"#,
            "batch_size",
        ),
        (
            "null-processes",
            r#"{"step": 1, "num_processes": null, "batch_size": 2}"#,
            "num_processes",
        ),
    ] {
        let state = training_step_file(&dir, label, body);
        let error = rerobot_train::checkpoint::TrainingStep::read(&state)
            .expect_err(&format!("{label}: a malformed field was accepted"));
        let message = error.to_string();
        assert!(
            message.contains(field),
            "{label}: the refusal does not name the field: {message}"
        );
    }
}

#[test]
fn a_training_step_field_out_of_range_is_refused() {
    let dir = TempDir::new("step-range");
    for (label, body, field) in [
        (
            "negative-processes",
            r#"{"step": 1, "num_processes": -1, "batch_size": 2}"#,
            "num_processes",
        ),
        (
            "zero-processes",
            r#"{"step": 1, "num_processes": 0, "batch_size": 2}"#,
            "num_processes",
        ),
        (
            "huge-batch",
            r#"{"step": 1, "num_processes": 1, "batch_size": 99999999999999999999}"#,
            "batch_size",
        ),
        (
            "negative-batch",
            r#"{"step": 1, "num_processes": 1, "batch_size": -2}"#,
            "batch_size",
        ),
    ] {
        let state = training_step_file(&dir, label, body);
        let error = rerobot_train::checkpoint::TrainingStep::read(&state)
            .expect_err(&format!("{label}: an out-of-range field was accepted"));
        assert!(
            error.to_string().contains(field),
            "{label}: the refusal does not name the field: {error}"
        );
    }
}

#[test]
fn a_training_step_omitting_the_optional_fields_still_reads_because_upstream_omits_them() {
    // `save_training_step` writes `num_processes` and `batch_size` only when it was
    // given them, so a checkpoint from a single-process run legitimately has neither.
    let dir = TempDir::new("step-absent");
    let state = training_step_file(&dir, "minimal", r#"{"step": 7}"#);
    let read = rerobot_train::checkpoint::TrainingStep::read(&state)
        .expect("upstream's own minimal file must still read");
    assert_eq!(read.step, 7);
    assert_eq!(read.num_processes, 1);
    assert_eq!(read.batch_size, 0);
}

#[test]
fn a_well_formed_training_step_round_trips() {
    let dir = TempDir::new("step-round-trip");
    let state = training_step_file(
        &dir,
        "full",
        r#"{"step": 12, "num_processes": 3, "batch_size": 8}"#,
    );
    let read = rerobot_train::checkpoint::TrainingStep::read(&state).unwrap();
    assert_eq!((read.step, read.num_processes, read.batch_size), (12, 3, 8));
}

// ---------------------------------------------------------------------------
// Concurrent marker writers must not share a temporary name
// ---------------------------------------------------------------------------

#[test]
fn threads_writing_the_marker_at_once_do_not_collide_on_one_temporary() {
    // The temporary was named after the process alone, so every thread of one process
    // used the same path: one thread's `rename` moved the file another was still
    // writing, and the loser failed with ENOENT -- turning a checkpoint into an error
    // for no reason the user can act on.
    let dir = TempDir::new("marker-threads");
    let checkpoints = dir.child("checkpoints");
    for step in 1..=4 {
        std::fs::create_dir_all(checkpoints.join(format!("00000{step}"))).unwrap();
    }

    let failures = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    std::thread::scope(|scope| {
        for step in 1..=4 {
            let checkpoints = checkpoints.clone();
            let failures = std::sync::Arc::clone(&failures);
            scope.spawn(move || {
                for _ in 0..200 {
                    let target = checkpoints.join(format!("00000{step}"));
                    if let Err(error) = rerobot_train::checkpoint::write_last_checkpoint(
                        &target,
                        LastCheckpointKind::PortableFile,
                    ) {
                        failures.lock().unwrap().push(error.to_string());
                    }
                }
            });
        }
    });

    let failures = failures.lock().unwrap();
    assert!(
        failures.is_empty(),
        "{} concurrent marker writes failed, first: {}",
        failures.len(),
        failures[0]
    );
    // Whoever wrote last, the marker resolves to a real checkpoint and no temporary
    // survives.
    checkpoint::read_last_checkpoint(&checkpoints).expect("the marker resolves");
    let strays: Vec<_> = std::fs::read_dir(&checkpoints)
        .unwrap()
        .filter_map(|entry| {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            name.ends_with(".tmp").then_some(name)
        })
        .collect();
    assert!(strays.is_empty(), "temporaries left behind: {strays:?}");
}

// ---------------------------------------------------------------------------
// A checkpoint is published whole or not at all
// ---------------------------------------------------------------------------
//
// `save_checkpoint` writes eleven files across two directories. Written straight
// into the destination, a failure anywhere in that sequence leaves a directory that
// looks exactly like a finished checkpoint minus whichever files came after the
// error — and `pretrained_model/` alone is enough for a loader to believe it. The
// contents are built in a sibling staging directory and published with one `rename`,
// so the destination only ever exists complete.

/// A session and config against the committed fixture, ready to checkpoint.
fn checkpointable(
    dir: &TempDir,
) -> (
    rerobot_train::config::TrainConfig,
    rerobot_train::run::TrainSession,
) {
    let config = reduced_config(fixture_dataset(), dir.child("out"));
    let session = rerobot_train::run::TrainSession::new(&config).expect("the session builds");
    (config, session)
}

/// Every file a complete checkpoint holds, relative to its directory.
const CHECKPOINT_FILES: [&str; 11] = [
    "pretrained_model/config.json",
    "pretrained_model/model.safetensors",
    "pretrained_model/train_config.json",
    "pretrained_model/policy_preprocessor.json",
    "pretrained_model/policy_postprocessor.json",
    "pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors",
    "pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "training_state/training_step.json",
    "training_state/rng_state.safetensors",
    "training_state/optimizer_state.safetensors",
    "training_state/optimizer_param_groups.json",
];

#[test]
fn a_successful_save_publishes_every_file_and_leaves_no_staging_directory() {
    let dir = TempDir::new("save-complete");
    let (config, session) = checkpointable(&dir);
    let checkpoints = dir.child("checkpoints");
    let destination = checkpoints.join("000001");
    rerobot_train::run::save_checkpoint(&config, &session, 1, &destination).expect("the save");

    for name in CHECKPOINT_FILES {
        assert!(
            destination.join(name).is_file(),
            "the published checkpoint has no {name}"
        );
    }
    let siblings: Vec<_> = std::fs::read_dir(&checkpoints)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        siblings,
        vec!["000001".to_owned()],
        "the staging directory outlived the save: {siblings:?}"
    );

    // And nothing *but* those eleven files. A checkpoint published by renaming a
    // freshly created staging directory cannot inherit anything, which is the other
    // half of the guarantee: no file from an earlier run, and no file a third party
    // left in the way, can survive inside a checkpoint this call published.
    let mut published = Vec::new();
    let mut stack = vec![destination.clone()];
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                published.push(
                    path.strip_prefix(&destination)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }
    published.sort();
    let mut expected: Vec<String> = CHECKPOINT_FILES
        .iter()
        .map(|name| (*name).to_owned())
        .collect();
    expected.sort();
    assert_eq!(
        published, expected,
        "the published checkpoint holds files the save did not write"
    );
}

#[test]
fn a_save_that_cannot_write_leaves_no_partial_checkpoint_behind() {
    // The parent is made unwritable, so the very first thing the save does fails.
    // What matters is not *which* write failed but that the failure is reported and
    // the destination does not exist afterwards: a caller that sees an error must be
    // able to trust that nothing was published.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new("save-readonly");
        let (config, session) = checkpointable(&dir);
        let checkpoints = dir.child("checkpoints");
        std::fs::create_dir_all(&checkpoints).unwrap();
        let destination = checkpoints.join("000001");
        std::fs::set_permissions(&checkpoints, std::fs::Permissions::from_mode(0o555)).unwrap();

        let error = rerobot_train::run::save_checkpoint(&config, &session, 1, &destination)
            .expect_err("an unwritable parent must fail the save");

        std::fs::set_permissions(&checkpoints, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(
            !destination.exists(),
            "a failed save published a checkpoint anyway: {error}"
        );
        let leftovers: Vec<_> = std::fs::read_dir(&checkpoints)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
    }
}

#[test]
fn saving_over_an_existing_checkpoint_is_refused_and_cannot_merge_into_it() {
    // Renaming onto a populated directory would either fail with a platform errno or,
    // worse, merge -- leaving one run's weights beside another's optimizer state. The
    // destination is refused before anything is written, and what is already there is
    // left exactly as it was.
    let dir = TempDir::new("save-existing");
    let (config, session) = checkpointable(&dir);
    let destination = dir.child("checkpoints").join("000001");
    std::fs::create_dir_all(destination.join("pretrained_model")).unwrap();
    std::fs::write(destination.join("foreign.txt"), b"from another run").unwrap();

    let error = rerobot_train::run::save_checkpoint(&config, &session, 1, &destination)
        .expect_err("an occupied destination must be refused");
    assert!(
        error.to_string().contains("already exists"),
        "the refusal does not say why: {error}"
    );
    assert_eq!(
        std::fs::read(destination.join("foreign.txt")).unwrap(),
        b"from another run"
    );
    assert!(
        !destination
            .join("pretrained_model/model.safetensors")
            .exists(),
        "a refused save wrote into the destination anyway"
    );
}

#[test]
fn the_destination_is_never_visible_half_written() {
    // The distinguishing property of staging, and the one the other tests cannot see:
    // an observer watching the destination must never find it existing-but-incomplete.
    // Written directly, the directory appears at the first `create_dir_all` and fills
    // up file by file, and anything that lists `checkpoints/` during that window sees
    // what looks like a finished checkpoint missing whichever files come last.
    //
    // The assertion is one-sided: correct code publishes with a single `rename`, so a
    // partial sighting is impossible rather than unlikely.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    let dir = TempDir::new("save-atomic");
    let (config, session) = checkpointable(&dir);
    let checkpoints = dir.child("checkpoints");
    std::fs::create_dir_all(&checkpoints).unwrap();
    let destination = checkpoints.join("000001");

    let partial_sightings = Arc::new(Mutex::new(Vec::<String>::new()));
    let done = Arc::new(AtomicBool::new(false));
    let watcher = {
        let destination = destination.clone();
        let partial_sightings = Arc::clone(&partial_sightings);
        let done = Arc::clone(&done);
        std::thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                if destination.is_dir() {
                    let missing: Vec<&str> = CHECKPOINT_FILES
                        .iter()
                        .copied()
                        .filter(|name| !destination.join(name).is_file())
                        .collect();
                    if !missing.is_empty() {
                        partial_sightings
                            .lock()
                            .unwrap()
                            .push(format!("{} of 11 files missing", missing.len()));
                    }
                }
            }
        })
    };

    rerobot_train::run::save_checkpoint(&config, &session, 1, &destination).expect("the save");
    done.store(true, Ordering::Relaxed);
    watcher.join().unwrap();

    let sightings = partial_sightings.lock().unwrap();
    assert!(
        sightings.is_empty(),
        "the destination was visible incomplete {} times, first: {}",
        sightings.len(),
        sightings[0]
    );
}

// ---------------------------------------------------------------------------
// A Windows directory symlink is a directory to the filesystem API
// ---------------------------------------------------------------------------
//
// `update_last_checkpoint` writes a *directory* symlink where the platform allows
// one, so the marker a later call finds may be one. On Unix that is just a link and
// both `remove_file` and `rename` handle it. On Windows a directory symlink is a
// reparse point with FILE_ATTRIBUTE_DIRECTORY: `DeleteFileW` refuses it, and
// `MoveFileExW` with MOVEFILE_REPLACE_EXISTING refuses to replace it. Either turns
// the second checkpoint of a run into an error.
//
// These run on every platform: the behaviour must be identical, and a Unix run is
// what keeps the shared path honest between Windows CI runs.

/// Create a directory symlink, or `None` when the platform will not allow one.
fn directory_symlink(target: &Path, link: &Path) -> Option<()> {
    #[cfg(unix)]
    let result = std::os::unix::fs::symlink(target, link);
    #[cfg(windows)]
    let result = std::os::windows::fs::symlink_dir(target, link);
    #[cfg(not(any(unix, windows)))]
    let result: std::io::Result<()> = Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "no symlinks",
    ));
    // Windows needs SeCreateSymbolicLinkPrivilege or developer mode; when it is not
    // available the marker never becomes a symlink in the first place.
    result.ok()
}

#[test]
fn a_portable_marker_replaces_a_directory_symlink_without_deleting_its_target() {
    let dir = TempDir::new("portable-over-dirlink");
    let (checkpoints, step) = checkpoints_with_one(&dir);
    let second = checkpoints.join("000002");
    std::fs::create_dir_all(&second).unwrap();
    std::fs::write(second.join("payload.txt"), "payload").unwrap();

    let link = checkpoints.join("last");
    let Some(()) = directory_symlink(Path::new("000002"), &link) else {
        return; // No symlink privilege: the marker can never be one here.
    };

    checkpoint::write_last_checkpoint(&step, LastCheckpointKind::PortableFile)
        .expect("a directory symlink must not block the portable marker");
    assert_eq!(
        checkpoint::read_last_checkpoint(&checkpoints).unwrap(),
        step
    );
    assert!(
        second.join("payload.txt").is_file(),
        "replacing the marker followed the link and deleted its target"
    );
}

#[test]
fn removing_a_directory_symlink_marker_unlinks_it_rather_than_its_target() {
    let dir = TempDir::new("unlink-dirlink");
    let (checkpoints, step) = checkpoints_with_one(&dir);
    let second = checkpoints.join("000002");
    std::fs::create_dir_all(&second).unwrap();
    std::fs::write(second.join("payload.txt"), "payload").unwrap();

    let link = checkpoints.join("last");
    let Some(()) = directory_symlink(Path::new("000002"), &link) else {
        return;
    };

    // `update_last_checkpoint` removes whatever marker it finds before writing its
    // own; on Windows that removal is `RemoveDirectoryW`, not `DeleteFileW`.
    checkpoint::update_last_checkpoint(&step)
        .expect("a directory symlink marker must be replaceable");
    assert_eq!(
        checkpoint::read_last_checkpoint(&checkpoints).unwrap(),
        step
    );
    assert!(second.join("payload.txt").is_file());
    assert!(second.is_dir(), "the link's target directory was removed");
}
