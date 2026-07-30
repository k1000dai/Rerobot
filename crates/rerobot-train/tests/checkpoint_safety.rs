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
