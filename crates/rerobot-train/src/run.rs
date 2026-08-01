//! Port of the offline training loop in `lerobot/scripts/lerobot_train.py`.
//!
//! One step is, in upstream's order: pull the next batch from the episode-aware
//! sampler, normalize it, forward, compute the loss, backward, clip the gradient
//! norm, step AdamW, zero the gradients, and then — after `step` has been
//! incremented — log and checkpoint.
//!
//! What upstream wraps around that and this does not: `accelerate` (so no
//! distributed training, no mixed precision, no gradient accumulation), the
//! `DataLoader`'s worker processes, `wandb`, the LR scheduler, environment
//! evaluation, the held-out eval split, and resume. Every one of those is refused
//! by [`crate::config::TrainConfig::validate`] rather than ignored.

use crate::checkpoint;
use crate::config::TrainConfig;
use crate::data::batch::{collate, Batch};
use crate::data::dataset::StateOnlyDataset;
use crate::data::meta::ACTION;
use crate::error::{Result, TrainError};
use crate::model::act::{ActModel, Pass, Randomness};
use crate::optim::{act_parameter_groups, clip_grad_norm, parameter_l2, AdamW};
use candle_core::Device;
use indexmap::IndexMap;
use rerobot_core::dataset::delta::action_delta_timestamps;
use rerobot_core::dataset::sampler::EpisodeAwareSampler;
use rerobot_core::policy::normalize::Normalizer;
use rerobot_core::random::SplitMix64;
use rerobot_core::BigInt;
use std::path::PathBuf;

/// What one optimization step produced.
#[derive(Debug, Clone, PartialEq)]
pub struct StepMetrics {
    /// The step number, counted from one after the update.
    pub step: u64,
    /// `loss.item()`.
    pub loss: f64,
    /// `loss_dict["l1_loss"]`.
    pub l1_loss: f64,
    /// `loss_dict["kld_loss"]`, when the VAE is on.
    pub kld_loss: Option<f64>,
    /// The pre-clip total gradient norm, which is what upstream logs.
    pub grad_norm: f64,
    /// The learning rate of the first parameter group.
    pub lr: f64,
    /// The L2 distance the parameters moved on this step.
    ///
    /// Not an upstream metric. It exists because "the optimizer ran" and "the
    /// weights changed" are different claims, and this slice's tests assert the
    /// second one.
    pub parameter_delta: f64,
    /// The dataset-absolute frame indices the step consumed, in batch order.
    pub frame_indices: Vec<i64>,
}

/// What a run produced.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainOutcome {
    /// One entry per optimization step.
    pub steps: Vec<StepMetrics>,
    /// Checkpoint directories written, in order.
    pub checkpoints: Vec<PathBuf>,
    /// How many trainable scalars the policy has.
    pub num_parameters: usize,
    /// `dataset.num_frames`.
    pub num_frames: usize,
    /// `dataset.num_episodes`.
    pub num_episodes: usize,
}

/// Everything a run needs, built but not yet stepped.
///
/// Exposed so that a test can drive one step at a time and inspect the model in
/// between, which [`train`] itself gives no way to do.
pub struct TrainSession {
    /// The dataset being trained on.
    pub dataset: StateOnlyDataset,
    /// The policy.
    pub model: ActModel,
    /// The optimizer.
    pub optimizer: AdamW,
    /// The normalizer applied to every batch.
    pub normalizer: Normalizer,
    /// The sampler producing the data order.
    pub sampler: EpisodeAwareSampler,
    /// The generator driving dropout and the latent sample.
    pub rng: SplitMix64,
    /// `optimizer.grad_clip_norm`.
    pub grad_clip_norm: f64,
    /// Frame indices left over from the sampler's current epoch.
    queue: Vec<i64>,
    batch_size: usize,
    device: Device,
}

impl TrainSession {
    /// Build a session from a validated config.
    ///
    /// "Validated" is the caller's claim, not a fact this function can rely on:
    /// [`TrainConfig`]'s fields are public, so a library caller can construct one
    /// without ever calling [`TrainConfig::validate`]. The batch size is therefore
    /// re-checked here, because `next_batch` loops until it has that many frames --
    /// capping the initial reservation, which is all `next_batch` did, only slowed the
    /// growth down.
    pub fn new(config: &TrainConfig) -> Result<Self> {
        if config.batch_size == 0 {
            return Err(TrainError::Metadata(
                "batch_size must be positive".to_owned(),
            ));
        }
        crate::limits::within(
            config.batch_size,
            "batch_size",
            crate::limits::MAX_BATCH_SIZE,
        )?;
        // Resolved once, here, and then used for every tensor the run creates:
        // the collated batch, the normalized copy of it, the model's parameters,
        // the latent and dropout draws, the AdamW moments, and the optimizer state
        // the checkpoint serializes. Nothing downstream picks a device of its own.
        let device = crate::device::resolve(config.policy.device.as_deref())?;
        let seed = config.seed.unwrap_or(0);

        let metadata = crate::data::meta::DatasetMetadata::load(&config.dataset_root)?;
        let fps = metadata.fps()?;

        // `resolve_delta_timestamps`: ACT asks for an action window of
        // `range(chunk_size)` and no observation or reward history.
        let chunk_size = i64::try_from(&config.policy.chunk_size).map_err(|_| {
            TrainError::unsupported(format!(
                "chunk_size = {} does not fit in i64",
                config.policy.chunk_size
            ))
        })?;
        let mut delta_timestamps: IndexMap<String, Vec<f64>> = IndexMap::new();
        delta_timestamps.insert(ACTION.to_owned(), action_delta_timestamps(chunk_size, fps));

        let dataset =
            StateOnlyDataset::load(&config.dataset_root, &delta_timestamps, config.tolerance_s)?;

        // `make_policy`: the dataset's features become the policy's, split by
        // whether they are actions.
        let (inputs, outputs) = dataset.metadata().policy_feature_split();
        let mut policy_config = config.policy.clone();
        policy_config.input_features = Some(inputs);
        policy_config.output_features = Some(outputs);
        policy_config.validate()?;
        policy_config.validate_features()?;

        let normalizer = Normalizer::new(
            &policy_config
                .input_features
                .clone()
                .unwrap_or_default()
                .into_iter()
                .chain(policy_config.output_features.clone().unwrap_or_default())
                .collect(),
            &policy_config.normalization_mapping,
            &dataset.metadata().stats,
        )?;

        // The model's parameters are drawn from a sub-stream of the run seed so
        // that changing the number of steps cannot change the initial weights.
        let mut init_rng = SplitMix64::new(rerobot_core::random::mix64(seed ^ INIT_SUBSTREAM));
        let model = ActModel::new(&policy_config, &device, &mut init_rng)?;

        let preset = config.optimizer_preset();
        let optimizer = AdamW::new(
            act_parameter_groups(
                model.optimizer_parameter_groups(),
                &preset,
                config.policy.optimizer_lr_backbone,
            ),
            model.parameters().len(),
        )?;

        let sampler = dataset.sampler(
            config.dataset_episodes.as_deref(),
            // ACT declares no `drop_n_last_frames`, so upstream's `getattr`
            // default of 0 applies.
            0,
            true,
            seed,
        )?;

        Ok(Self {
            dataset,
            model,
            optimizer,
            normalizer,
            sampler,
            rng: SplitMix64::new(rerobot_core::random::mix64(seed ^ RUN_SUBSTREAM)),
            grad_clip_norm: preset.grad_clip_norm,
            queue: Vec::new(),
            batch_size: config.batch_size,
            device,
        })
    }

    /// The policy config the model was actually built from, features resolved.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The next batch, cycling through epochs the way `utils.cycle` does.
    pub fn next_batch(&mut self) -> Result<Batch> {
        // `Vec::with_capacity(batch_size)` is an allocation request, and `batch_size`
        // comes from the command line. `TrainConfig::validate` bounds it, and this
        // caps the reservation independently so that a library caller constructing a
        // session by hand cannot turn one number into an abort. Growth from a smaller
        // reservation costs an amortized reallocation and nothing else.
        let mut frames = Vec::with_capacity(self.batch_size.min(crate::limits::MAX_BATCH_SIZE));
        while frames.len() < self.batch_size {
            if self.queue.is_empty() {
                self.queue = self.sampler.next_epoch();
                self.queue.reverse();
                if self.queue.is_empty() {
                    return Err(TrainError::Metadata(
                        "the sampler yielded an empty epoch".to_owned(),
                    ));
                }
            }
            let index = self.queue.pop().expect("the queue is non-empty");
            let row = usize::try_from(index).map_err(|_| {
                TrainError::Metadata(format!(
                    "the sampler produced a negative frame index {index}"
                ))
            })?;
            frames.push(self.dataset.get(row)?);
        }
        collate(&frames, &self.device)
    }

    /// One optimization step: forward, loss, backward, clip, AdamW, zero.
    ///
    /// # Errors
    ///
    /// Among the ordinary failures, [`TrainError::NonFinite`] when the loss, the KL
    /// term, the gradient norm or the resulting parameters are not finite. That is a
    /// guard rather than a nicety: a NaN loss means the step trained nothing, and
    /// reporting it and carrying on writes a checkpoint of NaN weights that looks
    /// like a successful run. `TrainConfig::validate` refuses the configurations that
    /// *cause* this; a run can still diverge on its own, and this is where that is
    /// caught.
    pub fn step(&mut self, step_number: u64) -> Result<StepMetrics> {
        let raw = self.next_batch()?;
        let batch = raw.normalized(&self.normalizer)?;

        let before = parameter_l2(self.model.parameters())?;

        let output = self
            .model
            .forward(&batch, Pass::Train(Randomness::Seeded(&mut self.rng)))?;
        let loss = self.model.loss(&batch, &output)?;

        // Checked before the optimizer runs, so a poisoned gradient cannot reach the
        // weights: AdamW would turn every parameter it touches into NaN, and the
        // model would then be unrecoverable rather than merely wrong for one step.
        finite(step_number, "loss", loss.total)?;
        finite(step_number, "l1_loss", loss.l1_loss)?;
        if let Some(kld) = loss.kld_loss {
            finite(step_number, "kld_loss", kld)?;
        }

        let mut gradients = loss.loss.backward()?;
        let grad_norm =
            clip_grad_norm(self.model.parameters(), &mut gradients, self.grad_clip_norm)?;
        finite(step_number, "grad_norm", grad_norm)?;
        self.optimizer.step(self.model.parameters(), &gradients)?;
        // candle has no persistent `.grad`, so the store simply goes out of scope
        // here; that is `optimizer.zero_grad()`.
        drop(gradients);

        let after = parameter_l2(self.model.parameters())?;
        // And after: an update can overflow even from a finite gradient, and the next
        // step would then start from a NaN weight.
        finite(step_number, "the parameter norm after the update", after)?;

        Ok(StepMetrics {
            step: step_number,
            loss: loss.total,
            l1_loss: loss.l1_loss,
            kld_loss: loss.kld_loss,
            grad_norm,
            lr: self.optimizer.first_lr(),
            parameter_delta: (after - before).abs(),
            frame_indices: batch.indices.clone(),
        })
    }
}

/// Refuse a quantity a training step cannot continue from.
fn finite(step: u64, quantity: &str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(TrainError::NonFinite {
            step,
            quantity: quantity.to_owned(),
            value: value.to_string(),
        })
    }
}

/// Sub-stream tags, mixed with the run seed so that the parameter draw and the
/// per-step draw cannot alias each other.
const INIT_SUBSTREAM: u64 = 0x494E_4954_5F41_4354;
const RUN_SUBSTREAM: u64 = 0x5255_4E5F_4143_5420;

/// Whether upstream saves a checkpoint at this step.
///
/// A non-positive frequency disables periodic checkpoints but never suppresses
/// the final checkpoint. `BigInt` preserves the source `int` domain without a
/// narrowing conversion merely to make a modulo decision.
pub fn should_save_checkpoint(step: u64, save_freq: &BigInt, total_steps: u64) -> bool {
    step == total_steps
        || (save_freq > &BigInt::from(0_u8) && BigInt::from(step) % save_freq == BigInt::from(0_u8))
}

/// Train for `config.steps` steps, writing checkpoints as configured.
///
/// `observe` receives one line per log-worthy event, in the order upstream's
/// `logging.info` calls emit them. It is a callback rather than a returned
/// transcript so that a long run reports progress as it happens.
pub fn train(config: &TrainConfig, observe: &mut dyn FnMut(&str)) -> Result<TrainOutcome> {
    let mut config = config.clone();
    config.validate()?;

    if config.output_dir.is_dir() {
        // Upstream raises `FileExistsError` unless resuming, and resume is not
        // ported, so the check is unconditional here.
        return Err(TrainError::io_message(
            &config.output_dir,
            "output directory already exists and resume is not supported by this slice; \
             choose another --output_dir",
        ));
    }

    observe("Creating dataset");
    let mut session = TrainSession::new(&config)?;

    observe(&format!("Output dir: {}", config.output_dir.display()));
    observe(&format!("cfg.steps={}", config.steps));
    observe(&format!(
        "dataset.num_frames={}",
        session.dataset.num_frames()
    ));
    observe(&format!(
        "dataset.num_episodes={}",
        session.dataset.num_episodes()
    ));
    observe(&format!("Effective batch size: {}", config.batch_size));
    observe(&format!(
        "num_learnable_params={}",
        session.model.num_parameters()
    ));
    observe("Start offline training on a fixed dataset");

    // Not `with_capacity(config.steps)`: at the budget's ceiling that is a billion
    // `StepMetrics`, reserved before the first one exists. The metrics are appended
    // one per step, so the vector grows to whatever the run actually reaches.
    let mut steps: Vec<StepMetrics> = Vec::new();
    let mut checkpoints = Vec::new();

    for step_number in 1..=config.steps {
        let metrics = session.step(step_number)?;

        let is_log_step = config.log_freq > 0 && step_number % config.log_freq == 0;
        if is_log_step || step_number == config.steps {
            observe(&format!(
                "step:{} loss:{:.3} grdn:{:.3} lr:{:.1e}",
                metrics.step, metrics.loss, metrics.grad_norm, metrics.lr
            ));
        }
        let is_saving_step = should_save_checkpoint(step_number, &config.save_freq, config.steps);
        if config.save_checkpoint && is_saving_step {
            observe(&format!("Checkpoint policy after step {step_number}"));
            let directory =
                checkpoint::step_checkpoint_dir(&config.output_dir, config.steps, step_number);
            save_checkpoint(&config, &session, step_number, &directory)?;
            checkpoint::update_last_checkpoint(&directory)?;
            checkpoints.push(directory);
        }
        steps.push(metrics);
    }

    observe("End of training");

    Ok(TrainOutcome {
        steps,
        checkpoints,
        num_parameters: session.model.num_parameters(),
        num_frames: session.dataset.num_frames(),
        num_episodes: session.dataset.num_episodes(),
    })
}

/// `train_utils.save_checkpoint` for this slice's contents.
pub fn save_checkpoint(
    config: &TrainConfig,
    session: &TrainSession,
    step: u64,
    directory: &std::path::Path,
) -> Result<()> {
    write_staged_directory(directory, |staging| {
        write_checkpoint_contents(config, session, step, staging)
    })
}

/// Build a multi-file artifact out of sight and publish it with one rename.
fn write_staged_directory(
    destination: &std::path::Path,
    writer: impl FnOnce(&std::path::Path) -> Result<()>,
) -> Result<()> {
    if std::fs::symlink_metadata(destination).is_ok() {
        return Err(TrainError::checkpoint(
            destination,
            "already exists; refusing to overwrite a checkpoint destination",
        ));
    }
    let parent = destination.parent().ok_or_else(|| {
        TrainError::checkpoint(destination, "has no parent directory for staging")
    })?;
    std::fs::create_dir_all(parent).map_err(|error| TrainError::io(parent, &error))?;

    static NEXT_STAGING_DIRECTORY: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);
    let name = destination.file_name().ok_or_else(|| {
        TrainError::checkpoint(destination, "has no final path component for staging")
    })?;
    let staging = loop {
        let candidate = parent.join(format!(
            ".{}.{}.{}.tmp",
            name.to_string_lossy(),
            std::process::id(),
            NEXT_STAGING_DIRECTORY.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        match std::fs::create_dir(&candidate) {
            Ok(()) => break candidate,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(TrainError::io(&candidate, &error)),
        }
    };

    if let Err(error) = writer(&staging) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }
    // Repeat the no-clobber check after the potentially long write. `rename` then
    // publishes every file together rather than exposing a mixed checkpoint.
    if std::fs::symlink_metadata(destination).is_ok() {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(TrainError::checkpoint(
            destination,
            "appeared while the checkpoint was being written; refusing to overwrite it",
        ));
    }
    if let Err(error) = std::fs::rename(&staging, destination) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(TrainError::io(destination, &error));
    }
    Ok(())
}

fn write_checkpoint_contents(
    config: &TrainConfig,
    session: &TrainSession,
    step: u64,
    directory: &std::path::Path,
) -> Result<()> {
    let pretrained = directory.join(checkpoint::PRETRAINED_MODEL_DIR);
    let training_state = directory.join(checkpoint::TRAINING_STATE_DIR);
    std::fs::create_dir_all(&pretrained).map_err(|error| TrainError::io(&pretrained, &error))?;
    std::fs::create_dir_all(&training_state)
        .map_err(|error| TrainError::io(&training_state, &error))?;

    session
        .model
        .save(&pretrained.join(checkpoint::MODEL_FILE))?;

    // The config written is the one the model was actually built from -- features
    // resolved from the dataset -- not the one the user typed, which is what
    // `policy.config.save_pretrained` does upstream.
    let mut policy_config = config.policy.clone();
    let (inputs, outputs) = session.dataset.metadata().policy_feature_split();
    policy_config.input_features = Some(inputs);
    policy_config.output_features = Some(outputs);
    std::fs::write(
        pretrained.join(checkpoint::CONFIG_FILE),
        policy_config.to_checkpoint_json(),
    )
    .map_err(|error| TrainError::io(pretrained.join(checkpoint::CONFIG_FILE), &error))?;

    std::fs::write(
        pretrained.join(checkpoint::TRAIN_CONFIG_NAME),
        config.to_json_text(),
    )
    .map_err(|error| TrainError::io(pretrained.join(checkpoint::TRAIN_CONFIG_NAME), &error))?;

    // `train_utils.save_checkpoint` passes both processors, so their four artifacts
    // are part of the layout. They carry the dataset statistics the weights were
    // trained against, which nothing else in the checkpoint records.
    crate::processor::write_processor_artifacts(
        &pretrained,
        &policy_config,
        &session.dataset.metadata().stats,
    )?;

    checkpoint::write_json(
        &checkpoint::TrainingStep {
            step,
            num_processes: 1,
            batch_size: config.batch_size,
        }
        .to_json(),
        &training_state.join(checkpoint::TRAINING_STEP),
    )?;
    checkpoint::write_rng_state(&training_state, &session.rng)?;
    let state = session.optimizer.state_tensors(&session.device)?;
    candle_core::safetensors::save(&state, training_state.join(checkpoint::OPTIMIZER_STATE))?;
    checkpoint::write_json(
        &session.optimizer.param_groups_json(),
        &training_state.join(checkpoint::OPTIMIZER_PARAM_GROUPS),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::write_staged_directory;
    use crate::error::TrainError;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temporary_root(label: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rerobot-staged-writer-{}-{label}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn an_existing_destination_is_rejected_without_changing_it() {
        let root = temporary_root("existing");
        let destination = root.join("checkpoint");
        std::fs::create_dir(&destination).unwrap();
        std::fs::write(destination.join("sentinel"), b"original").unwrap();

        let error = write_staged_directory(&destination, |staging| {
            std::fs::write(staging.join("new"), b"new").unwrap();
            Ok(())
        })
        .expect_err("an existing destination must be refused");

        assert!(error.to_string().contains("already exists"));
        assert_eq!(
            std::fs::read(destination.join("sentinel")).unwrap(),
            b"original"
        );
        assert!(!destination.join("new").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_mid_write_failure_leaves_no_final_destination() {
        let root = temporary_root("failure");
        let destination = root.join("checkpoint");

        let error = write_staged_directory(&destination, |staging| {
            std::fs::write(staging.join("partial"), b"partial").unwrap();
            Err(TrainError::Metadata("injected failure".to_owned()))
        })
        .expect_err("the injected failure must escape");

        assert!(error.to_string().contains("injected failure"));
        assert!(!destination.exists());
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_destination_is_rejected_without_touching_its_target() {
        let root = temporary_root("alias");
        let target = root.join("target");
        let destination = root.join("checkpoint");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("sentinel"), b"original").unwrap();
        std::os::unix::fs::symlink(&target, &destination).unwrap();

        write_staged_directory(&destination, |_| Ok(()))
            .expect_err("a destination alias must be refused");

        assert_eq!(std::fs::read(target.join("sentinel")).unwrap(), b"original");
        assert!(destination.is_symlink());
        let _ = std::fs::remove_dir_all(root);
    }
}
