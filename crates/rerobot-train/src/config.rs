//! The resolved run configuration, and the `train_config.json` it writes.
//!
//! Upstream is `TrainPipelineConfig` (`lerobot/configs/train.py`), a Draccus
//! dataclass with 28 fields plus five nested config objects. This is not that
//! type: it is the subset the local ACT run actually consumes, and it
//! carries no field this slice would ignore.
//!
//! `train_config.json` is nevertheless written with **upstream's full field set**,
//! because a checkpoint whose `train_config.json` is missing half the schema is not
//! resumable by upstream. The fields this slice does not implement are written with
//! the values upstream defaults them to; `docs/compatibility.md` lists them, and
//! [`TrainConfig::unimplemented_fields`] names them in code so the doc and the
//! writer cannot drift.

use crate::error::{Result, TrainError};
use rerobot_core::dataset::json::{dumps_pretty_ascii, JsonLike, JsonObject};
use rerobot_core::policy::act::{ActConfig, AdamWConfig};
use rerobot_core::BigInt;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The default `TrainPipelineConfig.seed`.
pub const DEFAULT_SEED: u64 = 1000;
/// The default `TrainPipelineConfig.tolerance_s`.
pub const DEFAULT_TOLERANCE_S: f64 = 1e-4;
/// The default `DatasetConfig.use_imagenet_stats`.
pub const DEFAULT_USE_IMAGENET_STATS: bool = true;

/// A resolved, validated training run.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    /// `dataset.repo_id`.
    pub dataset_repo_id: String,
    /// `dataset.root`. Required here: this slice never downloads from the Hub.
    pub dataset_root: PathBuf,
    /// `dataset.episodes`.
    pub dataset_episodes: Option<Vec<i64>>,
    /// `dataset.use_imagenet_stats`.
    ///
    /// Upstream defaults it to `true`, and at `true` `LeRobotDataset` replaces every
    /// camera feature's statistics with `IMAGENET_STATS` — the per-channel mean and
    /// standard deviation the ResNet backbone was trained under. At `false` a camera
    /// keeps whatever `meta/stats.json` holds for it, and upstream's normalizer leaves
    /// a feature with no statistics entry untouched; this slice spells that second case
    /// [`crate::data::image::CameraNormalization::identity`].
    ///
    /// It only affects cameras. A state-only run reads and writes the same bytes
    /// whichever way it is set.
    pub dataset_use_imagenet_stats: bool,
    /// The ACT policy configuration.
    pub policy: ActConfig,
    /// `output_dir`.
    pub output_dir: PathBuf,
    /// Whether this run restores a previously saved local checkpoint.
    pub resume: bool,
    /// The checkpoint directory to restore when [`Self::resume`] is true.
    pub checkpoint_path: Option<PathBuf>,
    /// `job_name`.
    pub job_name: Option<String>,
    /// `seed`.
    pub seed: Option<u64>,
    /// `batch_size`.
    pub batch_size: usize,
    /// `steps`.
    pub steps: u64,
    /// `log_freq`.
    pub log_freq: u64,
    /// `save_freq`.
    pub save_freq: BigInt,
    /// `save_checkpoint`.
    pub save_checkpoint: bool,
    /// `tolerance_s`.
    pub tolerance_s: f64,
    /// `use_policy_training_preset`.
    pub use_policy_training_preset: bool,
    /// `num_workers`. Must be zero: this slice loads in the calling thread.
    pub num_workers: u32,
    /// `optimizer`, resolved from the policy preset when the preset is in use.
    pub optimizer: Option<AdamWConfig>,
}

impl TrainConfig {
    /// A run over `dataset_root`, writing into `output_dir`, with upstream's
    /// defaults everywhere else.
    pub fn new(dataset_repo_id: String, dataset_root: PathBuf, output_dir: PathBuf) -> Self {
        let mut policy = ActConfig::default();
        policy.device = Some("cpu".to_owned());
        // Upstream's `validate` demands a `repo_id` whenever `push_to_hub` is on,
        // and this slice cannot push, so the default run has it off. A user who
        // passes `--policy.repo_id` gets it back.
        policy.push_to_hub = false;
        Self {
            dataset_repo_id,
            dataset_root,
            dataset_episodes: None,
            dataset_use_imagenet_stats: DEFAULT_USE_IMAGENET_STATS,
            policy,
            output_dir,
            resume: false,
            checkpoint_path: None,
            job_name: None,
            seed: Some(DEFAULT_SEED),
            batch_size: 8,
            steps: 100_000,
            log_freq: 200,
            save_freq: 20_000.into(),
            save_checkpoint: true,
            tolerance_s: DEFAULT_TOLERANCE_S,
            use_policy_training_preset: true,
            num_workers: 0,
            optimizer: None,
        }
    }

    /// `TrainPipelineConfig.validate`, restricted to the checks this slice can
    /// make and extended with the ones its narrower scope requires.
    ///
    /// Resolves `job_name` and the optimizer preset as a side effect, exactly as
    /// upstream's `validate` does.
    pub fn validate(&mut self) -> Result<()> {
        // `__post_init__` only. `validate_features` is deliberately *not* called
        // here: upstream runs it inside `ACTPolicy.__init__`, which is after
        // `make_policy` has filled the feature maps in from the dataset. Calling it
        // now would reject every fresh run, because a config that has not met a
        // dataset yet has no features at all.
        self.policy.validate()?;

        match (self.resume, &self.checkpoint_path) {
            (true, None) => {
                return Err(TrainError::Metadata(
                    "resume requires a local checkpoint path".to_owned(),
                ))
            }
            (false, Some(_)) => {
                return Err(TrainError::Metadata(
                    "a checkpoint path requires resume=true".to_owned(),
                ))
            }
            _ => {}
        }

        self.validate_numeric_fields()?;

        if self.num_workers != 0 {
            return Err(TrainError::unsupported(format!(
                "num_workers = {}; this slice loads batches in the calling thread, so only 0 is \
                 accepted",
                self.num_workers
            )));
        }
        if self.batch_size == 0 {
            return Err(TrainError::Metadata(
                "batch_size must be positive".to_owned(),
            ));
        }
        crate::limits::within(self.batch_size, "batch_size", crate::limits::MAX_BATCH_SIZE)?;
        if self.steps == 0 {
            return Err(TrainError::Metadata("steps must be positive".to_owned()));
        }
        crate::limits::within_u64(self.steps, "steps", crate::limits::MAX_STEPS)?;
        self.validate_policy_dimensions()?;

        // Spelling and build support only: no device is initialized here, so a
        // configuration can be validated on a machine with no GPU. The hardware is
        // met in `TrainSession::new`, which resolves the same string again.
        crate::device::parse(self.policy.device.as_deref())?;
        if self.policy.use_amp {
            return Err(TrainError::unsupported(
                "policy.use_amp is set; mixed precision needs `accelerate`, which is not ported"
                    .to_owned(),
            ));
        }
        if self.policy.use_peft {
            return Err(TrainError::unsupported(
                "policy.use_peft is set; PEFT is not ported".to_owned(),
            ));
        }
        if !self.use_policy_training_preset {
            return Err(TrainError::unsupported(
                "use_policy_training_preset = false; a hand-specified optimizer or scheduler \
                 needs the Draccus optimizer registry, which is not ported"
                    .to_owned(),
            ));
        }
        if self.policy.push_to_hub {
            if self.policy.repo_id.is_none() {
                return Err(TrainError::Metadata(
                    "'repo_id' argument missing. Please specify it to push the model to the hub."
                        .to_owned(),
                ));
            }
            return Err(TrainError::unsupported(
                "policy.push_to_hub is set; this slice has no Hub client. Pass \
                 --policy.push_to_hub=false to train locally."
                    .to_owned(),
            ));
        }
        if let Some(episodes) = &self.dataset_episodes {
            if episodes.iter().any(|episode| *episode < 0) {
                return Err(TrainError::Metadata(
                    "Episode indices must be non-negative".to_owned(),
                ));
            }
            let mut sorted = episodes.clone();
            sorted.sort_unstable();
            let before = sorted.len();
            sorted.dedup();
            if sorted.len() != before {
                return Err(TrainError::Metadata(
                    "Episode indices contain duplicates".to_owned(),
                ));
            }
        }

        if self.job_name.is_none() {
            // Upstream: `f"{active_cfg.type}"` when there is no env.
            self.job_name = Some("act".to_owned());
        }
        if self.use_policy_training_preset {
            self.optimizer = Some(self.policy.optimizer_preset());
        }
        Ok(())
    }

    /// Bound every policy dimension before anything is built from it.
    ///
    /// These are the numbers that become tensor shapes, sequence lengths and loop
    /// counts. `chunk_size` is the sharpest of them: `resolve_delta_timestamps`
    /// collects `0..chunk_size` into a `Vec`, so a `chunk_size` of 10^29 was an
    /// allocation request rather than a configuration. Bounding them here, in
    /// `validate`, means the CLI refuses before the dataset is even opened.
    ///
    /// [`crate::model::act::ActModel::new`] bounds the same fields again, because a
    /// library caller can build a model without going through a [`TrainConfig`].
    fn validate_policy_dimensions(&self) -> Result<()> {
        use crate::limits::{
            bounded_positive_usize, bounded_usize, MAX_CHUNK_SIZE, MAX_DIM_FEEDFORWARD,
            MAX_DIM_MODEL, MAX_HEADS, MAX_LATENT_DIM, MAX_LAYERS,
        };
        bounded_positive_usize(&self.policy.dim_model, "policy.dim_model", MAX_DIM_MODEL)?;
        bounded_positive_usize(&self.policy.n_heads, "policy.n_heads", MAX_HEADS)?;
        bounded_positive_usize(
            &self.policy.dim_feedforward,
            "policy.dim_feedforward",
            MAX_DIM_FEEDFORWARD,
        )?;
        bounded_positive_usize(
            &self.policy.n_encoder_layers,
            "policy.n_encoder_layers",
            MAX_LAYERS,
        )?;
        bounded_positive_usize(
            &self.policy.n_decoder_layers,
            "policy.n_decoder_layers",
            MAX_LAYERS,
        )?;
        bounded_usize(
            &self.policy.n_vae_encoder_layers,
            "policy.n_vae_encoder_layers",
            MAX_LAYERS,
        )?;
        bounded_positive_usize(&self.policy.latent_dim, "policy.latent_dim", MAX_LATENT_DIM)?;
        bounded_positive_usize(&self.policy.chunk_size, "policy.chunk_size", MAX_CHUNK_SIZE)?;
        bounded_positive_usize(
            &self.policy.n_action_steps,
            "policy.n_action_steps",
            MAX_CHUNK_SIZE,
        )?;
        bounded_positive_usize(&self.policy.n_obs_steps, "policy.n_obs_steps", MAX_LAYERS)?;
        Ok(())
    }

    /// Every float the run consumes must be finite and inside the range in which it
    /// means anything.
    ///
    /// Both halves matter, and neither is theoretical.
    ///
    /// *Finiteness*: `--policy.dropout=nan` produced a run that reported success and
    /// wrote a checkpoint. It was not even a NaN model — `NaN > 0.0` is `false`, so
    /// the comparison that gates dropout silently turned it off, and the run trained
    /// a *different configuration* than the one asked for. `NaN` in a learning rate
    /// poisons every weight instead; either way the run cannot produce what was
    /// requested, so it must not claim to.
    ///
    /// *Range*: a negative dropout or a zero learning rate is finite and still
    /// cannot train. Upstream leaves these unchecked and produces nonsense; refusing
    /// is the honest behaviour for a port whose contract is applied-or-refused.
    fn validate_numeric_fields(&self) -> Result<()> {
        // `(name, value, lower, lower_inclusive, upper, upper_inclusive)`.
        let ranges: [(&str, f64, f64, bool, f64, bool); 6] = [
            // Torch's `nn.Dropout` accepts [0, 1]; at exactly 1 every activation is
            // zeroed and nothing can train, so the upper end is exclusive here.
            ("policy.dropout", self.policy.dropout, 0.0, true, 1.0, false),
            (
                "policy.kl_weight",
                self.policy.kl_weight,
                0.0,
                true,
                f64::MAX,
                true,
            ),
            (
                "policy.optimizer_lr",
                self.policy.optimizer_lr,
                0.0,
                false,
                f64::MAX,
                true,
            ),
            (
                "policy.optimizer_weight_decay",
                self.policy.optimizer_weight_decay,
                0.0,
                true,
                f64::MAX,
                true,
            ),
            (
                "policy.optimizer_lr_backbone",
                self.policy.optimizer_lr_backbone,
                0.0,
                true,
                f64::MAX,
                true,
            ),
            ("tolerance_s", self.tolerance_s, 0.0, true, f64::MAX, true),
        ];

        for (name, value, lower, lower_inclusive, upper, upper_inclusive) in ranges {
            if !value.is_finite() {
                return Err(TrainError::Metadata(format!(
                    "{name} must be finite, got {value}"
                )));
            }
            let below = if lower_inclusive {
                value < lower
            } else {
                value <= lower
            };
            let above = if upper_inclusive {
                value > upper
            } else {
                value >= upper
            };
            if below || above {
                let open = if lower_inclusive { '[' } else { '(' };
                let close = if upper_inclusive { ']' } else { ')' };
                return Err(TrainError::Metadata(format!(
                    "{name} must be in {open}{lower}, {upper}{close}, got {value}"
                )));
            }
        }

        // The optimizer preset's own constants travel with the config, so a preset
        // that has been edited into an unusable shape is refused here too.
        let preset = self.policy.optimizer_preset();
        for (name, value) in [
            ("optimizer.betas[0]", preset.betas[0]),
            ("optimizer.betas[1]", preset.betas[1]),
            ("optimizer.eps", preset.eps),
            ("optimizer.grad_clip_norm", preset.grad_clip_norm),
        ] {
            if !value.is_finite() {
                return Err(TrainError::Metadata(format!(
                    "{name} must be finite, got {value}"
                )));
            }
        }
        for (name, value) in [
            ("optimizer.betas[0]", preset.betas[0]),
            ("optimizer.betas[1]", preset.betas[1]),
        ] {
            if !(0.0..1.0).contains(&value) {
                return Err(TrainError::Metadata(format!(
                    "{name} must be in [0, 1), got {value}"
                )));
            }
        }
        if preset.eps <= 0.0 {
            return Err(TrainError::Metadata(format!(
                "optimizer.eps must be positive, got {}",
                preset.eps
            )));
        }
        Ok(())
    }

    /// The optimizer settings the run uses, after [`Self::validate`].
    pub fn optimizer_preset(&self) -> AdamWConfig {
        self.optimizer
            .clone()
            .unwrap_or_else(|| self.policy.optimizer_preset())
    }

    /// The per-channel statistics this run applies to every camera frame it decodes.
    ///
    /// The whole of what [`Self::dataset_use_imagenet_stats`] does:
    /// `LeRobotDataset.__init__` overwrites each camera's statistics with
    /// `IMAGENET_STATS` when the flag is set, and leaves them alone when it is not —
    /// and a camera with no statistics is returned unchanged by upstream's normalizer,
    /// which is [`crate::data::image::CameraNormalization::identity`].
    pub fn camera_normalization(&self) -> crate::data::image::CameraNormalization {
        if self.dataset_use_imagenet_stats {
            crate::data::image::CameraNormalization::imagenet()
        } else {
            crate::data::image::CameraNormalization::identity()
        }
    }

    /// The `checkpoints/` directory of this run.
    pub fn checkpoints_dir(&self) -> PathBuf {
        self.output_dir.join(crate::checkpoint::CHECKPOINTS_DIR)
    }

    /// Upstream fields that `train_config.json` records at their default value
    /// because this slice does not implement them.
    ///
    /// Named here rather than only in prose so that the compatibility document's
    /// list is checkable against the code.
    pub fn unimplemented_fields() -> &'static [&'static str] {
        &[
            "env",
            "reward_model",
            "resume",
            "cudnn_deterministic",
            "prefetch_factor",
            "persistent_workers",
            "dataloader_multiprocessing_context",
            "env_eval_freq",
            "eval_steps",
            "max_eval_samples",
            "scheduler",
            "eval",
            "wandb",
            "peft",
            "job",
            "save_checkpoint_to_hub",
            "sample_weighting",
            "rename_map",
        ]
    }

    /// `train_config.json`, in upstream's field order.
    pub fn to_json(&self) -> JsonLike {
        let mut root = JsonObject::new();
        root.insert("dataset".into(), self.dataset_json());
        root.insert("env".into(), JsonLike::Null);
        root.insert("policy".into(), self.policy.to_checkpoint_value());
        root.insert("reward_model".into(), JsonLike::Null);
        root.insert("output_dir".into(), JsonLike::Str(posix(&self.output_dir)));
        root.insert(
            "job_name".into(),
            match &self.job_name {
                Some(name) => JsonLike::Str(name.clone()),
                None => JsonLike::Null,
            },
        );
        root.insert("resume".into(), JsonLike::Bool(self.resume));
        root.insert(
            "seed".into(),
            match self.seed {
                Some(seed) => int(seed),
                None => JsonLike::Null,
            },
        );
        root.insert("cudnn_deterministic".into(), JsonLike::Bool(false));
        root.insert("num_workers".into(), int(u64::from(self.num_workers)));
        root.insert("batch_size".into(), int(self.batch_size as u64));
        root.insert("prefetch_factor".into(), int(4));
        root.insert("persistent_workers".into(), JsonLike::Bool(true));
        root.insert(
            "dataloader_multiprocessing_context".into(),
            JsonLike::Str("spawn".into()),
        );
        root.insert("steps".into(), int(self.steps));
        root.insert("env_eval_freq".into(), int(20_000));
        root.insert("log_freq".into(), int(self.log_freq));
        root.insert("eval_steps".into(), int(0));
        root.insert("max_eval_samples".into(), int(0));
        root.insert("tolerance_s".into(), JsonLike::Float(self.tolerance_s));
        root.insert(
            "save_checkpoint".into(),
            JsonLike::Bool(self.save_checkpoint),
        );
        root.insert("save_freq".into(), JsonLike::Int(self.save_freq.clone()));
        root.insert(
            "use_policy_training_preset".into(),
            JsonLike::Bool(self.use_policy_training_preset),
        );
        root.insert("optimizer".into(), self.optimizer_json());
        root.insert("scheduler".into(), JsonLike::Null);
        root.insert("eval".into(), eval_json());
        root.insert("wandb".into(), wandb_json());
        root.insert("peft".into(), JsonLike::Null);
        root.insert("job".into(), job_json());
        root.insert("save_checkpoint_to_hub".into(), JsonLike::Bool(false));
        root.insert("sample_weighting".into(), JsonLike::Null);
        root.insert("rename_map".into(), JsonLike::Object(JsonObject::new()));
        JsonLike::Object(root)
    }

    /// `train_config.json` as text: `draccus.dump(self, f, indent=4)`.
    pub fn to_json_text(&self) -> String {
        dumps_pretty_ascii(&self.to_json())
    }

    /// Reconstruct the local ACT training configuration stored in a checkpoint.
    ///
    /// This intentionally reads only the fields this native training boundary
    /// consumes. Missing or wrongly typed fields are reported as checkpoint
    /// corruption rather than replaced with a fresh-run default.
    pub fn from_checkpoint_dir(checkpoint_dir: &Path) -> Result<Self> {
        let path = checkpoint_dir
            .join(crate::checkpoint::PRETRAINED_MODEL_DIR)
            .join(crate::checkpoint::TRAIN_CONFIG_NAME);
        let file = std::fs::File::open(&path).map_err(|error| TrainError::io(&path, &error))?;
        let mut bytes = Vec::new();
        file.take(crate::limits::MAX_CHECKPOINT_JSON_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| TrainError::io(&path, &error))?;
        if bytes.len() as u64 > crate::limits::MAX_CHECKPOINT_JSON_BYTES {
            return Err(TrainError::checkpoint(
                &path,
                format!(
                    "train_config.json exceeds the {}-byte limit",
                    crate::limits::MAX_CHECKPOINT_JSON_BYTES
                ),
            ));
        }
        let text = String::from_utf8(bytes).map_err(|error| {
            TrainError::checkpoint(
                &path,
                format!("train_config.json is not valid UTF-8: {error}"),
            )
        })?;
        let document = rerobot_core::dataset::json::loads(&text).map_err(|error| {
            TrainError::checkpoint(
                &path,
                format!("train_config.json is not valid JSON: {error}"),
            )
        })?;
        let JsonLike::Object(root) = &document else {
            return Err(TrainError::checkpoint(
                &path,
                "train_config.json is not a JSON object",
            ));
        };
        let dataset = object_field(root, "dataset", &path)?;
        let dataset_repo_id = string_field(dataset, "repo_id", &path)?;
        let dataset_root = PathBuf::from(string_field(dataset, "root", &path)?);
        let dataset_episodes = match dataset.get("episodes") {
            None | Some(JsonLike::Null) => None,
            Some(JsonLike::Array(values)) => Some(
                values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        i64_field(value, &format!("dataset.episodes[{index}]"), &path)
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            Some(other) => {
                return Err(TrainError::checkpoint(
                    &path,
                    format!("dataset.episodes is {}, not an array", other.type_name()),
                ))
            }
        };
        let dataset_use_imagenet_stats = bool_field(
            dataset,
            "use_imagenet_stats",
            &path,
            DEFAULT_USE_IMAGENET_STATS,
        )?;
        let policy = ActConfig::from_checkpoint_value(value_field(root, "policy", &path)?)
            .map_err(|error| TrainError::checkpoint(&path, error.to_string()))?;
        let output_dir = PathBuf::from(string_field(root, "output_dir", &path)?);
        let mut config = Self::new(dataset_repo_id, dataset_root, output_dir);
        config.dataset_episodes = dataset_episodes;
        config.dataset_use_imagenet_stats = dataset_use_imagenet_stats;
        config.policy = policy;
        config.resume = true;
        config.checkpoint_path = Some(checkpoint_dir.to_owned());
        config.job_name = optional_string_field(root, "job_name", &path)?;
        config.seed = optional_u64_field(root, "seed", &path)?;
        config.num_workers = u32_field(root, "num_workers", &path, 0)?;
        config.batch_size = usize_field(root, "batch_size", &path)?;
        config.steps = u64_field(root, "steps", &path)?;
        config.log_freq = u64_field(root, "log_freq", &path)?;
        config.tolerance_s = f64_field(root, "tolerance_s", &path)?;
        config.save_checkpoint = bool_field(root, "save_checkpoint", &path, true)?;
        config.save_freq = bigint_field(root, "save_freq", &path)?;
        config.use_policy_training_preset =
            bool_field(root, "use_policy_training_preset", &path, true)?;
        Ok(config)
    }

    fn dataset_json(&self) -> JsonLike {
        let mut object = JsonObject::new();
        object.insert(
            "repo_id".into(),
            JsonLike::Str(self.dataset_repo_id.clone()),
        );
        object.insert("root".into(), JsonLike::Str(posix(&self.dataset_root)));
        object.insert(
            "episodes".into(),
            match &self.dataset_episodes {
                Some(episodes) => JsonLike::Array(
                    episodes
                        .iter()
                        .map(|episode| JsonLike::Int(num_bigint::BigInt::from(*episode)))
                        .collect(),
                ),
                None => JsonLike::Null,
            },
        );
        let mut transforms = JsonObject::new();
        transforms.insert("enable".into(), JsonLike::Bool(false));
        transforms.insert("max_num_transforms".into(), int(3));
        transforms.insert("random_order".into(), JsonLike::Bool(false));
        transforms.insert("tfs".into(), JsonLike::Object(JsonObject::new()));
        object.insert("image_transforms".into(), JsonLike::Object(transforms));
        object.insert("revision".into(), JsonLike::Null);
        object.insert(
            "use_imagenet_stats".into(),
            JsonLike::Bool(self.dataset_use_imagenet_stats),
        );
        object.insert("video_backend".into(), JsonLike::Str("pyav".into()));
        object.insert("return_uint8".into(), JsonLike::Bool(false));
        object.insert("depth_output_unit".into(), JsonLike::Str("m".into()));
        object.insert("streaming".into(), JsonLike::Bool(false));
        object.insert("eval_split".into(), JsonLike::Float(0.0));
        JsonLike::Object(object)
    }

    fn optimizer_json(&self) -> JsonLike {
        let preset = self.optimizer_preset();
        let mut object = JsonObject::new();
        object.insert("type".into(), JsonLike::Str("adamw".into()));
        object.insert("lr".into(), JsonLike::Float(preset.lr));
        object.insert("weight_decay".into(), JsonLike::Float(preset.weight_decay));
        object.insert(
            "grad_clip_norm".into(),
            JsonLike::Float(preset.grad_clip_norm),
        );
        object.insert(
            "betas".into(),
            JsonLike::Array(vec![
                JsonLike::Float(preset.betas[0]),
                JsonLike::Float(preset.betas[1]),
            ]),
        );
        object.insert("eps".into(), JsonLike::Float(preset.eps));
        JsonLike::Object(object)
    }
}

fn eval_json() -> JsonLike {
    let mut object = JsonObject::new();
    object.insert("n_episodes".into(), int(50));
    object.insert("batch_size".into(), int(50));
    object.insert("use_async_envs".into(), JsonLike::Bool(false));
    JsonLike::Object(object)
}

fn wandb_json() -> JsonLike {
    let mut object = JsonObject::new();
    object.insert("enable".into(), JsonLike::Bool(false));
    object.insert("disable_artifact".into(), JsonLike::Bool(false));
    object.insert("project".into(), JsonLike::Str("lerobot".into()));
    object.insert("entity".into(), JsonLike::Null);
    object.insert("notes".into(), JsonLike::Null);
    object.insert("run_id".into(), JsonLike::Null);
    object.insert("mode".into(), JsonLike::Null);
    JsonLike::Object(object)
}

fn job_json() -> JsonLike {
    let mut object = JsonObject::new();
    object.insert("target".into(), JsonLike::Str("local".into()));
    JsonLike::Object(object)
}

fn int(value: u64) -> JsonLike {
    JsonLike::Int(num_bigint::BigInt::from(value))
}

/// A path in the POSIX spelling `pathlib.PurePosixPath` would produce, so that a
/// `train_config.json` written on Windows names the same path as one written on
/// Linux.
fn object_field<'a>(root: &'a JsonObject, name: &str, path: &Path) -> Result<&'a JsonObject> {
    match root.get(name) {
        Some(JsonLike::Object(value)) => Ok(value),
        Some(other) => Err(TrainError::checkpoint(
            path,
            format!("{name} is {}, not an object", other.type_name()),
        )),
        None => Err(TrainError::checkpoint(path, format!("is missing {name}"))),
    }
}

fn value_field<'a>(root: &'a JsonObject, name: &str, path: &Path) -> Result<&'a JsonLike> {
    root.get(name)
        .ok_or_else(|| TrainError::checkpoint(path, format!("is missing {name}")))
}

fn string_field(root: &JsonObject, name: &str, path: &Path) -> Result<String> {
    match value_field(root, name, path)? {
        JsonLike::Str(value) => Ok(value.clone()),
        other => Err(TrainError::checkpoint(
            path,
            format!("{name} is {}, not a string", other.type_name()),
        )),
    }
}

fn optional_string_field(root: &JsonObject, name: &str, path: &Path) -> Result<Option<String>> {
    match root.get(name) {
        None | Some(JsonLike::Null) => Ok(None),
        Some(JsonLike::Str(value)) => Ok(Some(value.clone())),
        Some(other) => Err(TrainError::checkpoint(
            path,
            format!("{name} is {}, not a string or null", other.type_name()),
        )),
    }
}

fn bigint_value(value: &JsonLike, name: &str, path: &Path) -> Result<BigInt> {
    match value {
        JsonLike::Int(value) => Ok(value.clone()),
        other => Err(TrainError::checkpoint(
            path,
            format!("{name} is {}, not an integer", other.type_name()),
        )),
    }
}

fn bigint_field(root: &JsonObject, name: &str, path: &Path) -> Result<BigInt> {
    bigint_value(value_field(root, name, path)?, name, path)
}

fn i64_field(value: &JsonLike, name: &str, path: &Path) -> Result<i64> {
    i64::try_from(bigint_value(value, name, path)?).map_err(|_| {
        TrainError::checkpoint(
            path,
            format!("{name} does not fit in a signed 64-bit integer"),
        )
    })
}

fn u64_field(root: &JsonObject, name: &str, path: &Path) -> Result<u64> {
    u64::try_from(bigint_field(root, name, path)?).map_err(|_| {
        TrainError::checkpoint(path, format!("{name} is not a non-negative 64-bit integer"))
    })
}

fn optional_u64_field(root: &JsonObject, name: &str, path: &Path) -> Result<Option<u64>> {
    match root.get(name) {
        None | Some(JsonLike::Null) => Ok(None),
        Some(value) => Ok(Some(
            u64::try_from(bigint_value(value, name, path)?).map_err(|_| {
                TrainError::checkpoint(path, format!("{name} is not a non-negative 64-bit integer"))
            })?,
        )),
    }
}

fn usize_field(root: &JsonObject, name: &str, path: &Path) -> Result<usize> {
    usize::try_from(u64_field(root, name, path)?)
        .map_err(|_| TrainError::checkpoint(path, format!("{name} does not fit in usize")))
}

fn u32_field(root: &JsonObject, name: &str, path: &Path, default: u32) -> Result<u32> {
    match root.get(name) {
        None => Ok(default),
        Some(value) => u32::try_from(bigint_value(value, name, path)?)
            .map_err(|_| TrainError::checkpoint(path, format!("{name} does not fit in u32"))),
    }
}

fn bool_field(root: &JsonObject, name: &str, path: &Path, default: bool) -> Result<bool> {
    match root.get(name) {
        None => Ok(default),
        Some(JsonLike::Bool(value)) => Ok(*value),
        Some(other) => Err(TrainError::checkpoint(
            path,
            format!("{name} is {}, not a boolean", other.type_name()),
        )),
    }
}

fn f64_field(root: &JsonObject, name: &str, path: &Path) -> Result<f64> {
    match value_field(root, name, path)? {
        JsonLike::Float(value) => Ok(*value),
        other => Err(TrainError::checkpoint(
            path,
            format!("{name} is {}, not a float", other.type_name()),
        )),
    }
}

fn posix(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}
