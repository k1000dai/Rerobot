//! `lerobot-train`'s argument surface.
//!
//! Upstream parses `TrainPipelineConfig` with Draccus, which accepts a dotted
//! `--a.b.c=value` flag for every field of every nested dataclass, plus
//! `--config_path` for a YAML or checkpoint config. This is not that parser: it is
//! an explicit allow-list of the flags the local ACT run consumes. A saved local
//! `train_config.json` is accepted as a fresh-run configuration; general YAML and
//! arbitrary Draccus configurations remain outside this boundary.
//!
//! The design rule is that there are exactly three outcomes for any flag, and
//! never a fourth:
//!
//! * **applied** — the flag is in the allow-list and the run uses it;
//! * **refused as unsupported** — the flag names something upstream really does and
//!   this slice does not, and the error says which and why
//!   ([`UNSUPPORTED_ARGUMENTS`]);
//! * **refused as unknown** — the flag names nothing upstream has either.
//!
//! No flag is ever accepted and ignored. That is the property that keeps
//! `lerobot-train` honest: a command that would train a different model upstream
//! fails here rather than training a different model.

use rerobot_core::policy::act::ActConfig;
use rerobot_core::types::NormalizationMode;
use rerobot_core::BigInt;
use rerobot_train::config::TrainConfig;
use std::io::Read;
use std::path::PathBuf;

/// Why a `lerobot-train` command line could not be turned into a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgumentError {
    /// A flag names something upstream supports and this slice does not.
    Unsupported {
        /// The flag, without its value.
        flag: String,
        /// Why it is not supported.
        reason: String,
    },
    /// A flag names nothing upstream has.
    Unknown {
        /// The flag, without its value.
        flag: String,
    },
    /// A flag was given without a value, or with an unparseable one.
    Value {
        /// The flag.
        flag: String,
        /// What was wrong.
        reason: String,
    },
    /// A flag the run cannot proceed without was absent.
    Missing {
        /// The flag.
        flag: String,
        /// Why it is required here even though upstream can do without it.
        reason: String,
    },
    /// A bare positional argument, which Draccus has no meaning for either.
    Positional(String),
}

impl std::fmt::Display for ArgumentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported { flag, reason } => {
                write!(
                    formatter,
                    "--{flag} is not supported in this slice: {reason}"
                )
            }
            Self::Unknown { flag } => write!(
                formatter,
                "--{flag} is not a lerobot-train argument; try `lerobot-train --help`"
            ),
            Self::Value { flag, reason } => write!(formatter, "--{flag}: {reason}"),
            Self::Missing { flag, reason } => {
                write!(formatter, "--{flag} is required: {reason}")
            }
            Self::Positional(argument) => write!(
                formatter,
                "unexpected argument {argument:?}; every lerobot-train option is a \
                 --name=value flag"
            ),
        }
    }
}

impl std::error::Error for ArgumentError {}

/// Upstream flags that name a real feature this slice does not implement, with the
/// reason each is refused.
///
/// Kept as data rather than as branches so that `--help` can list them and the
/// compatibility document can be checked against them.
pub static UNSUPPORTED_ARGUMENTS: &[(&str, &str)] = &[
    ("env", "environment rollouts need Gymnasium, which is not ported"),
    ("env.type", "environment rollouts need Gymnasium, which is not ported"),
    ("eval", "environment evaluation needs Gymnasium, which is not ported"),
    ("eval.n_episodes", "environment evaluation needs Gymnasium, which is not ported"),
    ("eval.batch_size", "environment evaluation needs Gymnasium, which is not ported"),
    ("eval.use_async_envs", "environment evaluation needs Gymnasium, which is not ported"),
    ("env_eval_freq", "environment evaluation needs Gymnasium, which is not ported"),
    ("eval_steps", "the held-out eval split needs dataset splitting, which is not ported"),
    ("max_eval_samples", "the held-out eval split needs dataset splitting, which is not ported"),
    ("dataset.eval_split", "the held-out eval split needs dataset splitting, which is not ported"),
    ("reward_model", "reward-model training is a separate pipeline and is not ported"),
    ("peft", "PEFT adapters need the `peft` library, which is not ported"),
    ("wandb", "Weights & Biases logging is not ported; logs go to stdout"),
    ("wandb.project", "Weights & Biases logging is not ported; logs go to stdout"),
    ("wandb.entity", "Weights & Biases logging is not ported; logs go to stdout"),
    ("wandb.mode", "Weights & Biases logging is not ported; logs go to stdout"),
    ("job", "Hugging Face Jobs dispatch is not ported"),
    ("job.target", "Hugging Face Jobs dispatch is not ported"),
    ("save_checkpoint_to_hub", "there is no Hub client in this slice"),
    ("sample_weighting", "per-sample loss weighting is not ported"),
    ("rename_map", "the rename map only applies to a pretrained checkpoint, which cannot be loaded here"),
    ("scheduler", "learning-rate schedulers need the Draccus scheduler registry, which is not ported"),
    ("optimizer", "a hand-specified optimizer needs the Draccus optimizer registry; the ACT preset is used instead"),
    ("dataset.streaming", "the streaming dataset needs the Hub client"),
    ("dataset.revision", "pinning a Hub revision needs the Hub client"),
    ("dataset.image_transforms", "image transforms need the image pipeline, which is not ported"),
    ("dataset.video_backend", "video decoding is not ported; only embedded PNG/JPEG image columns are supported"),
    ("dataset.return_uint8", "the CLI decodes supported image columns to f32 tensors, so this option is not ported"),
    ("dataset.depth_output_unit", "depth cameras are not ported"),
    ("cudnn_deterministic", "there is no cuDNN here; the CPU backend is deterministic by construction"),
    ("prefetch_factor", "batches are loaded in the calling thread, so there is nothing to prefetch"),
    ("persistent_workers", "batches are loaded in the calling thread, so there are no workers"),
    ("dataloader_multiprocessing_context", "batches are loaded in the calling thread, so there are no workers"),
];

/// A parsed value in the small domain Draccus' YAML scalar parsing covers.
#[derive(Debug, Clone, PartialEq)]
enum Value {
    Null,
    Bool(bool),
    Int(BigInt),
    Float(f64),
    Str(String),
    List(Vec<Value>),
}

impl Value {
    /// Parse a CLI value the way Draccus' `yaml.safe_load` does for scalars.
    ///
    /// The domain is deliberately small and documented rather than a real YAML
    /// parser: `null`/`None`, the four boolean spellings Python and YAML share, an
    /// integer, a float, a bracketed list, and otherwise a string.
    fn parse(text: &str) -> Self {
        let trimmed = text.trim();
        match trimmed {
            "null" | "None" | "~" | "" => return Self::Null,
            "true" | "True" => return Self::Bool(true),
            "false" | "False" => return Self::Bool(false),
            _ => {}
        }
        if let Some(inner) = trimmed
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
        {
            if inner.trim().is_empty() {
                return Self::List(Vec::new());
            }
            return Self::List(inner.split(',').map(Self::parse).collect());
        }
        if let Ok(integer) = trimmed.parse::<BigInt>() {
            return Self::Int(integer);
        }
        if let Ok(float) = trimmed.parse::<f64>() {
            return Self::Float(float);
        }
        Self::Str(trimmed.to_owned())
    }

    fn as_bool(&self, flag: &str) -> Result<bool, ArgumentError> {
        match self {
            Self::Bool(value) => Ok(*value),
            _ => Err(ArgumentError::Value {
                flag: flag.to_owned(),
                reason: format!("expected true or false, got {}", self.describe()),
            }),
        }
    }

    fn as_u64(&self, flag: &str) -> Result<u64, ArgumentError> {
        self.as_integer(flag, "a non-negative integer")
    }

    fn as_bigint(&self, flag: &str) -> Result<BigInt, ArgumentError> {
        match self {
            Self::Int(value) => Ok(value.clone()),
            _ => Err(ArgumentError::Value {
                flag: flag.to_owned(),
                reason: format!("expected an integer, got {}", self.describe()),
            }),
        }
    }

    /// A machine integer of the exact width the field needs.
    ///
    /// Every integer flag goes through this rather than through `as_u64` followed by
    /// an `as` cast. A cast is silent, and silence is the failure mode this parser
    /// exists to prevent: `--num_workers=4294967296` narrowed to `u32` is `0`, the
    /// one value `TrainConfig::validate` accepts, so the run trained while appearing
    /// to honour a worker count it does not implement.
    fn as_integer<T>(&self, flag: &str, expectation: &str) -> Result<T, ArgumentError>
    where
        T: TryFrom<BigInt>,
    {
        match self {
            Self::Int(value) => T::try_from(value.clone()).map_err(|_| ArgumentError::Value {
                flag: flag.to_owned(),
                reason: format!(
                    "expected {expectation} in {}..={}, got {value}",
                    Self::describe_bound::<T>(false),
                    Self::describe_bound::<T>(true),
                ),
            }),
            _ => Err(ArgumentError::Value {
                flag: flag.to_owned(),
                reason: format!("expected an integer, got {}", self.describe()),
            }),
        }
    }

    /// The inclusive bound of `T`, for the message above.
    ///
    /// `TryFrom<BigInt>` gives no way to ask a type for its range, so the two
    /// integer widths this parser actually converts to are named here. Adding a
    /// third without extending this produces a less specific message, never a wrong
    /// conversion.
    fn describe_bound<T>(upper: bool) -> String {
        let name = std::any::type_name::<T>();
        match (name, upper) {
            ("u32", false) | ("u64", false) | ("usize", false) => "0".to_owned(),
            ("u32", true) => u32::MAX.to_string(),
            ("u64", true) => u64::MAX.to_string(),
            ("usize", true) => usize::MAX.to_string(),
            (_, false) => "the type's minimum".to_owned(),
            (_, true) => "the type's maximum".to_owned(),
        }
    }

    fn as_f64(&self, flag: &str) -> Result<f64, ArgumentError> {
        match self {
            Self::Float(value) => Ok(*value),
            Self::Int(value) => Ok(value.to_string().parse::<f64>().unwrap_or(f64::NAN)),
            _ => Err(ArgumentError::Value {
                flag: flag.to_owned(),
                reason: format!("expected a number, got {}", self.describe()),
            }),
        }
    }

    fn as_string(&self, flag: &str) -> Result<String, ArgumentError> {
        match self {
            Self::Str(value) => Ok(value.clone()),
            Self::Int(value) => Ok(value.to_string()),
            _ => Err(ArgumentError::Value {
                flag: flag.to_owned(),
                reason: format!("expected a string, got {}", self.describe()),
            }),
        }
    }

    fn as_int_list(&self, flag: &str) -> Result<Vec<i64>, ArgumentError> {
        match self {
            Self::List(items) => items
                .iter()
                .map(|item| match item {
                    Self::Int(value) => i64::try_from(value).map_err(|_| ArgumentError::Value {
                        flag: flag.to_owned(),
                        reason: format!("{value} does not fit in a 64-bit integer"),
                    }),
                    other => Err(ArgumentError::Value {
                        flag: flag.to_owned(),
                        reason: format!("expected an integer, got {}", other.describe()),
                    }),
                })
                .collect(),
            _ => Err(ArgumentError::Value {
                flag: flag.to_owned(),
                reason: format!("expected a list like [0,1,2], got {}", self.describe()),
            }),
        }
    }

    fn as_string_list(&self, flag: &str) -> Result<Vec<String>, ArgumentError> {
        match self {
            Self::List(items) => items.iter().map(|item| item.as_string(flag)).collect(),
            _ => Err(ArgumentError::Value {
                flag: flag.to_owned(),
                reason: format!("expected a list, got {}", self.describe()),
            }),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Null => "null".to_owned(),
            Self::Bool(value) => value.to_string(),
            Self::Int(value) => format!("the integer {value}"),
            Self::Float(value) => format!("the number {value}"),
            Self::Str(value) => format!("the string {value:?}"),
            Self::List(items) => format!("a list of {} items", items.len()),
        }
    }
}

/// One `--name=value` or `--name value` pair, split out of the argv.
fn split_flags(args: &[String]) -> Result<Vec<(String, String)>, ArgumentError> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let Some(body) = argument.strip_prefix("--") else {
            return Err(ArgumentError::Positional(argument.clone()));
        };
        match body.split_once('=') {
            Some((flag, value)) => {
                out.push((flag.to_owned(), value.to_owned()));
                index += 1;
            }
            None => {
                // Draccus also accepts `--flag value`.
                let value = args.get(index + 1).ok_or_else(|| ArgumentError::Value {
                    flag: body.to_owned(),
                    reason: "expected a value, either as --flag=value or --flag value".to_owned(),
                })?;
                if value.starts_with("--") {
                    return Err(ArgumentError::Value {
                        flag: body.to_owned(),
                        reason: format!("expected a value but found the flag {value:?}"),
                    });
                }
                out.push((body.to_owned(), value.clone()));
                index += 2;
            }
        }
    }
    Ok(out)
}

/// Turn a `lerobot-train` argv into a run configuration.
///
/// `args` excludes the executable name, `--help` and `--version`, which
/// [`crate::dispatch`] has already handled.
pub fn parse(args: &[String]) -> Result<TrainConfig, ArgumentError> {
    let flags = split_flags(args)?;

    let mut repo_id: Option<String> = None;
    let mut root: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut config_path: Option<PathBuf> = None;
    let mut resume: Option<bool> = None;
    let mut episodes: Option<Vec<i64>> = None;
    let mut policy_overrides: Vec<(String, Value)> = Vec::new();
    let mut policy_type_given = false;
    let mut policy_path: Option<PathBuf> = None;
    let mut policy = ActConfig::default();
    policy.device = Some("cpu".to_owned());
    policy.push_to_hub = false;

    let mut job_name: Option<String> = None;
    let mut seed: Option<Option<u64>> = None;
    let mut batch_size: Option<usize> = None;
    let mut steps: Option<u64> = None;
    let mut log_freq: Option<u64> = None;
    let mut save_freq: Option<BigInt> = None;
    let mut save_checkpoint: Option<bool> = None;
    let mut tolerance_s: Option<f64> = None;
    let mut num_workers: Option<u32> = None;
    let mut use_preset: Option<bool> = None;
    let mut use_imagenet_stats: Option<bool> = None;

    for (flag, raw) in &flags {
        let value = Value::parse(raw);
        match flag.as_str() {
            "dataset.repo_id" => repo_id = Some(value.as_string(flag)?),
            "dataset.root" => root = Some(PathBuf::from(value.as_string(flag)?)),
            "dataset.episodes" => episodes = Some(value.as_int_list(flag)?),
            "dataset.use_imagenet_stats" => use_imagenet_stats = Some(value.as_bool(flag)?),
            "output_dir" => output_dir = Some(PathBuf::from(value.as_string(flag)?)),
            "config_path" => config_path = Some(PathBuf::from(value.as_string(flag)?)),
            "resume" => resume = Some(value.as_bool(flag)?),
            "job_name" => job_name = Some(value.as_string(flag)?),
            "seed" => {
                seed = Some(match value {
                    Value::Null => None,
                    other => Some(other.as_u64(flag)?),
                })
            }
            "batch_size" => batch_size = Some(value.as_integer::<usize>(flag, "a batch size")?),
            "steps" => steps = Some(value.as_u64(flag)?),
            "log_freq" => log_freq = Some(value.as_u64(flag)?),
            "save_freq" => save_freq = Some(value.as_bigint(flag)?),
            "save_checkpoint" => save_checkpoint = Some(value.as_bool(flag)?),
            "tolerance_s" => tolerance_s = Some(value.as_f64(flag)?),
            "num_workers" => num_workers = Some(value.as_integer::<u32>(flag, "a worker count")?),
            "use_policy_training_preset" => use_preset = Some(value.as_bool(flag)?),
            "wandb.enable" => {
                // Accepted only in the form that changes nothing, so that a command
                // copied from upstream's README works instead of failing on a flag
                // whose value this slice already implements.
                if value.as_bool(flag)? {
                    return Err(ArgumentError::Unsupported {
                        flag: flag.clone(),
                        reason: "Weights & Biases logging is not ported; logs go to stdout"
                            .to_owned(),
                    });
                }
            }
            "policy.type" => {
                let name = value.as_string(flag)?;
                if name != "act" {
                    return Err(ArgumentError::Unsupported {
                        flag: flag.clone(),
                        reason: format!(
                            "{name:?} is not ported; ACT is the only policy with a tensor \
                             implementation in this slice"
                        ),
                    });
                }
                policy_type_given = true;
            }
            "policy.path" => {
                policy_path = Some(PathBuf::from(value.as_string(flag)?));
            }
            other if other.starts_with("policy.") => {
                // `apply_policy_flag` reports an unrecognized field as unknown; a
                // field that is a real upstream one this slice cannot honour is
                // reclassified here rather than duplicating the refusal list.
                match apply_policy_flag(&mut policy, other, &value) {
                    Err(ArgumentError::Unknown { flag }) => return Err(classify(&flag)),
                    other => other?,
                }
                policy_overrides.push((other.to_owned(), value));
            }
            _ => return Err(classify(flag)),
        }
    }

    if let Some(path) = &policy_path {
        if !path.is_dir() {
            return Err(ArgumentError::Value {
                flag: "policy.path".to_owned(),
                reason: format!(
                    "local pretrained policy directory {} does not exist; Hub model IDs are not \
                     supported by this native training path",
                    path.display()
                ),
            });
        }
        let config_path = path.join(rerobot_train::checkpoint::CONFIG_FILE);
        let mut bytes = Vec::new();
        std::fs::File::open(&config_path)
            .map_err(|error| ArgumentError::Value {
                flag: "policy.path".to_owned(),
                reason: format!("cannot read {}: {error}", config_path.display()),
            })?
            .take(rerobot_train::limits::MAX_CHECKPOINT_JSON_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| ArgumentError::Value {
                flag: "policy.path".to_owned(),
                reason: format!("cannot read {}: {error}", config_path.display()),
            })?;
        if bytes.len() as u64 > rerobot_train::limits::MAX_CHECKPOINT_JSON_BYTES {
            return Err(ArgumentError::Value {
                flag: "policy.path".to_owned(),
                reason: format!(
                    "{} exceeds the {}-byte checkpoint config limit",
                    config_path.display(),
                    rerobot_train::limits::MAX_CHECKPOINT_JSON_BYTES
                ),
            });
        }
        let text = String::from_utf8(bytes).map_err(|error| ArgumentError::Value {
            flag: "policy.path".to_owned(),
            reason: format!("{} is not valid UTF-8: {error}", config_path.display()),
        })?;
        let mut loaded =
            ActConfig::from_checkpoint_json(&text).map_err(|error| ArgumentError::Value {
                flag: "policy.path".to_owned(),
                reason: format!(
                    "{} is not a valid ACT policy checkpoint: {error}",
                    config_path.display()
                ),
            })?;
        loaded.pretrained_path = Some(path.to_string_lossy().into_owned());
        for (flag, value) in &policy_overrides {
            apply_policy_flag(&mut loaded, flag, value)?;
        }
        policy = loaded;
        policy_type_given = true;
    }

    if resume == Some(true) {
        let config_path = config_path.ok_or_else(|| ArgumentError::Unsupported {
            flag: "resume".to_owned(),
            reason: "resume needs the optimizer and RNG state loader plus sample-exact sampler positioning; pass --config_path to a local checkpoint".to_owned(),
        })?;
        let checkpoint_dir = resolve_checkpoint_path(&config_path)?;
        let mut config = TrainConfig::from_checkpoint_dir(&checkpoint_dir).map_err(|error| {
            ArgumentError::Value {
                flag: "config_path".to_owned(),
                reason: error.to_string(),
            }
        })?;
        for (flag, value) in policy_overrides {
            apply_policy_flag(&mut config.policy, &flag, &value)?;
        }
        if let Some(value) = repo_id {
            config.dataset_repo_id = value;
        }
        if let Some(value) = root {
            config.dataset_root = value;
        }
        if let Some(value) = output_dir {
            config.output_dir = value;
        }
        if let Some(value) = episodes {
            config.dataset_episodes = Some(value);
        }
        if let Some(value) = job_name {
            config.job_name = Some(value);
        }
        if let Some(value) = seed {
            config.seed = value;
        }
        if let Some(value) = batch_size {
            config.batch_size = value;
        }
        if let Some(value) = steps {
            config.steps = value;
        }
        if let Some(value) = log_freq {
            config.log_freq = value;
        }
        if let Some(value) = save_freq {
            config.save_freq = value;
        }
        if let Some(value) = save_checkpoint {
            config.save_checkpoint = value;
        }
        if let Some(value) = tolerance_s {
            config.tolerance_s = value;
        }
        if let Some(value) = num_workers {
            config.num_workers = value;
        }
        if let Some(value) = use_preset {
            if value != config.use_policy_training_preset {
                return Err(ArgumentError::Unsupported {
                    flag: "use_policy_training_preset".to_owned(),
                    reason: "the checkpoint restores its AdamW parameter-group hyperparameters; changing the optimizer preset during resume is not supported".to_owned(),
                });
            }
        }
        if let Some(value) = use_imagenet_stats {
            config.dataset_use_imagenet_stats = value;
        }
        config.resume = true;
        config.checkpoint_path = Some(checkpoint_dir);
        return Ok(config);
    }
    if let Some(path) = config_path {
        if policy_path.is_some() {
            return Err(ArgumentError::Value {
                flag: "policy.path".to_owned(),
                reason: "cannot be combined with --config_path; choose one configuration source"
                    .to_owned(),
            });
        }
        let mut config =
            TrainConfig::from_config_file(&path).map_err(|error| ArgumentError::Value {
                flag: "config_path".to_owned(),
                reason: error.to_string(),
            })?;
        for (flag, value) in policy_overrides {
            apply_policy_flag(&mut config.policy, &flag, &value)?;
        }
        if let Some(value) = repo_id {
            config.dataset_repo_id = value;
        }
        if let Some(value) = root {
            config.dataset_root = value;
        }
        if let Some(value) = output_dir {
            config.output_dir = value;
        }
        if let Some(value) = episodes {
            config.dataset_episodes = Some(value);
        }
        if let Some(value) = job_name {
            config.job_name = Some(value);
        }
        if let Some(value) = seed {
            config.seed = value;
        }
        if let Some(value) = batch_size {
            config.batch_size = value;
        }
        if let Some(value) = steps {
            config.steps = value;
        }
        if let Some(value) = log_freq {
            config.log_freq = value;
        }
        if let Some(value) = save_freq {
            config.save_freq = value;
        }
        if let Some(value) = save_checkpoint {
            config.save_checkpoint = value;
        }
        if let Some(value) = tolerance_s {
            config.tolerance_s = value;
        }
        if let Some(value) = num_workers {
            config.num_workers = value;
        }
        if let Some(value) = use_preset {
            config.use_policy_training_preset = value;
        }
        if let Some(value) = use_imagenet_stats {
            config.dataset_use_imagenet_stats = value;
        }
        config.resume = false;
        config.checkpoint_path = None;
        return Ok(config);
    }

    if resume == Some(false) {
        // `--resume=false` is a valid explicit default and otherwise has no effect.
    }

    if !policy_type_given {
        return Err(ArgumentError::Missing {
            flag: "policy.type=act".to_owned(),
            reason: "there is no default policy; upstream demands --policy.type or --policy.path"
                .to_owned(),
        });
    }
    let repo_id = repo_id.ok_or_else(|| ArgumentError::Missing {
        flag: "dataset.repo_id".to_owned(),
        reason: "the dataset identifier has no default".to_owned(),
    })?;
    let root = match root {
        Some(root) => root,
        None => rerobot_train::hub::default_dataset_root(&repo_id).map_err(|error| {
            ArgumentError::Value {
                flag: "dataset.repo_id".to_owned(),
                reason: error.to_string(),
            }
        })?,
    };
    let output_dir = output_dir.ok_or_else(|| ArgumentError::Missing {
        flag: "output_dir".to_owned(),
        reason: "upstream defaults it to a timestamped directory; a run here names its output \
                 explicitly so that the same command is reproducible"
            .to_owned(),
    })?;

    let mut config = TrainConfig::new(repo_id, root, output_dir);
    config.policy = policy;
    config.dataset_episodes = episodes;
    if let Some(name) = job_name {
        config.job_name = Some(name);
    }
    if let Some(value) = seed {
        config.seed = value;
    }
    if let Some(value) = batch_size {
        config.batch_size = value;
    }
    if let Some(value) = steps {
        config.steps = value;
    }
    if let Some(value) = log_freq {
        config.log_freq = value;
    }
    if let Some(value) = save_freq {
        config.save_freq = value;
    }
    if let Some(value) = save_checkpoint {
        config.save_checkpoint = value;
    }
    if let Some(value) = tolerance_s {
        config.tolerance_s = value;
    }
    if let Some(value) = num_workers {
        config.num_workers = value;
    }
    if let Some(value) = use_preset {
        config.use_policy_training_preset = value;
    }
    if let Some(value) = use_imagenet_stats {
        config.dataset_use_imagenet_stats = value;
    }
    Ok(config)
}

fn resolve_checkpoint_path(path: &std::path::Path) -> Result<PathBuf, ArgumentError> {
    let checkpoint = if path.is_file() {
        let is_train_config = path.file_name().and_then(|name| name.to_str())
            == Some(rerobot_train::checkpoint::TRAIN_CONFIG_NAME)
            && path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                == Some(rerobot_train::checkpoint::PRETRAINED_MODEL_DIR);
        if !is_train_config {
            return Err(ArgumentError::Value {
                flag: "config_path".to_owned(),
                reason: "expected a checkpoint directory or its pretrained_model/train_config.json"
                    .to_owned(),
            });
        }
        path.parent()
            .and_then(|parent| parent.parent())
            .map(PathBuf::from)
            .ok_or_else(|| ArgumentError::Value {
                flag: "config_path".to_owned(),
                reason: "the checkpoint config has no parent checkpoint directory".to_owned(),
            })?
    } else if path.is_dir()
        && path.file_name().and_then(|name| name.to_str())
            == Some(rerobot_train::checkpoint::PRETRAINED_MODEL_DIR)
        && path
            .join(rerobot_train::checkpoint::TRAIN_CONFIG_NAME)
            .is_file()
    {
        path.parent()
            .map(PathBuf::from)
            .ok_or_else(|| ArgumentError::Value {
                flag: "config_path".to_owned(),
                reason: "the pretrained_model directory has no checkpoint parent".to_owned(),
            })?
    } else if path.file_name().and_then(|name| name.to_str())
        == Some(rerobot_train::checkpoint::LAST_CHECKPOINT_LINK)
    {
        let checkpoints_dir = path.parent().ok_or_else(|| ArgumentError::Value {
            flag: "config_path".to_owned(),
            reason: "the last-checkpoint marker has no parent directory".to_owned(),
        })?;
        rerobot_train::checkpoint::read_last_checkpoint(checkpoints_dir).map_err(|error| {
            ArgumentError::Value {
                flag: "config_path".to_owned(),
                reason: error.to_string(),
            }
        })?
    } else if path.is_dir()
        && path
            .join(rerobot_train::checkpoint::PRETRAINED_MODEL_DIR)
            .is_dir()
    {
        path.to_owned()
    } else {
        return Err(ArgumentError::Value {
            flag: "config_path".to_owned(),
            reason: format!("{} is not a recognized local checkpoint", path.display()),
        });
    };
    Ok(checkpoint)
}

/// Decide whether an otherwise-unrecognized flag is a real upstream one this slice
/// refuses, or nothing upstream has either.
///
/// The refusal list is matched by prefix as well as exactly, because an unported
/// config object takes its own dotted fields: refusing `--optimizer` but calling
/// `--optimizer.lr` *unknown* would report a real upstream flag as a typo. The
/// longest match wins, so a specifically-listed field keeps its own reason.
///
/// This runs only after the accepted flags have had their chance, which is what
/// lets `--wandb.enable=false` be honoured while `--wandb.project` is refused.
fn classify(flag: &str) -> ArgumentError {
    match UNSUPPORTED_ARGUMENTS
        .iter()
        .filter(|(name, _)| *name == flag || flag.starts_with(&format!("{name}.")))
        .max_by_key(|(name, _)| name.len())
    {
        Some((_, reason)) => ArgumentError::Unsupported {
            flag: flag.to_owned(),
            reason: (*reason).to_owned(),
        },
        None => ArgumentError::Unknown {
            flag: flag.to_owned(),
        },
    }
}

fn apply_policy_flag(
    policy: &mut ActConfig,
    flag: &str,
    value: &Value,
) -> Result<(), ArgumentError> {
    let field = flag.strip_prefix("policy.").expect("checked by the caller");

    // `normalization_mapping.<FEATURE_TYPE>=<MODE>`, which is how Draccus spells a
    // dict entry on the command line.
    if let Some(feature_type) = field.strip_prefix("normalization_mapping.") {
        let mode: NormalizationMode = value.as_string(flag)?.parse().map_err(
            |error: rerobot_core::types::ParseEnumError| ArgumentError::Value {
                flag: flag.to_owned(),
                reason: error.to_string(),
            },
        )?;
        policy
            .normalization_mapping
            .insert(feature_type.to_owned(), mode);
        return Ok(());
    }

    let integer = |value: &Value| -> Result<BigInt, ArgumentError> {
        match value {
            Value::Int(number) => Ok(number.clone()),
            other => Err(ArgumentError::Value {
                flag: flag.to_owned(),
                reason: format!("expected an integer, got {}", other.describe()),
            }),
        }
    };

    match field {
        "n_obs_steps" => policy.n_obs_steps = integer(value)?,
        "chunk_size" => policy.chunk_size = integer(value)?,
        "n_action_steps" => policy.n_action_steps = integer(value)?,
        "dim_model" => policy.dim_model = integer(value)?,
        "n_heads" => policy.n_heads = integer(value)?,
        "dim_feedforward" => policy.dim_feedforward = integer(value)?,
        "n_encoder_layers" => policy.n_encoder_layers = integer(value)?,
        "n_decoder_layers" => policy.n_decoder_layers = integer(value)?,
        "n_vae_encoder_layers" => policy.n_vae_encoder_layers = integer(value)?,
        "latent_dim" => policy.latent_dim = integer(value)?,
        "use_vae" => policy.use_vae = value.as_bool(flag)?,
        "pre_norm" => policy.pre_norm = value.as_bool(flag)?,
        "dropout" => policy.dropout = value.as_f64(flag)?,
        "kl_weight" => policy.kl_weight = value.as_f64(flag)?,
        "optimizer_lr" => policy.optimizer_lr = value.as_f64(flag)?,
        "optimizer_weight_decay" => policy.optimizer_weight_decay = value.as_f64(flag)?,
        "optimizer_lr_backbone" => policy.optimizer_lr_backbone = value.as_f64(flag)?,
        "feedforward_activation" => policy.feedforward_activation = value.as_string(flag)?,
        "vision_backbone" => policy.vision_backbone = value.as_string(flag)?,
        "device" => policy.device = optional_string(value, flag)?,
        "use_amp" => policy.use_amp = value.as_bool(flag)?,
        "use_peft" => policy.use_peft = value.as_bool(flag)?,
        "push_to_hub" => policy.push_to_hub = value.as_bool(flag)?,
        "repo_id" => policy.repo_id = optional_string(value, flag)?,
        "private" => {
            policy.private = match value {
                Value::Null => None,
                other => Some(other.as_bool(flag)?),
            }
        }
        "license" => policy.license = optional_string(value, flag)?,
        "tags" => {
            policy.tags = match value {
                Value::Null => None,
                other => Some(other.as_string_list(flag)?),
            }
        }
        "temporal_ensemble_coeff" => {
            policy.temporal_ensemble_coeff = match value {
                Value::Null => None,
                other => Some(other.as_f64(flag)?),
            }
        }
        "pretrained_backbone_weights" => {
            policy.pretrained_backbone_weights = optional_string(value, flag)?
        }
        // Accepted only at its default, because the ResNet backbone it configures
        // is not ported: silently taking a value that cannot take effect would be
        // exactly the failure mode this parser exists to prevent.
        "replace_final_stride_with_dilation" => {
            if value.as_bool(flag).unwrap_or(true) {
                return Err(ArgumentError::Unsupported {
                    flag: flag.to_owned(),
                    reason: "it configures the ResNet backbone, which is not ported; only false \
                             is accepted"
                        .to_owned(),
                });
            }
        }
        "pretrained_revision" => policy.pretrained_revision = optional_string(value, flag)?,
        _ => {
            return Err(ArgumentError::Unknown {
                flag: flag.to_owned(),
            })
        }
    }
    Ok(())
}

fn optional_string(value: &Value, flag: &str) -> Result<Option<String>, ArgumentError> {
    match value {
        Value::Null => Ok(None),
        other => Ok(Some(other.as_string(flag)?)),
    }
}

/// The extra `--help` section listing what `lerobot-train` accepts.
///
/// The device lines are read from [`rerobot_train::device::CUDA_COMPILED`]
/// rather than from a `cfg` of this crate, so the help describes the binary the
/// user is holding and cannot drift from what the training crate will accept.
pub fn help_section() -> String {
    let (device_flag, scope_note) = if rerobot_train::device::CUDA_COMPILED {
        (
            "--policy.device=cpu|cuda",
            "This binary was built with CUDA support, so --policy.device=cuda (or cuda:0)\n\
             trains on NVIDIA GPU 0. A requested GPU that cannot be initialized is an\n\
             error; the run never falls back to the CPU.",
        )
    } else {
        (
            "--policy.device=cpu",
            "This binary has no CUDA backend compiled in, so `cpu` is the only device it\n\
             accepts. Rebuild with `--features cuda` on a machine with the NVIDIA CUDA\n\
             toolkit to enable --policy.device=cuda.",
        )
    };
    let mut text = String::from(
        "Supported arguments (every other upstream argument is refused, never ignored):\n\
         \n\
         Required:\n\
         \x20 --dataset.repo_id=ID        dataset identifier, recorded in the checkpoint\n\
         --dataset.root=DIR          optional local directory; absent means Hub snapshot cache\n\
         \x20 --output_dir=DIR            fresh-run output directory, must not exist\n\
         \x20 --policy.type=act           ACT is the only ported policy\n\
         \n\
         Pretrained policy:\n\
         \x20 --policy.path=DIR           local ACT checkpoint's pretrained_model directory;\n\
         \x20                              its config and weights seed this run (Hub model IDs\n\
         \x20                              are refused because native model download is not ported)\n\
         \n\
         Resume:\n\
         \x20 --resume=true --config_path=DIR|FILE\n\
         \x20                              restore a local checkpoint; FILE may be\n\
         \x20                              pretrained_model/train_config.json, or DIR\n\
         \x20                              may be the checkpoint directory or checkpoints/last\n\
         \x20                              (dataset and policy flags are loaded from it)\n\
         \n\
         Fresh config:\n\
         \x20 --config_path=FILE --resume=false\n\
         \x20                              load a local JSON train_config.json and start a\n\
         \x20                              new run; CLI values override fields in the file\n\
         \x20                              (general YAML/Draccus configs are not supported)\n\
         \n\
         Run:\n\
         \x20 --steps=N --batch_size=N --seed=N|null --log_freq=N --save_freq=N\n\
         \x20 --save_checkpoint=BOOL --tolerance_s=SECONDS --job_name=NAME\n\
         \x20 --dataset.episodes=[0,1,..] --num_workers=0\n\
         \x20 --dataset.use_imagenet_stats=BOOL  per-channel camera statistics;\n\
         \x20                                    true (the default) uses IMAGENET_STATS,\n\
         \x20                                    false leaves camera frames untouched\n\
         \n\
         Policy (ACTConfig fields):\n\
         \x20 --policy.chunk_size --policy.n_action_steps --policy.dim_model\n\
         \x20 --policy.n_heads --policy.dim_feedforward --policy.n_encoder_layers\n\
         \x20 --policy.n_decoder_layers --policy.n_vae_encoder_layers\n\
         \x20 --policy.latent_dim --policy.use_vae --policy.pre_norm\n\
         \x20 --policy.dropout --policy.kl_weight --policy.optimizer_lr\n\
         \x20 --policy.optimizer_weight_decay --policy.optimizer_lr_backbone\n\
         \x20 --policy.feedforward_activation\n",
    );
    text.push_str(&format!("   {device_flag}\n"));
    text.push_str(
        "   --policy.normalization_mapping.STATE=MEAN_STD\n\
         \n\
         Refused, with a reason naming what is missing:\n",
    );
    let mut names: Vec<&str> = UNSUPPORTED_ARGUMENTS
        .iter()
        .map(|(name, _)| *name)
        .collect();
    names.sort_unstable();
    names.dedup();
    for chunk in names.chunks(4) {
        text.push_str("  ");
        text.push_str(
            &chunk
                .iter()
                .map(|name| format!("--{name}"))
                .collect::<Vec<_>>()
                .join(" "),
        );
        text.push('\n');
    }
    text.push_str("\nScope: a LeRobot v3.0 dataset from local disk or the Hugging Face Hub, ACT. State and action columns\nare read, and so is a dtype=\"image\" camera column whose PNG or JPEG frames are embedded in the parquet file. Video features and distributed training, mixed precision, LR schedulers and environment evaluation are not ported.\n\n");
    text.push_str(scope_note);
    text
}
