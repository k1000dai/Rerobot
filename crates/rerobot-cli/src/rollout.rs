//! Argument parsing and execution for the hardware-independent rollout slice.
//!
//! Upstream's `lerobot-rollout` has a much larger robot/teleoperator/strategy
//! surface. This parser accepts only the local checkpoint + local dataset path
//! needed by the native inference boundary and refuses the rest by name.

use rerobot_core::BigInt;
use rerobot_train::deploy::InferenceSession;
use std::fmt;
use std::path::PathBuf;

/// The executable's supported offline rollout configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolloutConfig {
    /// Directory containing `config.json` and `model.safetensors`.
    pub policy_path: PathBuf,
    /// Local LeRobot dataset root used as the observation source.
    pub dataset_root: PathBuf,
    /// Number of observations/actions to emit.
    pub steps: usize,
    /// First dataset frame to use.
    pub start_index: usize,
    /// Optional device override.
    pub device: Option<String>,
}

/// Why a rollout command could not be started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgumentError {
    /// A real upstream option names a boundary this slice does not implement.
    Unsupported {
        /// Option name without leading dashes.
        flag: String,
        /// The unsupported boundary explanation.
        reason: String,
    },
    /// The option is not part of upstream's rollout configuration.
    Unknown {
        /// Option name without leading dashes.
        flag: String,
    },
    /// The option has a malformed or missing value.
    Value {
        /// Option name without leading dashes.
        flag: String,
        /// The value error explanation.
        reason: String,
    },
    /// A required offline option was absent.
    Missing {
        /// Option name without leading dashes.
        flag: String,
        /// The missing-value explanation.
        reason: String,
    },
    /// A positional argument is not accepted by Draccus or this parser.
    Positional(String),
}

impl fmt::Display for ArgumentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { flag, reason } => {
                write!(formatter, "--{flag} is not supported in this slice: {reason}")
            }
            Self::Unknown { flag } => write!(
                formatter,
                "--{flag} is not a lerobot-rollout argument; try `lerobot-rollout --help`"
            ),
            Self::Value { flag, reason } => write!(formatter, "--{flag}: {reason}"),
            Self::Missing { flag, reason } => write!(formatter, "--{flag} is required: {reason}"),
            Self::Positional(argument) => write!(
                formatter,
                "unexpected argument {argument:?}; every lerobot-rollout option is a --name=value flag"
            ),
        }
    }
}

impl std::error::Error for ArgumentError {}

fn split_flags(args: &[String]) -> Result<Vec<(String, String)>, ArgumentError> {
    let mut flags = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let Some(body) = argument.strip_prefix("--") else {
            return Err(ArgumentError::Positional(argument.clone()));
        };
        if let Some((flag, value)) = body.split_once('=') {
            flags.push((flag.to_owned(), value.to_owned()));
            index += 1;
            continue;
        }
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
        flags.push((body.to_owned(), value.clone()));
        index += 2;
    }
    Ok(flags)
}

fn parse_index(flag: &str, value: &str) -> Result<usize, ArgumentError> {
    let integer = value.parse::<BigInt>().map_err(|_| ArgumentError::Value {
        flag: flag.to_owned(),
        reason: "expected a decimal integer".to_owned(),
    })?;
    usize::try_from(integer).map_err(|_| ArgumentError::Value {
        flag: flag.to_owned(),
        reason: format!("integer is outside the supported range 0..={}", usize::MAX),
    })
}

/// Parse the supported `lerobot-rollout` arguments.
pub fn parse(args: &[String]) -> Result<RolloutConfig, ArgumentError> {
    let mut policy_path = None;
    let mut dataset_root = None;
    let mut steps = None;
    let mut start_index = 0usize;
    let mut device = None;

    for (flag, value) in split_flags(args)? {
        match flag.as_str() {
            "policy.path" => policy_path = Some(PathBuf::from(value)),
            "dataset.root" => dataset_root = Some(PathBuf::from(value)),
            "steps" => steps = Some(parse_index(&flag, &value)?),
            "start_index" => start_index = parse_index(&flag, &value)?,
            "policy.device" => device = Some(value),
            "robot" | "robot.type" | "teleop" | "teleop.type" => {
                return Err(ArgumentError::Unsupported {
                    flag,
                    reason: "robot drivers are hardware-gated and this path emits actions from a local dataset instead".to_owned(),
                })
            }
            "strategy" | "strategy.type" | "inference" | "inference.type" | "duration"
            | "task" | "display_data" | "display_mode" | "display_ip" | "display_port"
            | "use_torch_compile" => {
                return Err(ArgumentError::Unsupported {
                    flag,
                    reason: "the hardware rollout strategy is not ported; use the local dataset-backed deployment boundary".to_owned(),
                })
            }
            other if other.starts_with("dataset.") => {
                return Err(ArgumentError::Unsupported {
                    flag,
                    reason: "Hub/video dataset rollout options are not ported; only dataset.root is local and supported".to_owned(),
                })
            }
            other if other.starts_with("inference.") || other.starts_with("robot.") || other.starts_with("teleop.") => {
                return Err(ArgumentError::Unsupported {
                    flag: other.to_owned(),
                    reason: "the requested runtime backend is not ported".to_owned(),
                })
            }
            other => return Err(ArgumentError::Unknown {
                flag: other.to_owned(),
            }),
        }
    }

    let policy_path = policy_path.ok_or_else(|| ArgumentError::Missing {
        flag: "policy.path".to_owned(),
        reason: "the local directory containing an ACT checkpoint has no default".to_owned(),
    })?;
    let dataset_root = dataset_root.ok_or_else(|| ArgumentError::Missing {
        flag: "dataset.root".to_owned(),
        reason: "this hardware-independent path reads observations from a local dataset".to_owned(),
    })?;
    let steps = steps.ok_or_else(|| ArgumentError::Missing {
        flag: "steps".to_owned(),
        reason:
            "an explicit finite rollout bound is required; an unbounded hardware loop is not ported"
                .to_owned(),
    })?;
    if steps == 0 {
        return Err(ArgumentError::Value {
            flag: "steps".to_owned(),
            reason: "must be positive".to_owned(),
        });
    }
    if steps > rerobot_train::limits::MAX_ROLLOUT_TRACE_STEPS {
        return Err(ArgumentError::Value {
            flag: "steps".to_owned(),
            reason: format!(
                "rollout trace would exceed the supported bound {} steps",
                rerobot_train::limits::MAX_ROLLOUT_TRACE_STEPS
            ),
        });
    }

    Ok(RolloutConfig {
        policy_path,
        dataset_root,
        steps,
        start_index,
        device,
    })
}

/// Help section for the supported offline invocation.
pub fn help_section() -> &'static str {
    "Accepted offline options:\n  --policy.path=DIR       ACT checkpoint's pretrained_model directory\n  --dataset.root=DIR      local LeRobot dataset root\n  --steps=N               finite number of actions to emit\n  --start_index=N         first dataset frame (default: 0)\n  --policy.device=cpu     optional device override\n\nThis is a hardware-independent deployment path: it loads a checkpoint, reads local\nobservations, and emits actions. ACT action queues and temporal ensembling are\nsupported from checkpoint config. Robot drivers, teleoperators, environments,\nvisualization, and video shards are refused."
}

/// Parse and run one offline rollout, writing one machine-readable line per action.
pub fn run(
    config: &RolloutConfig,
    observe: &mut dyn FnMut(&str),
) -> rerobot_train::error::Result<()> {
    let mut session = InferenceSession::load(
        &config.policy_path,
        &config.dataset_root,
        config.device.as_deref(),
    )?;
    for step in session.rollout(config.start_index, config.steps)? {
        observe(&format!(
            "frame:{} action:{:?} queried:{}",
            step.frame_index, step.action, step.queried_policy
        ));
    }
    Ok(())
}
