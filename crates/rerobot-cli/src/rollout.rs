//! Argument parsing and execution for the local and state-only hardware rollout slices.
//!
//! Upstream's `lerobot-rollout` has a much larger robot/teleoperator/strategy
//! surface. This parser accepts the local checkpoint + local dataset path and a
//! calibrated SO-101 follower path; unsupported strategies and devices are still
//! refused by name.

use rerobot_core::BigInt;
use rerobot_hardware::feetech::open_serial;
use rerobot_hardware::so101::{load_calibration, So101Follower};
use rerobot_train::candle_core::{Device, Tensor};
use rerobot_train::data::batch::Batch;
use rerobot_train::deploy::{InferenceSession, InferenceStep};
use rerobot_train::error::TrainError;
use rerobot_train::indexmap::IndexMap;
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
    /// Optional SO-101 follower serial port. When set, observations come from
    /// the follower instead of a local dataset.
    pub robot_port: Option<PathBuf>,
    /// Upstream-compatible SO-101 calibration JSON path.
    pub calibration_path: Option<PathBuf>,
    /// Explicit acknowledgement that the hardware path may enable torque.
    pub confirm: bool,
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
    let mut robot_type = None;
    let mut robot_port = None;
    let mut calibration_path = None;
    let mut confirm = false;

    for (flag, value) in split_flags(args)? {
        match flag.as_str() {
            "policy.path" => policy_path = Some(PathBuf::from(value)),
            "dataset.root" => dataset_root = Some(PathBuf::from(value)),
            "steps" => steps = Some(parse_index(&flag, &value)?),
            "start_index" => start_index = parse_index(&flag, &value)?,
            "policy.device" => device = Some(value),
            "robot.type" if value == "so101_follower" => robot_type = Some(value),
            "robot.type" => {
                return Err(ArgumentError::Unsupported {
                    flag,
                    reason: "only the SO-101 follower hardware source is available in this slice"
                        .to_owned(),
                })
            }
            "robot.port" => robot_port = Some(PathBuf::from(value)),
            "robot.calibration" => calibration_path = Some(PathBuf::from(value)),
            "robot.confirm" => {
                confirm = value.parse::<bool>().map_err(|_| ArgumentError::Value {
                    flag: flag.to_owned(),
                    reason: "expected true or false".to_owned(),
                })?;
            }
            "robot" | "teleop" | "teleop.type" => {
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

    if robot_type.is_some() {
        if dataset_root.is_some() || start_index != 0 {
            return Err(ArgumentError::Value {
                flag: "dataset.root".to_owned(),
                reason: "a hardware rollout cannot also use a dataset source or start_index"
                    .to_owned(),
            });
        }
        if !confirm {
            return Err(ArgumentError::Value {
                flag: "robot.confirm".to_owned(),
                reason:
                    "robot.confirm=true is required before the hardware rollout can enable torque"
                        .to_owned(),
            });
        }
        if robot_port.is_none() {
            return Err(ArgumentError::Missing {
                flag: "robot.port".to_owned(),
                reason: "the SO-101 follower serial port is required".to_owned(),
            });
        }
        if calibration_path.is_none() {
            return Err(ArgumentError::Missing {
                flag: "robot.calibration".to_owned(),
                reason: "a calibrated SO-101 follower is required; defaults are not safe for policy control"
                    .to_owned(),
            });
        }
    } else if robot_port.is_some() || calibration_path.is_some() || confirm {
        return Err(ArgumentError::Missing {
            flag: "robot.type".to_owned(),
            reason: "robot.port, robot.calibration and robot.confirm require --robot.type=so101_follower"
                .to_owned(),
        });
    }

    let dataset_root = match dataset_root {
        Some(root) => root,
        None if robot_type.is_some() => PathBuf::new(),
        None => {
            return Err(ArgumentError::Missing {
                flag: "dataset.root".to_owned(),
                reason: "this hardware-independent path reads observations from a local dataset"
                    .to_owned(),
            })
        }
    };

    Ok(RolloutConfig {
        policy_path,
        dataset_root,
        steps,
        start_index,
        device,
        robot_port,
        calibration_path,
        confirm,
    })
}

/// Help section for the supported offline invocation.
pub fn help_section() -> &'static str {
    "Accepted rollout options:\n  --policy.path=DIR       ACT checkpoint's pretrained_model directory\n  --dataset.root=DIR      local LeRobot dataset root\n  --steps=N               finite number of actions to emit\n  --start_index=N         first dataset frame (default: 0)\n  --policy.device=cpu     optional device override\n\nHardware source (state-only SO-101 follower):\n  --robot.type=so101_follower\n  --robot.port=PATH       serial port at 1 Mbps\n  --robot.calibration=FILE upstream calibration JSON\n  --robot.confirm=true    required before enabling torque\n\nThe dataset path loads a local checkpoint and emits actions without hardware. The\nSO-101 path reads six calibrated joint observations, runs the local ACT checkpoint,\nsends finite position actions, and releases torque on exit. Cameras, environments,\nteleoperators, async inference, and video shards remain refused."
}

/// Build the single-observation batch expected by `InferenceSession`.
fn state_batch(
    state: [f32; 6],
    device: &Device,
    frame_index: i64,
) -> rerobot_train::error::Result<Batch> {
    if let Some(value) = state.iter().find(|value| !value.is_finite()) {
        return Err(TrainError::NonFinite {
            step: frame_index.max(0) as u64 + 1,
            quantity: "SO-101 observation".to_owned(),
            value: value.to_string(),
        });
    }
    let mut features = IndexMap::new();
    features.insert(
        "observation.state".to_owned(),
        Tensor::from_vec(state.to_vec(), (1, 6), device)?,
    );
    Ok(Batch {
        features,
        images: IndexMap::new(),
        padding: IndexMap::new(),
        tasks: vec![String::new()],
        indices: vec![frame_index],
    })
}

fn action_array(step: &InferenceStep) -> rerobot_train::error::Result<[f32; 6]> {
    step.action.clone().try_into().map_err(|values: Vec<f32>| {
        TrainError::unsupported(format!(
            "SO-101 policy must emit exactly six action values, got {}",
            values.len()
        ))
    })
}

fn hardware_error(error: impl fmt::Display) -> TrainError {
    TrainError::unsupported(format!("SO-101 hardware rollout failed: {error}"))
}

fn run_so101(
    config: &RolloutConfig,
    observe: &mut dyn FnMut(&str),
) -> rerobot_train::error::Result<()> {
    let port = config
        .robot_port
        .as_deref()
        .and_then(|path| path.to_str())
        .ok_or_else(|| hardware_error("robot.port must be valid UTF-8"))?;
    let calibration_path = config.calibration_path.as_deref().ok_or_else(|| {
        hardware_error("robot.calibration is required for the SO-101 hardware source")
    })?;
    if !config.confirm {
        return Err(hardware_error(
            "robot.confirm=true is required before torque can be enabled",
        ));
    }
    if config.steps == 0 || config.steps > rerobot_train::limits::MAX_ROLLOUT_TRACE_STEPS {
        return Err(hardware_error(format!(
            "steps must be in 1..={}",
            rerobot_train::limits::MAX_ROLLOUT_TRACE_STEPS
        )));
    }

    // Load and validate all policy files before opening the serial port. A malformed
    // checkpoint must never cause a hardware side effect.
    let mut session =
        InferenceSession::load_checkpoint(&config.policy_path, config.device.as_deref())?;
    let calibration = load_calibration(calibration_path).map_err(hardware_error)?;
    let bus = open_serial(port).map_err(hardware_error)?;
    let mut robot = So101Follower::with_calibration(bus, calibration);
    let mut torque_attempted = false;
    let result = (|| {
        let found = robot.ping_all().map_err(hardware_error)?;
        if !found.iter().all(|found| *found) {
            return Err(hardware_error(format!(
                "expected all six servo IDs {:?}, found {found:?}",
                rerobot_hardware::so101::SO101_MOTOR_IDS
            )));
        }
        // Upstream configures position mode with torque disabled. The first ACT
        // query also runs before torque is enabled, so model/schema errors remain
        // side-effect-free.
        robot.configure_position_mode().map_err(hardware_error)?;
        let mut send_step = |frame_index: i64| -> rerobot_train::error::Result<()> {
            let positions = robot.read_positions_ticks().map_err(hardware_error)?;
            let state = robot
                .positions_to_observation(positions)
                .map_err(hardware_error)?;
            let batch = state_batch(state, session.device(), frame_index)?;
            let step = session.select_action_on_batch(&batch)?;
            let action = action_array(&step)?;
            if !torque_attempted {
                torque_attempted = true;
                robot.enable_torque().map_err(hardware_error)?;
            }
            let sent = robot
                .set_positions_degrees(action)
                .map_err(hardware_error)?;
            observe(&format!(
                "source:so101 frame:{} action:{:?} sent_ticks:{sent:?} queried:{}",
                step.frame_index, step.action, step.queried_policy
            ));
            Ok(())
        };
        for frame_index in 0..config.steps {
            send_step(
                i64::try_from(frame_index)
                    .map_err(|_| hardware_error("rollout frame index does not fit in i64"))?,
            )?;
        }
        Ok(())
    })();
    let cleanup = if torque_attempted {
        robot.disable_torque().map_err(hardware_error)
    } else {
        Ok(())
    };
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(cleanup_error)) => Err(hardware_error(format!(
            "{error}; torque cleanup also failed: {cleanup_error}"
        ))),
    }
}

/// Parse and run one offline rollout, writing one machine-readable line per action.
pub fn run(
    config: &RolloutConfig,
    observe: &mut dyn FnMut(&str),
) -> rerobot_train::error::Result<()> {
    if config.robot_port.is_some() {
        return run_so101(config, observe);
    }
    let mut session = InferenceSession::load(
        &config.policy_path,
        &config.dataset_root,
        config.device.as_deref(),
    )?;
    session.rollout_with_sink(config.start_index, config.steps, |step| {
        observe(&format!(
            "frame:{} action:{:?} queried:{}",
            step.frame_index, step.action, step.queried_policy
        ));
        Ok(())
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rerobot_train::deploy::InferenceStep;

    #[test]
    fn so101_state_batch_is_one_float32_observation() {
        let batch = state_batch(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &rerobot_train::candle_core::Device::Cpu,
            7,
        )
        .unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch.indices, vec![7]);
        assert_eq!(batch.feature("observation.state").unwrap().dims(), &[1, 6]);
        assert_eq!(
            batch
                .feature("observation.state")
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        );
    }

    #[test]
    fn so101_action_requires_exactly_six_values() {
        let step = InferenceStep {
            frame_index: 0,
            action: vec![0.0; 5],
            queried_policy: true,
        };

        let error = action_array(&step).expect_err("SO-101 actions cannot be truncated or padded");

        assert!(error.to_string().contains("six"), "{error}");
    }
}
