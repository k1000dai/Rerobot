//! Argument parsing and execution for the local and state-only hardware rollout slices.
//!
//! Upstream's `lerobot-rollout` has a much larger robot/teleoperator/strategy
//! surface. This parser accepts the local checkpoint + local dataset path and a
//! calibrated SO-101 follower path; unsupported strategies and devices are still
//! refused by name.

use rerobot_core::BigInt;
use rerobot_hardware::feetech::{open_serial, FeetechBus};
use rerobot_hardware::so101::{load_calibration, JointCalibration, So101Follower};
use rerobot_train::candle_core::{Device, Tensor};
use rerobot_train::data::batch::Batch;
use rerobot_train::deploy::{InferenceSession, InferenceStep};
use rerobot_train::error::TrainError;
use rerobot_train::indexmap::IndexMap;
use std::fmt;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// The executable's supported offline rollout configuration.
#[derive(Debug, Clone)]
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
    /// Target hardware control-loop frequency, matching upstream's default.
    pub fps: f64,
}

impl PartialEq for RolloutConfig {
    fn eq(&self, other: &Self) -> bool {
        self.policy_path == other.policy_path
            && self.dataset_root == other.dataset_root
            && self.steps == other.steps
            && self.start_index == other.start_index
            && self.device == other.device
            && self.robot_port == other.robot_port
            && self.calibration_path == other.calibration_path
            && self.confirm == other.confirm
            && self.fps.to_bits() == other.fps.to_bits()
    }
}

impl Eq for RolloutConfig {}

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

fn parse_fps(flag: &str, value: &str) -> Result<f64, ArgumentError> {
    let fps = value.parse::<f64>().map_err(|_| ArgumentError::Value {
        flag: flag.to_owned(),
        reason: "expected a positive finite number".to_owned(),
    })?;
    if !fps.is_finite() || fps <= 0.0 {
        return Err(ArgumentError::Value {
            flag: flag.to_owned(),
            reason: "expected a positive finite number".to_owned(),
        });
    }
    Ok(fps)
}

/// Return the sleep remaining after one hardware control tick.
///
/// Upstream sleeps for `max(1 / fps - elapsed, 0)` after reading the
/// observation, selecting an action, and sending it. Invalid frequencies are
/// rejected by the CLI; returning zero here keeps the low-level helper
/// bounded for library callers too.
fn control_sleep_duration(fps: f64, elapsed: Duration) -> Duration {
    if !fps.is_finite() || fps <= 0.0 {
        return Duration::ZERO;
    }
    let interval_seconds = 1.0 / fps;
    if !interval_seconds.is_finite() || interval_seconds > Duration::MAX.as_secs_f64() {
        return Duration::ZERO;
    }
    Duration::from_secs_f64(interval_seconds).saturating_sub(elapsed)
}

/// Sleep until a control-loop deadline without oversleeping short intervals.
///
/// This mirrors upstream `robot_utils.precise_sleep`: on macOS and Windows it
/// sleeps most of a long remainder and spins for the final 10 ms; other targets
/// use the platform scheduler directly.
fn precise_sleep(duration: Duration) {
    if duration.is_zero() {
        return;
    }
    if !(cfg!(target_os = "macos") || cfg!(target_os = "windows")) {
        std::thread::sleep(duration);
        return;
    }
    let Some(deadline) = Instant::now().checked_add(duration) else {
        std::thread::sleep(duration);
        return;
    };
    let spin_threshold = Duration::from_millis(10);
    let sleep_margin = Duration::from_millis(5);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        if remaining > spin_threshold {
            std::thread::sleep(remaining.saturating_sub(sleep_margin));
        } else {
            std::hint::spin_loop();
        }
    }
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
    let mut fps = 30.0;

    for (flag, value) in split_flags(args)? {
        match flag.as_str() {
            "policy.path" => policy_path = Some(PathBuf::from(value)),
            "dataset.root" => dataset_root = Some(PathBuf::from(value)),
            "steps" => steps = Some(parse_index(&flag, &value)?),
            "start_index" => start_index = parse_index(&flag, &value)?,
            "fps" => fps = parse_fps(&flag, &value)?,
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
        fps,
    })
}

/// Help section for the supported offline invocation.
pub fn help_section() -> &'static str {
    "Accepted rollout options:\n  --policy.path=DIR       ACT checkpoint's pretrained_model directory\n  --dataset.root=DIR      local LeRobot dataset root\n  --steps=N               finite number of actions to emit\n  --start_index=N         first dataset frame (default: 0)\n  --fps=HZ                hardware control-loop target (default: 30)\n  --policy.device=cpu     optional device override\n\nHardware source (state-only SO-101 follower):\n  --robot.type=so101_follower\n  --robot.port=PATH       serial port at 1 Mbps\n  --robot.calibration=FILE upstream calibration JSON\n  --robot.confirm=true    required before enabling torque\n\nThe dataset path loads a local checkpoint and emits actions without hardware. The\nSO-101 path reads six calibrated joint observations, runs the local ACT checkpoint,\nsends finite position actions, and releases torque on exit. Cameras, environments,\nteleoperators, async inference, and video shards remain refused."
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
    run_so101_with_bus(config, observe, &mut session, bus, calibration).map(|_| ())
}

/// Run the hardware adapter over an already validated byte transport.
///
/// The serial-opening function above is deliberately kept outside this routine: the
/// policy and calibration are loaded before a hardware side effect, while this
/// generic boundary lets the same handshake/control loop be exercised against a
/// protocol-valid transport in tests. It returns the transport after a successful
/// run so tests can inspect the exact packets emitted by the real adapter.
fn run_so101_with_bus<T: Read + Write>(
    config: &RolloutConfig,
    observe: &mut dyn FnMut(&str),
    session: &mut InferenceSession,
    bus: FeetechBus<T>,
    calibration: [JointCalibration; 6],
) -> rerobot_train::error::Result<T> {
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
            let loop_start = Instant::now();
            send_step(
                i64::try_from(frame_index)
                    .map_err(|_| hardware_error("rollout frame index does not fit in i64"))?,
            )?;
            let sleep_time = control_sleep_duration(config.fps, loop_start.elapsed());
            precise_sleep(sleep_time);
        }
        Ok(())
    })();
    let cleanup = if torque_attempted {
        robot.disable_torque().map_err(hardware_error)
    } else {
        Ok(())
    };
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(robot.into_inner()),
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
    use rerobot_core::dataset::stats::{DatasetStats, FeatureStats};
    use rerobot_core::policy::act::ActConfig;
    use rerobot_core::random::SplitMix64;
    use rerobot_core::types::{FeatureType, PolicyFeature};
    use rerobot_hardware::feetech::FeetechBus;
    use rerobot_train::candle_core::Device;
    use rerobot_train::deploy::InferenceStep;
    use rerobot_train::model::act::ActModel;
    use std::collections::VecDeque;
    use std::io::{self, Read, Write};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "rerobot-rollout-unit-{}-{label}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("test directory creates");
            Self(path)
        }

        fn child(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[derive(Default)]
    struct RespondingPort {
        incoming: VecDeque<u8>,
        packets: Vec<Vec<u8>>,
    }

    impl RespondingPort {
        fn status(id: u8, parameters: &[u8]) -> Vec<u8> {
            let mut packet = vec![0xff, 0xff, id, (parameters.len() + 2) as u8, 0];
            packet.extend_from_slice(parameters);
            let checksum = !packet[2..]
                .iter()
                .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
            packet.push(checksum);
            packet
        }

        fn queue_status(&mut self, id: u8, parameters: &[u8]) {
            self.incoming.extend(Self::status(id, parameters));
        }
    }

    impl Read for RespondingPort {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            if self.incoming.is_empty() {
                return Ok(0);
            }
            let count = output.len().min(self.incoming.len());
            for byte in &mut output[..count] {
                *byte = self
                    .incoming
                    .pop_front()
                    .expect("count came from the queue");
            }
            Ok(count)
        }
    }

    impl Write for RespondingPort {
        fn write(&mut self, input: &[u8]) -> io::Result<usize> {
            self.packets.push(input.to_vec());
            if input.len() < 6 {
                return Ok(input.len());
            }
            let id = input[2];
            let instruction = input[4];
            let parameters = &input[5..input.len() - 1];
            match instruction {
                0x01 => self.queue_status(id, &[]),
                0x02 => {
                    let address = parameters[0];
                    let length = usize::from(parameters[1]);
                    let values = if address == rerobot_hardware::so101::MODEL_NUMBER {
                        vec![0x09, 0x03]
                    } else {
                        vec![0; length]
                    };
                    self.queue_status(id, &values);
                }
                0x82 => {
                    let length = usize::from(parameters[1]);
                    for servo_id in &parameters[2..] {
                        let values = if length == 2 {
                            vec![0x00, 0x08]
                        } else {
                            vec![0; length]
                        };
                        self.queue_status(*servo_id, &values);
                    }
                }
                _ => {}
            }
            Ok(input.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn six_joint_policy() -> (TempDir, PathBuf) {
        let directory = TempDir::new("mock-so101-policy");
        let checkpoint = directory.child("pretrained_model");
        std::fs::create_dir_all(&checkpoint).expect("checkpoint directory creates");
        let mut policy = ActConfig::default();
        policy.device = Some("cpu".to_owned());
        policy.input_features = Some(IndexMap::from([(
            "observation.state".to_owned(),
            PolicyFeature::new(FeatureType::State, [6]),
        )]));
        policy.output_features = Some(IndexMap::from([(
            "action".to_owned(),
            PolicyFeature::new(FeatureType::Action, [6]),
        )]));
        policy.chunk_size = 2.into();
        policy.n_action_steps = 2.into();
        policy.dim_model = 32.into();
        policy.n_heads = 4.into();
        policy.dim_feedforward = 64.into();
        policy.n_encoder_layers = 1.into();
        policy.n_decoder_layers = 1.into();
        policy.n_vae_encoder_layers = 1.into();
        policy.latent_dim = 8.into();
        policy.pretrained_backbone_weights = None;
        policy.validate().expect("the six-joint policy validates");

        let mut rng = SplitMix64::new(7);
        let model = ActModel::new(&policy, &Device::Cpu, &mut rng).expect("model builds");
        model
            .save(&checkpoint.join("model.safetensors"))
            .expect("model weights save");
        std::fs::write(checkpoint.join("config.json"), policy.to_checkpoint_json())
            .expect("policy config saves");

        let stats = DatasetStats::from_entries(IndexMap::from([
            (
                "observation.state".to_owned(),
                FeatureStats::from_entries(IndexMap::from([
                    ("mean".to_owned(), vec![0.0; 6]),
                    ("std".to_owned(), vec![1.0; 6]),
                    ("min".to_owned(), vec![-1.0; 6]),
                    ("max".to_owned(), vec![1.0; 6]),
                ])),
            ),
            (
                "action".to_owned(),
                FeatureStats::from_entries(IndexMap::from([
                    ("mean".to_owned(), vec![0.0; 6]),
                    ("std".to_owned(), vec![1.0; 6]),
                    ("min".to_owned(), vec![-1.0; 6]),
                    ("max".to_owned(), vec![1.0; 6]),
                ])),
            ),
        ]));
        rerobot_train::processor::write_processor_artifacts(&checkpoint, &policy, &stats)
            .expect("processor artifacts save");
        (directory, checkpoint)
    }

    #[test]
    fn mock_so101_rollout_runs_the_real_policy_boundary_and_releases_torque() {
        let (_checkpoint_dir, checkpoint) = six_joint_policy();
        let config = RolloutConfig {
            policy_path: checkpoint,
            dataset_root: PathBuf::new(),
            steps: 1,
            start_index: 0,
            device: Some("cpu".to_owned()),
            robot_port: Some(PathBuf::from("mock")),
            calibration_path: Some(PathBuf::from("mock-calibration.json")),
            confirm: true,
            fps: 1_000_000.0,
        };
        let mut logs = Vec::new();
        let port = RespondingPort::default();
        let calibration = [rerobot_hardware::so101::JointCalibration::default(); 6];
        let mut session = InferenceSession::load_checkpoint(&config.policy_path, Some("cpu"))
            .expect("the mock policy checkpoint loads before hardware access");

        let port = run_so101_with_bus(
            &config,
            &mut |line| logs.push(line.to_owned()),
            &mut session,
            FeetechBus::new(port),
            calibration,
        )
        .expect("the mock bus completes the real SO-101 adapter");

        assert_eq!(logs.len(), 1, "one finite control tick is emitted");
        assert!(logs[0].contains("source:so101"), "{}", logs[0]);
        let torque_on = port
            .packets
            .iter()
            .filter(|packet| {
                packet.len() >= 8 && packet[4] == 0x03 && packet[5] == 40 && packet[6] == 1
            })
            .count();
        let torque_off = port
            .packets
            .iter()
            .filter(|packet| {
                packet.len() >= 8 && packet[4] == 0x03 && packet[5] == 40 && packet[6] == 0
            })
            .count();
        assert_eq!(
            torque_on, 6,
            "all joints must be enabled before the first goal"
        );
        assert_eq!(torque_off, 6, "all joints must be released after the trace");
        assert!(
            port.packets
                .iter()
                .any(|packet| packet.len() >= 6 && packet[4] == 0x82),
            "the adapter must use the sync-read observation path"
        );
        assert!(
            port.packets
                .iter()
                .any(|packet| packet.len() >= 6 && packet[4] == 0x83),
            "the adapter must use the sync-write action path"
        );
    }

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

    #[test]
    fn control_sleep_duration_keeps_hardware_rollout_at_the_requested_fps() {
        assert_eq!(
            control_sleep_duration(20.0, std::time::Duration::from_millis(10)),
            std::time::Duration::from_millis(40)
        );
    }

    #[test]
    fn control_sleep_duration_does_not_sleep_after_an_overrun() {
        assert_eq!(
            control_sleep_duration(20.0, std::time::Duration::from_millis(60)),
            std::time::Duration::ZERO
        );
    }

    #[test]
    fn precise_sleep_waits_for_a_positive_duration() {
        let start = Instant::now();
        precise_sleep(std::time::Duration::from_millis(2));
        assert!(start.elapsed() >= std::time::Duration::from_millis(2));
    }
}
