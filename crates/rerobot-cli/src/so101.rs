//! Small hardware-gated SO-101 command surface.
//!
//! This is intentionally not presented as the full upstream teleoperation
//! strategy. It provides deterministic ping/read/center/set/release operations
//! so a real follower can be brought up and exercised without Python.

use rerobot_hardware::so101::So101Follower;
use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

/// Parsed direct-control arguments for one SO-101 follower.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// USB serial device path.
    pub port: PathBuf,
    /// One finite operation to perform.
    pub action: Action,
    /// Optional six-joint vector for [`Action::Set`]: five body-joint degrees
    /// followed by the gripper's `0..=100` command.
    pub positions_degrees: Option<[f32; 6]>,
    /// How long a position command holds torque before releasing it.
    pub hold_ms: u64,
    /// Explicit acknowledgement for operations that write to real servos.
    pub confirm: bool,
}

/// Direct SO-101 operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Ping the six expected IDs.
    Ping,
    /// Read present encoder positions.
    Read,
    /// Move all joints to their configured centers.
    Center,
    /// Move all joints to five degree values plus a gripper percentage.
    Set,
    /// Disable torque on all joints.
    Release,
}

impl fmt::Display for Action {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Ping => "ping",
            Self::Read => "read",
            Self::Center => "center",
            Self::Set => "set",
            Self::Release => "release",
        })
    }
}

/// Parse a direct-control command without opening a serial port.
pub fn parse(args: &[String]) -> Result<Config, String> {
    let mut port = None;
    let mut action = None;
    let mut positions = None;
    let mut hold_ms = 1000_u64;
    let mut confirm = false;
    let mut type_seen = false;
    for argument in args {
        let body = argument
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected positional argument {argument:?}"))?;
        let (flag, value) = body
            .split_once('=')
            .ok_or_else(|| format!("--{body} needs a value, for example --{body}=VALUE"))?;
        match flag {
            "robot.type" if value == "so101_follower" => type_seen = true,
            "robot.type" => {
                return Err(format!(
                    "--robot.type={value:?} is unsupported; only so101_follower is accepted"
                ))
            }
            "robot.port" => port = Some(PathBuf::from(value)),
            "robot.action" => {
                action = Some(match value {
                    "ping" => Action::Ping,
                    "read" => Action::Read,
                    "center" => Action::Center,
                    "set" => Action::Set,
                    "release" => Action::Release,
                    _ => {
                        return Err(
                            "--robot.action must be ping, read, center, set, or release".to_owned()
                        )
                    }
                })
            }
            "robot.positions" => positions = Some(parse_positions(value)?),
            "robot.hold_ms" => {
                hold_ms = value
                    .parse()
                    .map_err(|_| "--robot.hold_ms must be an unsigned integer".to_owned())?
            }
            "robot.confirm" => {
                confirm = value
                    .parse()
                    .map_err(|_| "--robot.confirm must be true or false".to_owned())?
            }
            other => return Err(format!("--{other} is not a SO-101 direct-control option")),
        }
    }
    if !type_seen {
        return Err("--robot.type=so101_follower is required".to_owned());
    }
    let port = port.ok_or_else(|| "--robot.port=PATH is required".to_owned())?;
    let action = action.ok_or_else(|| "--robot.action=ACTION is required".to_owned())?;
    if matches!(action, Action::Center | Action::Set | Action::Release) && !confirm {
        return Err(format!(
            "--robot.confirm=true is required before the {action} action can affect a real servo"
        ));
    }
    if action == Action::Set && positions.is_none() {
        return Err("--robot.positions=DEG,DEG,DEG,DEG,DEG,PERCENT is required for set".to_owned());
    }
    Ok(Config {
        port,
        action,
        positions_degrees: positions,
        hold_ms,
        confirm,
    })
}

/// Execute a parsed command, opening the serial port only here.
pub fn execute(config: &Config) -> Result<String, String> {
    let mut robot = So101Follower::open(
        config
            .port
            .to_str()
            .ok_or_else(|| "--robot.port must be valid UTF-8".to_owned())?,
    )
    .map_err(|error| error.to_string())?;
    match config.action {
        Action::Ping => {
            let found = robot.ping_all().map_err(|error| error.to_string())?;
            Ok(format!(
                "action=ping port={} found={found:?}",
                config.port.display()
            ))
        }
        Action::Read => {
            let positions = robot
                .read_positions_ticks()
                .map_err(|error| error.to_string())?;
            Ok(format!(
                "action=read port={} positions_ticks={positions:?}",
                config.port.display()
            ))
        }
        Action::Center => {
            robot
                .configure_position_mode()
                .map_err(|error| error.to_string())?;
            robot.enable_torque().map_err(|error| error.to_string())?;
            let result = robot.center().map_err(|error| error.to_string());
            hold_and_release(&mut robot, config.hold_ms, result)
        }
        Action::Set => {
            robot
                .configure_position_mode()
                .map_err(|error| error.to_string())?;
            robot.enable_torque().map_err(|error| error.to_string())?;
            let degrees = config
                .positions_degrees
                .expect("parse validates set positions");
            let positions = robot
                .set_positions_degrees(degrees)
                .map_err(|error| error.to_string());
            match positions {
                Ok(positions) => {
                    hold_and_release(&mut robot, config.hold_ms, Ok(()))?;
                    Ok(format!(
                        "action=set port={} positions_ticks={positions:?} hold_ms={}",
                        config.port.display(),
                        config.hold_ms
                    ))
                }
                Err(error) => {
                    let _ = robot.disable_torque();
                    Err(error)
                }
            }
        }
        Action::Release => {
            robot.disable_torque().map_err(|error| error.to_string())?;
            Ok(format!("action=release port={}", config.port.display()))
        }
    }
}

fn hold_and_release<T: std::io::Read + std::io::Write>(
    robot: &mut So101Follower<T>,
    hold_ms: u64,
    result: Result<(), String>,
) -> Result<String, String> {
    std::thread::sleep(Duration::from_millis(hold_ms));
    let release_result = robot.disable_torque().map_err(|error| error.to_string());
    result?;
    release_result.map(|()| format!("completed hold_ms={hold_ms}; torque=disabled"))
}

fn parse_positions(value: &str) -> Result<[f32; 6], String> {
    let values = value
        .split(',')
        .map(|item| {
            item.parse::<f32>()
                .map_err(|_| format!("position {item:?} is not a finite decimal value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let positions: [f32; 6] = values.try_into().map_err(|_| {
        "--robot.positions needs five degrees and one gripper percentage".to_owned()
    })?;
    if positions.iter().any(|position| !position.is_finite()) {
        return Err("--robot.positions must contain only finite values".to_owned());
    }
    Ok(positions)
}

/// Standalone help for the direct-control surface.
pub fn help() -> &'static str {
    "SO-101 direct follower control (hardware-gated)\n\
     Usage:\n\
     lerobot-teleoperate --robot.type=so101_follower --robot.port=PORT --robot.action=read|ping\n\
     lerobot-teleoperate --robot.type=so101_follower --robot.port=PORT --robot.action=center --robot.confirm=true\n\
     lerobot-teleoperate --robot.type=so101_follower --robot.port=PORT --robot.action=set --robot.positions=DEG,DEG,DEG,DEG,DEG,PERCENT --robot.confirm=true\n\
     lerobot-teleoperate --robot.type=so101_follower --robot.port=PORT --robot.action=release --robot.confirm=true\n\
     Position actions hold torque for --robot.hold_ms=1000 (default), then release it.\n\
     This is a deterministic actuator smoke/position path, not the full upstream leader-follower teleoperation strategy."
}

#[cfg(test)]
mod tests {
    use super::{parse, Action};

    fn base(action: &str) -> Vec<String> {
        vec![
            "--robot.type=so101_follower".to_owned(),
            "--robot.port=/dev/mock".to_owned(),
            format!("--robot.action={action}"),
        ]
    }

    #[test]
    fn read_is_parseable_without_confirming_a_write() {
        let config = parse(&base("read")).unwrap();
        assert_eq!(config.action, Action::Read);
        assert!(!config.confirm);
    }

    #[test]
    fn position_actions_require_an_explicit_confirmation() {
        let error = parse(&base("center")).unwrap_err();
        assert!(error.contains("confirm=true"));
    }

    #[test]
    fn set_requires_exactly_six_values() {
        let mut args = base("set");
        args.push("--robot.confirm=true".to_owned());
        args.push("--robot.positions=0,1,2".to_owned());
        assert!(parse(&args).unwrap_err().contains("five degrees"));
    }

    #[test]
    fn set_accepts_body_degrees_followed_by_gripper_percentage() {
        let mut args = base("set");
        args.push("--robot.confirm=true".to_owned());
        args.push("--robot.positions=0,1,2,3,4,25.5".to_owned());
        let config = parse(&args).unwrap();
        assert_eq!(
            config.positions_degrees,
            Some([0.0, 1.0, 2.0, 3.0, 4.0, 25.5])
        );
    }
}
