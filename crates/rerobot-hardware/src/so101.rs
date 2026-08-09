//! SO-101 follower configuration and safe joint-level commands.

use crate::feetech::{open_serial, FeetechBus, FeetechError};
use std::io::{Read, Write};

/// SO-101's six follower servos in the control order used by LeRobot actions.
pub const SO101_MOTOR_IDS: [u8; 6] = [1, 2, 3, 4, 5, 6];
/// Human-readable SO-101 joint names in action-vector order.
pub const SO101_JOINT_NAMES: [&str; 6] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
];

/// STS3215 torque-enable register.
pub const TORQUE_ENABLE: u8 = 40;
/// STS3215 lock register used by the Feetech SDK around EEPROM/config writes.
pub const LOCK: u8 = 55;
/// STS3215 operating-mode register; `0` is position mode.
pub const OPERATING_MODE: u8 = 33;
/// STS3215 position-mode P coefficient.
pub const POSITION_P: u8 = 21;
/// STS3215 position-mode D coefficient.
pub const POSITION_D: u8 = 22;
/// STS3215 position-mode I coefficient.
pub const POSITION_I: u8 = 23;
/// STS3215 goal-position register.
pub const GOAL_POSITION: u8 = 42;
/// STS3215 present-position register.
pub const PRESENT_POSITION: u8 = 56;
/// The encoder range used by STS3215 in one mechanical revolution.
pub const TICKS_PER_REVOLUTION: f32 = 4096.0;
/// The default SO-101 bus baudrate.
pub const SO101_BAUDRATE: u32 = 1_000_000;

/// Per-joint conversion and safety limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointCalibration {
    /// Encoder tick corresponding to the calibrated zero angle.
    pub center_ticks: i32,
    /// Direction multiplier; normally `1.0` or `-1.0`.
    pub sign: f32,
    /// Minimum permitted goal position in raw encoder ticks.
    pub min_ticks: i32,
    /// Maximum permitted goal position in raw encoder ticks.
    pub max_ticks: i32,
}

impl Default for JointCalibration {
    fn default() -> Self {
        Self {
            center_ticks: 2048,
            sign: 1.0,
            min_ticks: 0,
            max_ticks: 4095,
        }
    }
}

impl JointCalibration {
    /// Convert a joint angle in degrees to a checked raw goal position.
    pub fn degrees_to_ticks(self, degrees: f32) -> Result<u16, FeetechError> {
        if !degrees.is_finite() || !self.sign.is_finite() || self.sign == 0.0 {
            return Err(FeetechError::Invalid(
                "joint angle and calibration sign must be finite; sign must be non-zero".to_owned(),
            ));
        }
        let raw = f64::from(self.center_ticks)
            + f64::from(self.sign) * f64::from(degrees) * f64::from(TICKS_PER_REVOLUTION) / 360.0;
        if !raw.is_finite()
            || raw.round() < f64::from(self.min_ticks)
            || raw.round() > f64::from(self.max_ticks)
        {
            return Err(FeetechError::Invalid(format!(
                "angle {degrees}° maps to {:.1} ticks, outside {}..={}",
                raw, self.min_ticks, self.max_ticks
            )));
        }
        Ok(raw.round() as u16)
    }
}

/// A connected SO-101 follower.
pub struct So101Follower<T> {
    bus: FeetechBus<T>,
    calibration: [JointCalibration; 6],
    torque_enabled: bool,
}

impl<T> So101Follower<T> {
    /// Construct a follower with the documented default encoder centers.
    pub fn new(bus: FeetechBus<T>) -> Self {
        Self {
            bus,
            calibration: [JointCalibration::default(); 6],
            torque_enabled: false,
        }
    }

    /// Construct a follower with explicit per-joint calibration.
    pub fn with_calibration(bus: FeetechBus<T>, calibration: [JointCalibration; 6]) -> Self {
        Self {
            bus,
            calibration,
            torque_enabled: false,
        }
    }

    /// Replace calibration before enabling torque.
    pub fn set_calibration(
        &mut self,
        calibration: [JointCalibration; 6],
    ) -> Result<(), FeetechError> {
        if self.torque_enabled {
            return Err(FeetechError::Invalid(
                "calibration cannot change while torque is enabled".to_owned(),
            ));
        }
        self.calibration = calibration;
        Ok(())
    }

    /// Whether this process has explicitly enabled torque on all six joints.
    pub fn torque_enabled(&self) -> bool {
        self.torque_enabled
    }

    /// Access the underlying bus, for diagnostics and advanced register reads.
    pub fn bus_mut(&mut self) -> &mut FeetechBus<T> {
        &mut self.bus
    }
}

impl<T: Read + Write> So101Follower<T> {
    /// Ping all six expected servo IDs. No torque or EEPROM writes occur.
    pub fn ping_all(&mut self) -> Result<[bool; 6], FeetechError> {
        let mut found = [false; 6];
        for (index, id) in SO101_MOTOR_IDS.iter().copied().enumerate() {
            found[index] = self.bus.ping(id).is_ok();
        }
        Ok(found)
    }

    /// Enable torque on all SO-101 joints.
    pub fn enable_torque(&mut self) -> Result<(), FeetechError> {
        for id in SO101_MOTOR_IDS {
            self.bus.write_register(id, TORQUE_ENABLE, &[1])?;
            self.bus.write_register(id, LOCK, &[1])?;
        }
        self.torque_enabled = true;
        Ok(())
    }

    /// Disable torque on all SO-101 joints and mark the local state released.
    pub fn disable_torque(&mut self) -> Result<(), FeetechError> {
        for id in SO101_MOTOR_IDS {
            self.bus.write_register(id, TORQUE_ENABLE, &[0])?;
            self.bus.write_register(id, LOCK, &[0])?;
        }
        self.torque_enabled = false;
        Ok(())
    }

    /// Read all six present positions in raw encoder ticks.
    pub fn read_positions_ticks(&mut self) -> Result<[u16; 6], FeetechError> {
        let mut positions = [0_u16; 6];
        for (index, id) in SO101_MOTOR_IDS.iter().copied().enumerate() {
            let bytes = self.bus.read_register(id, PRESENT_POSITION, 2)?;
            positions[index] = u16::from_le_bytes([bytes[0], bytes[1]]);
        }
        Ok(positions)
    }

    /// Move all six joints to raw encoder positions. Torque must be enabled by
    /// an explicit call first; values are rejected rather than silently clamped.
    pub fn set_positions_ticks(&mut self, positions: [u16; 6]) -> Result<(), FeetechError> {
        if !self.torque_enabled {
            return Err(FeetechError::Invalid(
                "torque is disabled; call enable_torque() explicitly before moving".to_owned(),
            ));
        }
        let payloads = positions.map(|position| position.to_le_bytes());
        let values = SO101_MOTOR_IDS
            .iter()
            .copied()
            .zip(payloads.iter())
            .map(|(id, bytes)| (id, bytes.as_slice()))
            .collect::<Vec<_>>();
        self.bus.sync_write(GOAL_POSITION, 2, &values)
    }

    /// Configure the Feetech position mode and PID defaults used by LeRobot.
    ///
    /// This must run while torque is disabled. It does not enable torque itself;
    /// callers should explicitly call [`Self::enable_torque`] afterwards.
    pub fn configure_position_mode(&mut self) -> Result<(), FeetechError> {
        if self.torque_enabled {
            return Err(FeetechError::Invalid(
                "position-mode configuration requires torque to be disabled".to_owned(),
            ));
        }
        for id in SO101_MOTOR_IDS {
            self.bus.write_register(id, LOCK, &[0])?;
            self.bus.write_register(id, OPERATING_MODE, &[0])?;
            self.bus.write_register(id, POSITION_P, &[16])?;
            self.bus.write_register(id, POSITION_I, &[0])?;
            self.bus.write_register(id, POSITION_D, &[32])?;
        }
        Ok(())
    }

    /// Move all six joints to angles in degrees using the configured calibration.
    pub fn set_positions_degrees(&mut self, degrees: [f32; 6]) -> Result<[u16; 6], FeetechError> {
        let ticks = degrees
            .into_iter()
            .zip(self.calibration)
            .map(|(degree, calibration)| calibration.degrees_to_ticks(degree))
            .collect::<Result<Vec<_>, _>>()?;
        let positions: [u16; 6] = ticks.try_into().map_err(|_| {
            FeetechError::Invalid("SO-101 requires exactly six joint positions".to_owned())
        })?;
        self.set_positions_ticks(positions)?;
        Ok(positions)
    }

    /// Move the follower to the encoder centers after torque is enabled.
    pub fn center(&mut self) -> Result<(), FeetechError> {
        let positions = self.calibration.map(|joint| {
            joint
                .center_ticks
                .try_into()
                .map_err(|_| FeetechError::Invalid("center tick does not fit u16".to_owned()))
        });
        let positions: Result<[u16; 6], FeetechError> = positions
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| {
                FeetechError::Invalid("SO-101 requires exactly six center positions".to_owned())
            });
        self.set_positions_ticks(positions?)
    }
}

impl So101Follower<Box<dyn serialport::SerialPort>> {
    /// Open a SO-101 on a USB serial device at 1 Mbps.
    pub fn open(path: &str) -> Result<Self, FeetechError> {
        Ok(Self::new(open_serial(path)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::io;

    #[derive(Default)]
    struct MockPort {
        incoming: VecDeque<u8>,
        outgoing: Vec<u8>,
    }

    impl Read for MockPort {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            if self.incoming.is_empty() {
                return Ok(0);
            }
            let count = output.len().min(self.incoming.len());
            for byte in &mut output[..count] {
                *byte = self.incoming.pop_front().unwrap();
            }
            Ok(count)
        }
    }

    impl Write for MockPort {
        fn write(&mut self, input: &[u8]) -> io::Result<usize> {
            self.outgoing.extend_from_slice(input);
            Ok(input.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn default_angle_conversion_uses_4096_ticks_per_revolution() {
        assert_eq!(
            JointCalibration::default().degrees_to_ticks(90.0).unwrap(),
            3072
        );
    }

    #[test]
    fn torque_is_required_before_a_goal_write() {
        let mut follower = So101Follower::new(FeetechBus::new(MockPort::default()));
        let error = follower.set_positions_ticks([2048; 6]).unwrap_err();
        assert!(error.to_string().contains("torque is disabled"));
    }

    #[test]
    fn calibration_rejects_motion_outside_the_declared_range() {
        let calibration = JointCalibration {
            min_ticks: 1800,
            max_ticks: 2200,
            ..JointCalibration::default()
        };
        assert!(calibration.degrees_to_ticks(90.0).is_err());
    }
}
