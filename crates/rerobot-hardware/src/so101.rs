//! SO-101 follower configuration and safe joint-level commands.

use crate::feetech::{encode_sign_magnitude, open_serial, FeetechBus, FeetechError};
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
/// STS3215 return-delay-time register.
pub const RETURN_DELAY_TIME: u8 = 7;
/// STS3215 maximum-acceleration register.
pub const MAXIMUM_ACCELERATION: u8 = 85;
/// STS3215 acceleration register.
pub const ACCELERATION: u8 = 41;
/// STS3215 phase register used to select non-overflowing position feedback.
pub const PHASE: u8 = 18;
/// STS3215 maximum-torque-limit register.
pub const MAX_TORQUE_LIMIT: u8 = 16;
/// STS3215 protection-current register.
pub const PROTECTION_CURRENT: u8 = 28;
/// STS3215 overload-torque register.
pub const OVERLOAD_TORQUE: u8 = 36;
/// STS3215 operating-mode register; `0` is position mode.
pub const OPERATING_MODE: u8 = 33;
/// STS3215 position-mode P coefficient.
pub const POSITION_P: u8 = 21;
/// STS3215 position-mode D coefficient.
pub const POSITION_D: u8 = 22;
/// STS3215 position-mode I coefficient.
pub const POSITION_I: u8 = 23;
/// STS3215 minimum-position-limit register.
pub const MIN_POSITION_LIMIT: u8 = 9;
/// STS3215 maximum-position-limit register.
pub const MAX_POSITION_LIMIT: u8 = 11;
/// STS3215 sign-magnitude homing-offset register.
pub const HOMING_OFFSET: u8 = 31;
/// The sign bit used by STS3215's `Homing_Offset` field.
pub const HOMING_OFFSET_SIGN_BIT: u8 = 11;
/// STS3215 goal-position register.
pub const GOAL_POSITION: u8 = 42;
/// STS3215 present-position register.
pub const PRESENT_POSITION: u8 = 56;
/// The maximum encoder tick used by the upstream degree conversion (`resolution - 1`).
pub const POSITION_TICK_MAX: f32 = 4095.0;
/// The default SO-101 bus baudrate.
pub const SO101_BAUDRATE: u32 = 1_000_000;

/// Per-joint conversion and safety limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointCalibration {
    /// Legacy nominal center retained for source compatibility; command
    /// conversion derives the midpoint from `min_ticks`/`max_ticks`.
    pub center_ticks: i32,
    /// Direction multiplier; normally `1.0` or `-1.0`.
    pub sign: f32,
    /// Signed Feetech homing offset stored in sign-magnitude form.
    pub homing_offset: i32,
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
            homing_offset: 0,
            min_ticks: 0,
            max_ticks: 4095,
        }
    }
}

impl JointCalibration {
    /// Encode the signed homing offset as Feetech sign-magnitude bytes.
    pub fn homing_offset_bytes(self) -> Result<[u8; 2], FeetechError> {
        Ok(encode_sign_magnitude(self.homing_offset, HOMING_OFFSET_SIGN_BIT)?.to_le_bytes())
    }

    /// Encode the calibrated minimum position-limit register.
    pub fn min_ticks_bytes(self) -> Result<[u8; 2], FeetechError> {
        Self::limit_bytes(self.min_ticks, "minimum")
    }

    /// Encode the calibrated maximum position-limit register.
    pub fn max_ticks_bytes(self) -> Result<[u8; 2], FeetechError> {
        Self::limit_bytes(self.max_ticks, "maximum")
    }

    fn limit_bytes(value: i32, name: &str) -> Result<[u8; 2], FeetechError> {
        let value = u16::try_from(value).map_err(|_| {
            FeetechError::Invalid(format!("{name} position limit {value} does not fit a u16"))
        })?;
        Ok(value.to_le_bytes())
    }

    fn validate_range(self) -> Result<(), FeetechError> {
        let min = u16::try_from(self.min_ticks).map_err(|_| {
            FeetechError::Invalid(format!(
                "minimum position limit {} does not fit a u16",
                self.min_ticks
            ))
        })?;
        let max = u16::try_from(self.max_ticks).map_err(|_| {
            FeetechError::Invalid(format!(
                "maximum position limit {} does not fit a u16",
                self.max_ticks
            ))
        })?;
        if min >= max {
            return Err(FeetechError::Invalid(format!(
                "invalid calibration range {}..={}",
                self.min_ticks, self.max_ticks
            )));
        }
        Ok(())
    }

    /// Return the upstream calibration midpoint as a raw encoder tick.
    pub fn center_tick(self) -> Result<u16, FeetechError> {
        self.validate_range()?;
        let midpoint = (f64::from(self.min_ticks) + f64::from(self.max_ticks)) / 2.0;
        let midpoint = midpoint.trunc();
        if !midpoint.is_finite() || !(0.0..=f64::from(u16::MAX)).contains(&midpoint) {
            return Err(FeetechError::Invalid(format!(
                "calibration midpoint {midpoint} does not fit a u16"
            )));
        }
        Ok(midpoint as u16)
    }

    /// Convert a body-joint degree command using the calibration range and
    /// direction. This is the raw-unit inverse of LeRobot's `DEGREES` mode.
    pub fn degrees_to_ticks(self, degrees: f32) -> Result<u16, FeetechError> {
        if !degrees.is_finite() || !self.sign.is_finite() || self.sign == 0.0 {
            return Err(FeetechError::Invalid(
                "joint angle and calibration sign must be finite; sign must be non-zero".to_owned(),
            ));
        }
        self.validate_range()?;
        let midpoint = (f64::from(self.min_ticks) + f64::from(self.max_ticks)) / 2.0;
        let raw = midpoint
            + f64::from(self.sign) * f64::from(degrees) * f64::from(POSITION_TICK_MAX) / 360.0;
        let truncated = raw.trunc();
        if !raw.is_finite()
            || truncated < f64::from(self.min_ticks)
            || truncated > f64::from(self.max_ticks)
        {
            return Err(FeetechError::Invalid(format!(
                "angle {degrees}° maps to {:.1} ticks, outside {}..={}",
                raw, self.min_ticks, self.max_ticks
            )));
        }
        Ok(truncated as u16)
    }

    /// Convert the gripper's upstream `RANGE_0_100` command to a raw goal position.
    pub fn range_0_100_to_ticks(self, percent: f32) -> Result<u16, FeetechError> {
        if !percent.is_finite() || !self.sign.is_finite() || self.sign == 0.0 {
            return Err(FeetechError::Invalid(
                "gripper command and calibration sign must be finite; sign must be non-zero"
                    .to_owned(),
            ));
        }
        self.validate_range()?;

        let percent = if self.sign < 0.0 {
            100.0 - percent
        } else {
            percent
        };
        let bounded = f64::from(percent.clamp(0.0, 100.0));
        let span = f64::from(self.max_ticks - self.min_ticks);
        let raw = (bounded / 100.0) * span + f64::from(self.min_ticks);
        Ok(raw.trunc() as u16)
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

    /// Convert the five body-joint degree commands and the gripper percentage
    /// command into raw encoder positions without touching the transport.
    pub fn positions_to_ticks(&self, values: [f32; 6]) -> Result<[u16; 6], FeetechError> {
        let mut positions = [0_u16; 6];
        for (index, (value, calibration)) in values.into_iter().zip(self.calibration).enumerate() {
            positions[index] = if index == 5 {
                calibration.range_0_100_to_ticks(value)?
            } else {
                calibration.degrees_to_ticks(value)?
            };
        }
        Ok(positions)
    }

    /// Recover the underlying byte transport after all desired operations finish.
    pub fn into_inner(self) -> T {
        self.bus.into_inner()
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

    /// Apply the six calibration records to Feetech EEPROM/control-table registers.
    ///
    /// The write order matches upstream `write_calibration`: homing offset,
    /// minimum limit, then maximum limit for each motor. Torque is not enabled
    /// by this operation and must remain disabled while it runs.
    pub fn apply_calibration(&mut self) -> Result<(), FeetechError> {
        if self.torque_enabled {
            return Err(FeetechError::Invalid(
                "calibration writes require torque to be disabled".to_owned(),
            ));
        }
        for (id, calibration) in SO101_MOTOR_IDS.iter().copied().zip(self.calibration) {
            calibration.validate_range()?;
            let homing_offset = calibration.homing_offset_bytes()?;
            let min_ticks = calibration.min_ticks_bytes()?;
            let max_ticks = calibration.max_ticks_bytes()?;
            self.bus.write_register(id, HOMING_OFFSET, &homing_offset)?;
            self.bus
                .write_register(id, MIN_POSITION_LIMIT, &min_ticks)?;
            self.bus
                .write_register(id, MAX_POSITION_LIMIT, &max_ticks)?;
        }
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
            self.bus.write_register(id, RETURN_DELAY_TIME, &[0])?;
            self.bus.write_register(id, MAXIMUM_ACCELERATION, &[254])?;
            self.bus.write_register(id, ACCELERATION, &[254])?;
            let phase = self.bus.read_register(id, PHASE, 1)?[0];
            if phase & 0x10 != 0 {
                self.bus.write_register(id, PHASE, &[phase & !0x10])?;
            }
            self.bus.write_register(id, OPERATING_MODE, &[0])?;
            self.bus.write_register(id, POSITION_P, &[16])?;
            self.bus.write_register(id, POSITION_I, &[0])?;
            self.bus.write_register(id, POSITION_D, &[32])?;
            if id == SO101_MOTOR_IDS[5] {
                self.bus
                    .write_register(id, MAX_TORQUE_LIMIT, &[0xf4, 0x01])?;
                self.bus
                    .write_register(id, PROTECTION_CURRENT, &[0xfa, 0x00])?;
                self.bus.write_register(id, OVERLOAD_TORQUE, &[25])?;
            }
        }
        Ok(())
    }

    /// Move all six joints using the upstream SO-101 command convention:
    /// five body-joint values in degrees and the gripper value in `0..=100`.
    pub fn set_positions_degrees(&mut self, values: [f32; 6]) -> Result<[u16; 6], FeetechError> {
        let positions = self.positions_to_ticks(values)?;
        self.set_positions_ticks(positions)?;
        Ok(positions)
    }

    /// Move the follower to the encoder centers after torque is enabled.
    pub fn center(&mut self) -> Result<(), FeetechError> {
        let positions = self.calibration.map(JointCalibration::center_tick);
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
    fn calibrated_degree_conversion_uses_range_midpoint_and_4095_denominator() {
        let calibration = JointCalibration {
            center_ticks: 2048,
            min_ticks: 100,
            max_ticks: 2500,
            ..JointCalibration::default()
        };

        // Upstream uses (range_min + range_max) / 2 and truncates the resulting
        // float conversion; it does not use a conventional 4096/360 formula.
        assert_eq!(calibration.degrees_to_ticks(90.0).unwrap(), 2323);
    }

    #[test]
    fn six_joint_conversion_uses_range_mode_for_the_gripper() {
        let body = JointCalibration {
            min_ticks: 100,
            max_ticks: 2500,
            ..JointCalibration::default()
        };
        let gripper = JointCalibration {
            min_ticks: 500,
            max_ticks: 1500,
            ..JointCalibration::default()
        };
        let follower = So101Follower::with_calibration(
            FeetechBus::new(MockPort::default()),
            [body, body, body, body, body, gripper],
        );

        assert_eq!(
            follower
                .positions_to_ticks([0.0, 0.0, 0.0, 0.0, 0.0, 25.5])
                .unwrap(),
            [1300, 1300, 1300, 1300, 1300, 755]
        );
    }

    #[test]
    fn torque_is_required_before_a_goal_write() {
        let mut follower = So101Follower::new(FeetechBus::new(MockPort::default()));
        let error = follower.set_positions_ticks([2048; 6]).unwrap_err();
        assert!(error.to_string().contains("torque is disabled"));
    }

    #[test]
    fn gripper_range_conversion_clamps_zero_to_one_hundred() {
        let calibration = JointCalibration {
            min_ticks: 500,
            max_ticks: 1500,
            ..JointCalibration::default()
        };

        assert_eq!(calibration.range_0_100_to_ticks(-10.0).unwrap(), 500);
        assert_eq!(calibration.range_0_100_to_ticks(25.5).unwrap(), 755);
        assert_eq!(calibration.range_0_100_to_ticks(110.0).unwrap(), 1500);
    }

    #[test]
    fn center_tick_uses_calibrated_range_midpoint() {
        let calibration = JointCalibration {
            center_ticks: 2048,
            min_ticks: 100,
            max_ticks: 2500,
            ..JointCalibration::default()
        };
        assert_eq!(calibration.center_tick().unwrap(), 1300);
    }

    #[test]
    fn apply_calibration_writes_homing_offset_then_limits() {
        let mut calibration = [JointCalibration::default(); 6];
        calibration[0].homing_offset = -709;
        calibration[0].min_ticks = 43;
        calibration[0].max_ticks = 1335;
        let mut follower =
            So101Follower::with_calibration(FeetechBus::new(MockPort::default()), calibration);

        follower.apply_calibration().unwrap();
        let port = follower.into_inner();
        let expected = [
            crate::feetech::instruction_packet(
                1,
                crate::feetech::Instruction::Write,
                &[HOMING_OFFSET, 0xc5, 0x0a],
            ),
            crate::feetech::instruction_packet(
                1,
                crate::feetech::Instruction::Write,
                &[MIN_POSITION_LIMIT, 43, 0],
            ),
            crate::feetech::instruction_packet(
                1,
                crate::feetech::Instruction::Write,
                &[MAX_POSITION_LIMIT, 55, 5],
            ),
        ]
        .concat();
        assert!(port.outgoing.starts_with(&expected));
    }

    #[test]
    fn configure_position_mode_matches_upstream_setup_and_gripper_limits() {
        let mut incoming = Vec::new();
        for id in SO101_MOTOR_IDS {
            // Status response for a one-byte Phase read, with bit 4 set.
            let mut packet = vec![0xff, 0xff, id, 3, 0, 0x12];
            let checksum = !packet[2..]
                .iter()
                .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
            packet.push(checksum);
            incoming.extend(packet);
        }
        let mut follower = So101Follower::with_calibration(
            FeetechBus::new(MockPort {
                incoming: incoming.into_iter().collect(),
                outgoing: Vec::new(),
            }),
            [JointCalibration::default(); 6],
        );

        follower.configure_position_mode().unwrap();
        let port = follower.into_inner();
        let mut expected = Vec::new();
        for id in SO101_MOTOR_IDS {
            expected.extend(crate::feetech::instruction_packet(
                id,
                crate::feetech::Instruction::Write,
                &[RETURN_DELAY_TIME, 0],
            ));
            expected.extend(crate::feetech::instruction_packet(
                id,
                crate::feetech::Instruction::Write,
                &[MAXIMUM_ACCELERATION, 254],
            ));
            expected.extend(crate::feetech::instruction_packet(
                id,
                crate::feetech::Instruction::Write,
                &[ACCELERATION, 254],
            ));
            expected.extend(crate::feetech::instruction_packet(
                id,
                crate::feetech::Instruction::Read,
                &[PHASE, 1],
            ));
            expected.extend(crate::feetech::instruction_packet(
                id,
                crate::feetech::Instruction::Write,
                &[PHASE, 0x02],
            ));
            expected.extend(crate::feetech::instruction_packet(
                id,
                crate::feetech::Instruction::Write,
                &[OPERATING_MODE, 0],
            ));
            expected.extend(crate::feetech::instruction_packet(
                id,
                crate::feetech::Instruction::Write,
                &[POSITION_P, 16],
            ));
            expected.extend(crate::feetech::instruction_packet(
                id,
                crate::feetech::Instruction::Write,
                &[POSITION_I, 0],
            ));
            expected.extend(crate::feetech::instruction_packet(
                id,
                crate::feetech::Instruction::Write,
                &[POSITION_D, 32],
            ));
            if id == SO101_MOTOR_IDS[5] {
                expected.extend(crate::feetech::instruction_packet(
                    id,
                    crate::feetech::Instruction::Write,
                    &[MAX_TORQUE_LIMIT, 0xf4, 0x01],
                ));
                expected.extend(crate::feetech::instruction_packet(
                    id,
                    crate::feetech::Instruction::Write,
                    &[PROTECTION_CURRENT, 0xfa, 0x00],
                ));
                expected.extend(crate::feetech::instruction_packet(
                    id,
                    crate::feetech::Instruction::Write,
                    &[OVERLOAD_TORQUE, 25],
                ));
            }
        }
        assert_eq!(port.outgoing, expected);
    }

    #[test]
    fn configure_position_mode_skips_phase_write_when_bit_is_already_clear() {
        let incoming = SO101_MOTOR_IDS
            .into_iter()
            .flat_map(|id| {
                let mut packet = vec![0xff, 0xff, id, 3, 0, 0x00];
                packet.push(
                    !packet[2..]
                        .iter()
                        .fold(0_u8, |sum, byte| sum.wrapping_add(*byte)),
                );
                packet
            })
            .collect::<Vec<_>>();
        let mut follower = So101Follower::with_calibration(
            FeetechBus::new(MockPort {
                incoming: incoming.into_iter().collect(),
                outgoing: Vec::new(),
            }),
            [JointCalibration::default(); 6],
        );

        follower.configure_position_mode().unwrap();
        let port = follower.into_inner();
        let phase_write =
            crate::feetech::instruction_packet(1, crate::feetech::Instruction::Write, &[PHASE, 0]);
        assert!(!port
            .outgoing
            .windows(phase_write.len())
            .any(|window| window == phase_write));
    }

    #[test]
    fn calibration_register_values_encode_homing_offset_and_limits() {
        let calibration = JointCalibration {
            homing_offset: -709,
            min_ticks: 43,
            max_ticks: 1335,
            ..JointCalibration::default()
        };

        assert_eq!(calibration.homing_offset_bytes().unwrap(), [0xc5, 0x0a]);
        assert_eq!(calibration.min_ticks_bytes().unwrap(), [43, 0]);
        assert_eq!(calibration.max_ticks_bytes().unwrap(), [55, 5]);
    }

    #[test]
    fn negative_raw_position_ranges_are_refused_before_integer_cast() {
        let calibration = JointCalibration {
            min_ticks: -10,
            max_ticks: 10,
            ..JointCalibration::default()
        };
        assert!(calibration.degrees_to_ticks(0.0).is_err());
        assert!(calibration.range_0_100_to_ticks(50.0).is_err());
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
