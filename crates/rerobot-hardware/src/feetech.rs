//! Feetech STS/SMS protocol 0 transport.
//!
//! The packet codec and bus are generic over `Read + Write`, so every protocol
//! operation can be tested against a byte-buffer transport. The serial adapter
//! is only the final boundary that opens `/dev/cu.*` or `/dev/ttyUSB*`.

use std::fmt;
use std::io::{self, Read, Write};
use std::time::Duration;

/// Feetech protocol-0 instruction bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Instruction {
    /// Request the servo status packet.
    Ping = 0x01,
    /// Read a contiguous control-table range.
    Read = 0x02,
    /// Write a contiguous control-table range.
    Write = 0x03,
    /// Write a value without applying it until ACTION (not used by SO-101).
    RegWrite = 0x04,
    /// Apply previously registered writes.
    Action = 0x05,
    /// Reset EEPROM values (never issued by the runtime).
    FactoryReset = 0x06,
    /// Write one range to multiple servo IDs.
    SyncWrite = 0x83,
    /// Read one range from multiple servo IDs.
    SyncRead = 0x82,
}

/// A parsed status response from a servo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusPacket {
    /// Servo ID in the response.
    pub id: u8,
    /// Protocol error bitfield returned by the servo.
    pub error: u8,
    /// Response parameters after the error byte.
    pub parameters: Vec<u8>,
}

/// Errors from packet validation, transport I/O, or a servo status response.
#[derive(Debug)]
pub enum FeetechError {
    /// Underlying serial or mock transport error.
    Io(io::Error),
    /// The serial crate failed while opening/configuring a port.
    Transport(String),
    /// A packet was malformed or did not match the request.
    Protocol(String),
    /// The servo returned a non-zero status error field.
    Servo { id: u8, error: u8 },
    /// A caller supplied an invalid address, length, ID, or position.
    Invalid(String),
}

impl fmt::Display for FeetechError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "serial I/O failed: {error}"),
            Self::Transport(error) => write!(formatter, "serial transport failed: {error}"),
            Self::Protocol(error) => write!(formatter, "Feetech protocol error: {error}"),
            Self::Servo { id, error } => {
                write!(
                    formatter,
                    "Feetech servo {id} returned protocol error 0x{error:02x}"
                )
            }
            Self::Invalid(error) => write!(formatter, "invalid Feetech request: {error}"),
        }
    }
}

impl std::error::Error for FeetechError {}

impl From<io::Error> for FeetechError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// The default SO-101 USB bus configuration.
pub const DEFAULT_BAUDRATE: u32 = 1_000_000;
/// Feetech's protocol-0 packet header.
pub const HEADER: [u8; 2] = [0xff, 0xff];

/// Encode an integer using the sign-magnitude fields used by Feetech registers.
///
/// The sign bit is separate from the magnitude; this is not two's-complement
/// encoding. Feetech uses bit 11 for `Homing_Offset` and bit 15 for several
/// signed position/velocity fields.
pub fn encode_sign_magnitude(value: i32, sign_bit: u8) -> Result<u16, FeetechError> {
    if sign_bit >= 16 {
        return Err(FeetechError::Invalid(format!(
            "sign bit {sign_bit} does not fit in a 16-bit register"
        )));
    }
    let max_magnitude = (1_i64 << sign_bit) - 1;
    let magnitude = i64::from(value).abs();
    if magnitude > max_magnitude {
        return Err(FeetechError::Invalid(format!(
            "magnitude {magnitude} exceeds {max_magnitude} for sign bit {sign_bit}"
        )));
    }
    let sign = if value < 0 { 1_i64 << sign_bit } else { 0 };
    Ok((sign | magnitude) as u16)
}

/// Decode a Feetech sign-magnitude register value.
pub fn decode_sign_magnitude(encoded: u16, sign_bit: u8) -> Result<i32, FeetechError> {
    if sign_bit >= 16 {
        return Err(FeetechError::Invalid(format!(
            "sign bit {sign_bit} does not fit in a 16-bit register"
        )));
    }
    let magnitude = i32::from(encoded & ((1_u16 << sign_bit) - 1));
    if encoded & (1_u16 << sign_bit) != 0 {
        Ok(-magnitude)
    } else {
        Ok(magnitude)
    }
}

/// Build a protocol-0 instruction packet.
pub fn instruction_packet(id: u8, instruction: Instruction, parameters: &[u8]) -> Vec<u8> {
    let length = u8::try_from(parameters.len() + 2).expect("Feetech packet length exceeds u8");
    let mut packet = Vec::with_capacity(parameters.len() + 6);
    packet.extend_from_slice(&HEADER);
    packet.push(id);
    packet.push(length);
    packet.push(instruction as u8);
    packet.extend_from_slice(parameters);
    packet.push(checksum(&packet[2..]));
    packet
}

/// Parse a complete protocol-0 status packet.
pub fn parse_status_packet(packet: &[u8]) -> Result<StatusPacket, FeetechError> {
    if packet.len() < 6 {
        return Err(FeetechError::Protocol(format!(
            "status packet has {} bytes; at least 6 are required",
            packet.len()
        )));
    }
    if packet[..2] != HEADER {
        return Err(FeetechError::Protocol(
            "status packet header is not 0xffff".to_owned(),
        ));
    }
    let length = usize::from(packet[3]);
    let expected = 4 + length;
    if expected != packet.len() {
        return Err(FeetechError::Protocol(format!(
            "status length byte says {length} bytes after the header, but packet has {}",
            packet.len() - 4
        )));
    }
    if checksum(&packet[2..packet.len() - 1]) != packet[packet.len() - 1] {
        return Err(FeetechError::Protocol(
            "status checksum mismatch".to_owned(),
        ));
    }
    let id = packet[2];
    let error = packet[4];
    let parameters = packet[5..packet.len() - 1].to_vec();
    if error != 0 {
        return Err(FeetechError::Servo { id, error });
    }
    Ok(StatusPacket {
        id,
        error,
        parameters,
    })
}

fn checksum(bytes: &[u8]) -> u8 {
    !bytes.iter().fold(0_u8, |sum, byte| sum.wrapping_add(*byte))
}

/// A Feetech bus over any byte-oriented transport.
pub struct FeetechBus<T> {
    transport: T,
}

impl<T> FeetechBus<T> {
    /// Wrap an already configured transport.
    pub fn new(transport: T) -> Self {
        Self { transport }
    }

    /// Recover the underlying transport.
    pub fn into_inner(self) -> T {
        self.transport
    }
}

impl<T: Read + Write> FeetechBus<T> {
    /// Send a packet without waiting for a status response. Protocol-0 write
    /// instructions normally have return level 1, meaning READ/PING only.
    pub fn write_register(&mut self, id: u8, address: u8, data: &[u8]) -> Result<(), FeetechError> {
        validate_id(id)?;
        validate_range(address, data.len())?;
        let mut parameters = Vec::with_capacity(data.len() + 1);
        parameters.push(address);
        parameters.extend_from_slice(data);
        self.send(&instruction_packet(id, Instruction::Write, &parameters))
    }

    /// Read a contiguous control-table range and validate the returned length.
    pub fn read_register(
        &mut self,
        id: u8,
        address: u8,
        length: u8,
    ) -> Result<Vec<u8>, FeetechError> {
        validate_id(id)?;
        validate_range(address, usize::from(length))?;
        self.send(&instruction_packet(
            id,
            Instruction::Read,
            &[address, length],
        ))?;
        let status = self.read_status()?;
        if status.id != id {
            return Err(FeetechError::Protocol(format!(
                "read for servo {id} received status from servo {}",
                status.id
            )));
        }
        if status.parameters.len() != usize::from(length) {
            return Err(FeetechError::Protocol(format!(
                "read for servo {id} requested {length} bytes, received {}",
                status.parameters.len()
            )));
        }
        Ok(status.parameters)
    }

    /// Ping one servo and return its status packet.
    pub fn ping(&mut self, id: u8) -> Result<StatusPacket, FeetechError> {
        validate_id(id)?;
        self.send(&instruction_packet(id, Instruction::Ping, &[]))?;
        let status = self.read_status()?;
        if status.id != id {
            return Err(FeetechError::Protocol(format!(
                "ping for servo {id} received status from servo {}",
                status.id
            )));
        }
        Ok(status)
    }

    /// Write one register range to multiple servo IDs with one bus packet.
    pub fn sync_write(
        &mut self,
        address: u8,
        data_length: u8,
        values: &[(u8, &[u8])],
    ) -> Result<(), FeetechError> {
        if values.is_empty() {
            return Err(FeetechError::Invalid(
                "sync write needs at least one servo".to_owned(),
            ));
        }
        validate_range(address, usize::from(data_length))?;
        let mut parameters = vec![address, data_length];
        for (id, data) in values {
            validate_id(*id)?;
            if data.len() != usize::from(data_length) {
                return Err(FeetechError::Invalid(format!(
                    "sync write for servo {id} has {} bytes; expected {data_length}",
                    data.len()
                )));
            }
            parameters.push(*id);
            parameters.extend_from_slice(data);
        }
        self.send(&instruction_packet(
            0xfe,
            Instruction::SyncWrite,
            &parameters,
        ))
    }

    fn send(&mut self, packet: &[u8]) -> Result<(), FeetechError> {
        self.transport.write_all(packet)?;
        self.transport.flush()?;
        Ok(())
    }

    fn read_status(&mut self) -> Result<StatusPacket, FeetechError> {
        let mut prefix = [0_u8; 4];
        self.transport.read_exact(&mut prefix)?;
        if prefix[..2] != HEADER {
            return Err(FeetechError::Protocol(
                "status packet header is not 0xffff".to_owned(),
            ));
        }
        let length = usize::from(prefix[3]);
        if !(2..=252).contains(&length) {
            return Err(FeetechError::Protocol(format!(
                "status packet length {length} is outside the protocol range"
            )));
        }
        let mut packet = prefix.to_vec();
        packet.resize(4 + length, 0);
        self.transport.read_exact(&mut packet[4..])?;
        parse_status_packet(&packet)
    }
}

fn validate_id(id: u8) -> Result<(), FeetechError> {
    if id == 0 || id == 0xfe {
        return Err(FeetechError::Invalid(format!(
            "servo id {id} is reserved for this operation"
        )));
    }
    Ok(())
}

fn validate_range(address: u8, length: usize) -> Result<(), FeetechError> {
    if length == 0 || usize::from(address) + length > 256 {
        return Err(FeetechError::Invalid(format!(
            "control-table range address {address} length {length} does not fit in one byte"
        )));
    }
    Ok(())
}

/// Open a SO-101-compatible USB serial port at 1 Mbps.
pub fn open_serial(
    path: &str,
) -> Result<FeetechBus<Box<dyn serialport::SerialPort>>, FeetechError> {
    let port = serialport::new(path, DEFAULT_BAUDRATE)
        .timeout(Duration::from_millis(250))
        .open()
        .map_err(|error| FeetechError::Transport(error.to_string()))?;
    Ok(FeetechBus::new(port))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[derive(Default)]
    struct MockPort {
        incoming: VecDeque<u8>,
        outgoing: Vec<u8>,
    }

    impl MockPort {
        fn with_incoming(bytes: &[u8]) -> Self {
            Self {
                incoming: bytes.iter().copied().collect(),
                outgoing: Vec::new(),
            }
        }
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
    fn sign_magnitude_codec_matches_feetech_calibration_encoding() {
        assert_eq!(encode_sign_magnitude(-709, 11).unwrap(), 0x0ac5);
        assert_eq!(decode_sign_magnitude(0x0ac5, 11).unwrap(), -709);
        assert_eq!(encode_sign_magnitude(2047, 11).unwrap(), 2047);
        assert_eq!(
            encode_sign_magnitude(-2048, 11).unwrap_err().to_string(),
            "invalid Feetech request: magnitude 2048 exceeds 2047 for sign bit 11"
        );
    }

    #[test]
    fn write_packet_matches_protocol_zero_checksum_and_length() {
        assert_eq!(
            instruction_packet(1, Instruction::Write, &[40, 1]),
            vec![0xff, 0xff, 1, 4, 3, 40, 1, 206]
        );
    }

    #[test]
    fn read_register_sends_read_and_parses_little_endian_bytes() {
        let response = [0xff, 0xff, 1, 4, 0, 0x34, 0x12, 0xb4];
        let mut bus = FeetechBus::new(MockPort::with_incoming(&response));
        assert_eq!(bus.read_register(1, 56, 2).unwrap(), vec![0x34, 0x12]);
        let mock = bus.into_inner();
        assert_eq!(mock.outgoing, vec![0xff, 0xff, 1, 4, 2, 56, 2, 190]);
    }

    #[test]
    fn sync_write_contains_each_id_and_fixed_width_payload() {
        let mut bus = FeetechBus::new(MockPort::default());
        bus.sync_write(42, 2, &[(1, &[0x00, 0x08]), (2, &[0xff, 0x0f])])
            .unwrap();
        let mock = bus.into_inner();
        assert_eq!(
            mock.outgoing,
            vec![0xff, 0xff, 0xfe, 10, 0x83, 42, 2, 1, 0, 8, 2, 0xff, 0x0f, 0x2f]
        );
    }

    #[test]
    fn a_bad_status_checksum_is_rejected() {
        let error = parse_status_packet(&[0xff, 0xff, 1, 2, 0, 0]).unwrap_err();
        assert!(error.to_string().contains("checksum"));
    }
}
