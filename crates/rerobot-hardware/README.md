# rerobot-hardware

Pure-Rust Feetech servo transport and SO-101 follower hardware control for
ReRobot. The crate implements Feetech protocol 0 packet encoding, register
access, sync writes, STS3215 unit conversion, and a six-joint SO-101 follower
surface without a Python or libtorch runtime.

The transport is generic over `Read + Write`, so packet validation and control
logic can be tested without a connected servo bus. The serial-port adapter is
provided for real hardware use.

The SO-101 conversion path uses calibrated `min_ticks`/`max_ticks` ranges,
the upstream `(min + max) / 2` midpoint and `4095` denominator for body-joint
degrees, and the distinct `0..=100` gripper convention. Calibration JSON uses
upstream's six named motor records and is validated before the serial port is
opened. Position reads use one protocol-0 sync-read request and surface a
malformed status packet immediately, matching the current upstream follower's
`get_observation()` path. Calibration writes
encode `Homing_Offset` as Feetech sign-magnitude. The transport and mock tests
are hardware-independent; no physical servo was connected for this release.

This is an independent Rust implementation; it is not an official Feetech or
Hugging Face product.
