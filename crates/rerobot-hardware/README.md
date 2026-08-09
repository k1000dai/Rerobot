# rerobot-hardware

Pure-Rust Feetech servo transport and SO-101 follower hardware control for
ReRobot. The crate implements Feetech protocol 0 packet encoding, register
access, sync writes, STS3215 unit conversion, and a six-joint SO-101 follower
surface without a Python or libtorch runtime.

The transport is generic over `Read + Write`, so packet validation and control
logic can be tested without a connected servo bus. The serial-port adapter is
provided for real hardware use.

This is an independent Rust implementation compatible with the public
Feetech/LeRobot control surface; it is not an official Feetech or Hugging Face
product.
