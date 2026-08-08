//! `lerobot-rollout` — hardware-independent local ACT checkpoint deployment.
//!
//! The executable keeps the upstream name while the implemented boundary is
//! deliberately narrower than a physical rollout: it reads local observations
//! and emits finite action traces, and refuses robot, teleoperator, environment,
//! visualization, and video-shard options explicitly.

fn main() -> ! {
    rerobot_cli::run("lerobot-rollout")
}
