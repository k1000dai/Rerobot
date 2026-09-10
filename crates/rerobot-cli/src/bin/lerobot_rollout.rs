//! `lerobot-rollout` — local ACT or state-only SO-101 checkpoint deployment.
//!
//! The executable keeps the upstream name while the implemented boundary is
//! deliberately narrower than a full physical rollout: it supports local
//! dataset observations and a finite calibrated six-joint SO-101 follower path,
//! and refuses teleoperators, environments, visualization, video shards, and
//! interactive strategies explicitly.

fn main() -> ! {
    rerobot_cli::run("lerobot-rollout")
}
