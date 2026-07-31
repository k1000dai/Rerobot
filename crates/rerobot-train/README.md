# rerobot-train

The pure-Rust ACT training slice of [Rerobot][repo], a behaviour-compatible port
of [Hugging Face LeRobot][upstream].

This crate is what makes `lerobot-train` run for one narrow, honest case: a
**state-only** LeRobot v3.0 dataset on local disk, the **ACT** policy on **CPU**,
and the upstream checkpoint layout on the way out. There is no PyTorch, no Python
sidecar and no FFI: the tensor work is [candle], the parquet work is [arrow], and
the rest is this crate.

[repo]: https://github.com/k1000dai/Rerobot
[upstream]: https://github.com/huggingface/lerobot
[candle]: https://github.com/huggingface/candle
[arrow]: https://arrow.apache.org/rust/

## What is in scope

| Piece | Upstream source |
| --- | --- |
| `data` — `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, `meta/episodes/`, `data/` | `lerobot/datasets/{lerobot_dataset,dataset_reader,io_utils}.py` |
| `model` — the ACT transformer, VAE encoder, sinusoidal embeddings, L1 + KL loss | `lerobot/policies/act/modeling_act.py` |
| `optim` — AdamW and `clip_grad_norm_` | `torch.optim.AdamW`, `torch.nn.utils.clip_grad_norm_` |
| `checkpoint` — `checkpoints/<step>/{pretrained_model,training_state}/` | `lerobot/common/train_utils.py` |
| `run` — the step loop | `lerobot/scripts/lerobot_train.py` |

## What is deliberately out of scope

Images and video, the Hub, `accelerate`, distributed training, mixed precision,
LR schedulers, PEFT, environment evaluation, `wandb`, and every policy other than
ACT. Each of those is *refused with an error*, never silently ignored — see
`docs/compatibility.md` for the exact boundary and
`rerobot_core::random` for why Rerobot's random numbers are its own rather than
PyTorch's.

## Example

```no_run
use rerobot_train::config::TrainConfig;
use std::path::PathBuf;

let mut config = TrainConfig::new(
    "rerobot/state_only_slice".to_owned(),
    PathBuf::from("crates/rerobot-train/tests/fixtures/state_only"),
    PathBuf::from("outputs/train/demo"),
);
config.steps = 1;
config.batch_size = 2;
config.policy.chunk_size = rerobot_core::BigInt::from(2);
config.policy.n_action_steps = rerobot_core::BigInt::from(2);

let outcome = rerobot_train::run::train(&config, &mut |line| println!("{line}"))?;
assert_eq!(outcome.steps.len(), 1);
# Ok::<(), rerobot_train::error::TrainError>(())
```
