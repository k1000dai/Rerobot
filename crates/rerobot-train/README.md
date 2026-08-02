# rerobot-train

The pure-Rust ACT training slice of [Rerobot][repo], a behaviour-compatible port
of [Hugging Face LeRobot][upstream].

This crate is what makes `lerobot-train` run for one narrow, honest case: a
LeRobot v3.0 dataset on local disk with state/action columns, or an ACT policy
fed camera tensors in memory, on **CPU** or **CUDA**, with the upstream checkpoint
layout on the way out. There is no PyTorch, no Python sidecar and no FFI: the
tensor work is [candle], the parquet work is [arrow], and the rest is this crate.

[repo]: https://github.com/k1000dai/Rerobot
[upstream]: https://github.com/huggingface/lerobot
[candle]: https://github.com/huggingface/candle
[arrow]: https://arrow.apache.org/rust/

## What is in scope

| Piece | Upstream source |
| --- | --- |
| `data` — `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, `meta/episodes/`, `data/` | `lerobot/datasets/{lerobot_dataset,dataset_reader,io_utils}.py` |
| `model` — the ACT transformer, VAE encoder, ResNet18/34 camera backbone, 1-D/2-D sinusoidal embeddings, L1 + KL loss | `lerobot/policies/act/modeling_act.py` |
| `optim` — AdamW and `clip_grad_norm_` | `torch.optim.AdamW`, `torch.nn.utils.clip_grad_norm_` |
| `checkpoint` — `checkpoints/<step>/{pretrained_model,training_state}/` | `lerobot/common/train_utils.py` |
| `run` — the step loop | `lerobot/scripts/lerobot_train.py` |

## Devices

`--policy.device` accepts `cpu` by default, and `cuda` (or `cuda:0`, the
spelling torch also accepts) when this crate is built with its **`cuda`**
feature:

```sh
cargo build --release -p rerobot-cli --features cuda
```

That feature switches candle onto its CUDA backend, so it needs the NVIDIA CUDA
toolkit at build time — `candle-kernels` compiles PTX from source. It is off by
default and stays off in CI, because none of the hosted runners has a toolkit.

Two properties hold on purpose:

* **No fallback.** A run that asks for `cuda` and cannot have it — the feature
  was not compiled, or the driver/GPU is missing — stops with a non-zero exit and
  says which of the two it was. Upstream downgrades to the CPU with a warning;
  this slice does not, because a run that reports success from a device nobody
  chose is indistinguishable from one that worked.
* **One device.** [`device::resolve`](crate::device::resolve) is called once in
  `TrainSession::new`, and the batch, the normalized copy of it, the model
  parameters, the latent and dropout draws, the AdamW moments and the optimizer
  state all come from that one `Device`. Only the safetensors writer leaves it,
  because serialization reads the bytes back to host memory.

**Validation status.** The CUDA path is implemented and covered by
`tests/device_smoke.rs`, which is compiled only under `--features cuda`. It has
not yet been executed on real NVIDIA hardware from this repository, so treat GPU
support as *available but not hardware-validated*. The CPU path is validated by
CI on Linux, macOS and Windows, and element by element against upstream PyTorch.

## What is deliberately out of scope

On-disk image/video decoding, the Hub, `accelerate`, distributed training, mixed
precision, LR schedulers, PEFT, environment evaluation, `wandb`, and every policy
other than ACT. On-disk camera features are refused rather than silently dropped;
ACT camera inputs supplied as `f32` Candle tensors through `Batch::with_images` and
`TrainSession::step_on` are supported. See `docs/compatibility.md` for the exact
boundary and
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
