# rerobot-train

The pure-Rust ACT training slice of [Rerobot][repo], a behaviour-compatible port
of [Hugging Face LeRobot][upstream].

This crate is what makes `lerobot-train` run for one narrow, honest case: a
LeRobot v3.0 dataset on local disk or a native Hub snapshot with state/action columns
and embedded PNG/JPEG camera columns, or an ACT policy fed camera tensors in memory,
on **CPU** or **CUDA**, with the upstream checkpoint layout on the way out. There is
no PyTorch, no Python sidecar and no FFI: the tensor work is [candle], the parquet
work is [arrow], and the rest is this crate. `meta/stats.json` camera mean/std entries
are retained per feature, so `dataset_use_imagenet_stats=false` trains and deploys
with the dataset's own per-camera normalization; the default `true` path writes
ImageNet statistics.

[repo]: https://github.com/k1000dai/Rerobot
[upstream]: https://github.com/huggingface/lerobot
[candle]: https://github.com/huggingface/candle
[arrow]: https://arrow.apache.org/rust/

## What is in scope

| Piece | Upstream source |
| --- | --- |
| `data` — `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, `meta/episodes/`, `data/`, and a staged native Hub snapshot | `lerobot/datasets/{lerobot_dataset,dataset_reader,io_utils}.py` |
| `model` — the ACT transformer, VAE encoder, ResNet18/34 camera backbone, 1-D/2-D sinusoidal embeddings, L1 + KL loss | `lerobot/policies/act/modeling_act.py` |
| `optim` — AdamW and `clip_grad_norm_` | `torch.optim.AdamW`, `torch.nn.utils.clip_grad_norm_` |
| `checkpoint` — `checkpoints/<step>/{pretrained_model,training_state}/` | `lerobot/common/train_utils.py` |
| `run` — the step loop, including local ACT warm-start from a `pretrained_model` directory | `lerobot/scripts/lerobot_train.py` |
| `deploy` — local ACT checkpoint loading, feature normalization, action queue, temporal ensembling, finite dataset-backed inference, and checkpoint-only caller-batch inference | `lerobot/policies/act/modeling_act.py`, `lerobot/scripts/lerobot_rollout.py`'s local observation boundary |

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
`tests/device_smoke.rs`, which is compiled only under `--features cuda`. Those
tests have been run on real NVIDIA hardware — an RTX 5080 Laptop (sm_120) on CUDA
12.8 — and so has a 10 000-step ACT training run on a LIBERO dataset whose
checkpoint upstream `lerobot` then loaded and evaluated in simulation; see
`docs/red-green.md`, cycle 13. CI still has no GPU, so the CUDA path is not
covered by any automated gate. The CPU path is validated by CI on Linux, macOS and
Windows, and element by element against upstream PyTorch.

## What is deliberately out of scope

Video decoding, external image files, image transforms, Hub streaming/sync,
`accelerate`,
distributed training, mixed precision, LR schedulers, PEFT, environment evaluation,
`wandb`, and every policy other than ACT. Embedded PNG/JPEG camera columns are
decoded into ACT inputs; camera inputs supplied as `f32` Candle tensors through
`Batch::with_images` and `TrainSession::step_on` are also supported. See
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

A trained local ACT checkpoint can be exercised without a robot or simulator:

```no_run
use rerobot_train::deploy::InferenceSession;
use std::path::Path;

let mut session = InferenceSession::load(
    Path::new("outputs/train/demo/checkpoints/000001/pretrained_model"),
    Path::new("crates/rerobot-train/tests/fixtures/state_only"),
    Some("cpu"),
)?;
let action = session.select_action(0)?;
assert_eq!(action.frame_index, 0);
# Ok::<(), rerobot_train::error::TrainError>(())
```

When observations come from a simulator, camera adapter, or another runtime,
the checkpoint can be loaded without opening a dataset and consumed through the
same single-observation `Batch` boundary as upstream `ACTPolicy.select_action`.
Use `session.device()` for Candle tensor placement and attach **raw** camera
tensors in `[0, 1]` to `Batch::images`; `select_action_on_batch` applies the
checkpoint's saved per-camera normalization and observation rename map:

```no_run
use rerobot_train::data::batch::Batch;
use rerobot_train::deploy::InferenceSession;
use std::path::Path;

let mut session = InferenceSession::load_checkpoint(
    Path::new("outputs/train/demo/checkpoints/000001/pretrained_model"),
    Some("cpu"),
)?;
let raw_batch: Batch = todo!("assemble one raw observation and camera tensors");
let action = session.select_action_on_batch(&raw_batch)?;
assert!(action.action.iter().all(|value| value.is_finite()));
# Ok::<(), rerobot_train::error::TrainError>(())
```
