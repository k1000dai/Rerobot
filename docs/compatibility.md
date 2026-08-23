# Compatibility with Hugging Face LeRobot

Rerobot is a from-scratch Rust port of [Hugging Face LeRobot][upstream]. This
document is the authoritative statement of what is and is not ported.

It is hand-written, and its two status tables — "Console entry points" and
"Module families" — are checked row by row against the machine-readable
inventory in `crates/rerobot-compat` by
`crates/rerobot-compat/tests/docs_consistency.rs`: executable or family name,
status, upstream target or module count, and port note, in inventory order, plus
the pinned upstream version and commit and the two counts quoted in the prose
around them. The build fails if any of those disagree. Prose outside those two
tables is not machine-checked; nothing generates this file.

[upstream]: https://github.com/huggingface/lerobot

## Upstream reference point

| Field | Value |
| --- | --- |
| Package | `lerobot` |
| Version | `0.6.1` |
| Commit | `f37be3edbee60f3a09a5183788b91eb19f0c07d1` |
| License | Apache-2.0 |
| Python requirement | `>=3.12` |
| Source | <https://github.com/huggingface/lerobot> |

## Status labels

| Label | Meaning |
| --- | --- |
| `implemented` | Behaviour parity with upstream is demonstrated by tests in this workspace for the whole unit. **Nothing carries this label yet.** |
| `partial` | Some observable behaviour is ported and tested; the rest of the unit is absent. The scope is spelled out per row. |
| `unimplemented` | Not ported. Executables exist under the upstream name and fail with a stable error and a non-zero exit status; they never silently succeed. |
| `hardware-gated` | Requires physical hardware or a vendor SDK. Out of scope for a pure-Rust milestone; never faked or simulated. |

`implemented` is deliberately unused at this milestone. A one-module port does
not make a family compatible, and labelling it so would misrepresent the work.

## Console entry points (`[project.scripts]`)

All 18 upstream console scripts exist as executables with byte-identical names.
`--help` and `--version` work for every one of them and state the row's status.

| Executable | Status | Upstream target | What upstream does | Port note |
| --- | --- | --- | --- | --- |
| `lerobot-calibrate` | hardware-gated | `lerobot.scripts.lerobot_calibrate:main` | Recalibrate a robot or teleoperator device. | Drives physical hardware through a vendor SDK; nothing is faked, so it stays hardware-gated until a real driver layer exists. |
| `lerobot-find-cameras` | hardware-gated | `lerobot.scripts.lerobot_find_cameras:main` | List the camera devices available on the system. | Drives physical hardware through a vendor SDK; nothing is faked, so it stays hardware-gated until a real driver layer exists. |
| `lerobot-find-port` | hardware-gated | `lerobot.scripts.lerobot_find_port:main` | Find the USB port a MotorsBus is attached to. | Drives physical hardware through a vendor SDK; nothing is faked, so it stays hardware-gated until a real driver layer exists. |
| `lerobot-record` | hardware-gated | `lerobot.scripts.lerobot_record:main` | Record a dataset via teleoperation. | Drives physical hardware through a vendor SDK; nothing is faked, so it stays hardware-gated until a real driver layer exists. |
| `lerobot-replay` | hardware-gated | `lerobot.scripts.lerobot_replay:main` | Replay a recorded episode's actions on a robot. | Drives physical hardware through a vendor SDK; nothing is faked, so it stays hardware-gated until a real driver layer exists. |
| `lerobot-setup-motors` | hardware-gated | `lerobot.scripts.lerobot_setup_motors:main` | Set motor ids and baudrate on a motor bus. | Drives physical hardware through a vendor SDK; nothing is faked, so it stays hardware-gated until a real driver layer exists. |
| `lerobot-teleoperate` | hardware-gated | `lerobot.scripts.lerobot_teleoperate:main` | Drive a robot from a teleoperator. | Drives physical hardware through a vendor SDK; nothing is faked, so it stays hardware-gated until a real driver layer exists. |
| `lerobot-eval` | unimplemented | `lerobot.scripts.lerobot_eval:main` | Evaluate a policy by running environment rollouts. | Needs policy inference and a Gymnasium environment; neither is ported, and fabricating metrics would be worse than failing. |
| `lerobot-train` | partial | `lerobot.scripts.lerobot_train:main` | Train a policy. | Runnable for one vertical slice: the ACT policy on a local or native Hub-downloaded LeRobot v3.0 snapshot with state/action columns and embedded PNG/JPEG camera columns (including dataset-provided per-camera normalization when `use_imagenet_stats=false`), or through the in-memory camera batch API. A local `--policy.path=DIR` warm-start loads the ACT `config.json`, weights, and four saved processor artifacts, then applies supported CLI policy overrides; `--config_path=FILE --resume=false` also loads a native JSON `train_config.json` as a fresh run with CLI overrides; general YAML/Draccus config files remain refused; Hub model IDs remain refused because native model download is not part of this boundary. `--policy.device` takes `cpu`, and `cuda`/`cuda:0` when built with the `cuda` feature; a GPU that was asked for and cannot be provided is an error rather than a silent fallback. A local one-process checkpoint can be resumed with `--resume=true --config_path=...`, restoring model, AdamW state, RNG and sampler position; Python's three-generator/distributed/accelerate resume semantics remain outside the boundary. Video decoding, external image files, image transforms, Hub streaming/revision flags, accelerate, mixed precision, LR schedulers, PEFT and environment evaluation are refused with a reason rather than ignored. |
| `lerobot-train-tokenizer` | unimplemented | `lerobot.scripts.lerobot_train_tokenizer:main` | Train the FAST action tokenizer. | Needs LeRobotDataset loading and the tokenizer training stack. |
| `lerobot-dataset-viz` | unimplemented | `lerobot.scripts.lerobot_dataset_viz:main` | Visualize every frame of a dataset episode. | Needs the dataset reader plus a Rerun/Foxglove viewer bridge. |
| `lerobot-info` | partial | `lerobot.scripts.lerobot_info:main` | Print a markdown summary of the system configuration. | Ported and runnable. Keys that report Python package versions cannot apply to a Rust build and are reported as not ported rather than invented. |
| `lerobot-find-joint-limits` | hardware-gated | `lerobot.scripts.lerobot_find_joint_limits:main` | Discover joint limits and end-effector bounds via teleoperation. | Drives physical hardware through a vendor SDK; nothing is faked, so it stays hardware-gated until a real driver layer exists. |
| `lerobot-imgtransform-viz` | unimplemented | `lerobot.scripts.lerobot_imgtransform_viz:main` | Render examples of the configured image transforms. | Needs the image transform pipeline and dataset loading. |
| `lerobot-edit-dataset` | unimplemented | `lerobot.scripts.lerobot_edit_dataset:main` | Delete, split, merge, and otherwise edit LeRobot datasets. | Needs the LeRobotDataset on-disk format (parquet chunks, video shards). |
| `lerobot-setup-can` | hardware-gated | `lerobot.scripts.lerobot_setup_can:main` | Set up and debug CAN interfaces for Damiao motors. | Drives physical hardware through a vendor SDK; nothing is faked, so it stays hardware-gated until a real driver layer exists. |
| `lerobot-annotate` | unimplemented | `lerobot.scripts.lerobot_annotate:main` | Populate language annotation columns on a dataset. | Needs dataset editing plus an OpenAI-compatible inference backend. |
| `lerobot-rollout` | partial | `lerobot.scripts.lerobot_rollout:main` | Run a trained policy on a real robot with pluggable strategies. | Runnable for a hardware-independent local ACT deployment: it loads a checkpoint, consumes the saved normalizer/unnormalizer processor state, reads local dataset observations, and emits action-unit outputs from the action queue or temporal ensembler. The library also loads a checkpoint without a dataset and accepts a caller-owned single-observation `Batch`, matching the policy's simulator/camera adapter boundary. Robot drivers, teleoperators, environments, visualization and video shards remain explicitly refused; the physical rollout boundary is not faked. |

Exit-status contract for unsupported executables:

* stdout is empty;
* stderr carries exactly one line beginning `<name>: unsupported in Rerobot`;
* the process exits with status `2`.

`--help` and `--version` take precedence over every other argument, so help is
reachable even for commands that cannot run.

## Module families (`src/lerobot/`)

"Upstream modules" counts the `.py` files upstream ships in the package at the
pinned commit; it is a size indicator, not a progress metric.

| Family | Status | Upstream modules | Port note |
| --- | --- | --- | --- |
| `lerobot/annotations` | unimplemented | 15 | Language annotation pipeline; needs dataset editing and an LLM backend. |
| `lerobot/async_inference` | unimplemented | 6 | gRPC policy server/client split. |
| `lerobot/cameras` | hardware-gated | 17 | OpenCV / RealSense capture backends. |
| `lerobot/common` | unimplemented | 4 | Shared constants and mixins used by the unported families. |
| `lerobot/configs` | partial | 11 | `configs.types` str-enums and `PolicyFeature` are ported and tested. The ACT policy's concrete config is too, including the `from_pretrained`/`_save_pretrained` checkpoint JSON path and the Draccus value conversions it decodes through. The Draccus CLI parser is not, and neither is `configs.train`: `lerobot-train` consumes an explicit allow-list of flags instead, refusing everything else by name. |
| `lerobot/data_processing` | unimplemented | 3 | Dataset-level batch processing helpers. |
| `lerobot/datasets` | partial | 22 | State/action columns and embedded PNG/JPEG `dtype: "image"` columns of a LeRobot v3.0 dataset on local disk are ported and tested end to end: `utils`' path constants, `DatasetInfo`, `io_utils`' four loaders including `load_stats` (with nested per-camera mean/std retained separately), the tasks and episodes parquet tables, the frame data files, `feature_utils`' delta indices and tolerance check, `dataset_reader`'s episode-clamped windows and `<key>_is_pad` flags, explicit `episodes=[...]` filtering with absolute-to-relative row remapping, and `sampler`'s `EpisodeAwareSampler` structure. A native Hub snapshot downloader also lists paginated trees, rejects unsafe paths, stages files atomically, and refuses video shards; Hub streaming/sync, video shards, external image files, image transforms, episode predicates and dataset editing are not, and the sampler's per-epoch order is Rerobot's own rather than `torch.randperm`'s. ACT's separate in-memory camera batch contract is implemented by `rerobot-train` rather than this dataset reader. |
| `lerobot/envs` | unimplemented | 10 | Gymnasium environment factories. |
| `lerobot/jobs` | unimplemented | 4 | Hugging Face Jobs launchers. |
| `lerobot/model` | unimplemented | 2 | Shared model plumbing. |
| `lerobot/motors` | hardware-gated | 16 | Feetech / Dynamixel / CAN motor buses. |
| `lerobot/optim` | partial | 4 | `torch.optim.AdamW` and `torch.nn.utils.clip_grad_norm_` are reimplemented with upstream's update order and save format, and checked against PyTorch. The Draccus optimizer and LR-scheduler registries, every other optimizer, and all schedulers are not ported. |
| `lerobot/policies` | partial | 128 | ACTConfig validation, presets, delta indices and byte-exact checkpoint JSON read/write are ported, as is the ACT tensor model for state/action and embedded or in-memory camera inputs: the VAE encoder, ResNet18/34 backbone, 1-D/2-D camera position embeddings, transformer, action head and the L1-plus-KL objective. The ACT temporal ensembler, checkpoint normalizer/unnormalizer boundary, and checkpoint-only caller-batch inference boundary are ported in `rerobot_train::deploy`; every other policy architecture is not. |
| `lerobot/processor` | partial | 19 | `rename_processor` (step + `rename_stats`) and the value transform/stateless lifecycle of `newline_task_processor.NewLineTaskProcessorStep` are ported and tested, as is `normalize_processor`'s numeric transform for all four of its modes. The four pre/postprocessor artifacts a checkpoint carries are written byte-identically to upstream's, and the native ACT deployment path now validates and consumes their saved numeric state for observation normalization and action unnormalization. Python aliasing, general registry/config reconstruction, and the tokenizer, device, batch and full multi-step pipeline runtime remain outside this boundary. |
| `lerobot/rewards` | unimplemented | 24 | Reward classifiers and success detectors. |
| `lerobot/rl` | unimplemented | 21 | HIL-SERL actor/learner infrastructure. |
| `lerobot/robots` | hardware-gated | 53 | Per-robot drivers (SO-100/101, LeKiwi, Reachy2, Unitree, ...). |
| `lerobot/rollout` | partial | 18 | `ring_buffer.RolloutRingBuffer` is ported and tested, including its byte-accounting quirks, as is the DAgger event state machine (`strategies.dagger.DAggerPhase`, its four transitions and `DAggerEvents`). The DAgger strategy itself, the input devices it listens to, the other rollout strategies and the hardware/environment policy loop are not. The local dataset-backed ACT action queue, temporal ensembler, and checkpoint-only caller-batch inference boundary are ported in `rerobot_train::deploy`; it does not claim a robot or Gymnasium boundary. |
| `lerobot/scripts` | partial | 20 | `lerobot_info`, `lerobot_train`, and the hardware-independent local ACT path of `lerobot_rollout` are ported and runnable. The other 15 entry points exist only as executables that fail with a stable unsupported error. |
| `lerobot/teleoperators` | hardware-gated | 59 | Leader arms, gamepads, keyboards, phone teleop. |
| `lerobot/templates` | unimplemented | 0 | Non-Python scaffolding templates; nothing to port yet. |
| `lerobot/transforms` | unimplemented | 2 | Image augmentation transforms built on torchvision. |
| `lerobot/transport` | unimplemented | 4 | gRPC transport for async inference. |
| `lerobot/utils` | partial | 25 | `action_interpolator` is ported and tested, as are `io_utils.load_json` and `io_utils.write_json` for local paths. `common.train_utils`' checkpoint layout is ported by `rerobot-train`. `random_utils` is deliberately *not* ported: Rerobot seeds its own SplitMix64 rather than Python's, NumPy's and PyTorch's three generators, so its random values differ. Hub utilities and the video and image writers in `io_utils` are not ported. |

## What is actually ported

Each item below is a port of specific upstream source, pinned by tests derived
from upstream's own test-suite and from direct execution of the upstream Python.

| Rerobot item | Upstream source | Tests |
| --- | --- | --- |
| `rerobot_core::types` | `lerobot/configs/types.py`, `lerobot/types.py` (`TransitionKey`) | `crates/rerobot-core/tests/types.rs` |
| `rerobot_core::action_interpolator` | `lerobot/utils/action_interpolator.py` | `crates/rerobot-core/tests/action_interpolator.rs` |
| `rerobot_core::ring_buffer` | `lerobot/rollout/ring_buffer.py` | `crates/rerobot-core/tests/ring_buffer.rs` |
| `rerobot_core::rollout::dagger` | `lerobot/rollout/strategies/dagger.py` lines 83-159 (`DAggerPhase`, `_DAGGER_TRANSITIONS`, `DAggerEvents`) only | `crates/rerobot-core/tests/dagger.rs`, plus the poison-recovery unit tests in `crates/rerobot-core/src/rollout/dagger.rs` |
| `rerobot_core::byte_count` | no upstream counterpart — it is the unbounded integer the byte accounting needs, standing in for a Python `int` | `crates/rerobot-core/tests/byte_count.rs` |
| `rerobot_core::processor::rename` | `lerobot/processor/rename_processor.py` | `crates/rerobot-core/tests/rename_processor.rs` |
| `rerobot_core::processor::newline_task` | `lerobot/processor/newline_task_processor.py` (`NewLineTaskProcessorStep` value transform, stateless lifecycle, and registry-name spelling only) | `crates/rerobot-core/tests/newline_task_processor.rs` |
| `rerobot_core::dataset` (constants) | `lerobot/datasets/utils.py` lines 78-97 | `crates/rerobot-core/tests/dataset_info.rs` |
| `rerobot_core::dataset::info` | `lerobot/datasets/utils.py` lines 104-225 (`DatasetInfo`, minus the deprecated dict-style layer) | `crates/rerobot-core/tests/dataset_info.rs` |
| `rerobot_core::dataset::json` | `lerobot/utils/io_utils.py`'s `JsonLike` alias, and CPython 3.12's `json` reader/writer for that domain | `crates/rerobot-core/tests/dataset_json.rs` |
| `rerobot_core::dataset::io` | `lerobot/datasets/io_utils.py` lines 120-134 (`write_info`, `load_info`), `lerobot/utils/io_utils.py` lines 26-50 (`load_json`, `write_json`) | `crates/rerobot-core/tests/dataset_io.rs` |
| `rerobot_core::policy::act` | `lerobot/policies/act/configuration_act.py` (`ACTConfig` only), and the `from_pretrained`/`_save_pretrained` checkpoint path in `lerobot/configs/policies.py` | `crates/rerobot-core/tests/act_config.rs`, `crates/rerobot-core/tests/act_checkpoint.rs` |
| `rerobot_core::policy::draccus` | Draccus 0.10.0's `parsers/decoding.py` conversions, and CPython's `int()`/`float()`/`str()`/`pathlib.PurePosixPath` for the values they reach | `crates/rerobot-core/tests/act_checkpoint.rs` |
| `rerobot_core::sysinfo` | `lerobot/scripts/lerobot_info.py` (pure parts) | `crates/rerobot-core/tests/sysinfo.rs` |
| `rerobot_cli::which` | `shutil.which`, as called by `get_ffmpeg_version` | `crates/rerobot-cli/tests/which.rs` |
| `rerobot_core::random` | no upstream counterpart — SplitMix64, standing in for the three generators `lerobot/utils/random_utils.py` seeds. Rerobot's random *values* are its own; see the boundary note below | `crates/rerobot-core/tests/random.rs` |
| `rerobot_core::dataset::delta` | `lerobot/datasets/feature_utils.py` (`get_delta_indices`, `check_delta_timestamps`), `dataset_reader.py`'s `_get_query_indices`, and `datasets/factory.py`'s `resolve_delta_timestamps` | `crates/rerobot-core/tests/dataset_delta.rs` |
| `rerobot_core::dataset::sampler` | `lerobot/datasets/sampler.py` (`EpisodeAwareSampler` structure and `compute_sampler_state`; **not** the per-epoch order) | `crates/rerobot-core/tests/dataset_sampler.rs` |
| `rerobot_core::dataset::stats` | `lerobot/datasets/io_utils.py`'s `load_stats` and `cast_stats_to_numpy` | `crates/rerobot-core/tests/dataset_stats.rs` |
| `rerobot_core::policy::normalize` | `lerobot/processor/normalize_processor.py`'s `_NormalizationMixin._apply_transform` (all four numeric modes) | `crates/rerobot-core/tests/policy_normalize.rs` |
| `rerobot_train::data` | `lerobot/datasets/{lerobot_dataset,dataset_reader,io_utils}.py`, local-directory state/action plus embedded PNG/JPEG image columns | `crates/rerobot-train/tests/dataset.rs`, `tests/embedded_image.rs` |
| `rerobot_train::model` | `lerobot/policies/act/modeling_act.py` (`ACT`, `ACTEncoder`, `ACTDecoder`, `create_sinusoidal_pos_embedding`, `get_activation_fn`, and `ACTPolicy.forward`'s loss), plus `torch.nn.{Linear,LayerNorm,MultiheadAttention}` | `crates/rerobot-train/tests/model.rs`, and `tests/goldens.rs` against PyTorch |
| `rerobot_train::optim` | `torch.optim.AdamW` and `torch.nn.utils.clip_grad_norm_`, plus `lerobot/optim/optimizers.py`'s save format | `crates/rerobot-train/tests/optimizer.rs`, and `tests/goldens.rs` against PyTorch |
| `rerobot_train::checkpoint` | `lerobot/common/train_utils.py`'s directory layout and `training_step.json` | `crates/rerobot-train/tests/train.rs`, `crates/rerobot-train/tests/checkpoint_safety.rs` |
| `rerobot_train::processor` | the four pre/postprocessor artifacts `save_checkpoint` writes, from `lerobot/policies/factory.py`'s `make_pre_post_processors` and `lerobot/processor/pipeline.py`'s `_save_pretrained`; the saved normalizer/unnormalizer state consumed by the local ACT deployment boundary | `crates/rerobot-train/tests/processor.rs`, byte for byte against upstream's own output, plus `crates/rerobot-train/tests/deploy.rs`'s changed-dataset-statistics differential |
| `rerobot_train::limits` | no upstream counterpart — the resource budget the reader and the model enforce on untrusted sizes | `crates/rerobot-train/tests/limits.rs`, `crates/rerobot-train/tests/parquet_budget.rs` |
| `rerobot_train::run` | `lerobot/scripts/lerobot_train.py`'s offline step loop | `crates/rerobot-train/tests/train.rs`, `crates/rerobot-cli/tests/train_cli.rs` |
| `rerobot_train::deploy` | `lerobot/policies/act/modeling_act.py`'s `select_action` queue and `ACTTemporalEnsembler`, plus the local observation side of `lerobot/scripts/lerobot_rollout.py` | `crates/rerobot-train/tests/deploy.rs`, `crates/rerobot-cli/tests/rollout_cli.rs` |
| `rerobot_cli::train` | `lerobot-train`'s argument surface, as an explicit allow-list rather than a Draccus port | `crates/rerobot-cli/tests/train_cli.rs` |

### The ACT training slice is checked against PyTorch, not only against itself

`crates/rerobot-train/tests/goldens.rs` compares Rerobot's ACT path element by
element against upstream running on PyTorch, at the reduced state-only
configuration described in `tools/goldens/README.md`. The expected values were
produced once by `tools/goldens/make_act_goldens.py` at the pinned commit and
committed; the Rust tests read them and never invoke Python.

What is compared: the normalized batch, the predicted action chunk, the latent
distribution's `mu` and `log(sigma^2)`, the masked L1 loss, the mean KL
divergence, the weighted total, eleven representative parameter gradients, the
total gradient norm `clip_grad_norm_` reports, and the parameters after one AdamW
step. Agreement is to `f32` round-off — measured worst case 9.4e-7 of each
tensor's own scale, and better than 1e-7 relative on every scalar.

The comparison is anchored at both ends. Upstream's exported `state_dict` loads
into Rerobot's model, which only works because the 62 tensor names and shapes are
upstream's; and `tools/goldens/verify_checkpoint_upstream.py` loads a checkpoint
`lerobot-train` wrote back into a real `ACTPolicy` with `strict=True` and runs a
forward pass on it.

### What a training run refuses, and why that is part of the contract

The training slice treats every number it did not compute itself as hostile, because
each of them arrives from a command line, a `meta/info.json`, an episode table or a
parquet footer, and two of its dependencies act on them: candle allocates tensors and
Arrow decodes columns, both in code with a large unsafe surface that
`#![forbid(unsafe_code)]` on this crate says nothing about.

| Refused | Because |
| --- | --- |
| a non-finite or out-of-range float (`policy.dropout`, `kl_weight`, the three learning rates, `tolerance_s`) | `NaN` in `dropout` silently *disables* dropout, since `NaN > 0.0` is false, so the run trains a different configuration than the one asked for; `NaN` in a learning rate poisons every weight |
| a non-finite loss, KL term, gradient norm or post-update parameter norm | the step trained nothing, and reporting it writes a checkpoint of `NaN` weights that looks like a successful run. Checked before the optimizer runs, so a poisoned gradient never reaches the weights |
| a policy dimension, batch size or step count above `rerobot_train::limits` | each becomes a tensor shape or an allocation; a `chunk_size` of 10^29 is a request to abort the process, not a configuration |
| a parquet file above its byte, row, column, value, text or element budget | checked from the footer before any decode, and accumulated across files so a dataset of many small files is bounded too |
| a shape product that overflows | wrapping is worse than panicking: the allocation succeeds at the wrong size and the reader then walks past its end |
| an episode range that is negative, inverted, overlapping, past `total_frames`, or disagrees with its own `length` | the reader treats these as arithmetic, and `i64::MIN` made the window clamp wrap onto unrelated frames |
| a real directory at `checkpoints/last` | maintaining a one-line marker must never recursively delete a tree. A symlink is unlinked without being followed; anything else is refused |
| a checkpoint tensor of the wrong dtype, shape or name, an RNG state of the wrong shape or with extra tensors, or an optimizer state with an unknown key, an out-of-range parameter index, a mismatched moment, or a non-finite step | each would otherwise be a silently wrong resume rather than an error. The optimizer validates every entry before installing any, so a rejected file leaves it untouched |

None of these limits is below what upstream's own defaults need; static assertions in
`crates/rerobot-train/tests/limits.rs` enforce that, so the budget cannot be tightened
into refusing a command upstream accepts.

### Where this slice's randomness diverges, and why

`lerobot/utils/random_utils.py` seeds Python's `random`, NumPy's global
`RandomState` and PyTorch's Mersenne Twister, and every random choice upstream
makes draws from one of those. Rerobot reproduces none of those streams: it uses
SplitMix64 (`rerobot_core::random`), whose entire state is one 64-bit word.

The consequences are stated rather than hidden:

* **parameter initialization** — the *distributions* are torch's
  (`kaiming_uniform_(a=sqrt(5))` for `nn.Linear`, `xavier_uniform_` for the
  transformer and for `nn.MultiheadAttention`'s packed projection, `N(0, 1)` for
  embeddings), but the values differ. A same-seeded run does not reproduce
  upstream's weights.
* **the sampler's per-epoch order** — upstream derives it from `torch.randperm`
  seeded through `numpy.random.SeedSequence([seed, epoch])`. Rerobot substitutes
  its own documented permutation, which keeps every property the training loop
  needs (a pure function of `(seed, epoch)`, reproducible across processes and
  platforms, resumable from an offset) but is a different sequence.
* **the VAE latent draw and dropout masks** — same seed, different numbers.
* **`rng_state.safetensors`** — holds one tensor named
  `rerobot_splitmix64_state`, not upstream's `random_state`,
  `numpy_random_state` and `torch_random_state`. A reader expecting those fails
  to find them rather than finding something that looks like them and is not.

This is why the differential oracle supplies the weights and the latent draw
instead of seeding both sides.
| `lerobot-info` | `lerobot/scripts/lerobot_info.py` | `crates/rerobot-cli/tests/{info,cli,which}.rs` |

`lerobot-info` prints upstream's 15 `get_sys_info` keys, in upstream's order,
and no others — including `Using GPU in script?`, whose `<fill in>` placeholder
is a prompt to whoever pastes the report rather than a probe. Rerobot's own
compatibility status is not in that output; it is in `<command> --help` and in
this document.

### Deliberate divergences

These are the only places where Rerobot knowingly differs from upstream. Each is
a consequence of the target language, not a shortcut.

| Upstream behaviour | Rerobot behaviour | Why |
| --- | --- | --- |
| `ActionInterpolator` operates on `torch.Tensor` | Operates on `&[f32]` / `&[f64]` slices, with 1-D broadcasting ported explicitly | No tensor runtime in a pure-Rust core. The control loop feeds it a 1-D action vector, which a slice models exactly. Length-1-against-length-N broadcasting, in both directions, is reproduced; a non-broadcastable pair is `InterpolatorError::NotBroadcastable`, carrying PyTorch's own message. |
| `ActionInterpolator.add` raises `RuntimeError` from tensor broadcasting | Returns `Err(InterpolatorError::NotBroadcastable)` after clearing the buffer and leaving `prev`/index untouched | Rust has no exceptions; the failure is still an error, not a silent success. The observable post-failure state matches upstream, which assigns `self._buffer = []` before the arithmetic that raises. |
| `RolloutRingBuffer.__init__` rejects a bad `maxlen` with `ValueError`/`OverflowError` | `RolloutRingBuffer::new` returns `Err(RingBufferError::…)`, one variant per CPython failure and in CPython's order | Same reason. NaN, infinity, out-of-`Py_ssize_t`, and plain-negative capacities stay distinguishable. |
| Byte accounting uses unbounded Python `int`s | `rerobot_core::byte_count::ByteCount`, a newtype over `num_bigint::BigUint`, for every frame estimate and running total; an `i128` byte cap | Not a divergence in the values, only in the type: `ByteCount` is exact at every magnitude, so there is no frame and no accrual on which the total differs from Python's. The cap stays fixed-width because `int(max_memory_mb * 1024 * 1024)` for an `i64` megabyte count fits an `i128` exactly. See "Numeric domain" below. |
| `PolicyFeature.shape` is a `tuple[int, ...]` of unbounded signed Python `int`s; CPython 3.12's `json.dumps` rejects decimal conversions above its default 4,300-digit guard | `Vec<num_bigint::BigInt>`, written to JSON as the same bare decimal integer; serde accepts longer tokens too | The in-memory integer domain is preserved, including negatives and values above every machine word. For JSON accepted by default CPython the wire value is identical. Rerobot deliberately accepts a serialization-domain superset because serde has no process-wide equivalent of `sys.set_int_max_str_digits`; Python can be configured to accept the same longer values. The container differs — a Rust `Vec` rather than a Python tuple — and the port has no `numel()`, because upstream's dataclass has no methods. |
| A checkpoint `config.json` is read by `json.load`, whose object literal follows Python `dict` assignment (a duplicate key takes the last value and keeps its first position) and whose integers are unbounded | `ActConfig::from_checkpoint_json` reproduces both, because it reads through `dataset::json::loads` rather than `serde_json`. The derived `serde::Deserialize` impl still rejects a duplicate field, as serde's data model requires | Not a divergence at the checkpoint boundary. Plain serde stays strict so an in-memory Rust caller is not silently handed a value the document mentioned twice; the checkpoint API is the one that claims upstream parity. |
| `ACTConfig.action_delta_indices` eagerly returns `list(range(chunk_size))` | Returns a lazy arbitrary-precision iterator with the same values and order | Python can exhaust memory for an enormous accepted `int`. The Rust port preserves that integer domain without an eager machine-sized allocation. The collection type and failure timing differ; normal finite consumers observe the same sequence. |
| `PreTrainedConfig.__post_init__` probes Torch device/AMP availability and may mutate `device`/`use_amp` while logging warnings | `ActConfig::validate` runs only ACT-specific checks; it does not mutate runtime fields or synthesize hardware availability | Device selection belongs to the later runtime adapter. Config defaults, ACT validation order/messages, feature validation, presets, delta indices, and local checkpoint JSON are the validated boundary in this slice. |
| CPython's JSON reader and writer accept and emit the bare `NaN`/`Infinity`/`-Infinity` tokens, which standard JSON has no literal for | `ActConfig::from_checkpoint_json`/`to_checkpoint_json` accept and emit them, matching upstream byte for byte. The `serde::Serialize` impl, which has to produce a `serde_json` document, still returns an explicit error for a non-finite float rather than silently writing `null` | Not a divergence at the checkpoint boundary, which is the one upstream reads and writes. It is a boundary of `serde_json` alone: its data model has no non-finite float, so the two entry points are kept distinguishable instead of one of them corrupting the value. |
| `replace_final_stride_with_dilation` is annotated `int` but defaults to `False`; fresh construction stores/writes a bool, while Draccus checkpoint loading converts bool to `0`/`1` and accepts arbitrary integers | `PythonIntBool` retains either the fresh boolean or decoded `BigInt` form and compares `false == 0` / `true == 1` as Python does | A plain Rust bool would reject the declared integer domain; a plain integer would change fresh `config.json` from `false` to `0`. The sum type preserves both observable paths. |
| `ActionInterpolator.multiplier` is an unbounded Python `int` | `num_bigint::BigInt`; storage, the getter and `enabled` are exact at every magnitude | Not a divergence in the stored domain. The two operations that cannot cover that domain say so: see the two rows below. |
| Building the interpolated sequence for a huge multiplier ends in `MemoryError` after CPython grinds through the `list` | `ActionInterpolator::add` returns `Err(InterpolatorError::BufferNotAllocatable)` up front | The sequence is `multiplier` elements of a Rust `Vec`, so a multiplier that does not fit a `usize`, or whose slots cannot be reserved, is refused rather than truncated to a step count that does fit. Below that boundary the buffer is genuinely built, and an allocator that fails part-way aborts the process where CPython would raise `MemoryError`. |
| `fps * self.multiplier` raises `OverflowError: int too large to convert to float` for a multiplier outside the `f64` range | `get_control_interval` returns `Err(InterpolatorError::MultiplierNotFloatRepresentable)`, carrying that exact message | Rust has no exceptions, and the alternative is worse than an error: dividing by an infinity would silently report a control interval of `0.0`. Below the boundary the conversion is the nearest `f64`, taken through the decimal digits and Rust's correctly-rounded float parser. |
| `_estimate_frame_bytes` dispatches on Python runtime types | Callers tag values with `FrameValue` | Static typing. The per-variant cost model is a byte-for-byte port. |
| `NewLineTaskProcessorStep.complementary_data` returns the *same* `dict` object when `task` is absent or `None`, and otherwise makes a shallow copy whose untouched nested values remain shared | Always returns an independently owned `IndexMap` and deep-cloned `serde_json::Value`s | Deliberate ownership boundary. Values and key order match, but mutation-visible aliasing does not: mutating a nested Python value through either map can affect the other, while Rust's result is independent. The input is never modified by the Rust step. |
| `NewLineTaskProcessorStep.transform_features` returns the identical `features` object it was handed | Returns an independent clone that compares equal and keeps its stage order | Value/ordering identity is ported; object identity and mutation sharing are not. This is more than a Python `is` difference: later nested mutation is observable. |
| `complementary_data` takes any Python object as the `task` value and dispatches with `isinstance` | Takes a `serde_json::Value` | Static JSON-domain boundary. Strings, null, booleans, objects, representable finite numbers, and arrays built from them follow the matching Python branches. Oversized Python integers, NaN/infinities, tuples, bytes, arbitrary objects and ill-formed Unicode are outside the domain rather than approximated; see below. |
| Upstream registers the step as `smolvla_new_line_processor` and reconstructs it through the processor registry | Exposes that exact spelling as `REGISTRY_NAME` only | Registry lookup, pipeline serialization and config reconstruction are not yet ported. The constant prevents spelling drift but is not a claim that old serialized pipelines can already be loaded. |
| `DAggerPhase` is a plain `enum.Enum`, so `str(DAggerPhase.PAUSED)` is `"DAggerPhase.PAUSED"` and `json.dumps` refuses the member outright | `as_str`/`FromStr` are the member `.value` and upstream's by-value lookup `DAggerPhase("paused")`; `Display` is that same value, and there is deliberately **no** `serde` impl | The values are ported; Python's `str()` spelling is not, because it is a repr rather than a wire format. No `serde` impl is provided precisely because upstream has no serialization to be compatible with — unlike the `str`-backed enums in `configs.types`, which do. |
| `DAggerEvents.stop_recording` / `upload_requested` are `threading.Event`s | `EventFlag`, an `AtomicBool` exposing `set`/`clear`/`is_set` | Those three are the only operations the upstream DAgger path performs on them. `Event.wait()` and its timeout are not ported rather than approximated; nothing upstream blocks on these flags. |
| `DAggerEvents` guards its state with a `threading.Lock`, which has no poison flag | `std::sync::Mutex` whose poisoning is recovered with `into_inner` | A panic in one thread must not convert every later call into a panic, which the `unwrap()` idiom would do. The recovered state is the one the panicking section left behind, not a silent reset. Verified by unit tests inside the module, because the lock is private and no public method runs caller code while holding it. |
| `request_transition(event: str)` accepts any `str` | Takes `&str` | Same domain; unknown names are ignored rather than rejected, exactly as upstream's dict lookup does. |
| `DatasetInfo.__post_init__` rewrites `features[...]["shape"]` **in place**, so the caller's own dict — and, via `from_dict`, the dict `json.load` returned — comes back with a tuple in it | `DatasetInfo::new` / `from_dict` take or copy their input and mutate only the value they own | Deliberate ownership boundary, and the one upstream behaviour here a caller can *see* rather than infer. In Python, `features = {"a": {"shape": [1, 2]}}` passed to the constructor leaves `features["a"]["shape"] == (1, 2)` afterwards, and that happens even when construction then fails on `fps`. Rerobot cannot reproduce it without handing out aliased mutable state, so it does not. Values and ordering match; aliasing does not. |
| `DatasetInfo` is an unchecked dataclass: `codebase_version=5`, `fps=30.5` and `splits=7` are all accepted, and a bad type surfaces later (`fps="30"` raises `TypeError` from the `<=` comparison, `features=[]` from `.values()`) or never | Typed fields; a value outside the domain is `DatasetInfoError::WrongType`, naming the field, the required Python type and the one found | Static typing. The message is deliberately *not* one of upstream's — there is no upstream message to match, because upstream mostly does not fail at all. `fps=30.5` is the clearest case: upstream stores the float and writes `30.5` into `info.json` where the field is declared `int`. |
| `fps=True` is accepted, because Python's `bool` is an `int` subclass, and `json.dump` writes it back as `true` | A JSON `true` in any integer field is `DatasetInfoError::WrongType { expected: "int", found: "bool" }` | Reproducing it would mean every counter remembering whether it had been spelled as a bool, so that `to_dict` could write `true` again. The narrower domain is stated rather than silently coerced to `1`, which would change the file on the next write. |
| `logger.warning(f"Unknown fields in DatasetInfo: {unknown}. …")` is emitted from `from_dict` | `DatasetInfo::from_dict` emits the same sentence at warning level through Rust's `log` facade before attempting construction; `unknown_fields` and `unknown_fields_warning` also expose the sorted list and exact text | Logger handlers and filters remain application configuration, just as Python logging configuration is process-wide. The list itself is exact: Python's `sorted()` on `str` orders by code point, which is Rust `String`'s `Ord` order for valid Unicode. |
| `json.dump(..., indent=4)` goes through `open(fpath, "w")`, whose text mode translates `\n` to `\r\n` on Windows and encodes with the process's locale encoding | Always UTF-8, always LF | The JSON `json.dump` itself produces is identical; the translation and the encoding are artifacts of how the file was opened, not of the format. A locale-dependent `meta/info.json` is not a format Rerobot can be compatible with on both sides, and guessing an encoding is worse than naming one. On any platform whose locale is UTF-8 — which includes upstream's own CI — the bytes match exactly. |
| `DatasetInfo` carries a deprecated dict-style layer (`__getitem__`, `__setitem__`, `__contains__`, `get`) that warns and forwards to attribute access | Not ported | It exists to keep un-migrated `info["key"]` call-sites working during upstream's own migration. A Rust port has no such call-sites, and public fields already are the attribute access it forwards to. |
| `lerobot-info` reports installed Python package versions | Reports `N/A (not ported)` for those keys, distinct from upstream's `N/A` | A Rust build has no `torch`/`datasets`/`numpy`. Inventing versions would make bug reports actively misleading; reusing `N/A` would claim a check that never happened. |
| `lerobot-info`'s `LeRobot version` is the installed `lerobot` distribution version | `0.6.1 (upstream target; Rerobot 0.1.0, a partial Rust port)` | There is no `lerobot` Python distribution in a Rerobot install. The value names both versions rather than fabricating one or dropping the key. |
| `lerobot-info`'s `Platform` is `platform.platform()`, e.g. `macOS-15.0-arm64-arm-64bit` | `<os>-<arch>` from `std::env::consts`, e.g. `macos-aarch64` | Rust's standard library exposes no OS release or libc version. The port reports only what it can know for certain rather than shelling out to `uname` or guessing a release string. |
| `get_ffmpeg_version` distinguishes "absent" (`shutil.which` -> `N/A`) from "ran but failed" (`SubprocessError` -> parse-failed sentinel) | Same distinction, and the same two steps: `which::which` resolves the name on `PATH` with Python's acceptance rule, then the resolved path is run | `shutil.which` is ported rather than approximated by a spawn error, so a non-executable or shadowed `ffmpeg` is classified the way upstream classifies it. |
| A resolved `ffmpeg` that fails to spawn, or prints non-text, raises an uncaught `OSError`/`UnicodeDecodeError` upstream | Reported as `Installed (version parsing failed)` | Aborting the whole report over an unreadable version would lose the rest of it. Both cases are unreachable through `shutil.which`'s own checks except under a race. |

### Upstream quirks reproduced on purpose

Reproduced, not "fixed", because callers can observe them:

* `RolloutRingBuffer`: frames dropped by the `deque(maxlen=...)` length cap do
  **not** decrement the byte accounting — only the explicit memory-cap eviction
  branch does. After four 8-byte frames into a 2-frame buffer, `len()` is `2`
  and `estimated_bytes()` is `32`.
* `RolloutRingBuffer`: a zero-length cap (`int(max_seconds * fps) == 0`)
  discards every appended frame while still accruing its bytes.
* `_estimate_frame_bytes`: Python `bool` is an `int` subclass and costs 8 bytes;
  `str` costs `len(v)` in code points, not UTF-8 bytes; an all-unrecognised
  frame still costs 1 byte via `max(total, 1)`.
* `get_ffmpeg_version`: `str.split(" ")` keeps empty fields, so a double space
  in the banner shifts the parsed token. `str.splitlines()` also breaks on bare
  `\r`, which `str::lines` does not — ported explicitly.
* `RenameObservationsProcessorStep`: the rename map is applied exactly once per
  key, so `{"a": "b", "b": "c"}` does not cascade `a -> b -> c`; and colliding
  output keys follow Python `dict` assignment (last value wins, first position
  kept).
* `ActionInterpolator.add`: `self._prev` is replaced by the *action*, not by the
  broadcast result, so a length-1 action after a length-3 previous leaves the
  interpolator able to accept any length next time.
* `DAggerEvents.reset`: clears `upload_requested` and **not** `stop_recording`,
  so a session stopped with ESC stays stopped across a reset. The asymmetry is
  reproduced, not corrected.
* `DAggerEvents.request_transition`: an invalid or unknown event is ignored
  *without* clearing a valid request that is already pending, and a later valid
  request overwrites an earlier one — there is a single pending slot.
* `DAggerEvents.consume_transition`: the pending request is cleared *before* the
  transition table is consulted, so a request invalidated by an intervening
  `phase` write returns `None` and is dropped rather than held until its phase
  comes back.
* `NewLineTaskProcessorStep`: the list branch is all-or-nothing. One non-string
  element leaves the whole list untouched, so `["a", 1]` keeps `"a"` without its
  newline; and because `all(...)` over an empty list is `True`, an empty list
  takes the list branch and is rebuilt as an empty list.
* `DatasetInfo.to_dict`: `tools` is dropped only when it is `None`. An empty
  list is a *declaration of no tools* and is written as `"tools": []`, so the
  two are not interchangeable on disk.
* `DatasetInfo.__post_init__`: only the feature dict's own `shape` key is
  coerced. A `shape` nested deeper (`features.a.info.shape`) stays a list, a
  `shape` that is not a list at all (`"xy"`) is left alone, and `[]` becomes an
  empty tuple that `to_dict` turns back into `[]`.
* `DatasetInfo.__post_init__`: the shape coercion runs *before* the four
  positivity checks, so upstream rewrites the features even on the path that
  then raises. The four checks run in declaration order, so an info with both
  `fps=0` and `chunks_size=0` is reported against `fps`.
* `DatasetInfo`: assigning to a field does not re-run `__post_init__`, so
  `info.fps = 0` is reachable and `to_dict()` will happily write it. Rerobot's
  fields are public for the same reason, and `post_init` is public so a caller
  can opt back into the checks.
* `json.loads`: `NaN`, `Infinity` and `-Infinity` are accepted by default and
  `json.dump` emits them, so a non-finite float survives a round trip through
  `meta/info.json` even though JSON has no literal for one. The lowercase
  spellings (`nan`, `infinity`) are *not* accepted.
* `json.loads`: a number with no fraction and no exponent is an `int`, and any
  fraction or exponent makes it a `float` — so `1.0`, `1E5` and `1e+5` are
  floats whose values are integral, and `-0` is the integer `0` while `-0.0` is
  the float `-0.0`.
* `float.__repr__`, which is what `json.dump` writes: the decimal point moves to
  exponent notation only when it lands at or below `-4` or above `16`, so `1e15`
  is written `1000000000000000.0` and `1e16` is written `1e+16`.
* `NewLineTaskProcessorStep`: `str.endswith("\n")` is exact. `"pick\r"`,
  `"pick\u{2028}"` and `"pick\u{0085}"` all gain a newline; only a trailing
  `"\n"` — including the `"\n"` of a `"\r\n"` — counts as already terminated.
  `""` becomes `"\n"`, and a `bool` is not a `str`, so `[true, false]` is left
  alone.

### Python `task` values the newline step does not claim

`complementary_data` is typed as `IndexMap<String, serde_json::Value>`, which is
the JSON value domain. Upstream dispatches on Python runtime types, so there are
`task` values that reach it in Python and have no `Value` counterpart. They are
stated here rather than approximated. Each upstream behaviour below was observed
by running the pinned module under CPython 3.12, not inferred:

| Python `task` value | Upstream behaviour | Rerobot |
| --- | --- | --- |
| `("a", "b")` — a tuple of strings | unchanged, because `isinstance(task, list)` is `False` | Not representable. A caller who turns a tuple into a `Value::Array` gets the *list* answer (`["a\n", "b\n"]`), which is upstream's answer for a list and not for a tuple. |
| `[b"a"]` — `bytes` elements | unchanged | Not representable; the nearest `Value` is a string or an array of numbers, neither of which is `bytes`. |
| a `str` subclass, `numpy.str_`, a `torch.Tensor` | dispatches on the runtime type (`str` subclasses take the string branch) | Not representable; no attempt is made to model Python's type hierarchy. |
| `"\ud800"` — a `str` holding an unpaired surrogate | becomes `"\ud800\n"` | Not representable: Rust `String` is well-formed UTF-8. Every value `Value::String` *can* hold is handled identically to upstream, including astral-plane and combining characters. |
| an integer outside `serde_json::Number`'s enabled range | unchanged | Not representable by this build's `serde_json::Value`; the processor does not silently narrow it. |
| `float("nan")`, `float("inf")`, `float("-inf")` | unchanged | Not representable as `serde_json::Number`; JSON has no non-finite number literals. |

The keys are `String` for the same reason: a Python `dict` accepts any hashable
key, and upstream only ever tests for `"task"`.

### The `meta/info.json` value domain

`serde_json::Value` is not the value domain this file is written in, so
`rerobot_core::dataset::json::JsonLike` is used instead. It is a port of the
type alias upstream itself declares in `lerobot/utils/io_utils.py`:

```python
JsonLike = str | int | float | bool | None | list["JsonLike"]
         | dict[str, "JsonLike"] | tuple["JsonLike", ...]
```

Three differences from `serde_json::Value` are load-bearing here, and each is
exercised by a real `info.json`:

| Property | Why `serde_json::Value` is not enough | What `JsonLike` does |
| --- | --- | --- |
| A Python `int` is unbounded | Without `arbitrary_precision`, an integer past `u64`/`i64` is silently rounded through `f64`. Enabling that feature is rejected at the workspace level, because it rewrites `serde_json::Number` for every crate in the dependency graph. | `JsonLike::Int` is a `num_bigint::BigInt`; values have no fixed-width narrowing. The parser accepts bare decimal tokens up to its documented 100,000-character fail-closed budget, while programmatically constructed values remain unbounded. CPython's configurable decimal-conversion guard is another runtime boundary described below. |
| `json.load` accepts `NaN`, `Infinity` and `-Infinity`; `json.dump` emits them | JSON has no literal for a non-finite number and `serde_json` rejects all three, so a file CPython wrote could not be read back. | `JsonLike::Float` is an `f64` including the non-finite values, with CPython's spellings on both sides. |
| A `tuple` is not a `list` | `Value` has one sequence variant, so `(1, 2) != [1, 2]` — which `DatasetInfo` relies on — cannot be represented. | `JsonLike::Tuple` is separate from `JsonLike::Array`, compares unequal to it, and is written as a JSON array exactly as `json.dump` writes a tuple. `loads` never produces one. |

The reader and writer are consequently ports of CPython 3.12's `json` module
for this domain, not wrappers around a JSON library: `loads` reproduces the
acceptance rules and the `JSONDecodeError` message, line, column and character
offset of the **C scanner** CPython uses by default (whose wording differs from
the pure-Python fallback in `json/decoder.py` — `Invalid control character at`
rather than `Invalid control character '\x01' at`), counting code points rather
than bytes as CPython does. `dumps_pretty` reproduces `json.dump(..., indent=4,
ensure_ascii=False)` byte for byte.

`float.__repr__` is ported rather than delegated to Rust's `{}`, which never
uses exponent notation. Rust's `{:e}` supplies only the *number* of significant
digits; the digits themselves are recomputed from the double's exact value with
`BigInt` arithmetic and rounded half to even, because on an exact tie CPython's
`_Py_dg_dtoa` rounds to an even last digit and Rust's formatter rounds upward.
That is not hypothetical: eight doubles in a 30,623-value sweep differ, and
`tests/dataset_json.rs` pins all eight. The port is checked against CPython
3.12.13 over 747,248 doubles, 40,586 of them subnormal, with no disagreement.

**This is not full JSON parity, and is not claimed as such.** The scope is the
`meta/info.json` metadata domain. One input CPython accepts is refused:

| Input | CPython | Rerobot |
| --- | --- | --- |
| `"\ud800"` — a string escape naming an unpaired surrogate | succeeds, yielding a `str` holding the lone surrogate | `ParseError` whose message is deliberately none of CPython's (`Unpaired surrogate escape (not representable in Rust)`), because a Rust `String` is well-formed UTF-8 and borrowing CPython's wording would imply CPython's behaviour |
| an integer token or value beyond CPython 3.12's active `sys.get_int_max_str_digits()` limit (4,300 by default) | `json.load` / `json.dump` raises `ValueError` unless the process raises or disables the limit | accepted and written exactly as `BigInt`; Rerobot has no process-global decimal-conversion guard |
| more than 128 nested arrays/objects | accepted up to CPython's active recursion limit, then raises `RecursionError` | returns `ParseError` with `Rerobot JSON nesting limit exceeded`; the explicit lower bound prevents malformed metadata from aborting the process through native stack exhaustion |
| a caller programmatically constructs a value far deeper than the reader's 128-container limit | CPython's encoder and ordinary object destruction are governed by its recursion/runtime behavior | `dumps` / `dumps_pretty` use an explicit work stack, but derived `JsonLike` destruction, cloning, equality, and debug formatting remain recursive and can exhaust the native stack. Callers must keep constructed values within the documented reader depth when using those operations. |
| metadata exceeding a resource budget | limited only by available resources plus CPython's recursion/integer guards | local files and `loads` input are capped at 16 MiB; one parse is capped at 100,000 values, 1,000,000 decoded characters per string/key, and 100,000 source characters per number. Exceeding any bound returns `LoadError::ResourceLimit` or `ParseError`, rather than relying on allocator failure. These bounds are far above ordinary `info.json` metadata and are intentional fail-closed divergences. |
| a file whose bytes are not UTF-8 | decodes with the process's locale encoding | an IO error naming the path |
| a `dict` with a non-`str` key | accepted by `json.dump` for `int`/`float`/`bool`/`None` keys, which it coerces to strings | not representable; keys are `String`, and upstream's own `JsonLike` alias already declares `dict[str, ...]` |

The budgets bound predictable work caused by the input; they are not a promise
that every process-wide allocator failure is recoverable. Parser-owned byte,
character, string, array, object-table, and numeric-token reservations are
fallible and become typed errors. Allocation performed internally by `BigInt`
conversion and by caller operations still follows Rust's global allocator
behavior. ACT checkpoint decoding additionally uses fallible reservations for
its structural object/sequence clones, feature maps, shapes, string lists, and
extra-field diagnostics. `BigInt` internals and compatibility diagnostics that
render an arbitrary caller-built `JsonLike` through Python-style `repr` remain
subject to the same global-allocator boundary; the 16 MiB/100,000-node parser
budgets bound inputs originating from checkpoint JSON but cannot turn every
process-wide OOM into a typed Rust error.

Within those stated boundaries, the alias values exercised by metadata —
including astral-plane and combining characters, surrogate *pairs*, duplicate
keys, and numeric tokens within CPython's active conversion limit — are handled
identically to CPython, and the round trip is pinned by tests. This does not
claim identical resource exhaustion, recursion limits, or process-global
configuration outside the `meta/info.json` metadata domain.

### Numeric domain

Upstream counts frame bytes in Python `int`s, which are unbounded. Rerobot
counts them in `rerobot_core::byte_count::ByteCount`, which is also unbounded: a
newtype over `num_bigint::BigUint` exposing exactly the operations the
accounting needs (add, subtract-on-eviction, compare against the cap, and the
one `numel * element_size` product). **The exact domain is every value, without
limit.** There is no frame estimate and no running total at which the port
saturates, wraps, panics, or undercounts, in debug or release.

The arithmetic is `BigUint`'s rather than hand-written, so the exactness claim
does not rest on carry and borrow code audited once here. On top of the pinned
CPython values, `tests/byte_count.rs` checks every operation `ByteCount`
exposes against `BigUint` computed independently of it, over a deterministic
sweep that crosses the 64- and 128-bit boundaries in both directions.

That matters because the values are describable, not hypothetical. One
`FrameValue::Tensor` costs up to `usize::MAX * usize::MAX`; *two* of them in one
frame already exceed `u128::MAX`, and under the zero-length-cap quirk above a
running total has no upper bound at all. A 128-bit accumulator silently
under-counts all of those, which is a wrong answer rather than a narrower one.
`tests/ring_buffer.rs` pins four such cases against values computed by CPython
3.12 — two, three and four maximal tensors in one frame, and 1000 maximal frames
accrued — and `tests/byte_count.rs` pins the arithmetic itself.

The remaining fixed-width quantities are fixed-width because a fixed width is
already exact for them:

* the byte cap is `int(max_memory_mb * 1024 * 1024)`, and `i64::MAX * 2^20` and
  `i64::MIN * 2^20` both fit an `i128`, including the negative caps Python
  accepts;
* `int(max_seconds * fps)` is validated against `Py_ssize_t` before it becomes a
  `usize`, so it is never truncated on a 32-bit target.

The other two Python `int`s are unbounded here as well, for the same reason:

* `ActionInterpolator`'s multiplier is a `num_bigint::BigInt`. `__init__`
  validates `multiplier < 1` and nothing else, so `2**63` — the first value an
  `i64` cannot hold — is an ordinary argument. Storage, `multiplier()` and
  `enabled()` are exact at every magnitude. The two operations that cannot be
  are errors, not narrowings, and the boundaries are
  exact: `add` returns `BufferNotAllocatable` when the multiplier does not fit a
  `usize` or its buffer slots cannot be reserved, and `get_control_interval`
  returns `MultiplierNotFloatRepresentable` at precisely the point CPython's
  `int`-to-float conversion raises `OverflowError` — `2**1023` converts,
  `2**1024` does not.
* `PolicyFeature.shape` is a `Vec<num_bigint::BigInt>`. Upstream's annotation is
  `tuple[int, ...]`, and both properties of a Python `int` are used: it is
  signed, so `-1` is the ordinary dynamic axis, and it is unbounded. For values
  within CPython 3.12's default 4,300-digit conversion guard, the JSON wire form
  is the identical bare decimal integer. Rerobot also accepts longer tokens;
  Python accepts those only after increasing or disabling
  `sys.set_int_max_str_digits`. Neither representation depends on `usize` width.

`PolicyFeature` deliberately has no `numel()`. Upstream's dataclass declares
`type` and `shape` and no methods, so a product convenience had no upstream
behaviour to be compatible with, and the product it computed overflowed on
shapes the struct stores perfectly well. It was removed rather than hardened,
and `policy_feature_round_trips_dimensions_far_above_usize_max_exactly` pins
that such a shape still round-trips exactly.

### Platform validation boundaries

`rerobot_cli::which` is a port of CPython **3.12**'s `shutil.which` restricted
to the one mode `get_ffmpeg_version` uses, `os.F_OK | os.X_OK`. Not all of it
can be exercised on one machine, so this is exactly what is checked where.

| Behaviour | Where it is verified |
| --- | --- |
| `os.path.split` semantics, including a trailing separator and a lone root | `which` unit tests, every platform; expectations taken from CPython 3.12 `posixpath.split` / `ntpath.split` |
| Unset `PATH` falling back to `os.confstr("CS_PATH")` / `os.defpath` | `an_unset_search_path_falls_back_to_the_system_default`, on the Linux and macOS CI runners; the CI smoke job additionally runs `lerobot-info` with `PATH` removed from the environment |
| `os.access(F_OK \| X_OK)` rather than raw mode bits | `executability_is_decided_by_the_kernel_not_by_the_raw_mode_bits`, on the Linux and macOS CI runners. It self-skips when the **real** uid is 0, because `access(2)` accepts any execute bit for root and the two rules stop disagreeing; CI's `gates` job runs unprivileged |
| `PATHEXT` expansion, ordering, `.`-stripping, and the 3.12 rule that a non-`PATHEXT` extension is never tried bare | `cfg(windows)` unit and integration tests, run by the `gates` job on `windows-latest`, which is the only place they are compiled at all. Nothing cross-compiles or lints the `cfg(windows)` code from a Unix machine, so on a Unix developer checkout it is not type-checked and a change that breaks it fails first in CI |
| `NeedCurrentDirectoryForExePath` deciding whether `.` is searched | **Not directly asserted anywhere.** The Win32 call is made rather than reimplemented, so its `NoDefaultCurrentDirectoryInExePath` handling is whatever the OS does; the `gates` job runs the whole Windows suite a second time with that variable set, which proves neither branch breaks resolution but does not pin which branch was taken |
| `ntpath.normcase` case-folding | Rust's Unicode `to_lowercase`, not CPython's `LCMapStringEx(LOCALE_NAME_INVARIANT, ...)`. They agree on ASCII. A disagreement on a non-ASCII directory name can only cause a duplicated `PATH` entry to be searched twice; it cannot change which file is returned |
| `\\?\`-prefixed Windows device paths | Not modelled as their own drive syntax. They are not names that reach `which` in this port |

## Optional dependency and hardware boundaries

Upstream gates most of its surface behind extras. Rerobot crosses only the narrow
local dataset/training boundary listed below, using native Rust dependencies rather
than loading the Python extras; the remaining boundaries are not simulated.

| Upstream extra | Gates | Rerobot |
| --- | --- | --- |
| `dataset`, `training` | `datasets`, `torchcodec`, `pyarrow`, `wandb`, `accelerate` | partial: local or native Hub-downloaded state/action Parquet datasets, embedded PNG/JPEG image columns, ACT normalization/training, and ACT in-memory camera batches use Arrow/Parquet, the Rust image codecs and Candle. Hub streaming, video decoding, external image-file loading, image augmentation, W&B, distributed/mixed-precision training and evaluation are refused. |
| `hardware` | `pynput`, `pyserial`, `deepdiff` | hardware-gated |
| `feetech`, `dynamixel`, `damiao`, `robstride` | motor SDKs, `python-can` | hardware-gated |
| `intelrealsense`, `gamepad`, `hopejr`, `lekiwi`, `openarms`, `reachy2`, `rebot`, `unitree_g1`, `phone` | robot/teleop vendor SDKs | hardware-gated |
| `viz`, `dataset_viz` | `rerun-sdk`, `foxglove-sdk` | not ported |
| `pi`, `smolvla`, `groot`, `diffusion`, `wallx`, `molmoact2`, `sarm`, `xvla`, `eo1`, `evo1`, `fastwam`, `vla_jepa`, `lingbot_va`, `multi_task_dit`, `robometer`, `topreward`, `hilserl` | `transformers`, `diffusers`, `peft`, `scipy`, ... | not ported; model inference is never faked |
| `aloha`, `pusht`, `libero`, `metaworld` | Gymnasium simulation environments | not ported |
| `async`, `kinematics`, `annotations` | `grpcio`, `placo`, `openai` | not ported |

The training backend pins Candle to `0.9.1`: `0.9.2` uses a standard-library API
that is unavailable on the declared Rust 1.85 floor. Candle 0.9.1 transitively
contains both `gemm` 0.17 and 0.18 families; `cargo deny` reports those as duplicate
warnings. It also reaches archived macro crate `paste` 1.0.15. The corresponding
`RUSTSEC-2024-0436` entry is an unmaintained notice rather than a vulnerability and
has no safe upgrade, so `deny.toml` carries a documented temporary exception until
Candle can move without raising the MSRV. Arrow and Parquet are pinned to 56.2.0.

## Non-goals for this milestone

* No Python sidecar, FFI bridge, or subprocess shim for the implemented core.
* No video decoder, external image-file loader, image augmentation pipeline, Hub
  streaming/sync client, evaluation environment, or Python/accelerate-compatible full
  training resume. The implemented ACT path reads local state/action Parquet plus
  embedded PNG/JPEG camera columns and can additionally consume validated in-memory
  camera tensors through `Batch::with_images`; it loads weights and runs inference
  when validating a checkpoint round trip, and it resumes a local one-process
  checkpoint including model, AdamW, RNG and sampler state.
* No hardware access beyond invoking `ffmpeg -version` for `lerobot-info`.
