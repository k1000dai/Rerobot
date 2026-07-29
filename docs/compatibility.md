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
| `lerobot-train` | unimplemented | `lerobot.scripts.lerobot_train:main` | Train a policy. | Needs the PyTorch training stack; out of scope for a pure-Rust milestone. |
| `lerobot-train-tokenizer` | unimplemented | `lerobot.scripts.lerobot_train_tokenizer:main` | Train the FAST action tokenizer. | Needs LeRobotDataset loading and the tokenizer training stack. |
| `lerobot-dataset-viz` | unimplemented | `lerobot.scripts.lerobot_dataset_viz:main` | Visualize every frame of a dataset episode. | Needs the dataset reader plus a Rerun/Foxglove viewer bridge. |
| `lerobot-info` | partial | `lerobot.scripts.lerobot_info:main` | Print a markdown summary of the system configuration. | Ported and runnable. Keys that report Python package versions cannot apply to a Rust build and are reported as not ported rather than invented. |
| `lerobot-find-joint-limits` | hardware-gated | `lerobot.scripts.lerobot_find_joint_limits:main` | Discover joint limits and end-effector bounds via teleoperation. | Drives physical hardware through a vendor SDK; nothing is faked, so it stays hardware-gated until a real driver layer exists. |
| `lerobot-imgtransform-viz` | unimplemented | `lerobot.scripts.lerobot_imgtransform_viz:main` | Render examples of the configured image transforms. | Needs the image transform pipeline and dataset loading. |
| `lerobot-edit-dataset` | unimplemented | `lerobot.scripts.lerobot_edit_dataset:main` | Delete, split, merge, and otherwise edit LeRobot datasets. | Needs the LeRobotDataset on-disk format (parquet chunks, video shards). |
| `lerobot-setup-can` | hardware-gated | `lerobot.scripts.lerobot_setup_can:main` | Set up and debug CAN interfaces for Damiao motors. | Drives physical hardware through a vendor SDK; nothing is faked, so it stays hardware-gated until a real driver layer exists. |
| `lerobot-annotate` | unimplemented | `lerobot.scripts.lerobot_annotate:main` | Populate language annotation columns on a dataset. | Needs dataset editing plus an OpenAI-compatible inference backend. |
| `lerobot-rollout` | unimplemented | `lerobot.scripts.lerobot_rollout:main` | Run a trained policy on a real robot with pluggable strategies. | Needs policy inference and robot drivers. Its rollout ring buffer is ported and tested, but the command itself is not runnable. |

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
| `lerobot/configs` | partial | 11 | `configs.types` str-enums and `PolicyFeature` are ported and tested. Draccus-based config parsing, policy configs, and train/eval configs are not. |
| `lerobot/data_processing` | unimplemented | 3 | Dataset-level batch processing helpers. |
| `lerobot/datasets` | unimplemented | 22 | LeRobotDataset format, video decoding, and Hub sync. |
| `lerobot/envs` | unimplemented | 10 | Gymnasium environment factories. |
| `lerobot/jobs` | unimplemented | 4 | Hugging Face Jobs launchers. |
| `lerobot/model` | unimplemented | 2 | Shared model plumbing. |
| `lerobot/motors` | hardware-gated | 16 | Feetech / Dynamixel / CAN motor buses. |
| `lerobot/optim` | unimplemented | 4 | Optimizer and LR-scheduler configs bound to PyTorch. |
| `lerobot/policies` | unimplemented | 128 | All policy architectures. Requires model inference; never faked. |
| `lerobot/processor` | partial | 19 | `rename_processor` (step + `rename_stats`) and the value transform/stateless lifecycle of `newline_task_processor.NewLineTaskProcessorStep` are ported and tested. Python aliasing, registry/config reconstruction, the pipeline runtime, normalization, tokenizer, and device steps are not. |
| `lerobot/rewards` | unimplemented | 24 | Reward classifiers and success detectors. |
| `lerobot/rl` | unimplemented | 21 | HIL-SERL actor/learner infrastructure. |
| `lerobot/robots` | hardware-gated | 53 | Per-robot drivers (SO-100/101, LeKiwi, Reachy2, Unitree, ...). |
| `lerobot/rollout` | partial | 18 | `ring_buffer.RolloutRingBuffer` is ported and tested, including its byte-accounting quirks. Rollout strategies and the policy loop are not. |
| `lerobot/scripts` | partial | 20 | `lerobot_info` is ported and runnable. The other 17 entry points exist only as executables that fail with a stable unsupported error. |
| `lerobot/teleoperators` | hardware-gated | 59 | Leader arms, gamepads, keyboards, phone teleop. |
| `lerobot/templates` | unimplemented | 0 | Non-Python scaffolding templates; nothing to port yet. |
| `lerobot/transforms` | unimplemented | 2 | Image augmentation transforms built on torchvision. |
| `lerobot/transport` | unimplemented | 4 | gRPC transport for async inference. |
| `lerobot/utils` | partial | 25 | `action_interpolator` is ported and tested. Random/IO/hub/train utilities are not. |

## What is actually ported

Each item below is a port of specific upstream source, pinned by tests derived
from upstream's own test-suite and from direct execution of the upstream Python.

| Rerobot item | Upstream source | Tests |
| --- | --- | --- |
| `rerobot_core::types` | `lerobot/configs/types.py`, `lerobot/types.py` (`TransitionKey`) | `crates/rerobot-core/tests/types.rs` |
| `rerobot_core::action_interpolator` | `lerobot/utils/action_interpolator.py` | `crates/rerobot-core/tests/action_interpolator.rs` |
| `rerobot_core::ring_buffer` | `lerobot/rollout/ring_buffer.py` | `crates/rerobot-core/tests/ring_buffer.rs` |
| `rerobot_core::byte_count` | no upstream counterpart — it is the unbounded integer the byte accounting needs, standing in for a Python `int` | `crates/rerobot-core/tests/byte_count.rs` |
| `rerobot_core::processor::rename` | `lerobot/processor/rename_processor.py` | `crates/rerobot-core/tests/rename_processor.rs` |
| `rerobot_core::processor::newline_task` | `lerobot/processor/newline_task_processor.py` (`NewLineTaskProcessorStep` value transform, stateless lifecycle, and registry-name spelling only) | `crates/rerobot-core/tests/newline_task_processor.rs` |
| `rerobot_core::sysinfo` | `lerobot/scripts/lerobot_info.py` (pure parts) | `crates/rerobot-core/tests/sysinfo.rs` |
| `rerobot_cli::which` | `shutil.which`, as called by `get_ffmpeg_version` | `crates/rerobot-cli/tests/which.rs` |
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
| `ActionInterpolator.multiplier` is an unbounded Python `int` | `num_bigint::BigInt`; storage, the getter and `enabled` are exact at every magnitude | Not a divergence in the stored domain. The two operations that cannot cover that domain say so: see the two rows below. |
| Building the interpolated sequence for a huge multiplier ends in `MemoryError` after CPython grinds through the `list` | `ActionInterpolator::add` returns `Err(InterpolatorError::BufferNotAllocatable)` up front | The sequence is `multiplier` elements of a Rust `Vec`, so a multiplier that does not fit a `usize`, or whose slots cannot be reserved, is refused rather than truncated to a step count that does fit. Below that boundary the buffer is genuinely built, and an allocator that fails part-way aborts the process where CPython would raise `MemoryError`. |
| `fps * self.multiplier` raises `OverflowError: int too large to convert to float` for a multiplier outside the `f64` range | `get_control_interval` returns `Err(InterpolatorError::MultiplierNotFloatRepresentable)`, carrying that exact message | Rust has no exceptions, and the alternative is worse than an error: dividing by an infinity would silently report a control interval of `0.0`. Below the boundary the conversion is the nearest `f64`, taken through the decimal digits and Rust's correctly-rounded float parser. |
| `_estimate_frame_bytes` dispatches on Python runtime types | Callers tag values with `FrameValue` | Static typing. The per-variant cost model is a byte-for-byte port. |
| `NewLineTaskProcessorStep.complementary_data` returns the *same* `dict` object when `task` is absent or `None`, and otherwise makes a shallow copy whose untouched nested values remain shared | Always returns an independently owned `IndexMap` and deep-cloned `serde_json::Value`s | Deliberate ownership boundary. Values and key order match, but mutation-visible aliasing does not: mutating a nested Python value through either map can affect the other, while Rust's result is independent. The input is never modified by the Rust step. |
| `NewLineTaskProcessorStep.transform_features` returns the identical `features` object it was handed | Returns an independent clone that compares equal and keeps its stage order | Value/ordering identity is ported; object identity and mutation sharing are not. This is more than a Python `is` difference: later nested mutation is observable. |
| `complementary_data` takes any Python object as the `task` value and dispatches with `isinstance` | Takes a `serde_json::Value` | Static JSON-domain boundary. Strings, null, booleans, objects, representable finite numbers, and arrays built from them follow the matching Python branches. Oversized Python integers, NaN/infinities, tuples, bytes, arbitrary objects and ill-formed Unicode are outside the domain rather than approximated; see below. |
| Upstream registers the step as `smolvla_new_line_processor` and reconstructs it through the processor registry | Exposes that exact spelling as `REGISTRY_NAME` only | Registry lookup, pipeline serialization and config reconstruction are not yet ported. The constant prevents spelling drift but is not a claim that old serialized pipelines can already be loaded. |
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
* `NewLineTaskProcessorStep`: the list branch is all-or-nothing. One non-string
  element leaves the whole list untouched, so `["a", 1]` keeps `"a"` without its
  newline; and because `all(...)` over an empty list is `True`, an empty list
  takes the list branch and is rebuilt as an empty list.
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

Upstream gates most of its surface behind extras. None of these boundaries are
crossed by Rerobot at this milestone, and none are simulated.

| Upstream extra | Gates | Rerobot |
| --- | --- | --- |
| `dataset`, `training` | `datasets`, `torchcodec`, `pyarrow`, `wandb`, `accelerate` | not ported |
| `hardware` | `pynput`, `pyserial`, `deepdiff` | hardware-gated |
| `feetech`, `dynamixel`, `damiao`, `robstride` | motor SDKs, `python-can` | hardware-gated |
| `intelrealsense`, `gamepad`, `hopejr`, `lekiwi`, `openarms`, `reachy2`, `rebot`, `unitree_g1`, `phone` | robot/teleop vendor SDKs | hardware-gated |
| `viz`, `dataset_viz` | `rerun-sdk`, `foxglove-sdk` | not ported |
| `pi`, `smolvla`, `groot`, `diffusion`, `wallx`, `molmoact2`, `sarm`, `xvla`, `eo1`, `evo1`, `fastwam`, `vla_jepa`, `lingbot_va`, `multi_task_dit`, `robometer`, `topreward`, `hilserl` | `transformers`, `diffusers`, `peft`, `scipy`, ... | not ported; model inference is never faked |
| `aloha`, `pusht`, `libero`, `metaworld` | Gymnasium simulation environments | not ported |
| `async`, `kinematics`, `annotations` | `grpcio`, `placo`, `openai` | not ported |

## Non-goals for this milestone

* No Python sidecar, FFI bridge, or subprocess shim for the implemented core.
* No model inference, no weight loading, no dataset format reading.
* No hardware access beyond invoking `ffmpeg -version` for `lerobot-info`.
