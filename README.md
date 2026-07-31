# Rerobot

A behaviour-compatible Rust port of [Hugging Face LeRobot][upstream], built one
tested vertical slice at a time.

Rerobot is **not** a wrapper, a binding, or a Python sidecar. Everything that
works is native Rust with parity pinned by tests derived from upstream's own
suite. Everything that does not work says so and exits non-zero.

[upstream]: https://github.com/huggingface/lerobot

| | |
| --- | --- |
| Upstream target | `lerobot` 0.6.1 (`f37be3edbee60f3a09a5183788b91eb19f0c07d1`) |
| Milestone | 2 of N — core utility slice, full CLI surface, and the first runnable training slice |
| Runnable executables | 2 of 18 (`lerobot-info`, and `lerobot-train` for the ACT state-only slice); the other 16 exist and fail explicitly |
| Tests | 795 integration/unit tests + 56 rustdoc tests, all passing. The ACT training slice is compared element by element against upstream running on PyTorch |
| Minimum Rust | 1.85 — the floor of the locked dependency tree, built and tested on that exact toolchain by the `msrv` CI job |

**Read [`docs/compatibility.md`](docs/compatibility.md) before using this.** It
states, per module family and per executable, exactly what is implemented,
partial, unimplemented, or hardware-gated. Nothing in this repository carries
`implemented` status yet, because a one-module port does not make a family
compatible, and labelling it so would misrepresent the work.

## Layout

```text
crates/
  rerobot-core     pure logic plus bounded local metadata IO; no hardware
  rerobot-compat   machine-readable inventory of the upstream surface + status
  rerobot-train    the ACT training slice: dataset, tensor model, AdamW, checkpoints
  rerobot-cli      the 18 lerobot-* executables
docs/
  compatibility.md the authoritative port boundary
  red-green.md     the RED -> GREEN record for every test cycle
tools/
  goldens/         the Python scripts that produced the committed fixtures
```

Four crates, not one per Python module. `rerobot-core` is the behaviour port,
`rerobot-compat` is compatibility *metadata* — a different axis, consumed by
both the CLI and the documentation tests — `rerobot-train` is the one part of the
port that needs a tensor runtime and a parquet reader, kept separate so that
`rerobot-core` stays dependency-light, and `rerobot-cli` is the deployment
surface.

## Install

```shell
git clone https://github.com/k1000dai/Rerobot
cd Rerobot
cargo install --path crates/rerobot-cli --locked
```

This installs all 18 executables under their upstream names.

## Try it

```shell
lerobot-info                 # runs for real

lerobot-train --help         # works, and lists exactly what it accepts

lerobot-eval --help          # works, and states that it is unimplemented
lerobot-eval; echo $?        # -> 2, with a single-line error on stderr
```

Unsupported commands honour a fixed contract: empty stdout, exactly one line on
stderr beginning `<name>: unsupported in Rerobot`, and exit status `2`. `--help`
and `--version` take precedence over every other argument, so help is always
reachable.

## What is ported

| Slice | Upstream source |
| --- | --- |
| `ActionInterpolator` | `lerobot/utils/action_interpolator.py` |
| `RolloutRingBuffer` | `lerobot/rollout/ring_buffer.py` |
| `DAggerPhase`, `DAggerEvents` (the event state machine only) | `lerobot/rollout/strategies/dagger.py` |
| `RenameObservationsProcessorStep`, `rename_stats` | `lerobot/processor/rename_processor.py` |
| `FeatureType`, `NormalizationMode`, `PolicyFeature`, `TransitionKey`, … | `lerobot/configs/types.py`, `lerobot/types.py` |
| `DatasetInfo`, the `meta/` path constants, and `load_info`/`write_info` for a local dataset | `lerobot/datasets/utils.py`, `lerobot/datasets/io_utils.py`, `lerobot/utils/io_utils.py` |
| `ACTConfig`, its validation/presets/delta indices, and byte-exact `config.json` read/write | `lerobot/policies/act/configuration_act.py`, `lerobot/configs/policies.py` |
| The Draccus value conversions a checkpoint is decoded through | `draccus/parsers/decoding.py` |
| `lerobot-info` and its parsing helpers | `lerobot/scripts/lerobot_info.py` |

Worked examples for every one of these live in
[`crates/rerobot-core/README.md`](crates/rerobot-core/README.md), which is
included as the crate's rustdoc — so each example is compiled and executed by
`cargo test --doc`, and cannot rot.

Three details worth knowing up front:

* `ActionInterpolator` is generic over scalar width, because the choice is
  observable — at `multiplier = 3` the first sub-step of a 0 → 1 move is
  `0.33333334` in `f32` and `0.3333333333333333` in `f64`. `f32` is the default
  and matches a `torch.float32` action tensor. It also broadcasts a length-1
  action against a length-N one in both directions, because the tensors upstream
  hands it do, and reports a non-broadcastable pair with PyTorch's own message.
* Upstream quirks are reproduced, not fixed: the ring buffer's byte accounting
  is not decremented when the *length* cap evicts a frame, `str` frame values
  cost code points rather than UTF-8 bytes, and `ffmpeg` banner parsing splits
  on single spaces so a double space shifts the parsed token. All of these are
  listed in [`docs/compatibility.md`](docs/compatibility.md).
* `lerobot-info` prints upstream's 15 keys, in upstream's order, and no others —
  so its output is comparable with what a Python user pastes into a bug report.
  Rerobot's own port status is in `<command> --help`, not in that report.
* The DAgger slice is the *event state machine* — phases, the four transitions,
  and the thread-safe request/consume hand-off. The DAgger strategy itself, the
  keyboard and pedal listeners, dataset recording, teleoperator handover and
  policy inference are not ported, and nothing about them is stubbed.

## Development

```shell
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo test --workspace --doc
cargo build --workspace --release
python3 tools/verify_packages.py

# The published MSRV, on the toolchain it names:
cargo +1.85.0 build --workspace --all-features --locked
cargo +1.85.0 test --workspace --all-targets --all-features --locked
```

Every one of those runs in CI, on Linux, macOS, and Windows for the first five.
`.github/workflows/ci.yml` pins every third-party action to a full commit SHA and
takes only `contents: read`.

Tests come first. See [`docs/red-green.md`](docs/red-green.md) for the record
and [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to add the next slice.

## License

Apache-2.0, matching upstream. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Rerobot is an independent port and is not affiliated with or endorsed by
Hugging Face, Inc.
