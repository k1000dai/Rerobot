# rerobot-core

Pure-Rust, behaviour-compatible ports of [Hugging Face LeRobot][upstream] core
utilities, policy configuration, and local `meta/info.json` metadata IO. No
network IO, tensor runtime, hardware, or Python sidecar.

[upstream]: https://github.com/huggingface/lerobot

Every item is a port of a specific upstream symbol at `lerobot` 0.6.1, commit
`f37be3edbee60f3a09a5183788b91eb19f0c07d1`, with parity pinned by tests derived
from upstream's own suite. Where upstream has surprising behaviour, the quirk is
reproduced and documented rather than "fixed" — callers can observe it. See
`docs/compatibility.md` in the repository root for the full boundary, including
the deliberate divergences.

Every example below is compiled and run as a doctest.

## `action_interpolator` — `lerobot/utils/action_interpolator.py`

Configurable Nx control rate by linear interpolation between consecutive policy
actions. The first action after construction or `reset` passes through
unchanged, because there is no previous action to blend from.

```rust
use rerobot_core::action_interpolator::ActionInterpolator;

let mut interp: ActionInterpolator<f32> = ActionInterpolator::new(3).unwrap();
assert!(interp.enabled());
assert_eq!(interp.get_control_interval(30.0).unwrap(), 1.0 / 90.0);

interp.add(&[0.0, 0.0]).unwrap();
assert_eq!(interp.get(), Some([0.0, 0.0].as_slice())); // passthrough
assert_eq!(interp.get(), None);

interp.add(&[3.0, 6.0]).unwrap();
assert_eq!(interp.get(), Some([1.0, 2.0].as_slice()));
assert_eq!(interp.get(), Some([2.0, 4.0].as_slice()));
assert_eq!(interp.get(), Some([3.0, 6.0].as_slice())); // exactly the target
assert_eq!(interp.get(), None);
```

Consecutive actions broadcast the way the 1-D tensors upstream passes it do: a
length-1 action against a length-N previous action, or the other way round,
produces N interpolated elements, and any other length mismatch is an
`InterpolatorError::NotBroadcastable` carrying PyTorch's own message. A failed
`add` clears the buffer and leaves the previous action untouched, because that is
the state upstream is left in when the tensor subtraction raises.

It is generic over scalar width, because the choice is observable: at
`multiplier = 3` the first sub-step of a 0 → 1 move differs between widths.
`f32` is the default and matches a `torch.float32` action tensor. The `1 / N`
weight is narrowed to the scalar width *before* the multiplication, which is what
PyTorch does with a Python `float` scalar against a typed tensor, and which is
observable: for some operands the two orders give different `f32` results.

```rust
use rerobot_core::action_interpolator::ActionInterpolator;

let mut narrow: ActionInterpolator<f32> = ActionInterpolator::new(3).unwrap();
let mut wide: ActionInterpolator<f64> = ActionInterpolator::new(3).unwrap();
narrow.add(&[0.0]).unwrap();
wide.add(&[0.0]).unwrap();
narrow.get();
wide.get();
narrow.add(&[1.0]).unwrap();
wide.add(&[1.0]).unwrap();

assert_eq!(narrow.get().unwrap()[0], 1.0f32 / 3.0);
assert_eq!(wide.get().unwrap()[0], 1.0f64 / 3.0);
```

Invalid configuration is rejected with upstream's wording:

```rust
use rerobot_core::action_interpolator::{ActionInterpolator, InterpolatorError};
use rerobot_core::BigInt;

let err = ActionInterpolator::<f32>::new(0).unwrap_err();
assert_eq!(err, InterpolatorError::InvalidMultiplier(BigInt::from(0)));
assert_eq!(err.to_string(), "multiplier must be >= 1, got 0");
```

The multiplier is the unbounded Python `int` upstream stores, not a machine
word: `__init__` checks only `multiplier < 1`, so `2**63` and `10**100` are
values it holds. Storage, the getter, `enabled` and `get_control_interval` are
exact at every magnitude. The two operations that cannot be — allocating the
interpolated buffer, and the `int`-to-float conversion inside `fps *
multiplier` — return an error naming the value rather than narrowing it.

```rust
use rerobot_core::action_interpolator::{ActionInterpolator, InterpolatorError};
use rerobot_core::BigInt;

let huge = BigInt::from(2).pow(128);
let mut interp: ActionInterpolator<f32> = ActionInterpolator::new(huge.clone()).unwrap();
assert_eq!(*interp.multiplier(), huge); // stored exactly, not truncated

interp.add(&[0.0]).unwrap(); // the first action never touches the multiplier
interp.get();
assert_eq!(
    interp.add(&[1.0]).unwrap_err(),
    InterpolatorError::BufferNotAllocatable { multiplier: huge },
);
```

## `ring_buffer` — `lerobot/rollout/ring_buffer.py`

Memory-bounded telemetry buffer, capped by both frame count and bytes.

```rust
use rerobot_core::ring_buffer::{Frame, FrameValue, RolloutRingBuffer};

let mut buffer = RolloutRingBuffer::new(2.0 / 30.0, 1024, 30.0).unwrap();
assert_eq!(buffer.max_frames(), 2); // int(max_seconds * fps), truncated

for step in 0..4 {
    let mut frame = Frame::new();
    frame.insert("step".to_string(), FrameValue::Int(step));
    buffer.append(frame);
}

let drained = buffer.drain();
assert_eq!(drained.len(), 2);
assert_eq!(drained[0]["step"], FrameValue::Int(2)); // oldest two evicted
```

Frame cost follows Python's type dispatch exactly — `bool` is an `int` and
costs 8 bytes, `str` costs code points rather than UTF-8 bytes, and an
all-unrecognised frame still costs 1 byte:

```rust
use rerobot_core::ring_buffer::{estimate_frame_bytes, Frame, FrameValue};

let mut frame = Frame::new();
frame.insert("flag".to_string(), FrameValue::Int(1)); // Python bool -> int -> 8
frame.insert("task".to_string(), FrameValue::Str("héllo".to_string())); // 5, not 6
frame.insert("obs".to_string(), FrameValue::Tensor { numel: 4, element_size: 4 });
assert_eq!(estimate_frame_bytes(&frame), 8 + 5 + 16);

assert_eq!(estimate_frame_bytes(&Frame::new()), 1); // max(total, 1)
```

## `rollout::dagger` — `lerobot/rollout/strategies/dagger.py`

The DAgger event state machine, and only that: the hand-off between the
input-device threads that request phase changes and the main loop that applies
them. `DAggerStrategy` itself, the keyboard and pedal listeners, the
teleoperator handover, dataset recording and policy inference are **not**
ported.

Requests are validated against the phase when they are made and again when they
are consumed, so a phase moved in between cannot be driven into an impossible
state:

```rust
use rerobot_core::rollout::dagger::{
    DAggerEvents, DAggerPhase, CORRECTION_EVENT, PAUSE_RESUME_EVENT,
};

let events = DAggerEvents::new();
assert_eq!(events.phase(), DAggerPhase::Autonomous);

events.request_transition(PAUSE_RESUME_EVENT);
assert_eq!(
    events.consume_transition(),
    Some((DAggerPhase::Autonomous, DAggerPhase::Paused))
);

// `correction` is valid from PAUSED. A later misspelled (invalid) request does
// not clear that valid pending request.
events.request_transition(CORRECTION_EVENT);
events.request_transition("pause_resume_but_misspelled"); // ignored
assert_eq!(
    events.consume_transition(),
    Some((DAggerPhase::Paused, DAggerPhase::Correcting))
);
assert_eq!(events.consume_transition(), None); // consumed exactly once
```

`reset` restores a fresh session — except for `stop_recording`, which upstream
deliberately leaves alone, so a session stopped with ESC stays stopped:

```rust
use rerobot_core::rollout::dagger::{DAggerEvents, DAggerPhase, PAUSE_RESUME_EVENT};

let events = DAggerEvents::new();
events.set_phase(DAggerPhase::Correcting);
events.request_transition("correction");
events.upload_requested.set();
events.stop_recording.set();

events.reset();

assert_eq!(events.phase(), DAggerPhase::Autonomous);
assert_eq!(events.consume_transition(), None); // the pending request is gone
assert!(!events.upload_requested.is_set());
assert!(events.stop_recording.is_set()); // upstream does not clear this one
```

Every method takes `&self` and is safe to call from several threads, because
the phase and the pending request live behind one lock and are read and written
together. `DAggerPhase` carries upstream's member values (`autonomous`,
`paused`, `correcting`) and by-value lookup, but no `serde` support: upstream's
enum is a plain `enum.Enum` with no JSON wire form to be compatible with.

## `processor::rename` — `lerobot/processor/rename_processor.py`

```rust
use rerobot_core::processor::rename::{Observation, RenameObservationsProcessorStep};
use serde_json::json;

// The map is applied once per key; it does not cascade a -> b -> c.
let step = RenameObservationsProcessorStep::new([("a", "b"), ("b", "c")]);

let mut observation = Observation::new();
observation.insert("a".to_string(), json!(1));
observation.insert("b".to_string(), json!(2));
observation.insert("x".to_string(), json!(3));

let renamed = step.observation(&observation);
assert_eq!(renamed["b"], json!(1));
assert_eq!(renamed["c"], json!(2));
assert_eq!(renamed["x"], json!(3)); // unmapped keys are untouched

assert_eq!(step.get_config(), json!({"rename_map": {"a": "b", "b": "c"}}));
```

`rename_stats` keeps the same key ordering and turns `None` sub-dicts into empty
ones:

```rust
use indexmap::IndexMap;
use rerobot_core::processor::rename::{rename_stats, Stats};
use serde_json::json;

let mut stats = Stats::new();
stats.insert("observation.state".to_string(), Some(IndexMap::from([
    ("mean".to_string(), json!([0.0])),
])));
stats.insert("action".to_string(), None);

let mut map = IndexMap::new();
map.insert("observation.state".to_string(), "observation.robot_state".to_string());

let renamed = rename_stats(&stats, &map);
assert_eq!(renamed.keys().collect::<Vec<_>>(), vec!["observation.robot_state", "action"]);
assert!(renamed["action"].as_ref().unwrap().is_empty());
```

## `processor::newline_task` — `lerobot/processor/newline_task_processor.py`

`NewLineTaskProcessorStep` makes the `task` prompt end with a newline, which is
what a PaliGemma-style tokenizer expects. Only the `task` entry is rewritten;
every other key keeps its value and its position.

```rust
use rerobot_core::processor::newline_task::{NewLineTaskProcessorStep, REGISTRY_NAME};
use rerobot_core::processor::ComplementaryData;
use serde_json::json;

// This preserves upstream's spelling. Registry lookup/config reconstruction is
// part of the not-yet-ported processor pipeline runtime.
assert_eq!(REGISTRY_NAME, "smolvla_new_line_processor");

let step = NewLineTaskProcessorStep;
let mut data = ComplementaryData::new();
data.insert("index".to_string(), json!(0));
data.insert("task".to_string(), json!(["task1", "task2\n"]));

let out = step.complementary_data(&data);
assert_eq!(out["task"], json!(["task1\n", "task2\n"]));
assert_eq!(out.keys().collect::<Vec<_>>(), vec!["index", "task"]);
```

The list branch is all-or-nothing, and `str.endswith("\n")` is exact — a value
ending in `\r`, `\u{2028}` or `\u{0085}` does not count as already terminated:

```rust
use rerobot_core::processor::newline_task::NewLineTaskProcessorStep;
use rerobot_core::processor::ComplementaryData;
use serde_json::json;

let step = NewLineTaskProcessorStep;
let mut data = ComplementaryData::new();

data.insert("task".to_string(), json!(["task1", 1])); // not all strings
assert_eq!(step.complementary_data(&data)["task"], json!(["task1", 1]));

data.insert("task".to_string(), json!([])); // `all(...)` of nothing is true
assert_eq!(step.complementary_data(&data)["task"], json!([]));

data.insert("task".to_string(), json!("")); // "" does not end with "\n"
assert_eq!(step.complementary_data(&data)["task"], json!("\n"));

data.insert("task".to_string(), json!("pick\r\n")); // already terminated
assert_eq!(step.complementary_data(&data)["task"], json!("pick\r\n"));

data.insert("task".to_string(), json!(null)); // null is left alone
assert_eq!(step.complementary_data(&data)["task"], json!(null));
```

The step declares no configuration and no state, so `get_config` and
`state_dict` are empty while `load_state_dict` and `reset` are no-ops.
`transform_features` returns an equal owned clone; unlike upstream, it does not
alias its input. The processor registry and pipeline-config reconstruction are
not part of this slice.

## `dataset` — `lerobot/datasets/utils.py`, `lerobot/datasets/io_utils.py`

The `meta/info.json` slice, and only that: the path constants, the
`DatasetInfo` dataclass, and reading and writing that one file from a local
directory. `LeRobotDatasetMetadata`, tasks, stats, episodes, parquet, video and
the Hub are **not** ported and are not stubbed.

```rust
use indexmap::IndexMap;
use rerobot_core::dataset::info::{DatasetInfo, Feature};
use rerobot_core::dataset::json::{dumps_pretty, JsonLike};
use rerobot_core::dataset::{DEFAULT_DATA_PATH, INFO_PATH};
use num_bigint::BigInt;

assert_eq!(INFO_PATH, "meta/info.json");

let mut state = Feature::new();
state.insert("dtype".to_string(), JsonLike::Str("float32".to_string()));
state.insert("shape".to_string(), JsonLike::Array(vec![JsonLike::Int(BigInt::from(6))]));

let info = DatasetInfo::new("v3.0", 30, IndexMap::from([
    ("observation.state".to_string(), state),
])).unwrap();

assert_eq!(info.chunks_size, BigInt::from(1000)); // the eleven defaults
assert_eq!(info.data_path, DEFAULT_DATA_PATH);
assert_eq!(info.tools, None);

// `__post_init__` turned the shape into a tuple, so it no longer equals the
// list it came from — exactly as it does not in Python.
assert_eq!(
    info.features["observation.state"]["shape"],
    JsonLike::Tuple(vec![JsonLike::Int(BigInt::from(6))]),
);

// ... and `to_dict` turns it back into a list, and drops the unset `tools`.
let dict = info.to_dict();
assert!(!dict.contains_key("tools"));
assert!(dumps_pretty(&JsonLike::Object(dict)).contains("\"shape\": [\n                6\n            ]"));
```

Upstream's four positivity checks keep their exact wording, and every integer
field is a `BigInt`, because a Python `int` is unbounded and nothing upstream
clamps one:

```rust
use rerobot_core::dataset::info::{DatasetInfo, DatasetInfoError};
use indexmap::IndexMap;
use num_bigint::BigInt;

let err = DatasetInfo::new("v3.0", 0, IndexMap::new()).unwrap_err();
assert_eq!(err.to_string(), "fps must be positive, got 0");
assert_eq!(err, DatasetInfoError::NotPositive { field: "fps", value: BigInt::from(0) });

// Assignment does not re-run `__post_init__` — nor does it upstream — so the
// check is available on its own.
let mut info = DatasetInfo::new("v3.0", 30, IndexMap::new()).unwrap();
info.chunks_size = BigInt::from(2).pow(200); // held exactly, not narrowed
info.post_init().unwrap();
assert_eq!(info.chunks_size, BigInt::from(2).pow(200));
```

Unknown top-level keys are ignored for forward compatibility, as upstream
ignores them. `DatasetInfo::from_dict` emits the upstream-equivalent warning
through the `log` facade; callers can additionally inspect and render the same
sorted fields explicitly:

```rust
use rerobot_core::dataset::info::{unknown_fields_warning, DatasetInfo};
use rerobot_core::dataset::json::loads;

let raw = loads(r#"{"codebase_version": "v2.1", "fps": 30, "features": {}, "total_videos": 7}"#)
    .unwrap()
    .as_object()
    .unwrap()
    .clone();

let info = DatasetInfo::from_dict(&raw).unwrap(); // the extra key is ignored
assert_eq!(info.codebase_version, "v2.1");

let unknown = DatasetInfo::unknown_fields(&raw); // sorted, by code point
assert_eq!(unknown, vec!["total_videos"]);
assert_eq!(
    unknown_fields_warning(&unknown).unwrap(),
    "Unknown fields in DatasetInfo: ['total_videos']. These will be ignored.",
);
```

`dataset::json` is a port of the `JsonLike` alias `lerobot/utils/io_utils.py`
declares, and of CPython 3.12's reader and writer for it — not a wrapper around
a JSON library. It has to be: a Python `int` is unbounded, `json.load` accepts
`NaN`/`Infinity`/`-Infinity` by default and `json.dump` emits them, and a
`tuple` is not a `list`.

```rust
use rerobot_core::dataset::json::{dumps, loads, python_float_repr, JsonLike};
use num_bigint::BigInt;
use std::str::FromStr;

// Integers keep every digit without fixed-width narrowing; parser resource
// budgets are documented in the compatibility ledger.
let huge = "340282366920938463463374607431768211457";
assert_eq!(loads(huge).unwrap(), JsonLike::Int(BigInt::from_str(huge).unwrap()));
assert_eq!(dumps(&loads(huge).unwrap()), huge);

// The three non-finite tokens survive a round trip; the lowercase spellings
// are not values, exactly as in CPython.
assert_eq!(loads("Infinity").unwrap(), JsonLike::Float(f64::INFINITY));
assert_eq!(dumps(&JsonLike::Float(f64::NEG_INFINITY)), "-Infinity");
assert_eq!(loads("infinity").unwrap_err().msg, "Expecting value");

// Floats are written by `float.__repr__`, not by Rust's `{}`.
assert_eq!(python_float_repr(30.0), "30.0");
assert_eq!(python_float_repr(1e15), "1000000000000000.0");
assert_eq!(python_float_repr(1e16), "1e+16"); // repr switches over here

// A malformed document carries CPython's own message and coordinates, in code
// points rather than bytes.
let err = loads(r#"{"a":01}"#).unwrap_err();
assert_eq!(err.to_string(), "Expecting ',' delimiter: line 1 column 7 (char 6)");
```

Writing goes through `json.dump(..., indent=4, ensure_ascii=False)`: four-space
indentation, non-ASCII written literally, and no trailing newline.

```rust
use rerobot_core::dataset::io::{info_path, load_info, write_info};
use rerobot_core::dataset::info::DatasetInfo;
use indexmap::IndexMap;

let root = std::env::temp_dir().join(format!("rerobot-readme-{}", std::process::id()));
let _ = std::fs::remove_dir_all(&root);

let mut info = DatasetInfo::new("v3.0", 30, IndexMap::new()).unwrap();
info.robot_type = Some("bras-à-café".to_string());

write_info(&info, &root).unwrap(); // creates `meta/` on the way
assert_eq!(info_path(&root), root.join("meta").join("info.json"));

let written = std::fs::read_to_string(info_path(&root)).unwrap();
assert!(written.starts_with("{\n    \"codebase_version\": \"v3.0\","));
assert!(written.contains("\"bras-à-café\"")); // not escaped
assert!(!written.ends_with('\n'));            // `json.dump` writes none

assert_eq!(load_info(&root).unwrap(), info);
std::fs::remove_dir_all(&root).unwrap();
```

Three boundaries are worth knowing before you rely on this, and all three are
spelled out in [the compatibility ledger](https://github.com/k1000dai/Rerobot/blob/master/docs/compatibility.md):
`__post_init__` mutates the caller's own feature dicts in Python and cannot
here; the typed fields reject values upstream's unchecked dataclass would
accept, including the `bool` that Python treats as an `int`; and the file is
always written as UTF-8 with LF, where CPython's `open(fpath, "w")` follows the
locale encoding and translates newlines on Windows.

## `policy::act` — `lerobot/policies/act/configuration_act.py`

The concrete ACT policy configuration is checkpoint-wire compatible and keeps
every Python integer field arbitrary precision. Validation preserves upstream's
order, exception category, and exact message; it does not pretend that the ACT
tensor model is available yet.

```rust
use rerobot_core::policy::act::{ActConfig, ActConfigErrorKind};
use rerobot_core::BigInt;

let mut config = ActConfig::default();
assert_eq!(config.chunk_size, BigInt::from(100));
assert_eq!(
    config.action_delta_indices().take(3).collect::<Vec<_>>(),
    vec![BigInt::from(0), BigInt::from(1), BigInt::from(2)]
);

config.vision_backbone = "vit".into();
let error = config.validate().unwrap_err();
assert_eq!(error.kind(), ActConfigErrorKind::Value);
assert_eq!(
    error.to_string(),
    "`vision_backbone` must be one of the ResNet variants. Got vit."
);
```

`action_delta_indices` is lazy rather than eagerly building Python's list. That
is the explicit resource boundary which lets a syntactically valid, enormous
`chunk_size` remain an exact `BigInt` without allocating until the machine is
exhausted. Ordinary values yield the same ordered indices. JSON decoding
requires the registry tag `"type":"act"`, distinguishes null from an absent
defaulted field, rejects unknown fields, and round-trips negative and
thousand-digit integers as bare JSON numbers. It also reproduces Draccus'
checkpoint coercion: JSON strings accepted by Python `int()` and booleans become
integers; float fields likewise accept numeric strings and booleans, and bool
fields accept Draccus' exact lowercase `"true"` / `"false"` strings. Non-finite
float output fails explicitly instead of silently becoming JSON null; see the
compatibility ledger for that narrower wire boundary.

The upstream dilation field is uniquely declared `int = False`: a fresh config
stores and writes boolean `false`, while checkpoint decoding converts it to
integer `0` and accepts arbitrary Python integers. `PythonIntBool` retains both
forms instead of narrowing that field to a Rust `bool`.

## `types` — `lerobot/configs/types.py`, `lerobot/types.py`

The `str`-backed enums, with upstream's exact wire values and case-sensitive
lookup.

```rust
use rerobot_core::types::{BigInt, FeatureType, PolicyFeature, TransitionKey};
use std::str::FromStr;

assert_eq!(FeatureType::Visual.as_str(), "VISUAL");
assert_eq!(TransitionKey::ComplementaryData.as_str(), "complementary_data");
assert!(FeatureType::from_str("visual").is_err()); // exact, like a Python str enum

// `PolicyFeature` is upstream's two-field dataclass and nothing more.
let feature = PolicyFeature::new(FeatureType::Visual, [3, 96, 96]);
assert_eq!(feature.shape, vec![BigInt::from(3), BigInt::from(96), BigInt::from(96)]);
assert_eq!(
    serde_json::to_string(&feature).unwrap(),
    r#"{"type":"VISUAL","shape":[3,96,96]}"#
);
```

`shape` mirrors upstream's `tuple[int, ...]`, so a dimension is signed and
unbounded rather than a `usize`: `-1` is the ordinary dynamic axis. For values
within CPython 3.12's default 4,300-digit conversion guard, the JSON wire form
is the same bare decimal integer `json.dumps` writes. Serde also accepts longer
tokens; Python requires its integer-string limit to be increased or disabled
for those. Neither representation depends on the target's `usize` width.

```rust
use rerobot_core::types::{BigInt, FeatureType, PolicyFeature};
use std::str::FromStr;

let json = r#"{"type":"STATE","shape":[-1,340282366920938463463374607431768211457]}"#;
let feature: PolicyFeature = serde_json::from_str(json).unwrap();
assert_eq!(feature.shape[0], BigInt::from(-1));
assert_eq!(
    feature.shape[1],
    BigInt::from_str("340282366920938463463374607431768211457").unwrap()
);
assert_eq!(serde_json::to_string(&feature).unwrap(), json); // exact round trip
```

## `sysinfo` — `lerobot/scripts/lerobot_info.py`

The pure parts behind the `lerobot-info` executable. Upstream's parsing rules
are ported exactly, including the fact that Python's `str.split(" ")` keeps
empty fields:

```rust
use rerobot_core::sysinfo::{format_dict_for_markdown, parse_ffmpeg_version, FFMPEG_PARSE_FAILED};

assert_eq!(parse_ffmpeg_version("ffmpeg version 7.1 Copyright (c) 2000"), "7.1");
assert_eq!(parse_ffmpeg_version("ffmpeg  version 7.1"), "version"); // double space
assert_eq!(parse_ffmpeg_version(""), FFMPEG_PARSE_FAILED);

assert_eq!(
    format_dict_for_markdown([("Platform", "macos-aarch64"), ("FFmpeg version", "7.1")]),
    "- Platform: macos-aarch64\n- FFmpeg version: 7.1"
);
```

## License

Apache-2.0, matching upstream. See `LICENSE` and `NOTICE` in the repository
root. Rerobot is an independent port and is not affiliated with or endorsed by
Hugging Face, Inc.
