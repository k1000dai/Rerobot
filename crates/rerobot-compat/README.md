# rerobot-compat

Machine-readable inventory of the [Hugging Face LeRobot][upstream] surface and
[Rerobot]'s port status for each part of it.

[upstream]: https://github.com/huggingface/lerobot
[Rerobot]: https://github.com/k1000dai/Rerobot

This crate holds no behaviour. It exists so that one table drives the
`--help` output of every `lerobot-*` executable, the `lerobot-info` report, and
`docs/compatibility.md` — which is verified against this crate by a test, so the
documentation cannot quietly overstate what works.

Pinned to `lerobot` 0.6.1, commit `f37be3edbee60f3a09a5183788b91eb19f0c07d1`.

```rust
use rerobot_compat::{entry_point, module_family, Status, ENTRY_POINTS, MODULE_FAMILIES};

// All 18 upstream `[project.scripts]` console entry points, in upstream order.
assert_eq!(ENTRY_POINTS.len(), 18);
assert_eq!(ENTRY_POINTS[0].name, "lerobot-calibrate");

// Only `lerobot-info` is runnable at this milestone.
let runnable: Vec<&str> = ENTRY_POINTS
    .iter()
    .filter(|e| !e.status.is_unsupported())
    .map(|e| e.name)
    .collect();
assert_eq!(runnable, vec!["lerobot-info"]);

// Everything else must fail rather than silently succeed.
let train = entry_point("lerobot-train").unwrap();
assert_eq!(train.status, Status::Unimplemented);
assert_eq!(train.target, "lerobot.scripts.lerobot_train:main");
assert!(train.status.is_unsupported());

// Hardware is gated, never simulated.
assert_eq!(module_family("robots").unwrap().status, Status::HardwareGated);
assert_eq!(MODULE_FAMILIES.len(), 24);
```

## Status labels

| Label | Meaning |
| --- | --- |
| `implemented` | Behaviour parity demonstrated by tests for the whole unit. **Unused at this milestone.** |
| `partial` | Some observable behaviour ported and tested; scope stated per row. |
| `unimplemented` | Not ported; the executable fails with a stable error and a non-zero exit status. |
| `hardware-gated` | Needs physical hardware or a vendor SDK. Never faked. |

```rust
use rerobot_compat::Status;

assert_eq!(Status::HardwareGated.as_str(), "hardware-gated");
assert!(!Status::Partial.is_unsupported());
assert!(Status::Unimplemented.is_unsupported());
```

## License

Apache-2.0, matching upstream. See `LICENSE` and `NOTICE` in the repository
root.
