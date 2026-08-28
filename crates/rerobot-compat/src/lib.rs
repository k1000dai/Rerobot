#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

use std::fmt;

/// Upstream distribution name.
pub const UPSTREAM_PACKAGE: &str = "lerobot";

/// Upstream distribution version this inventory was taken from.
pub const UPSTREAM_VERSION: &str = "0.6.1";

/// Upstream git commit this inventory was taken from.
pub const UPSTREAM_COMMIT: &str = "f37be3edbee60f3a09a5183788b91eb19f0c07d1";

/// Upstream license identifier.
pub const UPSTREAM_LICENSE: &str = "Apache-2.0";

/// Upstream source URL.
pub const UPSTREAM_REPOSITORY: &str = "https://github.com/huggingface/lerobot";

/// Port status of one upstream unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Status {
    /// Behaviour parity demonstrated by tests in this workspace.
    Implemented,
    /// Some observable behaviour ported; the rest is absent.
    Partial,
    /// Not ported. Invoking it fails with a stable unsupported error.
    Unimplemented,
    /// Requires physical hardware or a vendor SDK; out of scope for a
    /// pure-Rust milestone and never faked.
    HardwareGated,
}

impl Status {
    /// Stable lowercase slug used in docs and CLI output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Implemented => "implemented",
            Self::Partial => "partial",
            Self::Unimplemented => "unimplemented",
            Self::HardwareGated => "hardware-gated",
        }
    }

    /// Whether invoking this unit must fail rather than silently succeed.
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unimplemented | Self::HardwareGated)
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One upstream `[project.scripts]` console entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryPoint {
    /// Executable name, byte-identical to upstream.
    pub name: &'static str,
    /// Upstream `module:function` target.
    pub target: &'static str,
    /// Port status.
    pub status: Status,
    /// One-line summary of what upstream does.
    pub summary: &'static str,
    /// Why the status is what it is.
    pub note: &'static str,
}

/// One upstream module family under `src/lerobot/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModuleFamily {
    /// Package name under `src/lerobot/`.
    pub name: &'static str,
    /// Port status.
    pub status: Status,
    /// Number of Python modules upstream ships in this family.
    pub upstream_modules: u32,
    /// Why the status is what it is.
    pub note: &'static str,
}

const HW: &str = "Drives physical hardware through a vendor SDK; nothing is faked, so it stays \
    hardware-gated until a real driver layer exists.";

/// All 18 upstream console entry points, in upstream declaration order.
pub static ENTRY_POINTS: &[EntryPoint] = &[
    EntryPoint {
        name: "lerobot-calibrate",
        target: "lerobot.scripts.lerobot_calibrate:main",
        status: Status::HardwareGated,
        summary: "Recalibrate a robot or teleoperator device.",
        note: HW,
    },
    EntryPoint {
        name: "lerobot-find-cameras",
        target: "lerobot.scripts.lerobot_find_cameras:main",
        status: Status::HardwareGated,
        summary: "List the camera devices available on the system.",
        note: HW,
    },
    EntryPoint {
        name: "lerobot-find-port",
        target: "lerobot.scripts.lerobot_find_port:main",
        status: Status::HardwareGated,
        summary: "Find the USB port a MotorsBus is attached to.",
        note: HW,
    },
    EntryPoint {
        name: "lerobot-record",
        target: "lerobot.scripts.lerobot_record:main",
        status: Status::HardwareGated,
        summary: "Record a dataset via teleoperation.",
        note: HW,
    },
    EntryPoint {
        name: "lerobot-replay",
        target: "lerobot.scripts.lerobot_replay:main",
        status: Status::HardwareGated,
        summary: "Replay a recorded episode's actions on a robot.",
        note: HW,
    },
    EntryPoint {
        name: "lerobot-setup-motors",
        target: "lerobot.scripts.lerobot_setup_motors:main",
        status: Status::HardwareGated,
        summary: "Set motor ids and baudrate on a motor bus.",
        note: HW,
    },
    EntryPoint {
        name: "lerobot-teleoperate",
        target: "lerobot.scripts.lerobot_teleoperate:main",
        status: Status::HardwareGated,
        summary: "Drive a robot from a teleoperator.",
        note: HW,
    },
    EntryPoint {
        name: "lerobot-eval",
        target: "lerobot.scripts.lerobot_eval:main",
        status: Status::Unimplemented,
        summary: "Evaluate a policy by running environment rollouts.",
        note: "Needs policy inference and a Gymnasium environment; neither is ported, and \
               fabricating metrics would be worse than failing.",
    },
    EntryPoint {
        name: "lerobot-train",
        target: "lerobot.scripts.lerobot_train:main",
        status: Status::Partial,
        summary: "Train a policy.",
        note: "Runnable for one vertical slice: the ACT policy on a local or native Hub-downloaded LeRobot v3.0 snapshot with state/action columns and embedded PNG/JPEG camera columns (including dataset-provided per-camera normalization when `use_imagenet_stats=false`), or through the in-memory camera batch API. A local `--policy.path=DIR` warm-start loads the ACT `config.json`, weights, and four saved processor artifacts, then applies supported CLI policy overrides; `--config_path=FILE --resume=false` also loads a native JSON `train_config.json` as a fresh run with CLI overrides; general YAML/Draccus config files remain refused; Hub model IDs remain refused because native model download is not part of this boundary. `--policy.device` takes `cpu`, and `cuda`/`cuda:0` when built with the `cuda` feature; a GPU that was asked for and cannot be provided is an error rather than a silent fallback. A local one-process checkpoint can be resumed with `--resume=true --config_path=...`, restoring model, AdamW state, RNG and sampler position; Python's three-generator/distributed/accelerate resume semantics remain outside the boundary. Video decoding, external image files, image transforms, Hub streaming/revision flags, accelerate, mixed precision, LR schedulers, PEFT and environment evaluation are refused with a reason rather than ignored.",
    },
    EntryPoint {
        name: "lerobot-train-tokenizer",
        target: "lerobot.scripts.lerobot_train_tokenizer:main",
        status: Status::Unimplemented,
        summary: "Train the FAST action tokenizer.",
        note: "Needs LeRobotDataset loading and the tokenizer training stack.",
    },
    EntryPoint {
        name: "lerobot-dataset-viz",
        target: "lerobot.scripts.lerobot_dataset_viz:main",
        status: Status::Unimplemented,
        summary: "Visualize every frame of a dataset episode.",
        note: "Needs the dataset reader plus a Rerun/Foxglove viewer bridge.",
    },
    EntryPoint {
        name: "lerobot-info",
        target: "lerobot.scripts.lerobot_info:main",
        status: Status::Partial,
        summary: "Print a markdown summary of the system configuration.",
        note: "Ported and runnable. Keys that report Python package versions cannot apply to a \
               Rust build and are reported as not ported rather than invented.",
    },
    EntryPoint {
        name: "lerobot-find-joint-limits",
        target: "lerobot.scripts.lerobot_find_joint_limits:main",
        status: Status::HardwareGated,
        summary: "Discover joint limits and end-effector bounds via teleoperation.",
        note: HW,
    },
    EntryPoint {
        name: "lerobot-imgtransform-viz",
        target: "lerobot.scripts.lerobot_imgtransform_viz:main",
        status: Status::Unimplemented,
        summary: "Render examples of the configured image transforms.",
        note: "Needs the image transform pipeline and dataset loading.",
    },
    EntryPoint {
        name: "lerobot-edit-dataset",
        target: "lerobot.scripts.lerobot_edit_dataset:main",
        status: Status::Unimplemented,
        summary: "Delete, split, merge, and otherwise edit LeRobot datasets.",
        note: "Needs the LeRobotDataset on-disk format (parquet chunks, video shards).",
    },
    EntryPoint {
        name: "lerobot-setup-can",
        target: "lerobot.scripts.lerobot_setup_can:main",
        status: Status::HardwareGated,
        summary: "Set up and debug CAN interfaces for Damiao motors.",
        note: HW,
    },
    EntryPoint {
        name: "lerobot-annotate",
        target: "lerobot.scripts.lerobot_annotate:main",
        status: Status::Unimplemented,
        summary: "Populate language annotation columns on a dataset.",
        note: "Needs dataset editing plus an OpenAI-compatible inference backend.",
    },
    EntryPoint {
        name: "lerobot-rollout",
        target: "lerobot.scripts.lerobot_rollout:main",
        status: Status::Partial,
        summary: "Run a trained policy on a real robot with pluggable strategies.",
        note: "Runnable for a hardware-independent local ACT deployment: it loads a checkpoint, consumes the saved normalizer/unnormalizer processor state, reads local dataset observations, and emits action-unit outputs from the action queue or temporal ensembler. The library also loads a checkpoint without a dataset and accepts a caller-owned single-observation `Batch`, matching the policy's simulator/camera adapter boundary. Robot drivers, teleoperators, environments, visualization and video shards remain explicitly refused; the physical rollout boundary is not faked.",
    },
];

/// All upstream module families under `src/lerobot/`, alphabetically.
pub static MODULE_FAMILIES: &[ModuleFamily] = &[
    ModuleFamily {
        name: "annotations",
        status: Status::Unimplemented,
        upstream_modules: 15,
        note: "Language annotation pipeline; needs dataset editing and an LLM backend.",
    },
    ModuleFamily {
        name: "async_inference",
        status: Status::Unimplemented,
        upstream_modules: 6,
        note: "gRPC policy server/client split.",
    },
    ModuleFamily {
        name: "cameras",
        status: Status::HardwareGated,
        upstream_modules: 17,
        note: "OpenCV / RealSense capture backends.",
    },
    ModuleFamily {
        name: "common",
        status: Status::Unimplemented,
        upstream_modules: 4,
        note: "Shared constants and mixins used by the unported families.",
    },
    ModuleFamily {
        name: "configs",
        status: Status::Partial,
        upstream_modules: 11,
        note:
            "`configs.types` str-enums and `PolicyFeature` are ported and tested. The ACT policy's \
               concrete config is too, including the `from_pretrained`/`_save_pretrained` \
               checkpoint JSON path and the Draccus value conversions it decodes through. The \
               Draccus CLI parser is not, and neither is `configs.train`: `lerobot-train` \
               consumes an explicit allow-list of flags instead, refusing everything else by \
               name.",
    },
    ModuleFamily {
        name: "data_processing",
        status: Status::Unimplemented,
        upstream_modules: 3,
        note: "Dataset-level batch processing helpers.",
    },
    ModuleFamily {
        name: "datasets",
        status: Status::Partial,
        upstream_modules: 22,
        note: "State/action columns and embedded PNG/JPEG `dtype: \"image\"` columns of a LeRobot v3.0 dataset on local disk are ported and tested end to end: `utils`' path constants, `DatasetInfo`, `io_utils`' four loaders including `load_stats` (with nested per-camera mean/std retained separately), the tasks and episodes parquet tables, the frame data files, `feature_utils`' delta indices and tolerance check, `dataset_reader`'s episode-clamped windows and `<key>_is_pad` flags, explicit `episodes=[...]` filtering with absolute-to-relative row remapping, and `sampler`'s `EpisodeAwareSampler` structure. A native Hub snapshot downloader also lists paginated trees, rejects unsafe paths, stages files atomically, and refuses video shards; Hub streaming/sync, video shards, external image files, image transforms, episode predicates and dataset editing are not, and the sampler's per-epoch order is Rerobot's own rather than `torch.randperm`'s. ACT's separate in-memory camera batch contract is implemented by `rerobot-train` rather than this dataset reader.",
    },
    ModuleFamily {
        name: "envs",
        status: Status::Unimplemented,
        upstream_modules: 10,
        note: "Gymnasium environment factories.",
    },
    ModuleFamily {
        name: "jobs",
        status: Status::Unimplemented,
        upstream_modules: 4,
        note: "Hugging Face Jobs launchers.",
    },
    ModuleFamily {
        name: "model",
        status: Status::Unimplemented,
        upstream_modules: 2,
        note: "Shared model plumbing.",
    },
    ModuleFamily {
        name: "motors",
        status: Status::HardwareGated,
        upstream_modules: 16,
        note: "Feetech / Dynamixel / CAN motor buses.",
    },
    ModuleFamily {
        name: "optim",
        status: Status::Partial,
        upstream_modules: 4,
        note: "`torch.optim.AdamW` and `torch.nn.utils.clip_grad_norm_` are reimplemented with \
               upstream's update order and save format, and checked against PyTorch. The Draccus \
               optimizer and LR-scheduler registries, every other optimizer, and all schedulers \
               are not ported.",
    },
    ModuleFamily {
        name: "policies",
        status: Status::Partial,
        upstream_modules: 128,
        note: "ACTConfig validation, presets, delta indices and byte-exact checkpoint JSON \
               read/write are ported, as is the ACT tensor model for state/action and embedded or \
               in-memory camera inputs: the VAE encoder, ResNet18/34 backbone, 1-D/2-D camera \
               position embeddings, transformer, action head and the L1-plus-KL objective. The \
               ACT temporal ensembler, checkpoint normalizer/unnormalizer boundary, and \
               checkpoint-only caller-batch inference boundary are ported in \
               `rerobot_train::deploy`; every other policy architecture is not.",
    },
    ModuleFamily {
        name: "processor",
        status: Status::Partial,
        upstream_modules: 19,
        note: "`rename_processor` (step + `rename_stats`) and the value transform/stateless \
               lifecycle of `newline_task_processor.NewLineTaskProcessorStep` are ported and \
               tested, as is `normalize_processor`'s numeric transform for all four of its \
               modes. The four pre/postprocessor artifacts a checkpoint carries are written \
               byte-identically to upstream's, and the native ACT training/deployment paths load the \
               saved pipeline's rename map before applying scalar/camera normalization and action \
               unnormalization. Python aliasing, arbitrary registry/config reconstruction, and the \
               tokenizer, device, batch and full multi-step pipeline runtime remain outside this \
               boundary.",
    },
    ModuleFamily {
        name: "rewards",
        status: Status::Unimplemented,
        upstream_modules: 24,
        note: "Reward classifiers and success detectors.",
    },
    ModuleFamily {
        name: "rl",
        status: Status::Unimplemented,
        upstream_modules: 21,
        note: "HIL-SERL actor/learner infrastructure.",
    },
    ModuleFamily {
        name: "robots",
        status: Status::HardwareGated,
        upstream_modules: 53,
        note: "Per-robot drivers (SO-100/101, LeKiwi, Reachy2, Unitree, ...).",
    },
    ModuleFamily {
        name: "rollout",
        status: Status::Partial,
        upstream_modules: 18,
        note: "`ring_buffer.RolloutRingBuffer` is ported and tested, including its byte-accounting quirks, as is the DAgger event state machine (`strategies.dagger.DAggerPhase`, its four transitions and `DAggerEvents`). The DAgger strategy itself, the input devices it listens to, the other rollout strategies and the hardware/environment policy loop are not. The local dataset-backed ACT action queue, temporal ensembler, and checkpoint-only caller-batch inference boundary are ported in `rerobot_train::deploy`; it does not claim a robot or Gymnasium boundary.",
    },
    ModuleFamily {
        name: "scripts",
        status: Status::Partial,
        upstream_modules: 20,
        note: "`lerobot_info`, `lerobot_train`, and the hardware-independent local ACT path of `lerobot_rollout` are ported and runnable. The other 15 entry points exist only as executables that fail with a stable unsupported error.",
    },
    ModuleFamily {
        name: "teleoperators",
        status: Status::HardwareGated,
        upstream_modules: 59,
        note: "Leader arms, gamepads, keyboards, phone teleop.",
    },
    ModuleFamily {
        name: "templates",
        status: Status::Unimplemented,
        upstream_modules: 0,
        note: "Non-Python scaffolding templates; nothing to port yet.",
    },
    ModuleFamily {
        name: "transforms",
        status: Status::Unimplemented,
        upstream_modules: 2,
        note: "Image augmentation transforms built on torchvision.",
    },
    ModuleFamily {
        name: "transport",
        status: Status::Unimplemented,
        upstream_modules: 4,
        note: "gRPC transport for async inference.",
    },
    ModuleFamily {
        name: "utils",
        status: Status::Partial,
        upstream_modules: 25,
        note: "`action_interpolator` is ported and tested, as are `io_utils.load_json` and \
               `io_utils.write_json` for local paths. `common.train_utils`' checkpoint layout is \
               ported by `rerobot-train`. `random_utils` is deliberately *not* ported: Rerobot \
               seeds its own SplitMix64 rather than Python's, NumPy's and PyTorch's three \
               generators, so its random values differ. Hub utilities and the video and image \
               writers in `io_utils` are not ported.",
    },
];

/// Look up an entry point by executable name.
pub fn entry_point(name: &str) -> Option<&'static EntryPoint> {
    ENTRY_POINTS.iter().find(|e| e.name == name)
}

/// Look up a module family by package name.
pub fn module_family(name: &str) -> Option<&'static ModuleFamily> {
    MODULE_FAMILIES.iter().find(|f| f.name == name)
}
