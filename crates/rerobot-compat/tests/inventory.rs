//! The inventory must stay faithful to upstream `pyproject.toml` and honest
//! about what this workspace actually implements.

use rerobot_compat::{
    entry_point, module_family, EntryPoint, Status, ENTRY_POINTS, MODULE_FAMILIES, UPSTREAM_COMMIT,
    UPSTREAM_LICENSE, UPSTREAM_PACKAGE, UPSTREAM_VERSION,
};

/// Verbatim `[project.scripts]` from upstream `pyproject.toml`, in file order.
const UPSTREAM_SCRIPTS: &[(&str, &str)] = &[
    (
        "lerobot-calibrate",
        "lerobot.scripts.lerobot_calibrate:main",
    ),
    (
        "lerobot-find-cameras",
        "lerobot.scripts.lerobot_find_cameras:main",
    ),
    (
        "lerobot-find-port",
        "lerobot.scripts.lerobot_find_port:main",
    ),
    ("lerobot-record", "lerobot.scripts.lerobot_record:main"),
    ("lerobot-replay", "lerobot.scripts.lerobot_replay:main"),
    (
        "lerobot-setup-motors",
        "lerobot.scripts.lerobot_setup_motors:main",
    ),
    (
        "lerobot-teleoperate",
        "lerobot.scripts.lerobot_teleoperate:main",
    ),
    ("lerobot-eval", "lerobot.scripts.lerobot_eval:main"),
    ("lerobot-train", "lerobot.scripts.lerobot_train:main"),
    (
        "lerobot-train-tokenizer",
        "lerobot.scripts.lerobot_train_tokenizer:main",
    ),
    (
        "lerobot-dataset-viz",
        "lerobot.scripts.lerobot_dataset_viz:main",
    ),
    ("lerobot-info", "lerobot.scripts.lerobot_info:main"),
    (
        "lerobot-find-joint-limits",
        "lerobot.scripts.lerobot_find_joint_limits:main",
    ),
    (
        "lerobot-imgtransform-viz",
        "lerobot.scripts.lerobot_imgtransform_viz:main",
    ),
    (
        "lerobot-edit-dataset",
        "lerobot.scripts.lerobot_edit_dataset:main",
    ),
    (
        "lerobot-setup-can",
        "lerobot.scripts.lerobot_setup_can:main",
    ),
    ("lerobot-annotate", "lerobot.scripts.lerobot_annotate:main"),
    ("lerobot-rollout", "lerobot.scripts.lerobot_rollout:main"),
];

#[test]
fn upstream_coordinates_are_pinned() {
    assert_eq!(UPSTREAM_PACKAGE, "lerobot");
    assert_eq!(UPSTREAM_VERSION, "0.6.1");
    assert_eq!(UPSTREAM_LICENSE, "Apache-2.0");
    assert_eq!(UPSTREAM_COMMIT.len(), 40);
    assert!(UPSTREAM_COMMIT.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn all_eighteen_entry_points_are_present_in_upstream_order() {
    let got: Vec<(&str, &str)> = ENTRY_POINTS.iter().map(|e| (e.name, e.target)).collect();
    assert_eq!(got, UPSTREAM_SCRIPTS.to_vec());
    assert_eq!(ENTRY_POINTS.len(), 18);
}

#[test]
fn entry_point_names_are_unique() {
    let mut names: Vec<&str> = ENTRY_POINTS.iter().map(|e| e.name).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(names.len(), before);
}

#[test]
fn every_entry_point_keeps_the_upstream_executable_name() {
    for e in ENTRY_POINTS {
        assert!(
            e.name.starts_with("lerobot-"),
            "{} is not a lerobot-* name",
            e.name
        );
        assert!(!e.name.contains('_'), "{} must use hyphens", e.name);
    }
}

#[test]
fn every_entry_point_documents_itself() {
    for e in ENTRY_POINTS {
        assert!(!e.summary.is_empty(), "{} has no summary", e.name);
        assert!(!e.note.is_empty(), "{} has no status note", e.name);
    }
}

#[test]
fn lookup_finds_entry_points_and_rejects_unknown_names() {
    let info: &EntryPoint = entry_point("lerobot-info").unwrap();
    assert_eq!(info.target, "lerobot.scripts.lerobot_info:main");
    assert!(entry_point("lerobot-nope").is_none());
    assert!(entry_point("").is_none());
    assert!(
        entry_point("LEROBOT-INFO").is_none(),
        "lookup must be case sensitive"
    );
}

#[test]
fn exactly_one_entry_point_is_runnable_in_this_milestone() {
    let runnable: Vec<&str> = ENTRY_POINTS
        .iter()
        .filter(|e| !e.status.is_unsupported())
        .map(|e| e.name)
        .collect();
    assert_eq!(runnable, vec!["lerobot-info"]);
}

#[test]
fn no_entry_point_claims_full_implementation() {
    for e in ENTRY_POINTS {
        assert_ne!(
            e.status,
            Status::Implemented,
            "{} must not claim parity that is not demonstrated",
            e.name
        );
    }
}

#[test]
fn status_slugs_are_stable() {
    assert_eq!(Status::Implemented.as_str(), "implemented");
    assert_eq!(Status::Partial.as_str(), "partial");
    assert_eq!(Status::Unimplemented.as_str(), "unimplemented");
    assert_eq!(Status::HardwareGated.as_str(), "hardware-gated");
    assert_eq!(Status::HardwareGated.to_string(), "hardware-gated");
}

#[test]
fn only_implemented_and_partial_are_supported() {
    assert!(!Status::Implemented.is_unsupported());
    assert!(!Status::Partial.is_unsupported());
    assert!(Status::Unimplemented.is_unsupported());
    assert!(Status::HardwareGated.is_unsupported());
}

#[test]
fn module_families_cover_every_upstream_package() {
    let expected = [
        "annotations",
        "async_inference",
        "cameras",
        "common",
        "configs",
        "data_processing",
        "datasets",
        "envs",
        "jobs",
        "model",
        "motors",
        "optim",
        "policies",
        "processor",
        "rewards",
        "rl",
        "robots",
        "rollout",
        "scripts",
        "teleoperators",
        "templates",
        "transforms",
        "transport",
        "utils",
    ];
    let got: Vec<&str> = MODULE_FAMILIES.iter().map(|f| f.name).collect();
    assert_eq!(got, expected.to_vec());
}

#[test]
fn module_families_are_sorted_and_annotated() {
    let mut sorted: Vec<&str> = MODULE_FAMILIES.iter().map(|f| f.name).collect();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        MODULE_FAMILIES.iter().map(|f| f.name).collect::<Vec<_>>()
    );
    for f in MODULE_FAMILIES {
        assert!(!f.note.is_empty(), "{} has no status note", f.name);
    }
}

#[test]
fn no_module_family_claims_full_implementation() {
    for f in MODULE_FAMILIES {
        assert_ne!(
            f.status,
            Status::Implemented,
            "{} overstates parity",
            f.name
        );
    }
}

#[test]
fn partially_ported_families_are_exactly_the_ones_with_tests() {
    let partial: Vec<&str> = MODULE_FAMILIES
        .iter()
        .filter(|f| f.status == Status::Partial)
        .map(|f| f.name)
        .collect();
    assert_eq!(
        partial,
        vec!["configs", "processor", "rollout", "scripts", "utils"]
    );
}

#[test]
fn hardware_families_are_marked_hardware_gated() {
    for name in ["cameras", "motors", "robots", "teleoperators"] {
        assert_eq!(
            module_family(name).unwrap().status,
            Status::HardwareGated,
            "{name} must be hardware-gated"
        );
    }
}

#[test]
fn module_family_lookup_rejects_unknown_names() {
    assert!(module_family("policies").is_some());
    assert!(module_family("nope").is_none());
}

#[test]
fn module_counts_are_recorded() {
    assert_eq!(module_family("policies").unwrap().upstream_modules, 128);
    assert_eq!(module_family("templates").unwrap().upstream_modules, 0);
    for f in MODULE_FAMILIES {
        if f.name != "templates" {
            assert!(
                f.upstream_modules > 0,
                "{} has no recorded module count",
                f.name
            );
        }
    }
}
