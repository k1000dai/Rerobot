//! `docs/compatibility.md` is the human-facing rendering of this crate's
//! inventory. These tests are what makes the doc's claim about itself true: the
//! two status tables are checked row by row — name, status, upstream target or
//! module count, and port note — plus the pinned upstream coordinates and the
//! stated entry-point count. Prose outside those tables is *not* machine
//! checked, and the doc must not claim otherwise.

use rerobot_compat::{ENTRY_POINTS, MODULE_FAMILIES, UPSTREAM_COMMIT, UPSTREAM_VERSION};

fn compatibility_doc() -> String {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/compatibility.md");
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

/// Cells of every body row of the markdown table whose header row contains
/// `header`, in document order. The separator row is skipped and the table ends
/// at the first line that is not a table row.
fn table_rows(doc: &str, header: &str) -> Vec<Vec<String>> {
    let mut lines = doc.lines();
    lines
        .by_ref()
        .find(|line| line.contains(header))
        .unwrap_or_else(|| panic!("docs/compatibility.md has no table headed {header:?}"));
    lines
        .skip(1) // the `| --- | --- |` separator
        .take_while(|line| line.trim_start().starts_with('|'))
        .map(|line| {
            let trimmed = line.trim().trim_start_matches('|').trim_end_matches('|');
            trimmed
                .split('|')
                .map(|cell| cell.trim().to_string())
                .collect()
        })
        .collect()
}

#[test]
fn the_doc_pins_the_same_upstream_coordinates() {
    let doc = compatibility_doc();
    assert!(doc.contains(UPSTREAM_VERSION), "missing upstream version");
    assert!(doc.contains(UPSTREAM_COMMIT), "missing upstream commit");
}

#[test]
fn the_doc_lists_every_entry_point_with_its_status() {
    let doc = compatibility_doc();
    for e in ENTRY_POINTS {
        let row = doc
            .lines()
            .find(|line| line.contains(&format!("`{}`", e.name)))
            .unwrap_or_else(|| panic!("docs/compatibility.md has no row for {}", e.name));
        assert!(
            row.contains(e.status.as_str()),
            "row for {} does not state status {}: {row}",
            e.name,
            e.status
        );
        assert!(
            row.contains(e.target),
            "row for {} does not name its upstream target",
            e.name
        );
    }
}

#[test]
fn the_entry_point_table_is_the_inventory_row_for_row() {
    let doc = compatibility_doc();
    let rows = table_rows(&doc, "| Executable | Status | Upstream target |");
    assert_eq!(
        rows.len(),
        ENTRY_POINTS.len(),
        "the entry-point table has {} rows for {} inventory entries",
        rows.len(),
        ENTRY_POINTS.len()
    );
    for (row, e) in rows.iter().zip(ENTRY_POINTS) {
        assert_eq!(row.len(), 5, "unexpected column count in {row:?}");
        assert_eq!(row[0], format!("`{}`", e.name), "row order diverges");
        assert_eq!(row[1], e.status.as_str(), "status for {}", e.name);
        assert_eq!(row[2], format!("`{}`", e.target), "target for {}", e.name);
        assert_eq!(row[3], e.summary, "summary for {}", e.name);
        assert_eq!(row[4], collapse(e.note), "note for {}", e.name);
    }
}

#[test]
fn the_module_family_table_is_the_inventory_row_for_row() {
    let doc = compatibility_doc();
    let rows = table_rows(&doc, "| Family | Status | Upstream modules |");
    assert_eq!(rows.len(), MODULE_FAMILIES.len());
    for (row, f) in rows.iter().zip(MODULE_FAMILIES) {
        assert_eq!(row.len(), 4, "unexpected column count in {row:?}");
        assert_eq!(
            row[0],
            format!("`lerobot/{}`", f.name),
            "row order diverges"
        );
        assert_eq!(row[1], f.status.as_str(), "status for {}", f.name);
        assert_eq!(
            row[2],
            f.upstream_modules.to_string(),
            "module count for {}",
            f.name
        );
        assert_eq!(row[3], collapse(f.note), "note for {}", f.name);
    }
}

#[test]
fn the_doc_states_the_entry_point_count_from_the_inventory() {
    let doc = compatibility_doc();
    let sentence = format!(
        "All {} upstream console scripts exist as executables",
        ENTRY_POINTS.len()
    );
    assert!(doc.contains(&sentence), "docs must state {sentence:?}");
}

#[test]
fn the_doc_states_the_unsupported_entry_point_count_from_the_inventory() {
    let doc = compatibility_doc();
    let unsupported = ENTRY_POINTS
        .iter()
        .filter(|e| e.status.is_unsupported())
        .count();
    assert!(
        doc.contains(&format!("The other {unsupported} entry points")),
        "docs must state the {unsupported} unsupported entry points"
    );
}

#[test]
fn the_doc_never_claims_a_family_is_fully_implemented() {
    let doc = compatibility_doc();
    for line in doc.lines() {
        if line.starts_with('|') && line.contains("| implemented ") {
            panic!("docs/compatibility.md claims unproven parity: {line}");
        }
    }
}

#[test]
fn the_doc_documents_every_status_label() {
    let doc = compatibility_doc();
    for label in ["implemented", "partial", "unimplemented", "hardware-gated"] {
        assert!(doc.contains(label), "status label {label} is not explained");
    }
}

#[test]
fn the_doc_does_not_claim_to_be_generated() {
    // Finding 6: nothing generates this file, and only its two status tables
    // plus the pinned coordinates are machine-checked. Saying more than that
    // would be a false claim about the project's own process.
    let doc = compatibility_doc();
    for claim in [
        "generated from",
        "auto-generated",
        "autogenerated",
        "do not edit",
    ] {
        assert!(
            !doc.to_lowercase().contains(claim),
            "docs/compatibility.md claims to be generated ({claim:?}), but no generator exists"
        );
    }
}

#[test]
fn the_doc_says_exactly_which_parts_are_checked() {
    let doc = compatibility_doc();
    assert!(
        doc.contains("crates/rerobot-compat/tests/docs_consistency.rs"),
        "the doc must name the test that checks it"
    );
    assert!(
        doc.contains("hand-written"),
        "the doc must say it is hand-written"
    );
    // Markdown hard-wraps prose, so compare against the unwrapped text.
    assert!(
        collapse(&doc).contains("Prose outside those two tables is not machine-checked"),
        "the doc must narrow its own consistency claim"
    );
}

/// Inventory notes are Rust string continuations, so they carry the source
/// indentation; markdown cells are single-line.
fn collapse(note: &str) -> String {
    note.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Every crate ships the licence documents its README points at
// ---------------------------------------------------------------------------

/// The root `LICENSE` and `NOTICE`, and each crate's packaged copy of them.
///
/// Real copies rather than symlinks: cargo does follow a symlink when packaging, but a
/// Windows checkout without symlink support leaves git a *text file containing the
/// path*, which would package silently-wrong content. Copies are robust everywhere and
/// this test is what stops them drifting.
#[test]
fn every_crate_ships_a_verbatim_copy_of_the_licence_and_notice() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("the repository root");

    for document in ["LICENSE", "NOTICE"] {
        let canonical = std::fs::read(root.join(document))
            .unwrap_or_else(|error| panic!("the root {document} is unreadable: {error}"));
        assert!(!canonical.is_empty(), "the root {document} is empty");
        for crate_name in [
            "rerobot-core",
            "rerobot-compat",
            "rerobot-train",
            "rerobot-cli",
        ] {
            let path = root.join("crates").join(crate_name).join(document);
            let shipped = std::fs::read(&path).unwrap_or_else(|error| {
                panic!(
                    "{crate_name} does not ship {document} ({}): {error}. Apache-2.0 section \
                     4(d) requires the NOTICE to travel with a redistribution, and the \
                     packaged README tells recipients to consult these files.",
                    path.display()
                )
            });
            assert_eq!(
                shipped, canonical,
                "{crate_name}/{document} is not a verbatim copy of the repository root's"
            );
        }
    }
}

#[test]
fn the_notice_carries_the_upstream_attribution_the_licence_points_at() {
    // The reason `NOTICE` has to travel: it is where upstream LeRobot is credited.
    // A copy that had lost that would satisfy the byte-comparison above only because
    // the root had lost it too.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("the repository root");
    let notice = std::fs::read_to_string(root.join("NOTICE")).unwrap();
    assert!(
        notice.contains("LeRobot"),
        "NOTICE does not credit upstream LeRobot"
    );
    assert!(
        notice.contains("Apache"),
        "NOTICE does not name the licence"
    );
}

// ---------------------------------------------------------------------------
// The packaged CLI README cannot outlive its own claim
// ---------------------------------------------------------------------------

#[test]
fn the_cli_readme_agrees_with_the_inventory_about_how_many_commands_run() {
    // `crates/rerobot-cli/README.md` is included in `rerobot-cli-0.1.0.crate`, so
    // whatever it says appears on crates.io. It said "exactly one of them runs" and
    // that `lerobot-train` is unimplemented and exits 2, after `lerobot-train` had
    // become runnable -- a false claim about the release, in the one document a
    // prospective user reads first.
    //
    // Checked against the inventory rather than against a hardcoded number, so
    // promoting a third command forces the README to be updated in the same change.
    let readme = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/")
            .join("rerobot-cli/README.md"),
    )
    .expect("the CLI README is readable");

    let runnable: Vec<&str> = ENTRY_POINTS
        .iter()
        .filter(|entry| !entry.status.is_unsupported())
        .map(|entry| entry.name)
        .collect();
    let unsupported = ENTRY_POINTS.len() - runnable.len();

    // The count, spelled the way the prose spells it.
    let spelled = match runnable.len() {
        1 => "one",
        2 => "Two",
        3 => "Three",
        other => panic!("the README's prose has no spelling for {other} runnable commands"),
    };
    assert!(
        readme.contains(spelled),
        "the CLI README does not state that {} of the entry points run",
        runnable.len()
    );
    assert!(
        readme.contains(&unsupported.to_string()) || readme.contains("sixteen"),
        "the CLI README does not state how many are unported ({unsupported})"
    );

    // And it must not still be describing a runnable command as unported.
    for name in &runnable {
        let stale = format!("{name} --help         # works, and states that it is unimplemented");
        assert!(
            !readme.contains(&stale),
            "the CLI README still calls the runnable {name} unimplemented"
        );
    }
    // Every runnable command has to be mentioned at all.
    for name in &runnable {
        assert!(
            readme.contains(name),
            "the CLI README never mentions the runnable {name}"
        );
    }
}
