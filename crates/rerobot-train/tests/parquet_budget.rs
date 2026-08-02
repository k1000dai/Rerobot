//! What the parquet reader refuses to decode, and — importantly — *when* it refuses.
//!
//! The budget is exercised by shrinking it rather than by committing an oversized
//! fixture. That is not a weakening: [`ReadBudget::default`] is the production
//! budget, one test below asserts the shrunken budgets are genuinely below the real
//! ones, and another asserts the committed fixture passes at the default. What
//! shrinking buys is the ability to check the *ordering* of the checks, which is the
//! part that was wrong: the row limit used to be applied after Arrow had already
//! decoded a batch, so the memory had been allocated and the work done by the time
//! the file was refused.

mod common;

use common::fixture_dataset;
use rerobot_train::data::parquet::{ReadBudget, Table};
use rerobot_train::limits;
use std::path::PathBuf;

fn data_file() -> PathBuf {
    fixture_dataset().join("data/chunk-000/file-000.parquet")
}

fn episodes_file() -> PathBuf {
    fixture_dataset().join("meta/episodes/chunk-000/file-000.parquet")
}

fn tasks_file() -> PathBuf {
    fixture_dataset().join("meta/tasks.parquet")
}

/// A budget that permits everything, so a test can lower exactly one dimension.
fn generous() -> ReadBudget {
    ReadBudget {
        max_file_bytes: u64::MAX,
        max_rows: usize::MAX,
        max_columns: usize::MAX,
        max_values: usize::MAX,
        max_string_bytes: usize::MAX,
        max_list_elements: usize::MAX,
        max_cells: usize::MAX,
        max_decoded_bytes: usize::MAX,
        max_image_bytes: usize::MAX,
    }
}

// ---------------------------------------------------------------------------
// The default budget is the real one
// ---------------------------------------------------------------------------

#[test]
fn the_default_budget_is_the_documented_one() {
    // The shrunken budgets below would be meaningless if the default were shrunken
    // too. This is what ties the tests to the production limits.
    let budget = ReadBudget::default();
    assert_eq!(budget.max_file_bytes, limits::MAX_PARQUET_FILE_BYTES);
    assert_eq!(budget.max_rows, limits::MAX_DATASET_ROWS);
    assert_eq!(budget.max_columns, limits::MAX_PARQUET_COLUMNS);
    assert_eq!(budget.max_values, limits::MAX_DECODED_VALUES);
    assert_eq!(budget.max_string_bytes, limits::MAX_STRING_BYTES);
    assert_eq!(budget.max_list_elements, limits::MAX_LIST_ELEMENTS);
}

#[test]
fn every_file_of_the_committed_fixture_reads_at_the_default_budget() {
    for path in [data_file(), episodes_file(), tasks_file()] {
        let table = Table::read(&path)
            .unwrap_or_else(|error| panic!("{} was refused: {error}", path.display()));
        assert!(table.rows() > 0, "{} decoded no rows", path.display());
    }
}

// ---------------------------------------------------------------------------
// The file itself
// ---------------------------------------------------------------------------

#[test]
fn a_file_larger_than_the_byte_budget_is_refused_without_being_parsed() {
    let path = data_file();
    let actual = std::fs::metadata(&path).unwrap().len();
    let budget = ReadBudget {
        max_file_bytes: actual - 1,
        ..generous()
    };
    let error = Table::read_within(&path, &budget).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains(&actual.to_string()),
        "the refusal does not report the size: {message}"
    );
    assert!(
        message.contains("bytes"),
        "the refusal does not say what was too big: {message}"
    );
}

#[test]
fn a_file_at_exactly_the_byte_budget_is_accepted() {
    let path = data_file();
    let actual = std::fs::metadata(&path).unwrap().len();
    let budget = ReadBudget {
        max_file_bytes: actual,
        ..generous()
    };
    Table::read_within(&path, &budget).expect("the bound is inclusive");
}

#[test]
fn a_row_count_above_the_budget_is_refused_from_the_footer_before_decoding() {
    // The fixture has four rows. A budget of three must refuse it, and the message
    // must report the *declared* count — proof the footer was consulted rather than
    // the refusal happening after a batch had been decoded.
    let budget = ReadBudget {
        max_rows: 3,
        ..generous()
    };
    let error = Table::read_within(&data_file(), &budget).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("declares 4 rows"),
        "the refusal did not come from the footer: {message}"
    );
    assert!(
        message.contains('3'),
        "the limit is not reported: {message}"
    );
}

#[test]
fn a_row_count_at_the_budget_is_accepted() {
    let budget = ReadBudget {
        max_rows: 4,
        ..generous()
    };
    let table = Table::read_within(&data_file(), &budget).expect("four rows fit a budget of four");
    assert_eq!(table.rows(), 4);
}

#[test]
fn a_column_count_above_the_budget_is_refused() {
    // The episode table is the wide one: upstream writes ten statistics per feature.
    let budget = ReadBudget {
        max_columns: 5,
        ..generous()
    };
    let error = Table::read_within(&episodes_file(), &budget).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("columns"),
        "the refusal does not name the dimension: {message}"
    );
}

// ---------------------------------------------------------------------------
// Decoded values
// ---------------------------------------------------------------------------

#[test]
fn a_vector_column_above_the_value_budget_is_refused_before_it_is_materialized() {
    // Four rows of width two is eight values. Rows alone do not bound this: the
    // product is what the decode costs, which is why the budget is on the product.
    let budget = ReadBudget {
        max_values: 7,
        ..generous()
    };
    let path = data_file();
    let table = Table::read_within(&path, &budget).expect("the file itself is inside the budget");
    let error = table
        .vector_column(&path, "observation.state", 2)
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("observation.state"),
        "the refusal does not name the column: {message}"
    );
    assert!(
        message.contains("8 values"),
        "the refusal does not report the cost: {message}"
    );
}

#[test]
fn a_vector_column_at_the_value_budget_is_accepted() {
    let budget = ReadBudget {
        max_values: 8,
        ..generous()
    };
    let path = data_file();
    let table = Table::read_within(&path, &budget).unwrap();
    let rows = table.vector_column(&path, "observation.state", 2).unwrap();
    assert_eq!(rows.len(), 4);
}

#[test]
fn a_width_that_would_overflow_the_product_is_refused_rather_than_wrapping() {
    // `rows * width` with `width` near `usize::MAX` wraps to a small number in
    // release, and the allocation then succeeds at the wrong size. The declared width
    // comes from `info.json`, so it is attacker-controlled.
    let path = data_file();
    let table = Table::read(&path).unwrap();
    let error = table
        .vector_column(&path, "observation.state", usize::MAX)
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("overflow") || message.contains("too large"),
        "the refusal does not say it overflowed: {message}"
    );
}

// ---------------------------------------------------------------------------
// Text and lists
// ---------------------------------------------------------------------------

#[test]
fn a_string_column_above_the_byte_budget_is_refused() {
    // The fixture's one task is "reach the target", sixteen bytes.
    let budget = ReadBudget {
        max_string_bytes: 15,
        ..generous()
    };
    let path = tasks_file();
    let table = Table::read_within(&path, &budget).unwrap();
    let error = table.string_column(&path, "task").unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("bytes of text"),
        "the refusal does not name the dimension: {message}"
    );
    assert!(
        message.contains("16"),
        "the refusal does not report the size: {message}"
    );
}

#[test]
fn a_string_column_at_the_byte_budget_is_accepted() {
    let budget = ReadBudget {
        max_string_bytes: 16,
        ..generous()
    };
    let path = tasks_file();
    let table = Table::read_within(&path, &budget).unwrap();
    assert_eq!(
        table.string_column(&path, "task").unwrap(),
        vec!["reach the target".to_owned()]
    );
}

#[test]
fn a_list_column_above_the_element_budget_is_refused() {
    let budget = ReadBudget {
        max_list_elements: 0,
        ..generous()
    };
    let path = episodes_file();
    let table = Table::read_within(&path, &budget).unwrap();
    let error = table.string_list_column(&path, "tasks").unwrap_err();
    assert!(
        error.to_string().contains("list elements"),
        "unexpected: {error}"
    );
}

#[test]
fn a_list_columns_text_is_budgeted_too_not_only_its_element_count() {
    // One element holding a gigabyte is one element. Bounding the count alone would
    // miss it entirely.
    let budget = ReadBudget {
        max_string_bytes: 4,
        ..generous()
    };
    let path = episodes_file();
    let table = Table::read_within(&path, &budget).unwrap();
    let error = table.string_list_column(&path, "tasks").unwrap_err();
    assert!(
        error.to_string().contains("bytes of text"),
        "unexpected: {error}"
    );
}

#[test]
fn a_list_column_at_both_budgets_is_accepted() {
    let budget = ReadBudget {
        max_list_elements: 1,
        max_string_bytes: 16,
        ..generous()
    };
    let path = episodes_file();
    let table = Table::read_within(&path, &budget).unwrap();
    assert_eq!(
        table.string_list_column(&path, "tasks").unwrap(),
        vec![vec!["reach the target".to_owned()]]
    );
}

// ---------------------------------------------------------------------------
// Aggregate budgets, from the footer, before Arrow decodes anything
// ---------------------------------------------------------------------------
//
// Rows, columns and compressed bytes were each bounded separately, and the
// per-column decoded-value check ran only *after* `read_within` had already decoded
// every batch. So a file of a thousand rows and a thousand wide columns passed all
// three individual checks and then decoded a gibibyte before any per-column budget
// was consulted. The cost of a decode is `rows x columns` cells and the bytes behind
// them, so those are what has to be bounded, from the footer.

#[test]
fn the_aggregate_cell_and_byte_budgets_are_declared() {
    let budget = ReadBudget::default();
    assert_eq!(budget.max_cells, limits::MAX_PARQUET_CELLS);
    assert_eq!(budget.max_decoded_bytes, limits::MAX_DECODED_BYTES);
}

#[test]
fn a_rows_times_columns_product_above_the_budget_is_refused_from_the_footer() {
    // The fixture's data file is 4 rows by 8 columns, so 32 cells. A budget of 31
    // must refuse it, and the message must report the product rather than either
    // factor -- that is what shows the aggregate is being checked.
    let budget = ReadBudget {
        max_cells: 31,
        ..generous()
    };
    let error = Table::read_within(&data_file(), &budget).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("32"),
        "the refusal does not report the cell count: {message}"
    );
    assert!(
        message.contains("cells"),
        "the refusal does not name the dimension: {message}"
    );
    // And it came from the footer, so no batch was decoded.
    assert!(
        message.contains("declares"),
        "the refusal is not from the footer: {message}"
    );
}

#[test]
fn a_rows_times_columns_product_at_the_budget_is_accepted() {
    let budget = ReadBudget {
        max_cells: 32,
        ..generous()
    };
    let table = Table::read_within(&data_file(), &budget).expect("32 cells fit a budget of 32");
    assert_eq!(table.rows(), 4);
}

#[test]
fn an_aggregate_decoded_byte_estimate_above_the_budget_is_refused_from_the_footer() {
    // The cell count does not bound the bytes: one cell of a wide list column costs
    // far more than one cell of an `int64`. The footer records each column's
    // uncompressed size, which is what the decode will actually cost.
    let path = episodes_file();
    let generous_table = Table::read(&path).expect("the fixture reads");
    let estimate = generous_table.declared_decoded_bytes();
    assert!(
        estimate > 0,
        "the footer reported no uncompressed size, so the budget could not be checked"
    );

    let budget = ReadBudget {
        max_decoded_bytes: estimate - 1,
        ..generous()
    };
    let error = Table::read_within(&path, &budget).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("decode"),
        "the refusal does not say what was too big: {message}"
    );
    assert!(
        message.contains(&estimate.to_string()),
        "the refusal does not report the estimate: {message}"
    );

    let budget = ReadBudget {
        max_decoded_bytes: estimate,
        ..generous()
    };
    Table::read_within(&path, &budget).expect("the bound is inclusive");
}

#[test]
fn the_aggregate_budgets_are_checked_before_the_per_column_ones() {
    // Ordering matters: a file that violates both must be refused by the cheaper,
    // earlier check, because that is the one that runs before any allocation.
    let budget = ReadBudget {
        max_cells: 1,
        max_values: 1,
        ..generous()
    };
    let error = Table::read_within(&data_file(), &budget).unwrap_err();
    assert!(
        error.to_string().contains("cells"),
        "the per-column check ran first: {error}"
    );
}

#[test]
fn every_fixture_file_is_inside_the_aggregate_budgets_by_a_wide_margin() {
    for path in [data_file(), episodes_file(), tasks_file()] {
        let table = Table::read(&path).unwrap();
        let cells = table.rows() * table.column_names().len();
        assert!(
            cells * 1_000 < limits::MAX_PARQUET_CELLS,
            "{} uses {cells} cells, uncomfortably close to the budget",
            path.display()
        );
        assert!(
            table.declared_decoded_bytes().saturating_mul(1_000) < limits::MAX_DECODED_BYTES,
            "{} is uncomfortably close to the decoded-byte budget",
            path.display()
        );
    }
}
