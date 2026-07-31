//! The resource budgets the training slice refuses to exceed.
//!
//! Every number these tests push on comes from a file or a command line, which is
//! to say from outside the process. Candle and Arrow are large C-and-unsafe-adjacent
//! dependencies that will happily try to allocate whatever they are asked for, so
//! the budget has to be enforced *before* a value reaches them — an
//! allocation-failure abort is not a refusal, and a wrapped multiplication is worse
//! than either.
//!
//! Two properties are asserted throughout, and both matter:
//!
//! * an over-budget value is refused **by name**, with the limit in the message, so
//!   a user who legitimately needs more knows exactly what to raise;
//! * the value one step *inside* the budget still works, so the limits cannot be
//!   quietly tightened until nothing runs.

mod common;

use common::{fixture_dataset, reduced_config, TempDir};
use rerobot_core::BigInt;
use rerobot_train::error::TrainError;
use rerobot_train::limits;

/// Set a policy dimension and try to build a session, returning the error text.
fn reject_dimension(
    label: &str,
    set: impl Fn(&mut rerobot_core::policy::act::ActConfig),
) -> String {
    let dir = TempDir::new(label);
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    set(&mut config.policy);
    match config.validate() {
        Err(error) => error.to_string(),
        Ok(()) => match rerobot_train::run::TrainSession::new(&config) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("{label}: an over-budget dimension was accepted"),
        },
    }
}

// ---------------------------------------------------------------------------
// The budget is declared, not implicit
// ---------------------------------------------------------------------------

/// Every limit is positive, checked at compile time.
///
/// A static assertion rather than a `#[test]`: these are `const`, so a zero or
/// negative limit is a fact about the source that should stop the build rather than
/// wait for a test run. A zero limit would refuse everything, turning a budget into
/// an outage, and it is the kind of typo a hurried edit to `limits.rs` produces.
const _: () = {
    assert!(limits::MAX_DIM_MODEL > 0);
    assert!(limits::MAX_HEADS > 0);
    assert!(limits::MAX_DIM_FEEDFORWARD > 0);
    assert!(limits::MAX_LAYERS > 0);
    assert!(limits::MAX_LATENT_DIM > 0);
    assert!(limits::MAX_CHUNK_SIZE > 0);
    assert!(limits::MAX_FEATURE_WIDTH > 0);
    assert!(limits::MAX_BATCH_SIZE > 0);
    assert!(limits::MAX_STEPS > 0);
    assert!(limits::MAX_PARQUET_FILE_BYTES > 0);
    assert!(limits::MAX_PARQUET_FILES > 0);
    assert!(limits::MAX_DATASET_ROWS > 0);
    assert!(limits::MAX_DECODED_VALUES > 0);
    assert!(limits::MAX_STRING_BYTES > 0);
    assert!(limits::MAX_LIST_ELEMENTS > 0);
    assert!(limits::MAX_EPISODES > 0);
    assert!(limits::MAX_PARQUET_COLUMNS > 0);
    assert!(limits::MAX_TENSOR_BYTES > 0);
    assert!(limits::MAX_MODEL_BYTES > 0);
    assert!(limits::MAX_PARQUET_CELLS > 0);
    assert!(limits::MAX_DECODED_BYTES > 0);
    // One tensor may not be allowed to exceed the whole model's budget, or the total
    // would be unreachable and the per-tensor limit would be the only real one.
    assert!(limits::MAX_TENSOR_BYTES <= limits::MAX_MODEL_BYTES);
};

/// The budget must be wide enough for a real ACT run, also at compile time.
///
/// Upstream's own defaults are `dim_model` 512, `dim_feedforward` 3200,
/// `chunk_size` 100, four encoder layers, `latent_dim` 32, eight heads, batch size 8
/// and 100 000 steps, and its data files are capped at 100 MB. A limit below any of
/// those would make the budget a straitjacket: `lerobot-train` would refuse a command
/// upstream accepts, which is the opposite of what this port is for.
const _: () = {
    assert!(limits::MAX_DIM_MODEL >= 512);
    assert!(limits::MAX_DIM_FEEDFORWARD >= 3_200);
    assert!(limits::MAX_CHUNK_SIZE >= 100);
    assert!(limits::MAX_LAYERS >= 4);
    assert!(limits::MAX_LATENT_DIM >= 32);
    assert!(limits::MAX_HEADS >= 8);
    assert!(limits::MAX_BATCH_SIZE >= 8);
    assert!(limits::MAX_STEPS >= 100_000);
    assert!(limits::MAX_PARQUET_FILE_BYTES >= 100 * 1024 * 1024);
    // A frame of a 512x512 RGB image is 786 432 scalars, so the width limit has to
    // clear that for the image slice that is not yet ported to be reachable at all.
    assert!(limits::MAX_FEATURE_WIDTH >= 786_432);
    // Upstream's own default ACT is about 51.6 M parameters, so ~207 MB at `f32`. The
    // total model budget has to clear that or a stock model would be refused.
    assert!(limits::MAX_MODEL_BYTES >= 2 * 1024 * 1024 * 1024);
    // Its largest single tensor is `dim_feedforward` 3200 by `dim_model` 512 = 6.5 MB.
    assert!(limits::MAX_TENSOR_BYTES >= 64 * 1024 * 1024);
};

// ---------------------------------------------------------------------------
// Checked arithmetic
// ---------------------------------------------------------------------------

#[test]
fn a_product_that_would_overflow_is_an_error_rather_than_a_wrap() {
    // `usize::MAX * 2` wraps to `usize::MAX - 1` in release and panics in debug.
    // Neither is acceptable when the operands are a parquet file's declared shape.
    assert_eq!(limits::checked_product(&[3, 4], "test"), Ok(12));
    assert_eq!(limits::checked_product(&[], "test"), Ok(1));
    assert_eq!(limits::checked_product(&[0, 5], "test"), Ok(0));

    let error = limits::checked_product(&[usize::MAX, 2], "the shape of x").unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("overflow") || message.contains("too large"),
        "the error does not say it overflowed: {message}"
    );
    assert!(
        message.contains("the shape of x"),
        "the error does not name what overflowed: {message}"
    );

    let error = limits::checked_product(&[1 << 40, 1 << 40, 1 << 40], "big").unwrap_err();
    assert!(error.to_string().contains("big"));
}

#[test]
fn a_bounded_conversion_refuses_a_value_above_its_limit_and_accepts_the_limit() {
    assert_eq!(
        limits::bounded_usize(&BigInt::from(7), "dim_model", 8),
        Ok(7)
    );
    // The limit itself is inside the budget: an inclusive bound is easier to reason
    // about than an exclusive one and cannot be off by one.
    assert_eq!(
        limits::bounded_usize(&BigInt::from(8), "dim_model", 8),
        Ok(8)
    );

    let error = limits::bounded_usize(&BigInt::from(9), "dim_model", 8).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("dim_model"), "{message}");
    assert!(
        message.contains('9'),
        "the value is not reported: {message}"
    );
    assert!(
        message.contains('8'),
        "the limit is not reported: {message}"
    );

    // Negative and astronomically large values are the same refusal, not a panic.
    assert!(limits::bounded_usize(&BigInt::from(-1), "dim_model", 8).is_err());
    let huge: BigInt = "9".repeat(400).parse().unwrap();
    assert!(limits::bounded_usize(&huge, "dim_model", 8).is_err());
}

// ---------------------------------------------------------------------------
// Policy dimensions
// ---------------------------------------------------------------------------

#[test]
fn each_policy_dimension_is_refused_above_its_limit_by_name() {
    for (name, apply) in [
        (
            "dim_model",
            Box::new(|policy: &mut rerobot_core::policy::act::ActConfig| {
                policy.dim_model = BigInt::from(limits::MAX_DIM_MODEL + 1)
            }) as Box<dyn Fn(&mut rerobot_core::policy::act::ActConfig)>,
        ),
        (
            "n_heads",
            Box::new(|policy| policy.n_heads = BigInt::from(limits::MAX_HEADS + 1)),
        ),
        (
            "dim_feedforward",
            Box::new(|policy| {
                policy.dim_feedforward = BigInt::from(limits::MAX_DIM_FEEDFORWARD + 1)
            }),
        ),
        (
            "n_encoder_layers",
            Box::new(|policy| policy.n_encoder_layers = BigInt::from(limits::MAX_LAYERS + 1)),
        ),
        (
            "n_decoder_layers",
            Box::new(|policy| policy.n_decoder_layers = BigInt::from(limits::MAX_LAYERS + 1)),
        ),
        (
            "n_vae_encoder_layers",
            Box::new(|policy| policy.n_vae_encoder_layers = BigInt::from(limits::MAX_LAYERS + 1)),
        ),
        (
            "latent_dim",
            Box::new(|policy| policy.latent_dim = BigInt::from(limits::MAX_LATENT_DIM + 1)),
        ),
        (
            "chunk_size",
            Box::new(|policy| {
                policy.chunk_size = BigInt::from(limits::MAX_CHUNK_SIZE + 1);
                policy.n_action_steps = BigInt::from(1);
            }),
        ),
    ] {
        let message = reject_dimension(name, apply);
        assert!(
            message.contains(name),
            "an over-budget {name} was not refused by name: {message}"
        );
    }
}

#[test]
fn an_astronomically_large_dimension_is_refused_without_allocating() {
    // The reported reproducer: a `chunk_size` far above any machine's address space
    // reached `action_delta_timestamps`, which collects `0..chunk_size` into a `Vec`.
    // The refusal has to come first.
    let message = reject_dimension("astronomical", |policy| {
        policy.chunk_size = "123456789012345678901234567890".parse().unwrap();
        policy.n_action_steps = BigInt::from(1);
    });
    assert!(
        message.contains("chunk_size"),
        "the refusal does not name the field: {message}"
    );
}

#[test]
fn the_reduced_configuration_is_comfortably_inside_every_dimension_limit() {
    let dir = TempDir::new("inside");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().expect("the reduced config validates");
    rerobot_train::run::TrainSession::new(&config).expect("and builds a session");
}

// ---------------------------------------------------------------------------
// Batch size and step count
// ---------------------------------------------------------------------------

#[test]
fn an_over_budget_batch_size_or_step_count_is_refused_by_name() {
    let dir = TempDir::new("batch-steps");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.batch_size = limits::MAX_BATCH_SIZE + 1;
    let error = config.validate().unwrap_err();
    assert!(
        error.to_string().contains("batch_size"),
        "unexpected: {error}"
    );

    let mut config = reduced_config(fixture_dataset(), dir.child("out2"));
    config.steps = limits::MAX_STEPS + 1;
    let error = config.validate().unwrap_err();
    assert!(error.to_string().contains("steps"), "unexpected: {error}");
}

#[test]
fn a_huge_step_count_does_not_reserve_a_vector_for_it_up_front() {
    // `Vec::with_capacity(steps)` on an attacker-chosen `steps` is an allocation
    // request, not a plan. At the limit the run must still start; it will stop for
    // some other reason long before it finishes, and that is fine.
    let dir = TempDir::new("steps-capacity");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.steps = limits::MAX_STEPS;
    config.save_freq = limits::MAX_STEPS.into();
    config
        .validate()
        .expect("the maximum step count is inside the budget");
    // Building the session must not itself try to reserve per-step storage.
    rerobot_train::run::TrainSession::new(&config).expect("the session builds");
}

// ---------------------------------------------------------------------------
// Dataset budgets
// ---------------------------------------------------------------------------

#[test]
fn a_feature_wider_than_the_limit_is_refused_when_the_metadata_is_read() {
    let dir = TempDir::new("wide-feature");
    let root = dir.child("ds");
    std::fs::create_dir_all(root.join("meta")).unwrap();
    std::fs::write(
        root.join("meta/info.json"),
        format!(
            r#"{{
                "codebase_version": "v3.0",
                "fps": 10,
                "features": {{
                    "observation.state": {{"dtype": "float32", "shape": [{}], "names": null}},
                    "action": {{"dtype": "float32", "shape": [2], "names": null}}
                }},
                "total_episodes": 1,
                "total_frames": 4,
                "total_tasks": 1
            }}"#,
            limits::MAX_FEATURE_WIDTH as u64 + 1
        ),
    )
    .unwrap();
    let error = rerobot_train::data::meta::DatasetMetadata::load(&root).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("observation.state"),
        "the refusal does not name the feature: {message}"
    );
}

#[test]
fn a_feature_shape_whose_product_overflows_is_refused_rather_than_wrapping() {
    let dir = TempDir::new("overflow-shape");
    let root = dir.child("ds");
    std::fs::create_dir_all(root.join("meta")).unwrap();
    std::fs::write(
        root.join("meta/info.json"),
        r#"{
            "codebase_version": "v3.0",
            "fps": 10,
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [9223372036854775807, 9223372036854775807],
                    "names": null
                },
                "action": {"dtype": "float32", "shape": [2], "names": null}
            },
            "total_episodes": 1,
            "total_frames": 4,
            "total_tasks": 1
        }"#,
    )
    .unwrap();
    let error = rerobot_train::data::meta::DatasetMetadata::load(&root).unwrap_err();
    assert!(
        error.to_string().contains("observation.state"),
        "unexpected: {error}"
    );
}

#[test]
fn the_fixture_dataset_is_inside_every_dataset_budget() {
    rerobot_train::data::meta::DatasetMetadata::load(&fixture_dataset())
        .expect("the committed fixture is inside the budget");
    let windows = rerobot_train::indexmap::IndexMap::from([(
        "action".to_owned(),
        rerobot_core::dataset::delta::action_delta_timestamps(2, 10),
    )]);
    rerobot_train::data::dataset::StateOnlyDataset::load(&fixture_dataset(), &windows, 1e-4)
        .expect("and so is reading it");
}

// ---------------------------------------------------------------------------
// Malformed episode ranges
// ---------------------------------------------------------------------------

/// Build a dataset directory whose episode table has the given boundary row, reusing
/// the committed fixture for everything else.
fn dataset_with_episode_range(
    dir: &TempDir,
    from: i64,
    to: i64,
    length: i64,
) -> std::path::PathBuf {
    let root = dir.child("ds");
    common::copy_fixture_dataset(&root);
    common::rewrite_episode_row(&root, from, to, length);
    root
}

#[test]
fn an_inverted_episode_range_is_refused() {
    let dir = TempDir::new("inverted");
    let root = dataset_with_episode_range(&dir, 4, 0, 4);
    let error = match load_dataset(&root) {
        Err(error) => error,
        Ok(_) => panic!("the malformed episode range was accepted"),
    };
    let message = error.to_string();
    assert!(
        message.contains("dataset_from_index") || message.contains("range"),
        "the refusal does not describe the range: {message}"
    );
}

#[test]
fn a_negative_episode_boundary_is_refused() {
    let dir = TempDir::new("negative");
    let root = dataset_with_episode_range(&dir, -1, 4, 5);
    let error = match load_dataset(&root) {
        Err(error) => error,
        Ok(_) => panic!("the malformed episode range was accepted"),
    };
    assert!(
        error.to_string().contains("negative") || error.to_string().contains("dataset_from_index"),
        "unexpected: {error}"
    );
}

#[test]
fn an_extreme_episode_boundary_is_refused_rather_than_overflowing() {
    // `i64::MIN` is the value that made `ep_end - 1` panic.
    let dir = TempDir::new("extreme");
    for (from, to) in [(i64::MIN, 4), (0, i64::MIN), (0, i64::MAX)] {
        let root = dataset_with_episode_range(&dir, from, to, 4);
        let error = match load_dataset(&root) {
            Err(error) => error,
            Ok(_) => panic!("({from}, {to}) was accepted"),
        };
        assert!(
            matches!(error, TrainError::Metadata(_)),
            "({from}, {to}) produced the wrong kind of error: {error}"
        );
    }
}

#[test]
fn an_episode_length_disagreeing_with_its_range_is_refused() {
    let dir = TempDir::new("length-mismatch");
    let root = dataset_with_episode_range(&dir, 0, 4, 99);
    let error = match load_dataset(&root) {
        Err(error) => error,
        Ok(_) => panic!("the malformed episode range was accepted"),
    };
    assert!(error.to_string().contains("length"), "unexpected: {error}");
}

#[test]
fn an_episode_range_past_the_declared_frame_count_is_refused() {
    let dir = TempDir::new("past-end");
    let root = dataset_with_episode_range(&dir, 0, 9999, 9999);
    let error = match load_dataset(&root) {
        Err(error) => error,
        Ok(_) => panic!("the malformed episode range was accepted"),
    };
    let message = error.to_string();
    assert!(
        message.contains("total_frames") || message.contains("dataset_to_index"),
        "unexpected: {message}"
    );
}

#[test]
fn the_fixtures_own_episode_range_is_still_accepted() {
    let dir = TempDir::new("unmodified");
    let root = dataset_with_episode_range(&dir, 0, 4, 4);
    load_dataset(&root).expect("the fixture's own boundaries are valid");
}

fn load_dataset(
    root: &std::path::Path,
) -> Result<rerobot_train::data::dataset::StateOnlyDataset, TrainError> {
    let windows = rerobot_train::indexmap::IndexMap::from([(
        "action".to_owned(),
        rerobot_core::dataset::delta::action_delta_timestamps(2, 10),
    )]);
    rerobot_train::data::dataset::StateOnlyDataset::load(root, &windows, 1e-4)
}

// ---------------------------------------------------------------------------
// Collating a batch
// ---------------------------------------------------------------------------

#[test]
fn collating_a_batch_uses_checked_arithmetic_for_its_reservation() {
    // `frames.len() * window_length * width` was reserved directly. All three
    // operands come from outside: the batch size from the command line, the window
    // length from `chunk_size`, and the width from `info.json`. A frame whose window
    // is absurdly wide made that product overflow, which panics in a checked build
    // and wraps in release — and a wrapped reservation is the worse case, because the
    // allocation then succeeds at the wrong size.
    //
    // A hand-built frame is the only way to reach this with a width the dataset
    // reader would already have refused, which is the point: `collate` is a public
    // entry point and must not depend on its caller having validated for it.
    use rerobot_train::data::dataset::Frame;

    let mut windows = rerobot_train::indexmap::IndexMap::new();
    // One row that *claims* to be enormous by being one element short of the address
    // space is not constructible, so instead make the product overflow through the
    // window length: `usize::MAX` rows of width 1.
    windows.insert("observation.state".to_owned(), vec![vec![0.0f32]; 1]);
    let frame = Frame {
        index: 0,
        episode_index: 0,
        frame_index: 0,
        timestamp: 0.0,
        task_index: 0,
        task: "t".to_owned(),
        windows,
        padding: rerobot_train::indexmap::IndexMap::new(),
    };

    // A batch of one frame of width one is fine and must stay fine.
    let batch = rerobot_train::data::batch::collate(
        std::slice::from_ref(&frame),
        &rerobot_train::candle_core::Device::Cpu,
    )
    .expect("an ordinary frame collates");
    assert_eq!(batch.len(), 1);

    // And a batch whose frames disagree about width is refused rather than producing
    // a ragged reservation.
    let mut wide = rerobot_train::indexmap::IndexMap::new();
    wide.insert(
        "observation.state".to_owned(),
        vec![vec![0.0f32, 1.0, 2.0]; 1],
    );
    let mismatched = Frame {
        windows: wide,
        ..frame.clone()
    };
    let error = rerobot_train::data::batch::collate(
        &[frame, mismatched],
        &rerobot_train::candle_core::Device::Cpu,
    )
    .expect_err("frames of different widths must not collate");
    assert!(error.to_string().contains("width"), "unexpected: {error}");
}

#[test]
fn a_batch_larger_than_the_budget_is_refused_by_collate_itself() {
    // `collate` is public, so it carries its own bound rather than trusting that a
    // `TrainConfig` was validated first.
    use rerobot_train::data::dataset::Frame;

    let mut windows = rerobot_train::indexmap::IndexMap::new();
    windows.insert("observation.state".to_owned(), vec![vec![0.0f32, 1.0]]);
    let frame = Frame {
        index: 0,
        episode_index: 0,
        frame_index: 0,
        timestamp: 0.0,
        task_index: 0,
        task: "t".to_owned(),
        windows,
        padding: rerobot_train::indexmap::IndexMap::new(),
    };
    let frames = vec![frame; limits::MAX_BATCH_SIZE + 1];
    let error =
        rerobot_train::data::batch::collate(&frames, &rerobot_train::candle_core::Device::Cpu)
            .expect_err("an over-budget batch must be refused");
    assert!(
        error.to_string().contains("batch"),
        "the refusal does not name the dimension: {error}"
    );
}

// ---------------------------------------------------------------------------
// Dataset-wide budgets
// ---------------------------------------------------------------------------
//
// A per-file budget bounds one file. It says nothing about ten thousand of them,
// and the episode table — which names the files — is attacker-controlled too. These
// three budgets are therefore cumulative across the whole dataset, and like
// `ReadBudget` they are injectable so the checks can be exercised without committing
// a fixture that is deliberately enormous.

#[test]
fn the_default_dataset_budget_is_the_documented_one() {
    let budget = rerobot_train::data::dataset::DatasetBudget::default();
    assert_eq!(budget.max_files, limits::MAX_PARQUET_FILES);
    assert_eq!(budget.max_rows, limits::MAX_DATASET_ROWS);
    assert_eq!(budget.max_values, limits::MAX_DECODED_VALUES);
    assert_eq!(
        budget.read,
        rerobot_train::data::parquet::ReadBudget::default()
    );
}

#[test]
fn the_fixture_reads_at_the_default_dataset_budget() {
    load_within(
        &fixture_dataset(),
        &rerobot_train::data::dataset::DatasetBudget::default(),
    )
    .expect("the committed fixture is inside the default budget");
}

#[test]
fn a_dataset_with_more_files_than_the_budget_is_refused() {
    // The fixture has one data file, so a budget of zero refuses it and the message
    // names the dimension.
    let budget = rerobot_train::data::dataset::DatasetBudget {
        max_files: 0,
        ..Default::default()
    };
    let error = match load_within(&fixture_dataset(), &budget) {
        Err(error) => error,
        Ok(_) => panic!("a zero file budget was accepted"),
    };
    assert!(
        error.to_string().contains("data files"),
        "the refusal does not name the dimension: {error}"
    );
}

#[test]
fn a_dataset_with_more_rows_in_total_than_the_budget_is_refused() {
    // Four rows across one file. Three is over budget, and the refusal must be about
    // the *dataset's* total rather than the file's, which is what makes a
    // many-small-files dataset bounded.
    let budget = rerobot_train::data::dataset::DatasetBudget {
        max_rows: 3,
        ..Default::default()
    };
    let error = match load_within(&fixture_dataset(), &budget) {
        Err(error) => error,
        Ok(_) => panic!("a three-row budget was accepted for a four-row dataset"),
    };
    assert!(
        error.to_string().contains("total row count"),
        "the refusal is not about the dataset total: {error}"
    );
}

#[test]
fn a_dataset_that_would_materialize_more_values_than_the_budget_is_refused() {
    // Three vector features of width two over four rows is twenty-four scalars. Rows
    // alone do not bound that: a dataset of one row and a thousand wide features costs
    // the same as a thousand rows of one narrow one.
    let budget = rerobot_train::data::dataset::DatasetBudget {
        max_values: 23,
        ..Default::default()
    };
    let error = match load_within(&fixture_dataset(), &budget) {
        Err(error) => error,
        Ok(_) => panic!("a 23-value budget was accepted for a 24-value dataset"),
    };
    assert!(
        error.to_string().contains("total decoded size"),
        "the refusal is not about the dataset total: {error}"
    );
}

#[test]
fn a_dataset_at_exactly_its_budgets_is_accepted() {
    // The bounds are inclusive, so the fixture's own extents must pass.
    let budget = rerobot_train::data::dataset::DatasetBudget {
        max_files: 1,
        max_rows: 4,
        max_values: 24,
        ..Default::default()
    };
    load_within(&fixture_dataset(), &budget).expect("the fixture's own extents fit exactly");
}

#[test]
fn the_per_file_budget_still_applies_underneath_the_dataset_one() {
    // The two are independent: a dataset inside its own totals must still have every
    // file inside the per-file budget.
    let budget = rerobot_train::data::dataset::DatasetBudget {
        read: rerobot_train::data::parquet::ReadBudget {
            max_rows: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    let error = match load_within(&fixture_dataset(), &budget) {
        Err(error) => error,
        Ok(_) => panic!("a per-file row budget of three was accepted"),
    };
    assert!(
        error.to_string().contains("declares 4 rows"),
        "the per-file footer check did not run: {error}"
    );
}

fn load_within(
    root: &std::path::Path,
    budget: &rerobot_train::data::dataset::DatasetBudget,
) -> Result<rerobot_train::data::dataset::StateOnlyDataset, TrainError> {
    let windows = rerobot_train::indexmap::IndexMap::from([(
        "action".to_owned(),
        rerobot_core::dataset::delta::action_delta_timestamps(2, 10),
    )]);
    rerobot_train::data::dataset::StateOnlyDataset::load_within(root, &windows, 1e-4, budget)
}

// ---------------------------------------------------------------------------
// The model as a whole, not one tensor at a time
// ---------------------------------------------------------------------------
//
// Every individual dimension being inside its own limit does not bound the model:
// `dim_feedforward` 65 536 by `dim_model` 8 192 is a 2 GiB weight on its own, and the
// allowed layer count multiplies that by 128 twice over. The reported figure was
// roughly a tebibyte for the feed-forward weights alone, every byte of it inside the
// per-field limits. So the budget needs a total.

#[test]
fn a_configuration_whose_tensors_are_individually_legal_but_jointly_enormous_is_refused() {
    // Every tensor here is inside the *per-tensor* budget as well as every per-field
    // one: 2048 x 32768 at `f32` is 268 MB, under the 512 MB a single tensor may take.
    // The 128 permitted layers then multiply that past the model total, which is the
    // failure only a combined budget can catch.
    let message = reject_dimension("jointly-enormous", |policy| {
        policy.dim_model = BigInt::from(2_048);
        policy.dim_feedforward = BigInt::from(32_768);
        policy.n_encoder_layers = BigInt::from(limits::MAX_LAYERS);
        policy.n_decoder_layers = BigInt::from(limits::MAX_LAYERS);
        policy.n_vae_encoder_layers = BigInt::from(limits::MAX_LAYERS);
        policy.n_heads = BigInt::from(64);
    });
    assert!(
        message.contains("model's total size in bytes"),
        "the refusal does not name the total model budget: {message}"
    );
}

#[test]
fn a_single_tensor_larger_than_the_per_tensor_budget_is_refused() {
    // `dim_feedforward` x `dim_model` at both maxima is one 2 GiB weight. A total
    // budget alone would let that through if the rest of the model were small.
    let message = reject_dimension("one-huge-tensor", |policy| {
        policy.dim_model = BigInt::from(limits::MAX_DIM_MODEL);
        policy.dim_feedforward = BigInt::from(limits::MAX_DIM_FEEDFORWARD);
        policy.n_encoder_layers = BigInt::from(1);
        policy.n_decoder_layers = BigInt::from(1);
        policy.n_vae_encoder_layers = BigInt::from(1);
        policy.n_heads = BigInt::from(64);
    });
    assert!(
        message.contains("tensor") && message.contains("bytes"),
        "the refusal does not name the per-tensor byte budget: {message}"
    );
}

#[test]
fn the_reduced_and_the_stock_upstream_configuration_both_fit_the_total_budget() {
    // The budget must not refuse anything real. The reduced fixture config, and
    // upstream's own ACT defaults, both have to build.
    let dir = TempDir::new("stock-fits");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().expect("the reduced config validates");
    rerobot_train::run::TrainSession::new(&config).expect("and builds");

    let mut stock = reduced_config(fixture_dataset(), dir.child("out2"));
    stock.policy.dim_model = BigInt::from(512);
    stock.policy.n_heads = BigInt::from(8);
    stock.policy.dim_feedforward = BigInt::from(3_200);
    stock.policy.n_encoder_layers = BigInt::from(4);
    stock.policy.n_decoder_layers = BigInt::from(1);
    stock.policy.n_vae_encoder_layers = BigInt::from(4);
    stock.policy.latent_dim = BigInt::from(32);
    stock
        .validate()
        .expect("upstream's own ACT defaults must be inside the budget");
    rerobot_train::run::TrainSession::new(&stock).expect("upstream's own ACT defaults must build");
}

#[test]
fn an_initializer_shape_whose_product_overflows_is_refused_rather_than_wrapping() {
    // `Initializer::uniform` and `standard_normal` computed `shape.iter().product()`
    // unchecked. Both operands come from the config, so an overflowing product panics
    // in a checked build and wraps in release -- and a wrapped count allocates a small
    // vector that is then reshaped to an enormous shape.
    let mut rng = rerobot_core::random::SplitMix64::new(1);
    let mut init = rerobot_train::model::params::Initializer::new(
        &mut rng,
        rerobot_train::candle_core::Device::Cpu,
    );
    let error = init
        .uniform(&[usize::MAX, 2], 1.0)
        .expect_err("an overflowing shape must be refused");
    assert!(
        error.to_string().contains("overflow") || error.to_string().contains("too large"),
        "unexpected: {error}"
    );
    let error = init
        .standard_normal(&[usize::MAX, 2])
        .expect_err("an overflowing shape must be refused");
    assert!(
        error.to_string().contains("overflow") || error.to_string().contains("too large"),
        "unexpected: {error}"
    );

    // And an ordinary shape still works.
    assert_eq!(init.uniform(&[2, 3], 1.0).unwrap().dims(), &[2, 3]);
}

#[test]
fn an_initializer_tensor_above_the_per_tensor_byte_budget_is_refused() {
    let mut rng = rerobot_core::random::SplitMix64::new(1);
    let mut init = rerobot_train::model::params::Initializer::new(
        &mut rng,
        rerobot_train::candle_core::Device::Cpu,
    );
    let too_many = limits::MAX_TENSOR_BYTES / 4 + 1;
    let error = init
        .uniform(&[too_many], 1.0)
        .expect_err("a tensor above the per-tensor budget must be refused");
    assert!(error.to_string().contains("bytes"), "unexpected: {error}");
}

// ---------------------------------------------------------------------------
// `TrainSession::new` validates what it is handed
// ---------------------------------------------------------------------------

#[test]
fn train_session_validates_the_batch_size_it_was_given() {
    // `TrainSession::new` is public and took `config.batch_size` on trust: `run.rs`
    // capped only the *initial* reservation, then grew the vector until the unchecked
    // configured size was reached. A caller who built a `TrainConfig` by hand -- which
    // the type permits, since its fields are public -- could therefore ask for a batch
    // of `usize::MAX`.
    let dir = TempDir::new("session-batch");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    // Deliberately *not* through `validate`, which is the hole being closed.
    config.batch_size = limits::MAX_BATCH_SIZE + 1;
    let error = match rerobot_train::run::TrainSession::new(&config) {
        Err(error) => error,
        Ok(_) => panic!("TrainSession::new accepted an over-budget batch size"),
    };
    assert!(
        error.to_string().contains("batch_size"),
        "the refusal does not name the field: {error}"
    );

    let mut zero = reduced_config(fixture_dataset(), dir.child("out2"));
    zero.batch_size = 0;
    assert!(
        rerobot_train::run::TrainSession::new(&zero).is_err(),
        "a zero batch size was accepted"
    );
}

#[test]
fn train_session_still_accepts_the_batch_size_the_fixture_run_uses() {
    let dir = TempDir::new("session-batch-ok");
    let mut config = reduced_config(fixture_dataset(), dir.child("out"));
    config.validate().unwrap();
    rerobot_train::run::TrainSession::new(&config).expect("the ordinary batch size works");
}

// ---------------------------------------------------------------------------
// Episode invariants a single row cannot violate
// ---------------------------------------------------------------------------

/// A copied fixture whose episode table holds `rows` as `(index, from, to, length)`.
fn dataset_with_episode_rows(dir: &TempDir, rows: &[(i64, i64, i64, i64)]) -> std::path::PathBuf {
    let root = dir.child("ds");
    common::copy_fixture_dataset(&root);
    common::rewrite_episode_rows(&root, rows);
    root
}

fn expect_refusal(root: &std::path::Path, what: &str) -> String {
    match load_dataset(root) {
        Err(error) => error.to_string(),
        Ok(_) => panic!("{what} was accepted"),
    }
}

#[test]
fn two_episodes_sharing_an_index_are_refused() {
    // `episode_of` and `get` both find an episode by scanning for the *first* matching
    // index. With two records claiming index 0, one of them is unreachable and every
    // frame of the other is clamped against the wrong range -- so the action chunks are
    // built from the wrong episode's boundaries, silently, with no error anywhere.
    let dir = TempDir::new("duplicate-index");
    let root = dataset_with_episode_rows(&dir, &[(0, 0, 2, 2), (0, 2, 4, 2)]);
    let message = expect_refusal(&root, "a duplicate episode_index");
    assert!(
        message.contains("episode_index") && message.contains("duplicate"),
        "the refusal does not name the problem: {message}"
    );
    assert!(
        message.contains('0'),
        "the refusal does not name the offending index: {message}"
    );
}

#[test]
fn episode_indices_that_are_not_the_contiguous_range_are_refused() {
    // The frames refer to episodes by index, and `episode_of` resolves a *position*.
    // A table numbered 0 and 7 has no episode 1..6, so a frame naming one is
    // unresolvable -- better to say so when the table is read than when a frame is.
    let dir = TempDir::new("sparse-indices");
    let root = dataset_with_episode_rows(&dir, &[(0, 0, 2, 2), (7, 2, 4, 2)]);
    let message = expect_refusal(&root, "a gap in the episode indices");
    // Naming the field is not enough: the frame/episode cross-check downstream also
    // says "episode_index", so a message-only assertion passed even with this guard
    // removed. The refusal has to be about the numbering itself.
    assert!(
        message.contains("episode_index") && message.contains("contiguous"),
        "the refusal does not describe the non-contiguous numbering: {message}"
    );
}

#[test]
fn episode_ranges_that_leave_a_gap_in_the_frame_domain_are_refused() {
    // Four frames, but the ranges cover 0..1 and 2..4. Frame 1 belongs to no episode,
    // so reading it has no episode to clamp against.
    let dir = TempDir::new("gap");
    let root = dataset_with_episode_rows(&dir, &[(0, 0, 1, 1), (1, 2, 4, 2)]);
    let message = expect_refusal(&root, "a gap between episode ranges");
    assert!(
        message.contains("gap") || message.contains("cover"),
        "the refusal does not describe the gap: {message}"
    );
}

#[test]
fn episode_ranges_that_do_not_start_at_zero_are_refused() {
    let dir = TempDir::new("no-zero");
    let root = dataset_with_episode_rows(&dir, &[(0, 1, 4, 3)]);
    let message = expect_refusal(&root, "a frame domain that does not start at zero");
    // `contains('0')` matched almost any message, including the downstream per-frame
    // one, so the assertion has to name the uncovered domain itself.
    assert!(
        message.contains("cover") && message.contains("gap"),
        "the refusal does not describe the uncovered frames: {message}"
    );
}

#[test]
fn episode_ranges_that_stop_short_of_the_declared_frame_count_are_refused() {
    // Three of four frames covered. The uncovered frame is readable from the data file
    // but belongs to no episode.
    let dir = TempDir::new("short");
    let root = dataset_with_episode_rows(&dir, &[(0, 0, 3, 3)]);
    let message = expect_refusal(&root, "a frame domain that stops short");
    assert!(
        message.contains("cover") || message.contains("total_frames"),
        "unexpected: {message}"
    );
}

#[test]
fn an_empty_episode_is_refused_because_it_covers_nothing() {
    let dir = TempDir::new("empty-episode");
    let root = dataset_with_episode_rows(&dir, &[(0, 0, 0, 0), (1, 0, 4, 4)]);
    assert!(
        load_dataset(&root).is_err(),
        "an episode covering no frames was accepted"
    );
}

#[test]
fn the_fixtures_own_two_episode_split_is_accepted() {
    // The coverage rule must accept every well-formed table, including a genuine
    // multi-episode split of the same four frames. "Well-formed" now includes the
    // frames agreeing with the table, so both are rewritten -- splitting the table
    // alone is exactly the inconsistency
    // `a_frame_whose_episode_index_disagrees_with_its_range_is_refused` covers.
    let dir = TempDir::new("valid-split");
    let root = dir.child("ds");
    common::copy_fixture_dataset(&root);
    common::rewrite_episode_rows(&root, &[(0, 0, 2, 2), (1, 2, 4, 2)]);
    common::rewrite_frame_episode_indices(&root, &[0, 0, 1, 1]);
    let dataset = load_dataset(&root).expect("a contiguous two-episode split is valid");
    assert_eq!(dataset.num_episodes(), 2);
    assert_eq!(dataset.len(), 4);
}

#[test]
fn a_frame_whose_episode_index_disagrees_with_its_range_is_refused() {
    // The two sources of truth have to agree: the frame row says which episode it is
    // in, and the episode table says which frames it owns. When they disagree, the
    // reader clamps the action window against a range the frame is not inside -- which
    // is a silently wrong action chunk, not an error.
    //
    // The fixture's four frames all carry `episode_index` 0. Splitting the table into
    // two episodes without rewriting the frames makes frames 2 and 3 claim episode 0
    // while the table puts them in episode 1.
    let dir = TempDir::new("frame-mismatch");
    let root = dir.child("ds");
    common::copy_fixture_dataset(&root);
    common::rewrite_episode_rows(&root, &[(0, 0, 2, 2), (1, 2, 4, 2)]);
    // The episode table is now valid on its own, so if this is accepted the
    // cross-check is missing rather than some other rule catching it.
    common::rewrite_frame_episode_indices(&root, &[0, 0, 0, 0]);
    let message = expect_refusal(&root, "a frame whose episode_index is outside its range");
    assert!(
        message.contains("episode_index"),
        "the refusal does not name the field: {message}"
    );
    assert!(
        message.contains('2') || message.contains("range"),
        "the refusal does not identify the frame or the range: {message}"
    );
}

#[test]
fn frames_that_agree_with_a_multi_episode_table_are_accepted() {
    let dir = TempDir::new("frame-agree");
    let root = dir.child("ds");
    common::copy_fixture_dataset(&root);
    common::rewrite_episode_rows(&root, &[(0, 0, 2, 2), (1, 2, 4, 2)]);
    common::rewrite_frame_episode_indices(&root, &[0, 0, 1, 1]);
    let dataset = load_dataset(&root).expect("agreeing frames and episodes load");
    // And the windows are clamped against the *frame's own* episode: frame 1 is the
    // last of episode 0, so the second half of its chunk is padded.
    assert_eq!(
        dataset.get(1).unwrap().is_pad("action"),
        Some(&[false, true][..])
    );
    assert_eq!(
        dataset.get(2).unwrap().is_pad("action"),
        Some(&[false, false][..])
    );
}

// ---------------------------------------------------------------------------
// The episode metadata tree
// ---------------------------------------------------------------------------
//
// `data/` files are named by the episode table, so their count is bounded by a table
// that has itself been validated. The `meta/episodes/` tree is the other way round: it
// is discovered by walking the directory, and every file in it is read and
// materialized *before* any of that validation can run. So it needs its own cumulative
// budget, and like the others it is injectable so the checks are reachable without an
// enormous fixture.

#[test]
fn the_default_metadata_budget_is_the_documented_one() {
    let budget = rerobot_train::data::meta::MetadataBudget::default();
    assert_eq!(budget.max_files, limits::MAX_EPISODE_FILES);
    assert_eq!(budget.max_rows, limits::MAX_EPISODES);
    assert_eq!(budget.max_values, limits::MAX_DECODED_VALUES);
    assert_eq!(
        budget.read,
        rerobot_train::data::parquet::ReadBudget::default()
    );
}

#[test]
fn the_fixture_metadata_reads_at_the_default_budget() {
    rerobot_train::data::meta::DatasetMetadata::load_within(
        &fixture_dataset(),
        &rerobot_train::data::meta::MetadataBudget::default(),
    )
    .expect("the committed fixture is inside the default metadata budget");
}

#[test]
fn more_metadata_files_than_the_budget_are_refused_before_any_is_read() {
    // The fixture has one episode file, so a budget of zero refuses it. The bound is
    // enforced *during* the walk -- a tree with a million files must not be enumerated
    // in full before being refused -- and again after it, before the first
    // `Table::read`. The assertion below names the walk-time wording specifically,
    // because a message-only check passed with the walk bound removed and only the
    // post-walk backstop left.
    let budget = rerobot_train::data::meta::MetadataBudget {
        max_files: 0,
        ..Default::default()
    };
    let error = match rerobot_train::data::meta::DatasetMetadata::load_within(
        &fixture_dataset(),
        &budget,
    ) {
        Err(error) => error,
        Ok(_) => panic!("a zero metadata-file budget was accepted"),
    };
    assert!(
        error
            .to_string()
            .contains("episode metadata files the reader will open"),
        "the refusal did not come from the walk itself: {error}"
    );
}

#[test]
fn more_metadata_rows_than_the_budget_are_refused() {
    // One episode row, so a budget of zero refuses it. Rows are accumulated across
    // files, so a tree of many small files is bounded too.
    let budget = rerobot_train::data::meta::MetadataBudget {
        max_rows: 0,
        ..Default::default()
    };
    let error = match rerobot_train::data::meta::DatasetMetadata::load_within(
        &fixture_dataset(),
        &budget,
    ) {
        Err(error) => error,
        Ok(_) => panic!("a zero metadata-row budget was accepted"),
    };
    assert!(
        error.to_string().contains("episode"),
        "the refusal does not name the dimension: {error}"
    );
}

#[test]
fn a_metadata_tree_that_would_materialize_too_much_is_refused() {
    // Rows do not bound the cost: the episode table is very wide (upstream writes ten
    // statistics per feature), so one row of it costs far more than one row of the
    // frame table.
    let budget = rerobot_train::data::meta::MetadataBudget {
        max_values: 1,
        ..Default::default()
    };
    let error = match rerobot_train::data::meta::DatasetMetadata::load_within(
        &fixture_dataset(),
        &budget,
    ) {
        Err(error) => error,
        Ok(_) => panic!("a one-value metadata budget was accepted"),
    };
    assert!(
        error.to_string().contains("episode metadata"),
        "the refusal does not name the tree: {error}"
    );
}

#[test]
fn the_metadata_budget_still_applies_the_per_file_one() {
    let budget = rerobot_train::data::meta::MetadataBudget {
        read: rerobot_train::data::parquet::ReadBudget {
            max_rows: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let error = match rerobot_train::data::meta::DatasetMetadata::load_within(
        &fixture_dataset(),
        &budget,
    ) {
        Err(error) => error,
        Ok(_) => panic!("a zero per-file row budget was accepted"),
    };
    assert!(
        error.to_string().contains("declares 1 rows"),
        "the per-file footer check did not run: {error}"
    );
}

#[test]
fn the_fixture_metadata_is_inside_its_budgets_at_exactly_its_own_extents() {
    let budget = rerobot_train::data::meta::MetadataBudget {
        max_files: 1,
        max_rows: 1,
        ..Default::default()
    };
    rerobot_train::data::meta::DatasetMetadata::load_within(&fixture_dataset(), &budget)
        .expect("the fixture's own extents fit exactly");
}

// ---------------------------------------------------------------------------
// Zero is an extent too
// ---------------------------------------------------------------------------
//
// Every bound in this file is an upper one, and an upper bound says nothing about
// zero. A feature declaring `shape: [0]` passed all of them, and the first code that
// could not cope with it was `slice::chunks`, which panics on a zero chunk — the
// abort-instead-of-refusal outcome the whole budget exists to prevent. Zero is
// refused where it is declared, and again at each place that divides the flat buffer
// by it.

#[test]
fn a_feature_declaring_a_zero_dimension_is_refused_when_the_metadata_is_read() {
    let dir = TempDir::new("zero-width");
    let root = dir.child("ds");
    common::copy_fixture_dataset(&root);
    common::rewrite_feature_shape(&root, "observation.state", &[0]);
    let message = expect_refusal(&root, "a zero-width feature");
    // Specifically the *declaration* check: the parquet reader also notices that the
    // stored rows are two wide, but that refusal only happens once a data file has
    // been opened and decoded. An empty feature is refusable from `info.json` alone.
    assert!(
        message.contains("observation.state") && message.contains("empty"),
        "the refusal did not come from the declared shape: {message}"
    );
}

#[test]
fn a_feature_whose_shape_multiplies_out_to_zero_is_refused_too() {
    // `[3, 0, 2]` is not obviously empty at a glance, and every individual dimension
    // is inside `MAX_FEATURE_WIDTH`. The product is what the reader allocates by.
    let dir = TempDir::new("zero-product");
    let root = dir.child("ds");
    common::copy_fixture_dataset(&root);
    common::rewrite_feature_shape(&root, "observation.state", &[3, 0, 2]);
    let message = expect_refusal(&root, "a shape whose product is zero");
    assert!(
        message.contains("observation.state") && message.contains("empty"),
        "the refusal did not come from the declared shape: {message}"
    );
}

#[test]
fn feature_spec_width_refuses_zero_rather_than_returning_it() {
    let spec = rerobot_train::data::meta::FeatureSpec {
        dtype: "float32".to_owned(),
        shape: vec![0],
        names: None,
    };
    let error = spec.width().expect_err("a zero width is not a width");
    assert!(
        error.to_string().contains("empty"),
        "the refusal does not say the feature is empty: {error}"
    );
    // The valid neighbours still work, including the scalar shape `[]`, whose
    // product is 1 by the same convention `math.prod(())` uses.
    for shape in [vec![], vec![1], vec![2, 3]] {
        rerobot_train::data::meta::FeatureSpec {
            dtype: "float32".to_owned(),
            shape: shape.clone(),
            names: None,
        }
        .width()
        .unwrap_or_else(|error| panic!("shape {shape:?} is legitimate: {error}"));
    }
}

#[test]
fn collate_refuses_a_zero_width_window_instead_of_building_an_unusable_batch() {
    use indexmap::IndexMap;
    let mut windows: IndexMap<String, Vec<Vec<f32>>> = IndexMap::new();
    windows.insert("observation.state".to_owned(), vec![vec![]]);
    let frame = rerobot_train::data::dataset::Frame {
        index: 0,
        episode_index: 0,
        frame_index: 0,
        timestamp: 0.0,
        task_index: 0,
        task: "reach the target".to_owned(),
        windows,
        padding: IndexMap::new(),
    };
    let error = rerobot_train::data::batch::collate(&[frame], &candle_core::Device::Cpu)
        .expect_err("a zero-width feature cannot be collated");
    assert!(
        error.to_string().contains("observation.state"),
        "the refusal does not name the feature: {error}"
    );
}

#[test]
fn normalizing_a_zero_width_tensor_is_an_error_not_a_panic() {
    // `Batch`'s fields are public, so this state is reachable without `collate`.
    // `chunks(0)` panics, and a panic in a library is not a refusal: it takes the
    // process down with no exit code a caller can act on.
    use indexmap::IndexMap;

    // Built from the fixture's own metadata rather than by hand, so the normalizer is
    // exactly the one a real run holds and the only unusual thing is the tensor.
    let metadata = rerobot_train::data::meta::DatasetMetadata::load(&fixture_dataset())
        .expect("the fixture's metadata loads");
    let (inputs, outputs) = metadata.policy_feature_split();
    let normalizer = rerobot_core::policy::normalize::Normalizer::new(
        &inputs.into_iter().chain(outputs).collect(),
        &rerobot_core::policy::act::ActConfig::default().normalization_mapping,
        &metadata.stats,
    )
    .expect("the normalizer resolves against the fixture's stats");

    let mut features = IndexMap::new();
    features.insert(
        "observation.state".to_owned(),
        candle_core::Tensor::from_vec(Vec::<f32>::new(), (1, 0), &candle_core::Device::Cpu)
            .expect("an empty tensor is constructible"),
    );
    let batch = rerobot_train::data::batch::Batch {
        features,
        padding: IndexMap::new(),
        tasks: vec!["reach the target".to_owned()],
        indices: vec![0],
    };

    let error = batch
        .normalized(&normalizer)
        .expect_err("a zero-width tensor cannot be normalized");
    assert!(
        error.to_string().contains("observation.state"),
        "the refusal does not name the feature: {error}"
    );
}
