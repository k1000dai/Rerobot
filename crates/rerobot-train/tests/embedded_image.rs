//! Behaviour tests for the embedded camera column of a LeRobot v3.0 dataset:
//! `struct<bytes: binary, path: string>` cells of PNG or JPEG, decoded natively into
//! the RGB frames ACT trains on.
//!
//! Two fixtures are used, for two different jobs.
//!
//! * `tests/fixtures/embedded_image/` is committed and read offline. It is upstream's
//!   own state-only fixture with one camera column added — see
//!   `examples/make_embedded_image_fixture.rs`, which says exactly which bytes are
//!   upstream's and which are not — and it is what the happy path and the training run
//!   are asserted against.
//! * A copy of that fixture, corrupted one field at a time, is what every refusal is
//!   asserted against. Corrupting a copy rather than building a dataset from nothing
//!   keeps each test honest: everything except the field under test is exactly what the
//!   fixture holds, so a refusal can only be about that field.
//!
//! Every claim here is about the *reader*, not about a decoder library: which cells are
//! accepted, which are refused and by what name, and what the pixels become.

mod common;

use common::{embedded_image_fixture, reduced_config, TempDir};
use indexmap::IndexMap;
use rerobot_core::dataset::delta::action_delta_timestamps;
use rerobot_core::types::FeatureType;
use rerobot_train::data::batch::collate_images;
use rerobot_train::data::dataset::{DatasetBudget, StateOnlyDataset};
use rerobot_train::data::image::{CameraNormalization, DecodedImage, CAMERA_CHANNELS};
use rerobot_train::data::meta::DatasetMetadata;
use rerobot_train::error::TrainError;
use rerobot_train::run::TrainSession;
use std::path::Path;

/// The camera the fixture carries.
const CAMERA: &str = "observation.images.top";

/// The side of every fixture frame, which is also what `info.json` declares.
const EXTENT: usize = 32;

fn action_window(chunk_size: i64) -> IndexMap<String, Vec<f64>> {
    IndexMap::from([("action".to_owned(), action_delta_timestamps(chunk_size, 10))])
}

fn load_fixture() -> StateOnlyDataset {
    StateOnlyDataset::load(&embedded_image_fixture(), &action_window(2), 1e-4)
        .expect("the embedded-image fixture loads")
}

/// The fixture's own formula, restated here rather than imported.
///
/// A test that computed the expectation with the code under test would assert nothing;
/// this is the same arithmetic `examples/make_embedded_image_fixture.rs` encodes, and
/// the two agreeing is the claim.
fn expected_pixel(frame: usize, channel: usize, y: usize, x: usize) -> f32 {
    let sample = match channel {
        0 => (x * 8) as u8,
        1 => (y * 8) as u8,
        _ => (frame * 64) as u8,
    };
    f32::from(sample) / 255.0
}

// ---------------------------------------------------------------------------
// The committed fixture
// ---------------------------------------------------------------------------

#[test]
fn info_json_declares_the_camera_beside_the_state_features() {
    let metadata = DatasetMetadata::load(&embedded_image_fixture()).expect("the fixture loads");
    let camera = metadata.feature(CAMERA).expect("the camera is declared");
    assert_eq!(camera.dtype, "image");
    assert_eq!(camera.shape, vec![3, EXTENT as i64, EXTENT as i64]);
    assert_eq!(
        metadata.feature_keys().collect::<Vec<_>>(),
        vec![
            "observation.state",
            "observation.environment_state",
            CAMERA,
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]
    );
}

#[test]
fn the_camera_is_a_visual_input_feature_of_the_policy() {
    let metadata = DatasetMetadata::load(&embedded_image_fixture()).unwrap();
    let (inputs, outputs) = metadata.policy_feature_split();
    assert_eq!(inputs[CAMERA].r#type, FeatureType::Visual);
    assert_eq!(
        inputs.keys().collect::<Vec<_>>(),
        vec!["observation.state", "observation.environment_state", CAMERA]
    );
    assert_eq!(outputs.keys().collect::<Vec<_>>(), vec!["action"]);
}

#[test]
fn every_frame_decodes_to_the_pixels_the_fixture_encoded() {
    let dataset = load_fixture();
    assert_eq!(dataset.len(), 4);
    for frame_index in 0..4 {
        let frame = dataset.get(frame_index).expect("the frame loads");
        let image = frame.image(CAMERA).expect("the camera is present");
        assert_eq!(
            (image.channels, image.height, image.width),
            (CAMERA_CHANNELS, EXTENT, EXTENT),
            "frame {frame_index} decoded to the wrong extent"
        );
        assert_eq!(image.pixels.len(), CAMERA_CHANNELS * EXTENT * EXTENT);
        assert_eq!(
            image.path.as_deref(),
            Some(format!("images/{CAMERA}/episode-000000/frame-{frame_index:06}.png").as_str()),
            "the path field beside the bytes is not carried through"
        );
    }
}

#[test]
fn the_pixels_are_channel_first_and_scaled_into_zero_to_one() {
    let dataset = load_fixture();
    // Corners and a middle sample of every channel of every frame: enough to catch a
    // transposed axis, an off-by-one row stride, or a missing division by 255, none of
    // which a shape assertion would notice.
    for frame_index in 0..4 {
        let image = dataset.get(frame_index).unwrap();
        let image = image.image(CAMERA).unwrap();
        let plane = EXTENT * EXTENT;
        for channel in 0..CAMERA_CHANNELS {
            for (y, x) in [(0, 0), (0, EXTENT - 1), (EXTENT - 1, 0), (17, 5)] {
                assert_eq!(
                    image.pixels[channel * plane + y * EXTENT + x],
                    expected_pixel(frame_index, channel, y, x),
                    "frame {frame_index} channel {channel} pixel ({y}, {x})"
                );
            }
        }
        assert!(
            image.pixels.iter().all(|value| (0.0..=1.0).contains(value)),
            "frame {frame_index} decoded a value outside [0, 1]"
        );
    }
}

#[test]
fn the_frames_stack_into_a_batch_camera_tensor_in_frame_order() {
    let dataset = load_fixture();
    let frames = vec![dataset.get(0).unwrap(), dataset.get(3).unwrap()];
    let images = collate_images(&frames, &rerobot_train::candle_core::Device::Cpu).unwrap();

    assert_eq!(images.keys().collect::<Vec<_>>(), vec![CAMERA]);
    let tensor = &images[CAMERA];
    assert_eq!(tensor.dims(), &[2, CAMERA_CHANNELS, EXTENT, EXTENT]);
    let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let plane = EXTENT * EXTENT;
    let per_frame = CAMERA_CHANNELS * plane;
    // Channel 2 is the frame number in the fixture's formula, so the two stacked
    // frames must differ there and nowhere in the batch axis may they be swapped.
    assert_eq!(values[2 * plane], expected_pixel(0, 2, 0, 0));
    assert_eq!(values[per_frame + 2 * plane], expected_pixel(3, 2, 0, 0));
}

// ---------------------------------------------------------------------------
// Training on it
// ---------------------------------------------------------------------------

#[test]
fn a_session_over_the_fixture_builds_the_camera_into_the_model_and_steps_on_it() {
    let dir = TempDir::new("embedded-train");
    let mut config = reduced_config(embedded_image_fixture(), dir.child("out"));
    config.validate().unwrap();

    let mut session = TrainSession::new(&config).expect("the session builds");
    assert_eq!(
        session
            .model
            .shape()
            .cameras
            .iter()
            .map(|camera| camera.key.as_str())
            .collect::<Vec<_>>(),
        vec![CAMERA],
        "the dataset's own camera did not reach the model"
    );

    // The batch the sampler produces already carries the camera: nothing is attached
    // by the caller, which is the whole difference from the in-memory path.
    let batch = session.next_batch().expect("the batch collates");
    assert_eq!(
        batch.image(CAMERA).unwrap().dims(),
        &[config.batch_size, CAMERA_CHANNELS, EXTENT, EXTENT]
    );

    let metrics = session.step(1).expect("the step runs");
    assert!(metrics.loss.is_finite(), "loss is {}", metrics.loss);
    assert!(metrics.grad_norm > 0.0, "the loss reached no parameter");
    assert!(
        metrics.parameter_delta > 0.0,
        "AdamW ran but nothing moved: {metrics:?}"
    );
}

#[test]
fn use_imagenet_stats_selects_between_the_imagenet_statistics_and_leaving_frames_alone() {
    let dir = TempDir::new("embedded-stats");

    let mut with_stats = reduced_config(embedded_image_fixture(), dir.child("a"));
    assert!(
        with_stats.dataset_use_imagenet_stats,
        "upstream's default is true"
    );
    with_stats.validate().unwrap();
    let mut off = reduced_config(embedded_image_fixture(), dir.child("b"));
    off.dataset_use_imagenet_stats = false;
    off.validate().unwrap();

    let normalized = TrainSession::new(&with_stats)
        .unwrap()
        .next_batch()
        .unwrap();
    let raw = TrainSession::new(&off).unwrap().next_batch().unwrap();

    let normalized = normalized
        .image(CAMERA)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let raw = raw
        .image(CAMERA)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // `false` is the identity, so the batch holds exactly the decoded pixels.
    assert!(
        raw.iter().all(|value| (0.0..=1.0).contains(value)),
        "use_imagenet_stats=false must leave the [0, 1] frame untouched"
    );
    assert_eq!(raw[0], expected_pixel(0, 0, 0, 0));

    // `true` is `(value - mean) / (std + eps)` per channel, which moves every pixel.
    let expected = (raw[0] - rerobot_train::data::image::IMAGENET_MEAN[0])
        / rerobot_train::data::image::IMAGENET_STD[0];
    assert!(
        (normalized[0] - expected).abs() < 1e-3,
        "expected roughly {expected}, got {}",
        normalized[0]
    );
    assert_ne!(normalized, raw);
}

#[test]
fn the_flag_reaches_train_config_json_and_nothing_else_of_it_moves() {
    let dir = TempDir::new("embedded-config-json");
    let mut config = reduced_config(embedded_image_fixture(), dir.child("out"));
    config.validate().unwrap();
    let with_stats = config.to_json_text();

    config.dataset_use_imagenet_stats = false;
    let without = config.to_json_text();

    assert!(with_stats.contains("\"use_imagenet_stats\": true"));
    assert!(without.contains("\"use_imagenet_stats\": false"));
    assert_eq!(
        with_stats.replace(
            "\"use_imagenet_stats\": true",
            "\"use_imagenet_stats\": false"
        ),
        without,
        "the flag must be the only difference in train_config.json"
    );
}

#[test]
fn nested_image_statistics_emitted_by_lerobot_are_ignored_for_scalar_normalization() {
    let dir = TempDir::new("embedded-nested-stats");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);
    std::fs::write(
        root.join("meta/stats.json"),
        r#"{
            "observation.images.top": {
                "min": [[[0.0]]],
                "max": [[[1.0]]],
                "mean": [[[0.5]]],
                "std": [[[0.25]]]
            }
        }"#,
    )
    .expect("nested image stats are writable");

    let metadata = DatasetMetadata::load(&root)
        .expect("a LeRobot image dataset's nested camera stats must not make metadata unreadable");
    assert!(metadata.stats.get(CAMERA).is_none());
}

// ---------------------------------------------------------------------------
// Refusals: the cell
// ---------------------------------------------------------------------------

#[test]
fn a_cell_that_is_not_an_image_at_all_is_refused_by_name() {
    let dir = TempDir::new("embedded-garbage");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);
    common::rewrite_image_cells(&root, &vec![b"not an image at all".to_vec(); 4]);

    let error = StateOnlyDataset::load(&root, &action_window(2), 1e-4)
        .expect_err("a cell that is not an image must be refused");
    assert!(
        error.to_string().contains(CAMERA),
        "the refusal does not name the camera: {error}"
    );
    assert!(
        error.to_string().contains("PNG") && error.to_string().contains("JPEG"),
        "the refusal does not say which formats are read: {error}"
    );
}

#[test]
fn a_truncated_png_is_refused_rather_than_decoded_to_a_partial_frame() {
    let dir = TempDir::new("embedded-truncated");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);
    let mut truncated = common::png_of(EXTENT, EXTENT);
    truncated.truncate(40);
    common::rewrite_image_cells(&root, &vec![truncated; 4]);

    let error = StateOnlyDataset::load(&root, &action_window(2), 1e-4)
        .expect_err("a truncated PNG must be refused");
    assert!(
        error.to_string().contains(CAMERA),
        "the refusal does not name the camera: {error}"
    );
}

#[test]
fn an_empty_cell_is_refused_rather_than_treated_as_a_black_frame() {
    let dir = TempDir::new("embedded-empty");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);
    common::rewrite_image_cells(&root, &[Vec::new(), Vec::new(), Vec::new(), Vec::new()]);

    let error = StateOnlyDataset::load(&root, &action_window(2), 1e-4)
        .expect_err("an empty cell must be refused");
    assert!(
        error.to_string().contains("no encoded bytes"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn a_null_cell_is_refused_with_its_row_named() {
    let dir = TempDir::new("embedded-null");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);
    common::rewrite_image_cells_with_nulls(&root, 2);

    let error = StateOnlyDataset::load(&root, &action_window(2), 1e-4)
        .expect_err("a null cell must be refused");
    assert!(
        error.to_string().contains("row 2") && error.to_string().contains("null"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn a_format_that_is_not_png_or_jpeg_is_refused_by_name_rather_than_by_decoder_failure() {
    // GIF's magic bytes. The `image` dependency is compiled with two codecs, so this
    // would fail inside the decoder anyway; the point is that it fails *before* one is
    // built, with a message naming the format and the boundary rather than the build.
    let gif = b"GIF89a\x01\x00\x01\x00\x80\x00\x00".to_vec();
    let error = DecodedImage::from_encoded(CAMERA, &gif, None, (3, EXTENT, EXTENT))
        .expect_err("GIF must be refused");
    assert!(
        matches!(error, TrainError::Unsupported(_)),
        "expected an explicit refusal, got {error}"
    );
    assert!(
        error.to_string().contains("Gif") && error.to_string().contains("PNG and JPEG"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn a_jpeg_cell_decodes_as_readily_as_a_png_one() {
    let jpeg = common::jpeg_of(EXTENT, EXTENT);
    let decoded = DecodedImage::from_encoded(CAMERA, &jpeg, None, (3, EXTENT, EXTENT))
        .expect("a JPEG cell decodes");
    assert_eq!(
        (decoded.channels, decoded.height, decoded.width),
        (CAMERA_CHANNELS, EXTENT, EXTENT)
    );
    assert!(decoded
        .pixels
        .iter()
        .all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn a_cell_larger_than_the_per_image_limit_never_reaches_a_decoder() {
    let oversized = vec![0u8; rerobot_train::limits::MAX_EMBEDDED_IMAGE_BYTES + 1];
    let error = DecodedImage::from_encoded(CAMERA, &oversized, None, (3, EXTENT, EXTENT))
        .expect_err("an oversized cell must be refused");
    assert!(
        error.to_string().contains("encoded bytes")
            && error
                .to_string()
                .contains(&rerobot_train::limits::MAX_EMBEDDED_IMAGE_BYTES.to_string()),
        "unexpected refusal: {error}"
    );
}

#[test]
fn the_parquet_reader_refuses_an_over_budget_cell_before_copying_it() {
    // The production limit is 16 MiB, and committing a 16 MiB fixture to prove it is
    // enforced is not worth the bytes. The budget is injectable for exactly this, so
    // the fixture's real cells are over-budget against a shrunken one.
    let dir = TempDir::new("embedded-budget");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);

    let mut budget = DatasetBudget::default();
    budget.read.max_image_bytes = 64;
    let error = StateOnlyDataset::load_within(&root, &action_window(2), 1e-4, &budget)
        .expect_err("an over-budget cell must be refused");
    assert!(
        error.to_string().contains("encoded bytes") && error.to_string().contains("64"),
        "unexpected refusal: {error}"
    );
}

// ---------------------------------------------------------------------------
// Refusals: the declaration
// ---------------------------------------------------------------------------

#[test]
fn a_frame_whose_size_contradicts_info_json_is_refused_rather_than_resized() {
    let dir = TempDir::new("embedded-wrong-size");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);
    common::rewrite_feature_shape(&root, CAMERA, &[3, 16, 16]);

    let error = StateOnlyDataset::load(&root, &action_window(2), 1e-4)
        .expect_err("a shape disagreement must be refused");
    let message = error.to_string();
    assert!(
        message.contains("[3, 32, 32]") && message.contains("[3, 16, 16]"),
        "the refusal does not state both shapes: {error}"
    );
    assert!(
        message.contains("not resized"),
        "the refusal does not say what it will not do: {error}"
    );
}

#[test]
fn a_camera_declared_with_a_channel_count_other_than_three_is_refused() {
    let dir = TempDir::new("embedded-channels");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);
    common::rewrite_feature_shape(&root, CAMERA, &[1, 32, 32]);

    let error = StateOnlyDataset::load(&root, &action_window(2), 1e-4)
        .expect_err("a one-channel camera must be refused");
    assert!(
        matches!(error, TrainError::Unsupported(_)),
        "expected an explicit refusal, got {error}"
    );
    assert!(
        error.to_string().contains("RGB") && error.to_string().contains(CAMERA),
        "unexpected refusal: {error}"
    );
}

#[test]
fn a_camera_declared_with_a_rank_other_than_three_is_refused() {
    let dir = TempDir::new("embedded-rank");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);
    common::rewrite_feature_shape(&root, CAMERA, &[3, 32]);

    let error = StateOnlyDataset::load(&root, &action_window(2), 1e-4)
        .expect_err("a rank-2 camera must be refused");
    assert!(
        error.to_string().contains("[channels, height, width]"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn a_declared_camera_with_no_column_in_the_data_file_names_the_layout_it_needs() {
    let dir = TempDir::new("embedded-absent");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);
    common::drop_image_column(&root);

    let error = StateOnlyDataset::load(&root, &action_window(2), 1e-4)
        .expect_err("a declared camera with no column must be refused");
    assert!(
        matches!(error, TrainError::Unsupported(_)),
        "expected an explicit refusal, got {error}"
    );
    assert!(
        error
            .to_string()
            .contains("struct<bytes: binary, path: string>"),
        "the refusal does not name the layout it reads: {error}"
    );
}

#[test]
fn a_camera_column_of_the_wrong_arrow_type_is_refused_by_spelling() {
    let dir = TempDir::new("embedded-arrow-type");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);
    common::replace_image_column_with_binary(&root);

    let error = StateOnlyDataset::load(&root, &action_window(2), 1e-4)
        .expect_err("a bare binary column must be refused");
    assert!(
        error
            .to_string()
            .contains("struct<bytes: binary, path: string>"),
        "the refusal does not state the accepted spelling: {error}"
    );
}

#[test]
fn a_delta_window_on_a_camera_is_refused_rather_than_stacked() {
    let mut windows = action_window(2);
    windows.insert(CAMERA.to_owned(), vec![0.0, -0.1]);
    let error = StateOnlyDataset::load(&embedded_image_fixture(), &windows, 1e-4)
        .expect_err("a camera history window must be refused");
    assert!(
        matches!(error, TrainError::Unsupported(_)),
        "expected an explicit refusal, got {error}"
    );
    assert!(
        error.to_string().contains(CAMERA),
        "the refusal does not name the camera: {error}"
    );
}

// ---------------------------------------------------------------------------
// Video is still refused
// ---------------------------------------------------------------------------

#[test]
fn a_video_feature_is_refused_even_beside_a_readable_image_one() {
    let dir = TempDir::new("embedded-video");
    let root = dir.child("dataset");
    common::copy_embedded_image_fixture(&root);
    common::declare_video_feature(&root, "observation.images.wrist");

    let error = DatasetMetadata::load(&root).expect_err("a video feature must be refused outright");
    assert!(
        matches!(error, TrainError::Unsupported(_)),
        "expected an explicit refusal, got {error}"
    );
    let message = error.to_string();
    assert!(
        message.contains("observation.images.wrist"),
        "the refusal does not name the feature: {error}"
    );
    assert!(
        !message.contains(CAMERA),
        "the readable camera must not be swept up in the video refusal: {error}"
    );
    for expected in ["MP4", "AV1 or H.264"] {
        assert!(
            message.contains(expected),
            "the refusal does not mention {expected}: {error}"
        );
    }
}

// ---------------------------------------------------------------------------
// The identity normalization is genuinely the identity
// ---------------------------------------------------------------------------

#[test]
fn the_identity_camera_normalization_returns_the_tensor_it_was_given() {
    let dataset = load_fixture();
    let frames = vec![dataset.get(0).unwrap()];
    let images = collate_images(&frames, &rerobot_train::candle_core::Device::Cpu).unwrap();
    let before = images[CAMERA]
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let after = CameraNormalization::identity()
        .apply(CAMERA, &images[CAMERA])
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(before, after);
}

/// The fixture directory has to exist for every test above to mean anything.
#[test]
fn the_committed_fixture_is_present_and_offline() {
    let root: &Path = &embedded_image_fixture();
    for relative in [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
    ] {
        assert!(
            root.join(relative).is_file(),
            "the fixture is missing {relative}"
        );
    }
}
