//! Shared helpers for the training-slice tests.
//!
//! Cargo compiles this module separately into every integration-test binary that
//! declares `mod common;`, so a helper only some of them need is dead code in the
//! rest. `goldens.rs`, for instance, reads committed fixtures and never needs a
//! temporary directory.
#![allow(dead_code)]

use rerobot_train::config::TrainConfig;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

/// The committed state-only dataset fixture, written by upstream itself.
///
/// See `tools/goldens/make_dataset_fixture.py`: one episode, four frames, 10 fps,
/// `observation.state` and `observation.environment_state` and `action` all of
/// width two.
pub fn fixture_dataset() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/state_only")
}

/// The committed embedded-camera fixture.
///
/// The state-only fixture with one `observation.images.top` column of 32×32 PNGs added
/// in upstream's `struct<bytes: binary, path: string>` spelling. See
/// `examples/make_embedded_image_fixture.rs`, which wrote it and records which of its
/// bytes are upstream's and which are not.
pub fn embedded_image_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedded_image")
}

/// A unique directory that deletes itself when the test ends.
pub struct TempDir(PathBuf);

impl TempDir {
    /// A fresh directory named after `label`.
    pub fn new(label: &str) -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rerobot-train-{}-{label}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("cannot create the test directory");
        Self(path)
    }

    /// The directory itself.
    #[allow(dead_code)]
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// A path inside it that does not exist yet.
    pub fn child(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The reduced ACT configuration every test in this crate trains: small enough to
/// run in well under a second, large enough that every architectural piece is
/// exercised (a two-step chunk so padding is reachable, four attention heads over
/// 32 channels, a VAE encoder, and a decoder).
#[allow(dead_code)]
pub fn reduced_config(dataset_root: PathBuf, output_dir: PathBuf) -> TrainConfig {
    use rerobot_core::BigInt;

    let mut config = TrainConfig::new(
        "rerobot/state_only_slice".to_owned(),
        dataset_root,
        output_dir,
    );
    config.steps = 1;
    config.batch_size = 2;
    config.log_freq = 1;
    config.save_freq = 1.into();
    config.seed = Some(1000);

    config.policy.chunk_size = BigInt::from(2);
    config.policy.n_action_steps = BigInt::from(2);
    config.policy.dim_model = BigInt::from(32);
    config.policy.n_heads = BigInt::from(4);
    config.policy.dim_feedforward = BigInt::from(64);
    config.policy.n_encoder_layers = BigInt::from(1);
    config.policy.n_decoder_layers = BigInt::from(1);
    config.policy.n_vae_encoder_layers = BigInt::from(1);
    config.policy.latent_dim = BigInt::from(8);
    config.policy.use_vae = true;
    // The fixture has no cameras, so the backbone weights would name a
    // torchvision checkpoint that is never loaded. Upstream leaves the string set;
    // clearing it keeps `config.json` honest about what this run used.
    config.policy.pretrained_backbone_weights = None;
    config
}

// ---------------------------------------------------------------------------
// Building deliberately malformed datasets
// ---------------------------------------------------------------------------

/// Copy the committed fixture into `root` so a test can corrupt one file of it.
///
/// Corrupting a copy rather than constructing a dataset from nothing keeps the test
/// honest: everything except the field under test is exactly what upstream wrote, so
/// a refusal can only be about that field.
pub fn copy_fixture_dataset(root: &Path) {
    fn copy_tree(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).expect("cannot create the destination");
        for entry in std::fs::read_dir(from).expect("cannot read the fixture") {
            let entry = entry.expect("cannot read a fixture entry");
            let target = to.join(entry.file_name());
            if entry.file_type().expect("file type").is_dir() {
                copy_tree(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), &target).expect("cannot copy a fixture file");
            }
        }
    }
    let _ = std::fs::remove_dir_all(root);
    copy_tree(&fixture_dataset(), root);
}

/// [`copy_fixture_dataset`] for the embedded-camera fixture.
pub fn copy_embedded_image_fixture(root: &Path) {
    fn copy_tree(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).expect("cannot create the destination");
        for entry in std::fs::read_dir(from).expect("cannot read the fixture") {
            let entry = entry.expect("cannot read a fixture entry");
            let target = to.join(entry.file_name());
            if entry.file_type().expect("file type").is_dir() {
                copy_tree(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), &target).expect("cannot copy a fixture file");
            }
        }
    }
    let _ = std::fs::remove_dir_all(root);
    copy_tree(&embedded_image_fixture(), root);
}

// ---------------------------------------------------------------------------
// Encoding images, and corrupting the camera column of a copied fixture
// ---------------------------------------------------------------------------

/// The camera key the embedded-image fixture carries.
pub const CAMERA_KEY: &str = "observation.images.top";

/// A `width`×`height` RGB PNG, encoded here rather than committed.
pub fn png_of(width: usize, height: usize) -> Vec<u8> {
    encode(width, height, image::ImageFormat::Png)
}

/// A `width`×`height` RGB JPEG.
pub fn jpeg_of(width: usize, height: usize) -> Vec<u8> {
    encode(width, height, image::ImageFormat::Jpeg)
}

fn encode(width: usize, height: usize, format: image::ImageFormat) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 8) as u8);
            pixels.push((y * 8) as u8);
            pixels.push(0);
        }
    }
    let buffer: image::RgbImage = image::ImageBuffer::from_raw(width as u32, height as u32, pixels)
        .expect("the pixel buffer matches the extent");
    let mut encoded = Vec::new();
    image::DynamicImage::ImageRgb8(buffer)
        .write_to(&mut std::io::Cursor::new(&mut encoded), format)
        .expect("the encoder runs");
    encoded
}

/// The `Fields` of an embedded camera column.
fn image_struct_fields() -> arrow_schema::Fields {
    arrow_schema::Fields::from(vec![
        arrow_schema::Field::new("bytes", arrow_schema::DataType::Binary, true),
        arrow_schema::Field::new("path", arrow_schema::DataType::Utf8, true),
    ])
}

/// Replace the camera column of a copied fixture's data file, column by column.
///
/// `build` receives the row count and returns the replacement column, or `None` to drop
/// the camera entirely. Every other column, and the schema of every other column, is
/// the fixture's own.
fn rewrite_image_column(
    root: &Path,
    build: impl FnOnce(usize) -> Option<(arrow_schema::DataType, arrow_array::ArrayRef)>,
) {
    use arrow_array::{ArrayRef, RecordBatch};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let path = root.join("data/chunk-000/file-000.parquet");
    let existing = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
        std::fs::File::open(&path).expect("the fixture data file opens"),
    )
    .expect("it is parquet")
    .build()
    .expect("the reader builds")
    .next()
    .expect("it has a batch")
    .expect("the batch decodes");

    let replacement = build(existing.num_rows());
    let mut fields: Vec<Arc<arrow_schema::Field>> = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();
    for (index, field) in existing.schema().fields().iter().enumerate() {
        if field.name() == CAMERA_KEY {
            if let Some((data_type, array)) = &replacement {
                fields.push(Arc::new(arrow_schema::Field::new(
                    CAMERA_KEY,
                    data_type.clone(),
                    true,
                )));
                columns.push(Arc::clone(array));
            }
            continue;
        }
        fields.push(Arc::clone(field));
        columns.push(Arc::clone(existing.column(index)));
    }

    let schema = Arc::new(arrow_schema::Schema::new(arrow_schema::Fields::from(
        fields,
    )));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), columns).expect("the frame batch is well formed");
    let file = std::fs::File::create(&path).expect("cannot create the data file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("cannot open the writer");
    writer.write(&batch).expect("cannot write the frames");
    writer.close().expect("cannot close the writer");
}

/// Replace every camera cell's bytes, keeping the column's arrow type.
pub fn rewrite_image_cells(root: &Path, cells: &[Vec<u8>]) {
    use arrow_array::{ArrayRef, BinaryArray, StringArray, StructArray};
    use std::sync::Arc;

    let cells = cells.to_vec();
    rewrite_image_column(root, move |rows| {
        assert_eq!(cells.len(), rows, "one replacement cell per frame");
        let bytes = Arc::new(BinaryArray::from(
            cells
                .iter()
                .map(|cell| Some(cell.as_slice()))
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        let paths = Arc::new(StringArray::from(vec![Some("frame.png"); rows])) as ArrayRef;
        let fields = image_struct_fields();
        Some((
            arrow_schema::DataType::Struct(fields.clone()),
            Arc::new(StructArray::new(fields, vec![bytes, paths], None)) as ArrayRef,
        ))
    });
}

/// Null the `bytes` field of one row, keeping every other row intact.
pub fn rewrite_image_cells_with_nulls(root: &Path, null_row: usize) {
    use arrow_array::{ArrayRef, BinaryArray, StringArray, StructArray};
    use std::sync::Arc;

    let png = png_of(32, 32);
    rewrite_image_column(root, move |rows| {
        let bytes = Arc::new(BinaryArray::from(
            (0..rows)
                .map(|row| {
                    if row == null_row {
                        None
                    } else {
                        Some(png.as_slice())
                    }
                })
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        let paths = Arc::new(StringArray::from(vec![Some("frame.png"); rows])) as ArrayRef;
        let fields = image_struct_fields();
        Some((
            arrow_schema::DataType::Struct(fields.clone()),
            Arc::new(StructArray::new(fields, vec![bytes, paths], None)) as ArrayRef,
        ))
    });
}

/// Remove the camera column while leaving `info.json` declaring it.
pub fn drop_image_column(root: &Path) {
    rewrite_image_column(root, |_| None);
}

/// Replace the camera column with a bare `Binary` one, which is the wrong spelling.
pub fn replace_image_column_with_binary(root: &Path) {
    use arrow_array::{ArrayRef, BinaryArray};
    use std::sync::Arc;

    let png = png_of(32, 32);
    rewrite_image_column(root, move |rows| {
        Some((
            arrow_schema::DataType::Binary,
            Arc::new(BinaryArray::from(vec![Some(png.as_slice()); rows])) as ArrayRef,
        ))
    });
}

/// Add a `dtype: "video"` feature to a copied fixture's `meta/info.json`.
pub fn declare_video_feature(root: &Path, key: &str) {
    let path = root.join("meta/info.json");
    let text = std::fs::read_to_string(&path).expect("the fixture has an info.json");
    let anchor = "        \"action\": {";
    assert!(
        text.contains(anchor),
        "info.json does not declare \"action\" where expected"
    );
    let declaration = format!(
        "        \"{key}\": {{\n\
         \x20           \"dtype\": \"video\",\n\
         \x20           \"shape\": [\n\
         \x20               3,\n\
         \x20               32,\n\
         \x20               32\n\
         \x20           ],\n\
         \x20           \"names\": null\n\
         \x20       }},\n"
    );
    std::fs::write(
        &path,
        text.replacen(anchor, &format!("{declaration}{anchor}"), 1),
    )
    .expect("cannot write the rewritten info.json");
}

/// Rewrite the single episode row of a copied fixture with the given boundaries.
///
/// Only the seven columns `DatasetMetadata` reads are written. The `stats/*` columns
/// upstream also stores are dropped, which is deliberate: the reader must not depend
/// on them, and a test that silently required them would be asserting the wrong
/// thing.
pub fn rewrite_episode_row(root: &Path, from: i64, to: i64, length: i64) {
    rewrite_episode_rows(root, &[(0, from, to, length)]);
}

/// Rewrite the episode table with several rows, as `(episode_index, from, to, length)`.
///
/// Needed to reach the invariants a single row cannot violate: duplicate indices, gaps
/// and overlaps between ranges, and coverage of the declared frame domain.
pub fn rewrite_episode_rows(root: &Path, rows: &[(i64, i64, i64, i64)]) {
    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let mut tasks = ListBuilder::new(StringBuilder::new());
    for _ in rows {
        tasks.values().append_value("reach the target");
        tasks.append(true);
    }
    let tasks = Arc::new(tasks.finish()) as ArrayRef;

    let schema = Arc::new(Schema::new(Fields::from(vec![
        Field::new("episode_index", DataType::Int64, false),
        Field::new(
            "tasks",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new("length", DataType::Int64, false),
        Field::new("data/chunk_index", DataType::Int64, false),
        Field::new("data/file_index", DataType::Int64, false),
        Field::new("dataset_from_index", DataType::Int64, false),
        Field::new("dataset_to_index", DataType::Int64, false),
    ])));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.0).collect::<Vec<_>>(),
            )) as ArrayRef,
            tasks,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.3).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(vec![0i64; rows.len()])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0i64; rows.len()])) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.1).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.2).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .expect("the episode batch is well formed");

    let path = root.join("meta/episodes/chunk-000/file-000.parquet");
    let file = std::fs::File::create(&path).expect("cannot create the episode file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("cannot open the writer");
    writer.write(&batch).expect("cannot write the episode row");
    writer.close().expect("cannot close the writer");
}

/// Rewrite one feature's declared `shape` in a copied fixture's `meta/info.json`.
///
/// The shape is the one number every later allocation is derived from, and nothing
/// downstream re-derives it from the parquet, so a declared shape that disagrees with
/// the stored data has to be caught where it is declared.
pub fn rewrite_feature_shape(root: &Path, feature: &str, shape: &[i64]) {
    let path = root.join("meta/info.json");
    let text = std::fs::read_to_string(&path).expect("the fixture has an info.json");
    // Rewritten as text rather than through a JSON library so the test depends on
    // nothing the reader under test also depends on.
    let key = format!("\"{feature}\": {{");
    let start = text.find(&key).expect("the fixture declares the feature");
    let shape_at = text[start..]
        .find("\"shape\": [")
        .expect("the feature has a shape")
        + start;
    let open = shape_at + "\"shape\": [".len();
    let close = text[open..].find(']').expect("the shape closes") + open;
    let replacement = shape
        .iter()
        .map(|dimension| dimension.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mut rewritten = String::with_capacity(text.len());
    rewritten.push_str(&text[..open]);
    rewritten.push_str(&replacement);
    rewritten.push_str(&text[close..]);
    std::fs::write(&path, rewritten).expect("cannot write the rewritten info.json");
}

/// Rewrite one feature's declared `names` in a copied fixture's `meta/info.json`.
///
/// `names` is not decoration for a camera: it is what tells a reader whether the
/// three numbers beside it are `[channel, height, width]` or `[height, width,
/// channel]`, and the two are not distinguishable by the numbers alone.
pub fn rewrite_feature_names(root: &Path, feature: &str, names: &[&str]) {
    let path = root.join("meta/info.json");
    let text = std::fs::read_to_string(&path).expect("the fixture has an info.json");
    // Text, for the same reason `rewrite_feature_shape` is text.
    let key = format!("\"{feature}\": {{");
    let start = text.find(&key).expect("the fixture declares the feature");
    let names_at = text[start..]
        .find("\"names\": [")
        .expect("the feature has a names list")
        + start;
    let open = names_at + "\"names\": [".len();
    let close = text[open..].find(']').expect("the names list closes") + open;
    let replacement = names
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let mut rewritten = String::with_capacity(text.len());
    rewritten.push_str(&text[..open]);
    rewritten.push_str(&replacement);
    rewritten.push_str(&text[close..]);
    std::fs::write(&path, rewritten).expect("cannot write the rewritten info.json");
}

/// Rewrite the `episode_index` column of a copied fixture's data file.
///
/// The other half of `rewrite_episode_rows`: the frame rows carry their own
/// `episode_index`, and the two sources of truth have to agree. Rewriting only one of
/// them is how a test reaches the disagreement.
pub fn rewrite_frame_episode_indices(root: &Path, indices: &[i64]) {
    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let path = root.join("data/chunk-000/file-000.parquet");
    // Read the existing values back so only `episode_index` changes.
    let existing = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
        std::fs::File::open(&path).expect("the fixture data file opens"),
    )
    .expect("it is parquet")
    .build()
    .expect("the reader builds")
    .next()
    .expect("it has a batch")
    .expect("the batch decodes");

    assert_eq!(
        indices.len(),
        existing.num_rows(),
        "the replacement episode_index column must have one entry per frame"
    );

    // The schema is the fixture's own, not a hand-written copy of it: an arrow
    // `FixedSizeList`'s child field carries a name and a nullability flag, and
    // reproducing them by hand is how this helper first produced a batch the writer
    // refused. Reusing the schema means only the one column being tested changes.
    let schema = existing.schema();
    let replaced: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .map(|field| match field.name().as_str() {
            "episode_index" => Arc::new(Int64Array::from(indices.to_vec())) as ArrayRef,
            name => Arc::clone(existing.column_by_name(name).expect("the fixture has it")),
        })
        .collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), replaced)
        .expect("the frame batch is well formed");

    let file = std::fs::File::create(&path).expect("cannot create the data file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("cannot open the writer");
    writer.write(&batch).expect("cannot write the frames");
    writer.close().expect("cannot close the writer");
}
