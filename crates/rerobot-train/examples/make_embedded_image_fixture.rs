//! Write the committed embedded-image dataset fixture,
//! `tests/fixtures/embedded_image/`.
//!
//! Run once, and then never again unless the fixture has to change:
//!
//! ```shell
//! cargo run -p rerobot-train --example make_embedded_image_fixture
//! ```
//!
//! # Why this one generator is Rust rather than Python
//!
//! Every other fixture under `tests/fixtures/` was written by upstream `lerobot`
//! itself (see `tools/goldens/README.md`), which is what makes the reader's tests
//! checks against upstream's writer rather than against a guess at it. This one
//! cannot be: producing an `image` feature through `LeRobotDataset.add_frame` needs
//! `torch`, `datasets` and `PIL` installed, and the point of the fixture is an offline
//! test of a decoder that must never depend on any of them.
//!
//! So it is built out of upstream's own state-only fixture instead. Every byte except
//! the camera comes from `tools/goldens/make_dataset_fixture.py`'s output: the episode
//! table, the task table, the statistics, the four frame rows and their arrow types are
//! copied through unchanged. What this adds is one column in upstream's spelling for a
//! `datasets.Image()` feature — `struct<bytes: binary, path: string>`, the frames
//! embedded in the data file — and the matching `info.json` entry.
//!
//! It is deterministic: the pixels come from the formula in [`frame_png`], so
//! re-running it reproduces the same images, and only the parquet footer (which embeds
//! the writer's version string) can differ.
//!
//! # What the fixture deliberately does not carry
//!
//! No `meta/stats.json` entry for the camera. Upstream writes per-channel `(3, 1, 1)`
//! statistics for an image feature, and `rerobot_core::dataset::stats::load_stats`
//! refuses a nested statistic by design. This slice never consumes a camera's own
//! statistics anyway: `dataset.use_imagenet_stats` selects between `IMAGENET_STATS` and
//! leaving frames untouched, so the entry would be read and then ignored. A dataset
//! that *does* carry it is refused by the stats loader, which `docs/compatibility.md`
//! records.

use arrow_array::{ArrayRef, BinaryArray, RecordBatch, StringArray, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The camera key, in upstream's `observation.images.<name>` spelling.
const CAMERA: &str = "observation.images.top";

/// The side of every frame. `resnet18` divides by 32, so this is the smallest extent
/// that still runs the whole stem and all four stages of the backbone.
const EXTENT: u32 = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = crate_root.join("tests/fixtures/state_only");
    let destination = crate_root.join("tests/fixtures/embedded_image");

    if !source.join("meta/info.json").is_file() {
        return Err(format!("the state-only fixture is missing at {}", source.display()).into());
    }
    let _ = std::fs::remove_dir_all(&destination);
    copy_tree(&source, &destination)?;

    add_image_column(&destination)?;
    declare_image_feature(&destination)?;

    println!("wrote {}", destination.display());
    Ok(())
}

/// One frame's PNG, from a formula rather than from a file.
///
/// Channel 0 varies along x, channel 1 along y and channel 2 with the frame number, so
/// a test can assert an exact pixel and a transposition or an off-by-one in the CHW
/// conversion cannot pass.
fn frame_png(frame: u32) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut pixels = Vec::with_capacity((EXTENT * EXTENT * 3) as usize);
    for y in 0..EXTENT {
        for x in 0..EXTENT {
            pixels.push((x * 8) as u8);
            pixels.push((y * 8) as u8);
            pixels.push((frame * 64) as u8);
        }
    }
    let buffer: image::RgbImage =
        image::ImageBuffer::from_raw(EXTENT, EXTENT, pixels).ok_or("the pixel buffer is ragged")?;
    let mut encoded = Vec::new();
    image::DynamicImage::ImageRgb8(buffer).write_to(
        &mut std::io::Cursor::new(&mut encoded),
        image::ImageFormat::Png,
    )?;
    Ok(encoded)
}

/// Rewrite the data file with the camera column appended to upstream's own columns.
fn add_image_column(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let path = root.join("data/chunk-000/file-000.parquet");
    let existing = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path)?)?
        .build()?
        .next()
        .ok_or("the state-only data file has no batch")??;

    let struct_fields = image_struct_fields();
    let mut fields: Vec<Arc<Field>> = existing.schema().fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(
        CAMERA,
        DataType::Struct(struct_fields.clone()),
        false,
    )));
    let schema = Arc::new(Schema::new(Fields::from(fields)));

    let rows = existing.num_rows();
    let encoded: Vec<Vec<u8>> = (0..rows)
        .map(|frame| frame_png(frame as u32))
        .collect::<Result<_, _>>()?;
    let bytes = Arc::new(BinaryArray::from(
        encoded
            .iter()
            .map(|png| Some(png.as_slice()))
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    // Upstream's `path` is the file the frame would have lived in had the dataset been
    // written with separate image files; it travels with the bytes either way.
    let names: Vec<String> = (0..rows)
        .map(|frame| format!("images/{CAMERA}/episode-000000/frame-{frame:06}.png"))
        .collect();
    let paths = Arc::new(StringArray::from(
        names
            .iter()
            .map(|name| Some(name.as_str()))
            .collect::<Vec<_>>(),
    )) as ArrayRef;

    let mut columns: Vec<ArrayRef> = existing.columns().to_vec();
    columns.push(Arc::new(StructArray::new(struct_fields, vec![bytes, paths], None)) as ArrayRef);

    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    let mut writer = ArrowWriter::try_new(std::fs::File::create(&path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// `struct<bytes: binary, path: string>`, which is what a `datasets.Image()` feature
/// is written as.
fn image_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("bytes", DataType::Binary, true),
        Field::new("path", DataType::Utf8, true),
    ])
}

/// Add the camera to `meta/info.json`, in upstream's formatting.
///
/// Edited as text rather than through a JSON library so that every other byte of the
/// file upstream wrote — key order, indentation, the trailing newline — survives
/// exactly.
fn declare_image_feature(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let path = root.join("meta/info.json");
    let text = std::fs::read_to_string(&path)?;
    let anchor = "        \"action\": {";
    if !text.contains(anchor) {
        return Err("info.json does not declare \"action\" where expected".into());
    }
    let declaration = format!(
        "        \"{CAMERA}\": {{\n\
         \x20           \"dtype\": \"image\",\n\
         \x20           \"shape\": [\n\
         \x20               3,\n\
         \x20               {EXTENT},\n\
         \x20               {EXTENT}\n\
         \x20           ],\n\
         \x20           \"names\": [\n\
         \x20               \"channels\",\n\
         \x20               \"height\",\n\
         \x20               \"width\"\n\
         \x20           ]\n\
         \x20       }},\n"
    );
    std::fs::write(
        &path,
        text.replacen(anchor, &format!("{declaration}{anchor}"), 1),
    )?;
    Ok(())
}

fn copy_tree(from: &Path, to: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}
