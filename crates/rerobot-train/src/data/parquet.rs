//! The narrow parquet reader the local dataset slice needs.
//!
//! Upstream reaches a parquet file through `datasets.load_dataset` and pandas,
//! which will coerce almost anything into almost anything. This reader does the
//! opposite: it accepts exactly the arrow types upstream's writer produces and
//! refuses everything else with the column name in the message, so a dataset
//! this slice cannot read says so instead of being read wrongly.
//!
//! The types accepted per column kind, and where they come from:
//!
//! | Column kind | Arrow type | Written by |
//! | --- | --- | --- |
//! | `observation.*`, `action` | `FixedSizeList<Float32>[n]`, or `List<Float32>` | `datasets.Sequence(Value("float32"), length=n)` |
//! | `timestamp` | `Float32` or `Float64` | `datasets.Value("float32")` |
//! | `frame_index`, `episode_index`, `index`, `task_index`, `length`, `dataset_*_index` | `Int64` or `Int32` | `datasets.Value("int64")` |
//! | `task` | `Utf8` or `LargeUtf8` | pandas' string index |
//! | `tasks` | `List<Utf8>` or `LargeList<Utf8>` | `datasets.Sequence(Value("string"))` |
//! | `observation.images.*` | `Struct<bytes: Binary, path: Utf8>` | `datasets.Image()` |
//!
//! `List<Float32>` is accepted alongside `FixedSizeList` because a dataset
//! converted from v2.1 carries the variable-length spelling of the same data; the
//! per-row width is checked either way.

use crate::error::{Result, TrainError};
use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

type ArrayWithOffsets<'a> = (&'a dyn Array, Box<dyn Fn(usize) -> (usize, usize) + 'a>);

/// Every row group of one parquet file, concatenated column-wise.
///
/// A LeRobot data file holds one chunk, which upstream caps at 100 MB, so
/// materializing it is bounded. `MAX_ROWS` is the backstop against a file that
/// claims otherwise.
#[derive(Debug)]
pub struct Table {
    batches: Vec<RecordBatch>,
    rows: usize,
    budget: ReadBudget,
    declared_decoded_bytes: usize,
}

/// What one parquet file may cost before the reader refuses to open it.
///
/// A budget rather than a set of constants because the checks have to be testable:
/// proving that an oversized file is refused otherwise needs an oversized file, and a
/// gibibyte fixture is not something to commit. Production reads use
/// [`ReadBudget::default`], which is the documented budget in
/// [`crate::limits`]; `tests/parquet_budget.rs` shrinks it so a small file exceeds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadBudget {
    /// Bytes the file on disk may occupy.
    pub max_file_bytes: u64,
    /// Rows it may declare, checked against the footer *before* any decoding.
    pub max_rows: usize,
    /// Columns it may declare.
    pub max_columns: usize,
    /// Decoded scalars any one column may materialize.
    pub max_values: usize,
    /// UTF-8 bytes a string column may hold in total.
    pub max_string_bytes: usize,
    /// Elements a list column may hold in total.
    pub max_list_elements: usize,
    /// Cells — `rows * columns` — the footer may declare.
    ///
    /// Rows, columns and compressed bytes bounded separately do not bound a decode: a
    /// thousand rows of a thousand wide columns is inside all three and still a
    /// gibibyte of cells. The product is what the work costs, and the footer states
    /// both factors.
    pub max_cells: usize,
    /// Uncompressed bytes the footer may declare across every column.
    ///
    /// Cells do not bound bytes either: one cell of a wide list column costs far more
    /// than one cell of an `int64`.
    pub max_decoded_bytes: usize,
    /// Compressed bytes one embedded camera cell may carry.
    ///
    /// Per *cell*, not per column: an image column inside every other bound can still
    /// hold one binary value large enough to be a decompression bomb on its own, and
    /// the decoder is where that would be paid for.
    pub max_image_bytes: usize,
}

impl Default for ReadBudget {
    fn default() -> Self {
        Self {
            max_file_bytes: crate::limits::MAX_PARQUET_FILE_BYTES,
            max_rows: crate::limits::MAX_DATASET_ROWS,
            max_columns: crate::limits::MAX_PARQUET_COLUMNS,
            max_values: crate::limits::MAX_DECODED_VALUES,
            max_string_bytes: crate::limits::MAX_STRING_BYTES,
            max_list_elements: crate::limits::MAX_LIST_ELEMENTS,
            max_cells: crate::limits::MAX_PARQUET_CELLS,
            max_decoded_bytes: crate::limits::MAX_DECODED_BYTES,
            max_image_bytes: crate::limits::MAX_EMBEDDED_IMAGE_BYTES,
        }
    }
}

impl Table {
    /// Read every row group of `path` within the default budget.
    pub fn read(path: &Path) -> Result<Self> {
        Self::read_within(path, &ReadBudget::default())
    }

    /// Read every row group of `path`, refusing anything outside `budget`.
    ///
    /// The order of the checks is the point. The file's size and its declared row and
    /// column counts are read from the footer and checked *before* a single batch is
    /// decoded, because the previous version checked rows only after Arrow had already
    /// allocated and decoded each batch — by which time the work had been done and the
    /// memory taken. A file claiming ten billion rows now costs one footer read.
    pub fn read_within(path: &Path, budget: &ReadBudget) -> Result<Self> {
        let metadata = std::fs::metadata(path).map_err(|error| TrainError::io(path, &error))?;
        if metadata.len() > budget.max_file_bytes {
            return Err(TrainError::column(
                path,
                format!(
                    "is {} bytes, above the {} the reader will open",
                    metadata.len(),
                    budget.max_file_bytes
                ),
            ));
        }

        let file = File::open(path).map_err(|error| TrainError::io(path, &error))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|error| TrainError::column(path, error.to_string()))?;

        // The footer, before any column is touched.
        let declared_rows = builder.metadata().file_metadata().num_rows();
        let declared_rows = usize::try_from(declared_rows).map_err(|_| {
            TrainError::column(
                path,
                format!("declares {declared_rows} rows, which is not a count"),
            )
        })?;
        if declared_rows > budget.max_rows {
            return Err(TrainError::column(
                path,
                format!(
                    "declares {declared_rows} rows, above the {} the reader will decode",
                    budget.max_rows
                ),
            ));
        }
        let columns = builder
            .metadata()
            .file_metadata()
            .schema_descr()
            .num_columns();
        if columns > budget.max_columns {
            return Err(TrainError::column(
                path,
                format!(
                    "declares {columns} columns, above the {} the reader will decode",
                    budget.max_columns
                ),
            ));
        }

        // The two *aggregate* budgets, still from the footer. Each factor being inside
        // its own limit says nothing about the product, and the per-column checks
        // further down run only after Arrow has decoded every batch -- by which point
        // the allocation has happened. These are the last checks that cost nothing.
        let cells = crate::limits::checked_mul(declared_rows, columns, "the declared cell count")?;
        if cells > budget.max_cells {
            return Err(TrainError::column(
                path,
                format!(
                    "declares {declared_rows} rows by {columns} columns = {cells} cells, above \
                     the {} the reader will decode",
                    budget.max_cells
                ),
            ));
        }
        let decoded_bytes = declared_decoded_bytes_of(builder.metadata());
        if decoded_bytes > budget.max_decoded_bytes {
            return Err(TrainError::column(
                path,
                format!(
                    "declares {decoded_bytes} uncompressed bytes, above the {} the reader will \
                     decode",
                    budget.max_decoded_bytes
                ),
            ));
        }

        let reader = builder
            .build()
            .map_err(|error| TrainError::column(path, error.to_string()))?;
        let mut batches = Vec::new();
        let mut rows = 0usize;
        for batch in reader {
            let batch = batch.map_err(|error| TrainError::column(path, error.to_string()))?;
            rows = crate::limits::checked_add(rows, batch.num_rows(), "the row count")?;
            // Re-checked while decoding: the footer is data too, and a file whose
            // row groups hold more rows than its footer claims must not be read past
            // the budget just because the footer lied.
            if rows > budget.max_rows {
                return Err(TrainError::column(
                    path,
                    format!(
                        "holds more than the {} rows the reader will decode",
                        budget.max_rows
                    ),
                ));
            }
            batches.push(batch);
        }
        Ok(Self {
            batches,
            rows,
            budget: *budget,
            declared_decoded_bytes: decoded_bytes,
        })
    }

    /// The uncompressed size every column of this file declared in its footer.
    ///
    /// Exposed so a caller accumulating a dataset-wide budget can add it up, and so
    /// `tests/parquet_budget.rs` can shrink the budget to just below a real file's own
    /// figure rather than needing an oversized fixture.
    pub fn declared_decoded_bytes(&self) -> usize {
        self.declared_decoded_bytes
    }

    /// The budget this table was read within, so column decoding uses the same one.
    pub fn budget(&self) -> &ReadBudget {
        &self.budget
    }

    /// How many rows the file holds.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Column names, in file order.
    pub fn column_names(&self) -> Vec<String> {
        self.batches
            .first()
            .map(|batch| {
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Whether a column is present.
    pub fn has_column(&self, name: &str) -> bool {
        self.batches
            .first()
            .is_some_and(|batch| batch.schema().index_of(name).is_ok())
    }

    fn columns<'a>(&'a self, path: &Path, name: &str) -> Result<Vec<&'a dyn Array>> {
        let mut out = Vec::with_capacity(self.batches.len());
        for batch in &self.batches {
            let index = batch.schema().index_of(name).map_err(|_| {
                TrainError::column(
                    path,
                    format!(
                        "column {name:?} is missing; the file has {:?}",
                        self.column_names()
                    ),
                )
            })?;
            out.push(batch.column(index).as_ref());
        }
        Ok(out)
    }

    /// An `Int64`-shaped column as `i64`.
    pub fn i64_column(&self, path: &Path, name: &str) -> Result<Vec<i64>> {
        let mut out = Vec::with_capacity(self.rows);
        for column in self.columns(path, name)? {
            match column.data_type() {
                DataType::Int64 => {
                    let values = column.as_primitive::<arrow_array::types::Int64Type>();
                    for row in 0..values.len() {
                        out.push(if nonnull(path, name, row, values.is_null(row))? {
                            values.value(row)
                        } else {
                            i64::default()
                        });
                    }
                }
                DataType::Int32 => {
                    let values = column.as_primitive::<arrow_array::types::Int32Type>();
                    for row in 0..values.len() {
                        nonnull(path, name, row, values.is_null(row))?;
                        out.push(i64::from(values.value(row)));
                    }
                }
                other => {
                    return Err(TrainError::column(
                        path,
                        format!("column {name:?} must be int64 or int32, found {other}"),
                    ))
                }
            }
        }
        Ok(out)
    }

    /// A `Float32`/`Float64`-shaped column as `f32`.
    pub fn f32_column(&self, path: &Path, name: &str) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(self.rows);
        for column in self.columns(path, name)? {
            match column.data_type() {
                DataType::Float32 => {
                    let values = column.as_primitive::<arrow_array::types::Float32Type>();
                    for row in 0..values.len() {
                        nonnull(path, name, row, values.is_null(row))?;
                        out.push(values.value(row));
                    }
                }
                DataType::Float64 => {
                    let values = column.as_primitive::<arrow_array::types::Float64Type>();
                    for row in 0..values.len() {
                        nonnull(path, name, row, values.is_null(row))?;
                        out.push(values.value(row) as f32);
                    }
                }
                other => {
                    return Err(TrainError::column(
                        path,
                        format!("column {name:?} must be float32 or float64, found {other}"),
                    ))
                }
            }
        }
        Ok(out)
    }

    /// A fixed-width vector column as one `Vec<f32>` of `width` per row.
    ///
    /// The `rows * width` product is checked and budgeted before anything is
    /// allocated: both operands come from the file, and their product is what the
    /// decode actually costs. Rows alone do not bound it — one row of a very wide
    /// feature costs as much as a million narrow ones.
    pub fn vector_column(&self, path: &Path, name: &str, width: usize) -> Result<Vec<Vec<f32>>> {
        let values = crate::limits::checked_mul(
            self.rows,
            width,
            &format!("the decoded size of column {name:?}"),
        )?;
        if values > self.budget.max_values {
            return Err(TrainError::column(
                path,
                format!(
                    "column {name:?} would decode {values} values ({} rows of width {width}), \
                     above the {} the reader will materialize",
                    self.rows, self.budget.max_values
                ),
            ));
        }
        let mut out = Vec::with_capacity(self.rows);
        for column in self.columns(path, name)? {
            let (values, offsets): ArrayWithOffsets<'_> = match column.data_type() {
                DataType::FixedSizeList(field, size) => {
                    require_float32(path, name, field.data_type())?;
                    let list = column.as_fixed_size_list();
                    let size = *size as usize;
                    (
                        list.values().as_ref(),
                        Box::new(move |row| (row * size, size)),
                    )
                }
                DataType::List(field) => {
                    require_float32(path, name, field.data_type())?;
                    let list = column.as_list::<i32>();
                    let raw = list.offsets().clone();
                    (
                        list.values().as_ref(),
                        Box::new(move |row| {
                            let start = raw[row] as usize;
                            let end = raw[row + 1] as usize;
                            (start, end - start)
                        }),
                    )
                }
                DataType::LargeList(field) => {
                    require_float32(path, name, field.data_type())?;
                    let list = column.as_list::<i64>();
                    let raw = list.offsets().clone();
                    (
                        list.values().as_ref(),
                        Box::new(move |row| {
                            let start = raw[row] as usize;
                            let end = raw[row + 1] as usize;
                            (start, end - start)
                        }),
                    )
                }
                other => {
                    return Err(TrainError::column(
                        path,
                        format!("column {name:?} must be a list of float32, found {other}"),
                    ))
                }
            };
            let scalars = values.as_primitive::<arrow_array::types::Float32Type>();
            for row in 0..column.len() {
                nonnull(path, name, row, column.is_null(row))?;
                let (start, length) = offsets(row);
                if length != width {
                    return Err(TrainError::column(
                        path,
                        format!(
                            "column {name:?} row {row} has width {length}, but the feature \
                             declares {width}"
                        ),
                    ));
                }
                out.push((start..start + length).map(|i| scalars.value(i)).collect());
            }
        }
        Ok(out)
    }

    /// Embedded LeRobot v3.0 camera cells, as `(encoded_bytes, path)` per row.
    ///
    /// The accepted Arrow spelling is exactly `struct<bytes: binary, path: string>`,
    /// which is what a `datasets.Image()` feature writes. Anything else is refused by
    /// name rather than coerced, on the same principle as every other column here.
    ///
    /// Every cell is bounded before it is copied — per cell against
    /// [`ReadBudget::max_image_bytes`] and cumulatively against
    /// [`ReadBudget::max_decoded_bytes`] — because a binary column is the one place
    /// where a row count says nothing about the cost of reading it. Decoding, the
    /// format allow-list and the shape check all happen afterwards, in
    /// [`crate::data::image::DecodedImage::from_encoded`].
    ///
    /// # Errors
    ///
    /// [`TrainError::Column`] naming the file, the column and the row: a wrong Arrow
    /// type, a null cell or null `bytes` field, or a cell or column past its budget.
    pub fn image_column(&self, path: &Path, name: &str) -> Result<Vec<(Vec<u8>, Option<String>)>> {
        let mut out = Vec::with_capacity(self.rows);
        let mut encoded_bytes = 0usize;
        let mut row_offset = 0usize;
        for column in self.columns(path, name)? {
            let structure = column
                .as_any()
                .downcast_ref::<arrow_array::StructArray>()
                .ok_or_else(|| {
                    TrainError::column(
                        path,
                        format!(
                            "column {name:?} must be struct<bytes: binary, path: string>, found {}",
                            column.data_type()
                        ),
                    )
                })?;
            let fields = structure.fields();
            let has_expected_fields = fields.len() == 2
                && fields
                    .iter()
                    .any(|field| field.name() == "bytes" && field.data_type() == &DataType::Binary)
                && fields
                    .iter()
                    .any(|field| field.name() == "path" && field.data_type() == &DataType::Utf8);
            if !has_expected_fields {
                return Err(TrainError::column(
                    path,
                    format!(
                        "column {name:?} must be struct<bytes: binary, path: string>, found {}",
                        column.data_type()
                    ),
                ));
            }
            let bytes = structure
                .column_by_name("bytes")
                .expect("the struct field was checked above")
                .as_binary::<i32>();
            let paths = structure
                .column_by_name("path")
                .expect("the struct field was checked above")
                .as_string::<i32>();
            for row in 0..column.len() {
                nonnull(path, name, row_offset + row, column.is_null(row))?;
                nonnull(path, name, row_offset + row, bytes.is_null(row))?;
                let encoded = bytes.value(row);
                if encoded.len() > self.budget.max_image_bytes {
                    return Err(TrainError::column(
                        path,
                        format!(
                            "column {name:?} row {} carries {} encoded bytes, above the {} the \
                             reader will hand to an image decoder",
                            row_offset + row,
                            encoded.len(),
                            self.budget.max_image_bytes
                        ),
                    ));
                }
                encoded_bytes = crate::limits::checked_add(
                    encoded_bytes,
                    encoded.len(),
                    &format!("the encoded size of image column {name:?}"),
                )?;
                if encoded_bytes > self.budget.max_decoded_bytes {
                    return Err(TrainError::column(
                        path,
                        format!(
                            "column {name:?} holds {encoded_bytes} encoded bytes, above the {} \
                             the reader will materialize",
                            self.budget.max_decoded_bytes
                        ),
                    ));
                }
                let optional_path = if paths.is_null(row) {
                    None
                } else {
                    Some(paths.value(row).to_owned())
                };
                out.push((encoded.to_vec(), optional_path));
            }
            row_offset =
                crate::limits::checked_add(row_offset, column.len(), "the image row count")?;
        }
        Ok(out)
    }

    /// A string column.
    ///
    /// Bounded by total decoded bytes, not by row count: a single row may hold an
    /// arbitrarily long string, and `task` is read into an owned `String` per row.
    pub fn string_column(&self, path: &Path, name: &str) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(self.rows);
        let mut bytes = 0usize;
        for column in self.columns(path, name)? {
            bytes = crate::limits::checked_add(
                bytes,
                string_column_bytes(path, name, column)?,
                &format!("the decoded size of column {name:?}"),
            )?;
            if bytes > self.budget.max_string_bytes {
                return Err(TrainError::column(
                    path,
                    format!(
                        "column {name:?} holds {bytes} bytes of text, above the {} the reader \
                         will materialize",
                        self.budget.max_string_bytes
                    ),
                ));
            }
        }
        for column in self.columns(path, name)? {
            match column.data_type() {
                DataType::Utf8 => {
                    let values = column.as_string::<i32>();
                    for row in 0..values.len() {
                        nonnull(path, name, row, values.is_null(row))?;
                        out.push(values.value(row).to_owned());
                    }
                }
                DataType::LargeUtf8 => {
                    let values = column.as_string::<i64>();
                    for row in 0..values.len() {
                        nonnull(path, name, row, values.is_null(row))?;
                        out.push(values.value(row).to_owned());
                    }
                }
                other => {
                    return Err(TrainError::column(
                        path,
                        format!("column {name:?} must be a string, found {other}"),
                    ))
                }
            }
        }
        Ok(out)
    }

    /// A list-of-strings column, one `Vec<String>` per row.
    ///
    /// Bounded by total element count and total text: a list column's rows have no
    /// fixed width, so neither the row count nor a per-row cap bounds the decode.
    pub fn string_list_column(&self, path: &Path, name: &str) -> Result<Vec<Vec<String>>> {
        let mut out = Vec::with_capacity(self.rows);
        let mut elements = 0usize;
        let mut bytes = 0usize;
        for column in self.columns(path, name)? {
            let (column_elements, column_bytes) = list_column_extent(path, name, column)?;
            elements = crate::limits::checked_add(
                elements,
                column_elements,
                &format!("the element count of column {name:?}"),
            )?;
            bytes = crate::limits::checked_add(
                bytes,
                column_bytes,
                &format!("the decoded size of column {name:?}"),
            )?;
            if elements > self.budget.max_list_elements {
                return Err(TrainError::column(
                    path,
                    format!(
                        "column {name:?} holds {elements} list elements, above the {} the reader \
                         will materialize",
                        self.budget.max_list_elements
                    ),
                ));
            }
            if bytes > self.budget.max_string_bytes {
                return Err(TrainError::column(
                    path,
                    format!(
                        "column {name:?} holds {bytes} bytes of text, above the {} the reader \
                         will materialize",
                        self.budget.max_string_bytes
                    ),
                ));
            }
        }
        for column in self.columns(path, name)? {
            let (values, offsets): ArrayWithOffsets<'_> = match column.data_type() {
                DataType::List(_) => {
                    let list = column.as_list::<i32>();
                    let raw = list.offsets().clone();
                    (
                        list.values().as_ref(),
                        Box::new(move |row| {
                            (raw[row] as usize, (raw[row + 1] - raw[row]) as usize)
                        }),
                    )
                }
                DataType::LargeList(_) => {
                    let list = column.as_list::<i64>();
                    let raw = list.offsets().clone();
                    (
                        list.values().as_ref(),
                        Box::new(move |row| {
                            (raw[row] as usize, (raw[row + 1] - raw[row]) as usize)
                        }),
                    )
                }
                other => {
                    return Err(TrainError::column(
                        path,
                        format!("column {name:?} must be a list of strings, found {other}"),
                    ))
                }
            };
            let strings = match values.data_type() {
                DataType::Utf8 => StringView::Small(values.as_string::<i32>()),
                DataType::LargeUtf8 => StringView::Large(values.as_string::<i64>()),
                other => {
                    return Err(TrainError::column(
                        path,
                        format!("column {name:?} must hold strings, found {other}"),
                    ))
                }
            };
            for row in 0..column.len() {
                nonnull(path, name, row, column.is_null(row))?;
                let (start, length) = offsets(row);
                out.push(
                    (start..start + length)
                        .map(|index| strings.value(index))
                        .collect(),
                );
            }
        }
        Ok(out)
    }
}

enum StringView<'a> {
    Small(&'a arrow_array::StringArray),
    Large(&'a arrow_array::LargeStringArray),
}

impl StringView<'_> {
    fn value(&self, index: usize) -> String {
        match self {
            Self::Small(array) => array.value(index).to_owned(),
            Self::Large(array) => array.value(index).to_owned(),
        }
    }
}

/// The uncompressed size a parquet footer declares, summed over every column chunk.
///
/// Saturating rather than checked: this is an *estimate* used to refuse work, so a
/// footer claiming an absurd total should reach the ceiling and be refused, not error
/// out on the arithmetic and hide why.
fn declared_decoded_bytes_of(metadata: &parquet::file::metadata::ParquetMetaData) -> usize {
    let mut total = 0usize;
    for group in 0..metadata.num_row_groups() {
        let row_group = metadata.row_group(group);
        for column in 0..row_group.num_columns() {
            let size = row_group.column(column).uncompressed_size();
            total = total.saturating_add(usize::try_from(size).unwrap_or(usize::MAX));
        }
    }
    total
}

/// The total UTF-8 length of a string column, without copying it.
fn string_column_bytes(path: &Path, name: &str, column: &dyn Array) -> Result<usize> {
    let total = match column.data_type() {
        DataType::Utf8 => {
            let values = column.as_string::<i32>();
            usize::try_from(*values.offsets().last().unwrap_or(&0)).unwrap_or(usize::MAX)
        }
        DataType::LargeUtf8 => {
            let values = column.as_string::<i64>();
            usize::try_from(*values.offsets().last().unwrap_or(&0)).unwrap_or(usize::MAX)
        }
        other => {
            return Err(TrainError::column(
                path,
                format!("column {name:?} must be a string, found {other}"),
            ))
        }
    };
    Ok(total)
}

/// The element count and total text length of a list-of-strings column.
fn list_column_extent(path: &Path, name: &str, column: &dyn Array) -> Result<(usize, usize)> {
    let values: &dyn Array = match column.data_type() {
        DataType::List(_) => column.as_list::<i32>().values().as_ref(),
        DataType::LargeList(_) => column.as_list::<i64>().values().as_ref(),
        other => {
            return Err(TrainError::column(
                path,
                format!("column {name:?} must be a list of strings, found {other}"),
            ))
        }
    };
    let bytes = string_column_bytes(path, name, values)?;
    Ok((values.len(), bytes))
}

fn require_float32(path: &Path, name: &str, found: &DataType) -> Result<()> {
    if matches!(found, DataType::Float32) {
        Ok(())
    } else {
        Err(TrainError::column(
            path,
            format!("column {name:?} must hold float32 elements, found {found}"),
        ))
    }
}

/// A null in any of these columns is a corrupt dataset, not a missing value:
/// upstream's writer never emits one and the reader would hand torch a `None`.
fn nonnull(path: &Path, name: &str, row: usize, is_null: bool) -> Result<bool> {
    if is_null {
        Err(TrainError::column(
            path,
            format!("column {name:?} row {row} is null"),
        ))
    } else {
        Ok(true)
    }
}
