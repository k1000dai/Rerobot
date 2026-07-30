//! The Draccus decoding primitives a policy `config.json` is read through.
//!
//! `PreTrainedConfig.from_pretrained` loads a checkpoint with
//! `json.load`, then hands the resulting Python values to
//! `draccus.parse(config_cls, ...)`. Draccus does not type-check those values:
//! it *converts* them, with the constructor of the annotated type
//! (`draccus/parsers/decoding.py`):
//!
//! ```python
//! for t in [str, float, bytes]:
//!     decode.register(t, partial(decode_from_init, t))   # str(raw_value)
//!
//! @decode.register(int)
//! def decode_int(raw_value, path):
//!     if isinstance(raw_value, float):                   # floats are refused
//!         raise ValueError(...)
//!     return int(raw_value)                              # bool, str, int
//!
//! @decode.register(bool)
//! def decode_bool(raw_value, path):                      # yaml 1.2 bools only
//!     ...
//! ```
//!
//! So a checkpoint field declared `str` accepts *any* JSON value and keeps
//! `str()` of it; one declared `int` accepts a bool or a string that Python's
//! `int()` parses; and one declared `bool` accepts only the four spellings
//! above. This module ports those conversions over
//! [`crate::dataset::json::JsonLike`], which is the value domain `json.load`
//! actually produces — unbounded integers and the three non-finite tokens
//! included.
//!
//! The string-to-number domain is CPython's, not Rust's: `int()` and `float()`
//! run `_PyUnicode_TransformDecimalAndSpaceToASCII` first, so any Unicode
//! decimal digit and any Unicode space are accepted, and PEP 515 underscores
//! are allowed between digits.

use crate::dataset::info::python_str_repr;
use crate::dataset::json::{python_float_repr, JsonLike};
use num_bigint::BigInt;
use std::fmt;
use std::str::FromStr;
use unicode_general_category::{get_general_category, GeneralCategory};

/// Port of `draccus.utils.DecodingError`: a message and the key path it was
/// raised at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodingError {
    key_path: Vec<String>,
    message: String,
}

impl DecodingError {
    /// A decoding failure at the document root.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            key_path: Vec::new(),
            message: message.into(),
        }
    }

    /// The same failure, reported one level deeper under `key`.
    pub fn under(mut self, key: impl Into<String>) -> Self {
        self.key_path.insert(0, key.into());
        self
    }

    /// The dotted key path this failure was raised at.
    pub fn key_path(&self) -> &[String] {
        &self.key_path
    }
}

impl fmt::Display for DecodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `DecodingError.__str__`.
        if self.key_path.is_empty() {
            formatter.write_str(&self.message)
        } else {
            write!(formatter, "`{}`: {}", self.key_path.join("."), self.message)
        }
    }
}

impl std::error::Error for DecodingError {}

/// Port of `decode_int`: refuse a `float`, otherwise `int(raw_value)`.
pub fn decode_int(value: &JsonLike) -> Result<BigInt, DecodingError> {
    match value {
        // A Python `bool` is an `int` subclass, so `int(True)` is 1.
        JsonLike::Bool(flag) => Ok(BigInt::from(u8::from(*flag))),
        JsonLike::Int(integer) => Ok(integer.clone()),
        JsonLike::Str(text) => python_int_from_str(text)
            .ok_or_else(|| DecodingError::new(format!("Couldn't parse '{text}' into an int"))),
        other => Err(DecodingError::new(format!(
            "Couldn't parse '{}' into an int",
            python_str(other)
        ))),
    }
}

/// Port of `decode_from_init(float, ...)`: `float(raw_value)`.
pub fn decode_float(value: &JsonLike) -> Result<f64, DecodingError> {
    let failed = || {
        DecodingError::new(format!(
            "Couldn't parse '{}' into a float",
            python_str(value)
        ))
    };
    match value {
        JsonLike::Bool(flag) => Ok(if *flag { 1.0 } else { 0.0 }),
        JsonLike::Float(number) => Ok(*number),
        // `float(10**400)` is an `OverflowError`, not an infinity — unlike
        // `float("1e400")`, which really is one.
        JsonLike::Int(integer) => {
            let converted = integer.to_string().parse::<f64>().map_err(|_| failed())?;
            if converted.is_finite() {
                Ok(converted)
            } else {
                Err(DecodingError::new(
                    "int too large to convert to float".to_string(),
                ))
            }
        }
        JsonLike::Str(text) => python_float_from_str(text).ok_or_else(failed),
        _ => Err(failed()),
    }
}

/// Port of `decode_bool`: `True`, `False`, `"true"` and `"false"`, nothing else.
pub fn decode_bool(value: &JsonLike) -> Result<bool, DecodingError> {
    match value {
        JsonLike::Bool(flag) => Ok(*flag),
        JsonLike::Str(text) if text == "true" => Ok(true),
        JsonLike::Str(text) if text == "false" => Ok(false),
        other => Err(DecodingError::new(format!(
            "Couldn't parse '{}' into a bool",
            python_str(other)
        ))),
    }
}

/// Port of `decode_from_init(str, ...)`: Python's `str(raw_value)`.
///
/// ```
/// use rerobot_core::dataset::json::loads;
/// use rerobot_core::policy::draccus::python_str;
///
/// assert_eq!(python_str(&loads("true").unwrap()), "True");
/// assert_eq!(python_str(&loads("null").unwrap()), "None");
/// assert_eq!(python_str(&loads("1e-5").unwrap()), "1e-05");
/// assert_eq!(python_str(&loads(r#"[1, "a"]"#).unwrap()), "[1, 'a']");
/// ```
pub fn python_str(value: &JsonLike) -> String {
    match value {
        // `str` of a `str` is the string itself, without repr's quoting.
        JsonLike::Str(text) => text.clone(),
        other => python_repr(other),
    }
}

/// Python's `repr(...)`, which is what `str()` applies to container elements.
pub fn python_repr(value: &JsonLike) -> String {
    match value {
        JsonLike::Null => "None".to_string(),
        JsonLike::Bool(true) => "True".to_string(),
        JsonLike::Bool(false) => "False".to_string(),
        JsonLike::Int(integer) => integer.to_string(),
        JsonLike::Float(number) => python_float_repr(*number),
        JsonLike::Str(text) => python_str_repr(text),
        JsonLike::Array(items) => {
            let rendered: Vec<String> = items.iter().map(python_repr).collect();
            format!("[{}]", rendered.join(", "))
        }
        JsonLike::Tuple(items) => {
            let rendered: Vec<String> = items.iter().map(python_repr).collect();
            // A one-element tuple keeps its trailing comma.
            if rendered.len() == 1 {
                format!("({},)", rendered[0])
            } else {
                format!("({})", rendered.join(", "))
            }
        }
        JsonLike::Object(map) => {
            let rendered: Vec<String> = map
                .iter()
                .map(|(key, value)| format!("{}: {}", python_str_repr(key), python_repr(value)))
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
    }
}

/// Port of `pathlib.PurePosixPath(value).__fspath__()`.
///
/// This is pure string work — `Path(...)` touches no filesystem, and neither
/// does this. Empty and `.` segments collapse, `..` is kept because resolving
/// one would need the filesystem, a trailing separator is dropped, and exactly
/// two leading slashes are preserved as POSIX permits an implementation-defined
/// meaning for them.
///
/// ```
/// use rerobot_core::policy::draccus::pure_posix_path;
///
/// assert_eq!(pure_posix_path("a//b"), "a/b");
/// assert_eq!(pure_posix_path(""), ".");
/// assert_eq!(pure_posix_path("./x/"), "x");
/// assert_eq!(pure_posix_path("//net/share"), "//net/share");
/// ```
pub fn pure_posix_path(value: &str) -> String {
    let leading = value.len() - value.trim_start_matches('/').len();
    // `PurePosixPath._parse_path`: one or three-or-more leading slashes are a
    // plain root; exactly two are kept.
    let root = match leading {
        0 => "",
        2 => "//",
        _ => "/",
    };
    let parts: Vec<&str> = value
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    let joined = parts.join("/");
    if root.is_empty() && joined.is_empty() {
        // `str(PurePosixPath(""))` is `'.'`.
        ".".to_string()
    } else {
        format!("{root}{joined}")
    }
}

/// CPython's `int(str)`: the decimal domain, with no base prefix.
pub fn python_int_from_str(text: &str) -> Option<BigInt> {
    let ascii = transform_decimal_and_space_to_ascii(text)?;
    let body = strip_python_underscores(trim_python_numeric_whitespace(&ascii))?;
    let digits = body.strip_prefix(['+', '-']).unwrap_or(&body);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    BigInt::from_str(&body).ok()
}

/// CPython's `float(str)`: the decimal domain plus the `inf`/`nan` spellings.
pub fn python_float_from_str(text: &str) -> Option<f64> {
    let ascii = transform_decimal_and_space_to_ascii(text)?;
    let body = strip_python_underscores(trim_python_numeric_whitespace(&ascii))?;
    // Rust's `f64::from_str` grammar is Python's: an optional sign, then
    // `inf`/`infinity`/`nan` case-insensitively or a decimal with an optional
    // fraction and exponent. It refuses the hex and base-prefixed forms Python
    // refuses too, and it is correctly rounded, as CPython's is.
    body.parse::<f64>().ok()
}

/// Whitespace skipped by CPython's ASCII numeric parsers after Unicode spaces
/// have been transformed to U+0020.
fn trim_python_numeric_whitespace(text: &str) -> &str {
    text.trim_matches(|ch| matches!(ch, ' ' | '\t' | '\n' | '\u{0b}' | '\u{0c}' | '\r'))
}

/// PEP 515: an underscore is only legal between two digits. Returns the text
/// with the underscores removed, or `None` if one is misplaced.
fn strip_python_underscores(text: &str) -> Option<String> {
    if !text.contains('_') {
        return Some(text.to_string());
    }
    let bytes = text.as_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        if *byte != b'_' {
            continue;
        }
        let before = index.checked_sub(1).map(|i| bytes[i]);
        let after = bytes.get(index + 1).copied();
        if !matches!(before, Some(b) if b.is_ascii_digit())
            || !matches!(after, Some(b) if b.is_ascii_digit())
        {
            return None;
        }
    }
    Some(text.replace('_', ""))
}

/// Port of `_PyUnicode_TransformDecimalAndSpaceToASCII`, which CPython runs
/// before parsing a `str` as an `int` or a `float`.
///
/// Every Unicode space becomes `U+0020` and every Unicode decimal digit becomes
/// its ASCII counterpart. Anything else non-ASCII is left in place, where the
/// numeric parser then rejects it — returning `None` here is that rejection.
fn transform_decimal_and_space_to_ascii(text: &str) -> Option<String> {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else if python_is_space(ch) {
            out.push(' ');
        } else {
            let digit = decimal_digit_value(ch)?;
            out.push(char::from(b'0' + digit as u8));
        }
    }
    Some(out)
}

/// CPython's `Py_UNICODE_ISSPACE`: the Unicode `White_Space` property plus the
/// four C0 information separators, which Rust's `char::is_whitespace` omits.
fn python_is_space(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '\u{1c}'..='\u{1f}')
}

/// The value of a Unicode decimal digit, as `Py_UNICODE_TODECIMAL` reports it.
///
/// Every `Nd` character belongs to an aligned run of exactly ten code points
/// starting at the run's zero (UAX #44), so the value is the distance back to
/// the first code point of the run.
fn decimal_digit_value(ch: char) -> Option<u32> {
    if ch.is_ascii_digit() {
        return Some(ch as u32 - u32::from(b'0'));
    }
    if get_general_category(ch) != GeneralCategory::DecimalNumber {
        return None;
    }
    let code = ch as u32;
    // The five Mathematical digit alphabets are adjacent with no non-Nd code
    // point between their ten-character runs. Looking only for the previous
    // non-Nd character therefore cannot identify the final four zeroes.
    if (0x1D7CE..=0x1D7FF).contains(&code) {
        return Some((code - 0x1D7CE) % 10);
    }
    for offset in 0..=9 {
        let previous = code
            .checked_sub(offset + 1)
            .and_then(char::from_u32)
            .map(|previous| get_general_category(previous) == GeneralCategory::DecimalNumber);
        if previous != Some(true) {
            return Some(offset);
        }
    }
    None
}
