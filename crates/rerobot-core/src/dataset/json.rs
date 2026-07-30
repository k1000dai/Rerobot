//! The Python JSON value domain `meta/info.json` is read and written in.
//!
//! Upstream's `lerobot/utils/io_utils.py` spells the domain out as a type
//! alias, and this module is a port of *that* alias rather than of any
//! particular JSON library:
//!
//! ```python
//! JsonLike = str | int | float | bool | None | list["JsonLike"]
//!          | dict[str, "JsonLike"] | tuple["JsonLike", ...]
//! ```
//!
//! `serde_json::Value` is close but not equal to it in three ways that
//! `meta/info.json` actually exercises, so the domain is modelled here instead:
//!
//! * a Python `int` is unbounded, and `json.dump` writes every digit of it;
//! * `json.load` accepts the bare tokens `NaN`, `Infinity` and `-Infinity` by
//!   default, and `json.dump` emits them, so a non-finite float round-trips
//!   through a file where JSON proper has no literal for one;
//! * a `tuple` is a distinct Python value that compares unequal to the `list`
//!   with the same elements, and `DatasetInfo.__post_init__` puts one into
//!   `features[...]["shape"]`, where a caller can observe it.
//!
//! The reader and writer are ports of CPython's `json` module for the metadata
//! values inside the explicitly documented runtime and representation
//! boundaries — [`loads`] of `json.loads` and [`dumps_pretty`] of
//! `json.dump(..., indent=4, ensure_ascii=False)` — pinned to CPython 3.12 and
//! to the C scanner it uses by default, whose diagnostics differ in wording
//! from the pure-Python fallback in `json/decoder.py`.
//!
//! A string containing an unpaired surrogate is the representation boundary:
//! CPython can hold it but a Rust `String` cannot. Conversely, this module has
//! no process-global equivalent of CPython's configurable decimal-integer
//! conversion limit and therefore accepts longer integers. Locale decoding,
//! recursion/resource limits, and these two boundaries are listed in
//! `docs/compatibility.md`; they are not hidden under a full-JSON parity claim.

use indexmap::IndexMap;
use num_bigint::BigInt;
use std::cell::Cell;
use std::fmt;
use std::str::FromStr;

/// A Python value in upstream's `JsonLike` domain.
///
/// `PartialEq` is Python's `==`, which is why there is no `Eq`: a
/// [`JsonLike::Float`] holding NaN is unequal to itself, exactly as
/// `float("nan") != float("nan")` in Python, and [`JsonLike::Tuple`] is
/// unequal to the [`JsonLike::Array`] with the same elements.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonLike {
    /// Python `None`; JSON `null`.
    Null,
    /// Python `bool`.
    ///
    /// Kept distinct from [`JsonLike::Int`] even though Python's `bool` is an
    /// `int` subclass, because `json.dump` writes `true`/`false` for one and a
    /// decimal integer for the other — the subclassing is observable in
    /// arithmetic, not on the wire.
    Bool(bool),
    /// Python `int`, unbounded and signed.
    Int(BigInt),
    /// Python `float`, an IEEE-754 double including the three non-finite
    /// values CPython's JSON reader and writer both accept.
    Float(f64),
    /// Python `str`.
    Str(String),
    /// Python `list`.
    Array(Vec<JsonLike>),
    /// Python `tuple`.
    ///
    /// `json.dump` writes it as a JSON array, and `json.load` never produces
    /// one; it exists because upstream constructs tuples in memory and
    /// compares against them.
    Tuple(Vec<JsonLike>),
    /// Python `dict` with `str` keys, in insertion order.
    Object(IndexMap<String, JsonLike>),
}

/// A Python `dict[str, JsonLike]` in insertion order.
pub type JsonObject = IndexMap<String, JsonLike>;

/// Maximum UTF-8 bytes accepted by [`loads`] and local [`super::io::load_json`].
pub const MAX_JSON_INPUT_BYTES: usize = 16 * 1024 * 1024;
/// Maximum scalar/container values produced by one parse.
pub const MAX_JSON_NODES: usize = 100_000;
/// Maximum decoded characters in one string or object key.
pub const MAX_JSON_STRING_CHARS: usize = 1_000_000;
/// Maximum source characters in one integer or floating-point token.
pub const MAX_JSON_NUMBER_CHARS: usize = 100_000;

impl JsonLike {
    /// The name CPython's `type(...).__name__` gives this value, used by the
    /// typed errors in [`crate::dataset::info`].
    pub fn type_name(&self) -> &'static str {
        match self {
            JsonLike::Null => "NoneType",
            JsonLike::Bool(_) => "bool",
            JsonLike::Int(_) => "int",
            JsonLike::Float(_) => "float",
            JsonLike::Str(_) => "str",
            JsonLike::Array(_) => "list",
            JsonLike::Tuple(_) => "tuple",
            JsonLike::Object(_) => "dict",
        }
    }

    /// The contained object, if this is one.
    pub fn as_object(&self) -> Option<&JsonObject> {
        match self {
            JsonLike::Object(map) => Some(map),
            _ => None,
        }
    }
}

/// Where a [`ParseError`] was raised, in Python's units.
///
/// CPython indexes a `str` by code point, so `position`, `line` and `column`
/// count code points and not bytes. That is what makes the rendered message
/// identical for a document containing non-ASCII text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// CPython's unformatted `JSONDecodeError.msg`.
    ///
    /// The one exception is the unpaired-surrogate message, which CPython has
    /// no counterpart for because CPython accepts that input.
    pub msg: String,
    /// Zero-based code-point offset (`JSONDecodeError.pos`).
    pub position: usize,
    /// One-based line (`JSONDecodeError.lineno`).
    pub line: usize,
    /// One-based code-point column (`JSONDecodeError.colno`).
    pub column: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `JSONDecodeError.__init__`: '%s: line %d column %d (char %d)'.
        write!(
            f,
            "{}: line {} column {} (char {})",
            self.msg, self.line, self.column, self.position
        )
    }
}

impl std::error::Error for ParseError {}

/// Port of `json.loads` for the [`JsonLike`] domain, with CPython's defaults.
///
/// Non-finite tokens are accepted, integers keep every digit within the
/// numeric-token budget, duplicate keys follow Python `dict` assignment, and a
/// malformed document produces the [`ParseError`] CPython's `JSONDecodeError`
/// would carry. Explicit fail-closed byte, nesting, node, string, and number
/// budgets are exposed as this module's `MAX_JSON_*` constants; exceeding one
/// returns [`ParseError`] rather than relying on stack or allocator exhaustion.
///
/// ```
/// use rerobot_core::dataset::json::{loads, JsonLike};
///
/// let value = loads(r#"{"fps": 30, "spread": Infinity}"#).unwrap();
/// let map = value.as_object().unwrap();
/// assert_eq!(map["spread"], JsonLike::Float(f64::INFINITY));
///
/// let err = loads(r#"{"fps": 01}"#).unwrap_err();
/// assert_eq!(err.to_string(), "Expecting ',' delimiter: line 1 column 10 (char 9)");
/// ```
pub fn loads(text: &str) -> Result<JsonLike, ParseError> {
    if text.len() > MAX_JSON_INPUT_BYTES {
        return Err(ParseError {
            msg: format!("Rerobot JSON input byte limit exceeded ({MAX_JSON_INPUT_BYTES})"),
            position: 0,
            line: 1,
            column: 1,
        });
    }
    if text.starts_with('\u{feff}') {
        return Err(ParseError {
            msg: "Unexpected UTF-8 BOM (decode using utf-8-sig)".to_string(),
            position: 0,
            line: 1,
            column: 1,
        });
    }
    let char_count = text.chars().count();
    let mut chars = Vec::new();
    chars
        .try_reserve_exact(char_count)
        .map_err(|_| ParseError {
            msg: "Rerobot JSON parser allocation failed".to_string(),
            position: 0,
            line: 1,
            column: 1,
        })?;
    chars.extend(text.chars());
    let parser = Parser {
        chars: &chars,
        nodes: Cell::new(0),
    };
    let start = parser.skip_ws(0);
    let (value, end) = parser.scan_once(start, 0).map_err(|f| parser.settle(f))?;
    let end = parser.skip_ws(end);
    if end != chars.len() {
        return Err(parser.error("Extra data", end));
    }
    Ok(value)
}

/// Port of `json.dump(obj, f, indent=4, ensure_ascii=False)`.
///
/// Four-space indentation, `","` item separators, non-ASCII written
/// literally, and **no trailing newline** — `json.dump` writes none. Writing
/// itself is iterative; derived `JsonLike` operations such as destruction,
/// cloning, equality, and debug formatting remain recursive for values that a
/// caller constructs beyond [`loads`]'s nesting limit.
///
/// ```
/// use rerobot_core::dataset::json::{dumps_pretty, loads};
///
/// let value = loads(r#"{"clé": [1, {}], "fps": 30.0}"#).unwrap();
/// assert_eq!(
///     dumps_pretty(&value),
///     "{\n    \"clé\": [\n        1,\n        {}\n    ],\n    \"fps\": 30.0\n}"
/// );
/// ```
pub fn dumps_pretty(value: &JsonLike) -> String {
    let mut out = String::new();
    write_value(&mut out, value, Some(4), 0);
    out
}

/// Compact writer with CPython's default separators (`", "` and `": "`) and
/// this module's `ensure_ascii=False` string policy.
pub fn dumps(value: &JsonLike) -> String {
    let mut out = String::new();
    write_value(&mut out, value, None, 0);
    out
}

/// Port of `json.dump(obj, f, indent=4)` — that is, with CPython's *default*
/// `ensure_ascii=True`.
///
/// Identical to [`dumps_pretty`] except that every character outside printable
/// ASCII is written as a `\uXXXX` escape, astral ones as a UTF-16 surrogate
/// pair. This is the spelling Draccus' JSON parser produces, and therefore the
/// one a policy `config.json` is written with; `meta/info.json` uses
/// [`dumps_pretty`] because upstream passes `ensure_ascii=False` there.
///
/// ```
/// use rerobot_core::dataset::json::{dumps_pretty_ascii, loads};
///
/// let value = loads(r#"{"clé": "😀"}"#).unwrap();
/// assert_eq!(
///     dumps_pretty_ascii(&value),
///     "{\n    \"cl\\u00e9\": \"\\ud83d\\ude00\"\n}"
/// );
/// ```
pub fn dumps_pretty_ascii(value: &JsonLike) -> String {
    let mut out = String::new();
    write_value_with(&mut out, value, Some(4), 0, encode_basestring_ascii);
    out
}

/// Port of `float.__repr__`, which is what `json.dump` writes for a finite
/// float.
///
/// Rust's `{}` is not it: it never uses exponent notation and prints `30` for
/// `30.0`, where CPython prints `30.0` and `1e+16`. Both emit a shortest decimal
/// that round-trips, but exact halfway cases can have two such decimals and the
/// implementations do not always choose the same one. The internal
/// shortest-digit routine therefore recomputes CPython's half-even choice with
/// exact arithmetic.
///
/// The non-finite values render as CPython's `repr` does, `nan` / `inf` /
/// `-inf`; the JSON writer does *not* use those spellings, because
/// `json.encoder.floatstr` substitutes `NaN` / `Infinity` / `-Infinity`.
///
/// ```
/// use rerobot_core::dataset::json::python_float_repr;
///
/// assert_eq!(python_float_repr(30.0), "30.0");
/// assert_eq!(python_float_repr(1e15), "1000000000000000.0");
/// assert_eq!(python_float_repr(1e16), "1e+16"); // repr switches over here
/// assert_eq!(python_float_repr(-0.0), "-0.0");
/// ```
pub fn python_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "inf"
        } else {
            "-inf"
        }
        .to_string();
    }

    let sign = if value.is_sign_negative() { "-" } else { "" };
    let (digits, decpt) = shortest_digits(value.abs());

    if decpt <= -4 || decpt > 16 {
        let mantissa = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let exp = decpt - 1;
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exp_sign}{:02}", exp.abs())
    } else if decpt <= 0 {
        let zeros = "0".repeat(-decpt as usize);
        format!("{sign}0.{zeros}{digits}")
    } else if (decpt as usize) >= digits.len() {
        let zeros = "0".repeat(decpt as usize - digits.len());
        format!("{sign}{digits}{zeros}.0")
    } else {
        let (whole, frac) = digits.split_at(decpt as usize);
        format!("{sign}{whole}.{frac}")
    }
}

/// The shortest decimal significand that round-trips `value`, chosen the way
/// CPython's `_Py_dg_dtoa` chooses it, plus the decimal-point position CPython
/// calls `decpt`: `value == 0.<digits> * 10^decpt`.
///
/// `value` must be finite and non-negative.
///
/// Rust's `{:e}` is used for one thing only — *how many* digits the shortest
/// representation needs. That length is a property of the double, so the two
/// implementations agree on it. Which digits to print at that length is *not*
/// settled by round-tripping alone: when the double sits exactly halfway
/// between two decimals of that length, both round-trip, and CPython breaks
/// the tie to an even last digit while Rust's `{:e}` breaks it upward. Eight
/// such doubles turned up in a 30,623-value sweep against CPython 3.12, so the
/// digits are not taken from `{:e}` at all — they are recomputed here from the
/// double's exact value with `BigInt` arithmetic, rounding half to even.
fn shortest_digits(value: f64) -> (String, i32) {
    debug_assert!(value.is_finite() && value.is_sign_positive());
    if value == 0.0 {
        return ("0".to_string(), 1);
    }

    let exponential = format!("{:e}", value);
    let (mantissa, exponent) = exponential
        .split_once('e')
        .expect("Rust's LowerExp for f64 always emits an `e`");
    let rust_digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let exponent: i32 = exponent.parse().expect("LowerExp emits a decimal exponent");
    let digit_count = rust_digits.len() as i32;
    let decpt = exponent + 1;

    match round_half_even_significand(value, digit_count, decpt) {
        // `value == N * 10^(decpt - digit_count)`, so a carry out of the top
        // digit (999... -> 1000...) moves the decimal point one place right.
        Some((digits, carried)) => {
            let decpt = decpt + i32::from(carried);
            let digits = digits.trim_end_matches('0');
            let digits = if digits.is_empty() { "0" } else { digits };
            if round_trips(digits, decpt, value) {
                (digits.to_string(), decpt)
            } else {
                (rust_digits, decpt - i32::from(carried))
            }
        }
        None => (rust_digits, decpt),
    }
}

/// `round_half_even(value * 10^(digit_count - decpt))` as a decimal string,
/// and whether it carried into an extra digit.
///
/// The arithmetic is exact: an IEEE-754 double is `m * 2^e` for integers `m`
/// and `e`, so the quantity being rounded is the rational `m * 2^e * 10^s`,
/// and both halves of it are `BigInt`s.
fn round_half_even_significand(value: f64, digit_count: i32, decpt: i32) -> Option<(String, bool)> {
    let bits = value.to_bits();
    let biased_exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    // Subnormals carry no implicit leading bit and are all at the same scale.
    let (mantissa, exponent) = if biased_exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), biased_exponent - 1075)
    };

    let scale = digit_count - decpt;
    let pow2 = |k: i32| BigInt::from(2u8).pow(k.max(0) as u32);
    let pow10 = |k: i32| BigInt::from(10u8).pow(k.max(0) as u32);

    let numerator = BigInt::from(mantissa) * pow2(exponent) * pow10(scale);
    let denominator = pow2(-exponent) * pow10(-scale);

    let quotient = &numerator / &denominator;
    let remainder = &numerator % &denominator;
    let doubled = &remainder * BigInt::from(2u8);
    let rounded = match doubled.cmp(&denominator) {
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Less => quotient,
        // The exact halfway case CPython resolves to an even last digit.
        std::cmp::Ordering::Equal => {
            if quotient.bit(0) {
                quotient + 1
            } else {
                quotient
            }
        }
    };

    let digits = rounded.to_string();
    match digits.len() as i32 - digit_count {
        0 => Some((digits, false)),
        1 => Some((digits, true)),
        // Anything else means the digit count taken from `{:e}` did not
        // describe this double; leave the caller on its fallback rather than
        // inventing a representation.
        _ => None,
    }
}

/// Whether `0.<digits> * 10^decpt` parses back to exactly `value`.
///
/// This is the property that makes a representation usable at all, so it is
/// checked rather than assumed — a wrong digit here would be a silently
/// corrupted `meta/info.json`.
fn round_trips(digits: &str, decpt: i32, value: f64) -> bool {
    let candidate = format!("{}e{}", digits, decpt - digits.len() as i32);
    candidate
        .parse::<f64>()
        .is_ok_and(|parsed| parsed.to_bits() == value.to_bits())
}

/// Port of `json.encoder.encode_basestring`, the `ensure_ascii=False` string
/// escaper: quotes, backslash and the C0 controls, nothing else.
///
/// The result includes the surrounding double quotes, as CPython's does.
///
/// ```
/// use rerobot_core::dataset::json::encode_basestring;
///
/// assert_eq!(encode_basestring("clé/✅"), "\"clé/✅\""); // no escaping at all
/// assert_eq!(encode_basestring("a\tb"), "\"a\\tb\"");
/// ```
pub fn encode_basestring(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // `ESCAPE_DCT.setdefault(chr(i), '\\u{0:04x}'.format(i))` for the
            // remaining C0 controls; lowercase hex, always four digits.
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Port of `json.encoder.py_encode_basestring_ascii`, the default
/// `ensure_ascii=True` escaper.
///
/// `ESCAPE_ASCII = re.compile(r'([\\"]|[^\ -~])')`: everything outside
/// `U+0020..U+007E` is escaped, so `DEL` is too, and a character above the BMP
/// becomes the `\uXXXX\uXXXX` surrogate pair CPython computes by hand.
///
/// The result includes the surrounding double quotes, as CPython's does.
///
/// ```
/// use rerobot_core::dataset::json::encode_basestring_ascii;
///
/// assert_eq!(encode_basestring_ascii("clé"), r#""cl\u00e9""#);
/// assert_eq!(encode_basestring_ascii("😀"), r#""\ud83d\ude00""#);
/// assert_eq!(encode_basestring_ascii("a\tb"), r#""a\tb""#);
/// assert_eq!(encode_basestring_ascii("~\u{7f}"), r#""~\u007f""#);
/// ```
pub fn encode_basestring_ascii(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ' '..='~' => out.push(ch),
            c if (c as u32) < 0x1_0000 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => {
                let n = c as u32 - 0x1_0000;
                let high = 0xd800 | ((n >> 10) & 0x3ff);
                let low = 0xdc00 | (n & 0x3ff);
                out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
            }
        }
    }
    out.push('"');
    out
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

fn write_value(out: &mut String, value: &JsonLike, indent: Option<usize>, level: usize) {
    write_value_with(out, value, indent, level, encode_basestring);
}

fn write_value_with(
    out: &mut String,
    value: &JsonLike,
    indent: Option<usize>,
    level: usize,
    encode_str: fn(&str) -> String,
) {
    enum Task<'a> {
        Value(&'a JsonLike, usize),
        Prefix { position: usize, level: usize },
        Key(&'a str),
        Close { bracket: char, level: usize },
    }

    // CPython's encoder is recursive and raises at its recursion limit. Rust
    // has no catchable stack-overflow exception, so use an explicit work stack:
    // programmatically constructed deep metadata remains serialisable instead
    // of aborting the process. Other derived operations (`Drop`, `Clone`,
    // equality, and `Debug`) are still recursive for values constructed beyond
    // the reader's documented nesting limit.
    let mut tasks = vec![Task::Value(value, level)];
    while let Some(task) = tasks.pop() {
        match task {
            Task::Value(value, level) => match value {
                JsonLike::Null => out.push_str("null"),
                JsonLike::Bool(true) => out.push_str("true"),
                JsonLike::Bool(false) => out.push_str("false"),
                JsonLike::Int(n) => out.push_str(&n.to_string()),
                JsonLike::Float(f) => out.push_str(&json_float(*f)),
                JsonLike::Str(s) => out.push_str(&encode_str(s)),
                JsonLike::Array(items) | JsonLike::Tuple(items) => {
                    out.push('[');
                    if items.is_empty() {
                        out.push(']');
                        continue;
                    }
                    tasks.push(Task::Close {
                        bracket: ']',
                        level,
                    });
                    for position in (0..items.len()).rev() {
                        tasks.push(Task::Value(&items[position], level + 1));
                        tasks.push(Task::Prefix { position, level });
                    }
                }
                JsonLike::Object(map) => {
                    out.push('{');
                    if map.is_empty() {
                        out.push('}');
                        continue;
                    }
                    tasks.push(Task::Close {
                        bracket: '}',
                        level,
                    });
                    for (position, (key, value)) in map.iter().enumerate().rev() {
                        tasks.push(Task::Value(value, level + 1));
                        tasks.push(Task::Key(key));
                        tasks.push(Task::Prefix { position, level });
                    }
                }
            },
            Task::Prefix { position, level } => {
                if position > 0 {
                    out.push_str(if indent.is_some() { "," } else { ", " });
                }
                if let Some(width) = indent {
                    out.push('\n');
                    out.push_str(&" ".repeat(width * (level + 1)));
                }
            }
            Task::Key(key) => {
                out.push_str(&encode_str(key));
                out.push_str(": ");
            }
            Task::Close { bracket, level } => {
                if let Some(width) = indent {
                    out.push('\n');
                    out.push_str(&" ".repeat(width * level));
                }
                out.push(bracket);
            }
        }
    }
}

/// `json.encoder.floatstr`: `repr` for the finite values, and CPython's three
/// non-JSON spellings otherwise.
fn json_float(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value == f64::INFINITY {
        "Infinity".to_string()
    } else if value == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        python_float_repr(value)
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// The two ways `_scan_once` can fail to yield a value. CPython signals the
/// first by raising `StopIteration(idx)`, which every caller turns into its
/// own `"Expecting value"` error at that index.
enum ScanFail {
    NoValue(usize),
    Error(ParseError),
}

struct Parser<'a> {
    chars: &'a [char],
    nodes: Cell<usize>,
}

/// Keep recursive-descent frames far below a native thread's stack limit.
/// Real `info.json` documents are fewer than ten levels deep. CPython has its
/// own configurable recursion limit; this explicit Rust boundary is documented
/// rather than allowing hostile input to abort the process with stack overflow.
/// Maximum nested array/object containers accepted by [`loads`].
pub const MAX_JSON_NESTING_DEPTH: usize = 128;

impl Parser<'_> {
    fn at(&self, index: usize) -> Option<char> {
        self.chars.get(index).copied()
    }

    fn starts_with(&self, index: usize, word: &str) -> bool {
        let end = index + word.chars().count();
        end <= self.chars.len() && self.chars[index..end].iter().copied().eq(word.chars())
    }

    /// `WHITESPACE.match(s, index).end()` — the four characters CPython's
    /// scanner skips, which is fewer than `char::is_whitespace`.
    fn skip_ws(&self, mut index: usize) -> usize {
        while matches!(
            self.at(index),
            Some(' ') | Some('\t') | Some('\n') | Some('\r')
        ) {
            index += 1;
        }
        index
    }

    fn error(&self, msg: &str, position: usize) -> ParseError {
        // `JSONDecodeError.__init__`: lineno counts '\n' before pos, and colno
        // is pos minus the index of the last one (or -1 when there is none).
        let before = &self.chars[..position.min(self.chars.len())];
        let line = before.iter().filter(|c| **c == '\n').count() + 1;
        let last_newline = before.iter().rposition(|c| *c == '\n');
        let column = match last_newline {
            Some(index) => position - index,
            None => position + 1,
        };
        ParseError {
            msg: msg.to_string(),
            position,
            line,
            column,
        }
    }

    fn settle(&self, failure: ScanFail) -> ParseError {
        match failure {
            ScanFail::NoValue(position) => self.error("Expecting value", position),
            ScanFail::Error(error) => error,
        }
    }

    fn fail(&self, msg: &str, position: usize) -> ScanFail {
        ScanFail::Error(self.error(msg, position))
    }

    fn scan_once(&self, index: usize, depth: usize) -> Result<(JsonLike, usize), ScanFail> {
        let nodes = self.nodes.get() + 1;
        if nodes > MAX_JSON_NODES {
            return Err(self.fail("Rerobot JSON node limit exceeded", index));
        }
        self.nodes.set(nodes);
        let Some(next) = self.at(index) else {
            return Err(ScanFail::NoValue(index));
        };
        match next {
            '"' => {
                let (text, end) = self.scan_string(index + 1)?;
                Ok((JsonLike::Str(text), end))
            }
            '{' if depth >= MAX_JSON_NESTING_DEPTH => {
                Err(self.fail("Rerobot JSON nesting limit exceeded", index))
            }
            '[' if depth >= MAX_JSON_NESTING_DEPTH => {
                Err(self.fail("Rerobot JSON nesting limit exceeded", index))
            }
            '{' => self.scan_object(index + 1, depth + 1),
            '[' => self.scan_array(index + 1, depth + 1),
            'n' if self.starts_with(index, "null") => Ok((JsonLike::Null, index + 4)),
            't' if self.starts_with(index, "true") => Ok((JsonLike::Bool(true), index + 4)),
            'f' if self.starts_with(index, "false") => Ok((JsonLike::Bool(false), index + 5)),
            _ => {
                // CPython tries the number pattern before the constants, which
                // is why `-Infinity` has to fail the number match first — it
                // does, because a `-` must be followed by a digit.
                if let Some((value, end)) = self.scan_number(index)? {
                    return Ok((value, end));
                }
                if next == 'N' && self.starts_with(index, "NaN") {
                    Ok((JsonLike::Float(f64::NAN), index + 3))
                } else if next == 'I' && self.starts_with(index, "Infinity") {
                    Ok((JsonLike::Float(f64::INFINITY), index + 8))
                } else if next == '-' && self.starts_with(index, "-Infinity") {
                    Ok((JsonLike::Float(f64::NEG_INFINITY), index + 9))
                } else {
                    Err(ScanFail::NoValue(index))
                }
            }
        }
    }

    /// `NUMBER_RE`: `(-?(?:0|[1-9][0-9]*))(\.[0-9]+)?([eE][-+]?[0-9]+)?`.
    ///
    /// Each optional group is all-or-nothing, which is why `1e` parses as the
    /// integer `1` followed by a stray `e` rather than as a bad number.
    fn scan_number(&self, index: usize) -> Result<Option<(JsonLike, usize)>, ScanFail> {
        let mut end = index;
        if self.at(end) == Some('-') {
            end += 1;
        }
        match self.at(end) {
            Some('0') => end += 1,
            Some(c) if c.is_ascii_digit() => {
                while self.at(end).is_some_and(|c| c.is_ascii_digit()) {
                    end += 1;
                }
            }
            _ => return Ok(None),
        }

        let mut is_float = false;
        if self.at(end) == Some('.') && self.at(end + 1).is_some_and(|c| c.is_ascii_digit()) {
            end += 1;
            while self.at(end).is_some_and(|c| c.is_ascii_digit()) {
                end += 1;
            }
            is_float = true;
        }

        if matches!(self.at(end), Some('e') | Some('E')) {
            let mut probe = end + 1;
            if matches!(self.at(probe), Some('+') | Some('-')) {
                probe += 1;
            }
            if self.at(probe).is_some_and(|c| c.is_ascii_digit()) {
                while self.at(probe).is_some_and(|c| c.is_ascii_digit()) {
                    probe += 1;
                }
                end = probe;
                is_float = true;
            }
        }

        if end - index > MAX_JSON_NUMBER_CHARS {
            return Err(self.fail("Rerobot JSON number limit exceeded", index));
        }
        let mut token = String::new();
        token
            .try_reserve_exact(end - index)
            .map_err(|_| self.fail("Rerobot JSON parser allocation failed", index))?;
        token.extend(self.chars[index..end].iter());
        let value = if is_float {
            // Rust's `f64::from_str` is correctly rounded, as CPython's
            // `float()` is, and overflows to an infinity rather than erroring —
            // also as CPython does.
            JsonLike::Float(
                token
                    .parse::<f64>()
                    .map_err(|_| self.fail("Rerobot JSON number conversion failed", index))?,
            )
        } else {
            JsonLike::Int(
                BigInt::from_str(&token)
                    .map_err(|_| self.fail("Rerobot JSON number conversion failed", index))?,
            )
        };
        Ok(Some((value, end)))
    }

    /// `scanstring`, starting one past the opening quote.
    fn scan_string(&self, mut index: usize) -> Result<(String, usize), ScanFail> {
        let begin = index - 1;
        let mut out = String::new();
        let mut output_chars = 0usize;
        loop {
            let Some(ch) = self.at(index) else {
                return Err(self.fail("Unterminated string starting at", begin));
            };
            match ch {
                '"' => return Ok((out, index + 1)),
                c if (c as u32) < 0x20 => {
                    return Err(self.fail("Invalid control character at", index));
                }
                '\\' => {
                    let backslash = index;
                    let Some(escape) = self.at(index + 1) else {
                        return Err(self.fail("Unterminated string starting at", begin));
                    };
                    if escape == 'u' {
                        let (ch, end) = self.scan_unicode_escape(index + 1)?;
                        output_chars += 1;
                        if output_chars > MAX_JSON_STRING_CHARS {
                            return Err(self.fail("Rerobot JSON string limit exceeded", index));
                        }
                        self.push_string_char(&mut out, ch, index)?;
                        index = end;
                    } else {
                        let decoded = match escape {
                            '"' => '"',
                            '\\' => '\\',
                            '/' => '/',
                            'b' => '\u{8}',
                            'f' => '\u{c}',
                            'n' => '\n',
                            'r' => '\r',
                            't' => '\t',
                            _ => return Err(self.fail("Invalid \\escape", backslash)),
                        };
                        output_chars += 1;
                        if output_chars > MAX_JSON_STRING_CHARS {
                            return Err(self.fail("Rerobot JSON string limit exceeded", index));
                        }
                        self.push_string_char(&mut out, decoded, index)?;
                        index += 2;
                    }
                }
                c => {
                    output_chars += 1;
                    if output_chars > MAX_JSON_STRING_CHARS {
                        return Err(self.fail("Rerobot JSON string limit exceeded", index));
                    }
                    self.push_string_char(&mut out, c, index)?;
                    index += 1;
                }
            }
        }
    }

    fn push_string_char(
        &self,
        out: &mut String,
        value: char,
        index: usize,
    ) -> Result<(), ScanFail> {
        out.try_reserve(value.len_utf8())
            .map_err(|_| self.fail("Rerobot JSON parser allocation failed", index))?;
        out.push(value);
        Ok(())
    }

    /// `_decode_uXXXX` plus CPython's surrogate-pair join, starting at the `u`.
    fn scan_unicode_escape(&self, u: usize) -> Result<(char, usize), ScanFail> {
        let first = self
            .hex4(u + 1)
            .ok_or_else(|| self.fail("Invalid \\uXXXX escape", u))?;

        if (0xd800..=0xdbff).contains(&first)
            && self.at(u + 5) == Some('\\')
            && self.at(u + 6) == Some('u')
        {
            if let Some(second) = self.hex4(u + 7) {
                if (0xdc00..=0xdfff).contains(&second) {
                    let combined = 0x10000 + (((first - 0xd800) << 10) | (second - 0xdc00));
                    let ch = char::from_u32(combined)
                        .expect("a joined surrogate pair is always a scalar value");
                    return Ok((ch, u + 11));
                }
            }
        }

        // CPython would hand back a `str` holding this lone surrogate. Rust
        // cannot, so this is the one input where the port refuses rather than
        // approximates, with a message that is deliberately not CPython's.
        let ch = char::from_u32(first)
            .ok_or_else(|| self.fail("Unpaired surrogate escape (not representable in Rust)", u))?;
        Ok((ch, u + 5))
    }

    /// `HEXDIGITS.match(s, index)` — exactly four, any case.
    fn hex4(&self, index: usize) -> Option<u32> {
        let mut value = 0u32;
        for offset in 0..4 {
            let digit = self.at(index + offset)?.to_digit(16)?;
            value = value * 16 + digit;
        }
        Some(value)
    }

    /// `JSONObject`, starting one past the `{`.
    fn scan_object(&self, start: usize, depth: usize) -> Result<(JsonLike, usize), ScanFail> {
        let mut map: JsonObject = IndexMap::new();
        let mut index = start;

        if self.at(index) != Some('"') {
            index = self.skip_ws(index);
            match self.at(index) {
                Some('}') => return Ok((JsonLike::Object(map), index + 1)),
                Some('"') => {}
                _ => {
                    return Err(
                        self.fail("Expecting property name enclosed in double quotes", index)
                    )
                }
            }
        }
        index += 1;

        loop {
            let (key, after_key) = self.scan_string(index)?;
            index = after_key;

            if self.at(index) != Some(':') {
                index = self.skip_ws(index);
                if self.at(index) != Some(':') {
                    return Err(self.fail("Expecting ':' delimiter", index));
                }
            }
            index = self.skip_ws(index + 1);

            let (value, after_value) = self.scan_once(index, depth)?;
            // Python `dict` assignment: the last value wins and an existing
            // key keeps the position it was first inserted at.
            if !map.contains_key(&key) {
                map.try_reserve(1)
                    .map_err(|_| self.fail("Rerobot JSON parser allocation failed", index))?;
            }
            map.insert(key, value);
            index = after_value;

            let next = self.at(self.skip_ws(index));
            index = self.skip_ws(index) + 1;
            match next {
                Some('}') => return Ok((JsonLike::Object(map), index)),
                Some(',') => {}
                _ => return Err(self.fail("Expecting ',' delimiter", index - 1)),
            }

            index = self.skip_ws(index);
            let next = self.at(index);
            index += 1;
            if next != Some('"') {
                return Err(self.fail(
                    "Expecting property name enclosed in double quotes",
                    index - 1,
                ));
            }
        }
    }

    /// `JSONArray`, starting one past the `[`.
    fn scan_array(&self, start: usize, depth: usize) -> Result<(JsonLike, usize), ScanFail> {
        let mut items = Vec::new();
        let mut index = self.skip_ws(start);
        if self.at(index) == Some(']') {
            return Ok((JsonLike::Array(items), index + 1));
        }

        loop {
            let (value, after_value) = self.scan_once(index, depth)?;
            items
                .try_reserve(1)
                .map_err(|_| self.fail("Rerobot JSON parser allocation failed", index))?;
            items.push(value);

            let next = self.at(self.skip_ws(after_value));
            index = self.skip_ws(after_value) + 1;
            match next {
                Some(']') => return Ok((JsonLike::Array(items), index)),
                Some(',') => {}
                _ => return Err(self.fail("Expecting ',' delimiter", index - 1)),
            }
            index = self.skip_ws(index);
        }
    }
}
