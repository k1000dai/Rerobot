//! Behaviour parity tests for the `JsonLike` reader and writer.
//!
//! Every expectation here was observed by running CPython 3.12.13's `json`
//! module directly (see `docs/red-green.md` for the probe), not inferred from
//! the JSON grammar: CPython accepts and emits more than JSON proper, and this
//! slice has to agree with CPython rather than with the RFC.

use indexmap::IndexMap;
use num_bigint::BigInt;
use rerobot_core::dataset::json::{
    dumps, dumps_pretty, encode_basestring, loads, python_float_repr, JsonLike, ParseError,
    MAX_JSON_INPUT_BYTES, MAX_JSON_NESTING_DEPTH, MAX_JSON_NODES, MAX_JSON_NUMBER_CHARS,
    MAX_JSON_STRING_CHARS,
};
use std::process::Command;
use std::str::FromStr;

fn obj(pairs: &[(&str, JsonLike)]) -> JsonLike {
    JsonLike::Object(
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect::<IndexMap<_, _>>(),
    )
}

fn int(value: i64) -> JsonLike {
    JsonLike::Int(BigInt::from(value))
}

fn s(value: &str) -> JsonLike {
    JsonLike::Str(value.to_string())
}

fn err(text: &str) -> ParseError {
    loads(text).expect_err("expected a parse error")
}

// ---------------------------------------------------------------------------
// Reading — the scalar domain
// ---------------------------------------------------------------------------

#[test]
fn the_four_json_literals_parse_to_their_python_values() {
    assert_eq!(loads("null").unwrap(), JsonLike::Null);
    assert_eq!(loads("true").unwrap(), JsonLike::Bool(true));
    assert_eq!(loads("false").unwrap(), JsonLike::Bool(false));
    assert_eq!(loads(r#""hi""#).unwrap(), s("hi"));
}

#[test]
fn a_number_without_a_fraction_or_exponent_is_an_int() {
    assert_eq!(loads("1").unwrap(), int(1));
    assert_eq!(loads("-0").unwrap(), int(0)); // Python: json.loads('-0') == 0
    assert_eq!(loads("0").unwrap(), int(0));
}

#[test]
fn a_number_with_a_fraction_or_an_exponent_is_a_float() {
    // `json.loads('1.0')` is a float, and `1E5` / `1e+5` are floats too even
    // though their values are integral.
    assert_eq!(loads("1.0").unwrap(), JsonLike::Float(1.0));
    assert_eq!(loads("1E5").unwrap(), JsonLike::Float(100000.0));
    assert_eq!(loads("1e+5").unwrap(), JsonLike::Float(100000.0));
    assert_eq!(loads("1.0e5").unwrap(), JsonLike::Float(100000.0));
    assert_eq!(loads("-0.0").unwrap(), JsonLike::Float(-0.0));
}

#[test]
fn an_integer_past_every_machine_word_keeps_all_of_its_digits() {
    let huge = "340282366920938463463374607431768211457";
    assert_eq!(
        loads(huge).unwrap(),
        JsonLike::Int(BigInt::from_str(huge).unwrap())
    );
    let negative = format!("-{huge}");
    assert_eq!(
        loads(&negative).unwrap(),
        JsonLike::Int(BigInt::from_str(&negative).unwrap())
    );
}

#[test]
fn the_three_non_finite_tokens_are_accepted_as_python_floats() {
    // `json.loads('{"a": NaN, "b": Infinity, "c": -Infinity}')` succeeds by
    // default; these are not JSON, they are CPython.
    let parsed = loads(r#"{"a": NaN, "b": Infinity, "c": -Infinity}"#).unwrap();
    let JsonLike::Object(map) = parsed else {
        panic!("expected an object")
    };
    assert!(matches!(map["a"], JsonLike::Float(f) if f.is_nan()));
    assert_eq!(map["b"], JsonLike::Float(f64::INFINITY));
    assert_eq!(map["c"], JsonLike::Float(f64::NEG_INFINITY));
}

#[test]
fn a_nan_float_is_unequal_to_itself_exactly_as_in_python() {
    let nan = JsonLike::Float(f64::NAN);
    assert_ne!(nan, nan.clone());
}

// ---------------------------------------------------------------------------
// Reading — containers, ordering, whitespace
// ---------------------------------------------------------------------------

#[test]
fn object_keys_keep_their_document_order() {
    let parsed = loads(r#"{"z": 1, "a": 2, "m": 3}"#).unwrap();
    let JsonLike::Object(map) = parsed else {
        panic!("expected an object")
    };
    assert_eq!(map.keys().collect::<Vec<_>>(), vec!["z", "a", "m"]);
}

#[test]
fn a_duplicate_key_takes_the_last_value_and_the_first_position() {
    // Python: json.loads('{"a":1,"b":2,"a":3}') == {'a': 3, 'b': 2}, and 'a'
    // is still first because dict assignment does not move an existing key.
    let parsed = loads(r#"{"a":1,"b":2,"a":3}"#).unwrap();
    assert_eq!(parsed, obj(&[("a", int(3)), ("b", int(2))]));
    let JsonLike::Object(map) = parsed else {
        panic!("expected an object")
    };
    assert_eq!(map.keys().collect::<Vec<_>>(), vec!["a", "b"]);
}

#[test]
fn empty_containers_and_nesting_parse() {
    assert_eq!(loads("{}").unwrap(), JsonLike::Object(IndexMap::new()));
    assert_eq!(loads("[]").unwrap(), JsonLike::Array(vec![]));
    assert_eq!(
        loads("[[],{}]").unwrap(),
        JsonLike::Array(vec![
            JsonLike::Array(vec![]),
            JsonLike::Object(IndexMap::new())
        ])
    );
}

#[test]
fn the_four_json_whitespace_characters_are_skipped_anywhere_they_may_appear() {
    assert_eq!(
        loads(" \t\r\n{\"a\" \t: \n1} \r\n").unwrap(),
        obj(&[("a", int(1))])
    );
}

#[test]
fn parsing_never_produces_a_tuple() {
    // `json.load` has no tuple in its output table; only upstream code makes
    // one, which is why `JsonLike::Tuple` is unreachable from `loads`.
    assert_eq!(
        loads("[1,2]").unwrap(),
        JsonLike::Array(vec![int(1), int(2)])
    );
    assert_ne!(
        loads("[1,2]").unwrap(),
        JsonLike::Tuple(vec![int(1), int(2)])
    );
}

// ---------------------------------------------------------------------------
// Reading — strings
// ---------------------------------------------------------------------------

#[test]
fn every_backslash_escape_python_knows_is_decoded() {
    assert_eq!(
        loads(r#""\b\f\n\r\t\/\\\"A""#).unwrap(),
        s("\u{8}\u{c}\n\r\t/\\\"A")
    );
}

#[test]
fn a_surrogate_pair_escape_becomes_one_astral_character() {
    assert_eq!(loads("\"\\ud83d\\ude00\"").unwrap(), s("\u{1f600}"));
}

#[test]
fn a_hex_escape_is_case_insensitive_and_exactly_four_digits() {
    assert_eq!(loads("\"\\u00e9\\u00E9\"").unwrap(), s("éé"));
}

#[test]
fn a_high_surrogate_escape_not_followed_by_a_low_one_is_outside_the_domain() {
    // CPython *succeeds* here, yielding a `str` holding a lone surrogate. A
    // Rust `String` cannot hold one, so this is refused rather than
    // approximated, with a message that is deliberately not one of CPython's —
    // claiming its wording would claim its behaviour. See
    // `docs/compatibility.md`, "Python values this slice does not claim".
    let e = err("\"\\ud800\"");
    assert_eq!(
        e.msg,
        "Unpaired surrogate escape (not representable in Rust)"
    );
    assert_eq!((e.line, e.column, e.position), (1, 3, 2));
}

#[test]
fn non_ascii_source_text_is_taken_verbatim() {
    assert_eq!(loads("\"héllo ✅ 😀\"").unwrap(), s("héllo ✅ 😀"));
}

// ---------------------------------------------------------------------------
// Reading — CPython's rejections, with its own messages
// ---------------------------------------------------------------------------

#[test]
fn a_leading_zero_ends_the_number_and_trips_the_delimiter_check() {
    // Python: Expecting ',' delimiter: line 1 column 7 (char 6)
    let e = err(r#"{"a":01}"#);
    assert_eq!(e.msg, "Expecting ',' delimiter");
    assert_eq!((e.line, e.column, e.position), (1, 7, 6));
    assert_eq!(
        e.to_string(),
        "Expecting ',' delimiter: line 1 column 7 (char 6)"
    );
}

#[test]
fn a_leading_plus_a_bare_fraction_and_a_lone_minus_are_not_values() {
    for (text, position) in [(r#"{"a":+1}"#, 5), (r#"{"a":.5}"#, 5), (r#"{"a": -}"#, 6)] {
        let e = err(text);
        assert_eq!(e.msg, "Expecting value", "for {text}");
        assert_eq!(e.position, position, "for {text}");
    }
}

#[test]
fn a_truncated_exponent_or_fraction_ends_the_number_early() {
    // Python stops the number regex before the bad tail and then complains
    // about the delimiter, not about the number.
    let e = err(r#"{"a":1e}"#);
    assert_eq!((e.msg.as_str(), e.position), ("Expecting ',' delimiter", 6));
    let e = err(r#"{"a": 1.}"#);
    assert_eq!((e.msg.as_str(), e.position), ("Expecting ',' delimiter", 7));
}

#[test]
fn a_trailing_comma_and_a_single_quoted_key_want_a_double_quoted_property_name() {
    let e = err(r#"{"a":1,}"#);
    assert_eq!(e.msg, "Expecting property name enclosed in double quotes");
    assert_eq!((e.line, e.column, e.position), (1, 8, 7));
    let e = err("{'a':1}");
    assert_eq!(e.msg, "Expecting property name enclosed in double quotes");
    assert_eq!((e.line, e.column, e.position), (1, 2, 1));
}

#[test]
fn content_after_the_document_is_extra_data() {
    let e = err(r#"{"a":1}x"#);
    assert_eq!(e.msg, "Extra data");
    assert_eq!((e.line, e.column, e.position), (1, 8, 7));
}

#[test]
fn an_empty_document_expects_a_value_at_the_start() {
    let e = err("");
    assert_eq!(e.msg, "Expecting value");
    assert_eq!((e.line, e.column, e.position), (1, 1, 0));
}

#[test]
fn lowercase_spellings_of_the_non_finite_tokens_are_not_values() {
    for text in [r#"{"a":nan}"#, r#"{"a": Nan}"#, r#"{"a": infinity}"#] {
        assert_eq!(err(text).msg, "Expecting value", "for {text}");
    }
}

#[test]
fn a_raw_control_character_inside_a_string_is_rejected() {
    // The C scanner CPython actually uses omits the character repr that
    // `json/decoder.py`'s pure-Python fallback would interpolate. Observed:
    // "Invalid control character at: line 1 column 7 (char 6)".
    let e = err("{\"a\":\"\u{1}\"}");
    assert_eq!(e.msg, "Invalid control character at");
    assert_eq!((e.line, e.column, e.position), (1, 7, 6));
}

#[test]
fn an_unterminated_string_points_at_its_opening_quote() {
    let e = err(r#"{"a": "abc"#);
    assert_eq!(e.msg, "Unterminated string starting at");
    assert_eq!(e.position, 6);
}

#[test]
fn an_unknown_or_short_escape_carries_pythons_wording() {
    // Again the C scanner's wording, and note its two different anchors: the
    // unknown escape is reported at the backslash, the short \u at the `u`.
    let e = err(r#""\q""#);
    assert_eq!(e.msg, r"Invalid \escape");
    assert_eq!((e.line, e.column, e.position), (1, 2, 1));
    let e = err(r#""\u12""#);
    assert_eq!(e.msg, r"Invalid \uXXXX escape");
    assert_eq!((e.line, e.column, e.position), (1, 3, 2));
}

#[test]
fn a_missing_colon_and_two_adjacent_values_are_reported_where_python_reports_them() {
    let e = err(r#"{"a" 1}"#);
    assert_eq!(e.msg, "Expecting ':' delimiter");
    assert_eq!(e.position, 5);
    let e = err(r#"{"a": 1 2}"#);
    assert_eq!((e.msg.as_str(), e.position), ("Expecting ',' delimiter", 8));
}

#[test]
fn an_unclosed_array_is_reported_at_the_end_of_the_document() {
    let e = err("[1,2");
    assert_eq!(e.msg, "Expecting ',' delimiter");
    assert_eq!((e.line, e.column, e.position), (1, 5, 4));
}

#[test]
fn line_and_column_count_code_points_not_bytes() {
    // "é" is two UTF-8 bytes and one Python character. The column CPython
    // reports is the character one.
    let e = err("{\n    \"é\": 1,\n}");
    assert_eq!(e.msg, "Expecting property name enclosed in double quotes");
    assert_eq!(e.line, 3);
    assert_eq!(e.column, 1);
    assert_eq!(e.position, 14);
}

// ---------------------------------------------------------------------------
// Writing — `float.__repr__`
// ---------------------------------------------------------------------------

#[test]
fn python_float_repr_matches_cpython_across_the_exponent_switchover() {
    // Observed from CPython 3.12: repr() switches to exponent form when the
    // decimal point lands at or below -4, or above 16.
    for (value, expected) in [
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (1.0, "1.0"),
        (30.0, "30.0"),
        (0.1, "0.1"),
        (1e15, "1000000000000000.0"),
        (1e16, "1e+16"),
        (1e-4, "0.0001"),
        (1e-5, "1e-05"),
        (1.5e300, "1.5e+300"),
        (9007199254740992.0, "9007199254740992.0"),
        (1.0 / 3.0, "0.3333333333333333"),
        (1e100, "1e+100"),
        (123456789012345678.0, "1.2345678901234568e+17"),
        (5e-324, "5e-324"),
        (f64::MAX, "1.7976931348623157e+308"),
        (-2.5, "-2.5"),
    ] {
        assert_eq!(python_float_repr(value), expected, "for {value:?}");
    }
}

#[test]
fn an_exact_decimal_tie_rounds_to_the_even_last_digit_as_pythons_dtoa_does() {
    // These eight doubles sit exactly halfway between two 16/17-digit decimals
    // that both round-trip. CPython's `_Py_dg_dtoa` breaks the tie to even;
    // Rust's `{:e}` breaks it upward, so the last digit differs. Found by
    // sweeping 30,623 doubles against CPython 3.12 — see `docs/red-green.md`.
    for (bits, expected) in [
        (4836584781728123821u64, "2181495296738027.2"),
        (14054228241957549514, "-937625523621561.2"),
        (4829635748597700618, "785068460487425.2"),
        (4835311044807928153, "1863061066689110.2"),
        (4814660501532195592, "75251554695404.12"),
        (14057521920747911625, "-1572770837991026.2"),
        (14058047203397768261, "-1704091500455185.2"),
        (4834353408920277085, "1623652094776343.2"),
    ] {
        let value = f64::from_bits(bits);
        assert_eq!(python_float_repr(value), expected, "for bits {bits}");
        // Whatever else is true, the digits still have to name this double.
        assert_eq!(expected.parse::<f64>().unwrap().to_bits(), bits);
    }
}

#[test]
fn a_tie_whose_lower_neighbour_is_odd_still_rounds_up_to_the_even_one() {
    // The mirror of the case above, so the fix cannot be "always round down".
    // 0.5 ties exactly between the 1-digit decimals 0.5 has no neighbours for,
    // so use a value whose upper neighbour carries the even digit.
    let value = 1.0000000000000002e16; // exactly 10000000000000002.0
    assert_eq!(python_float_repr(value), "1.0000000000000002e+16");
    let value = 2.5f64; // no tie: 2.5 is exact, one digit past the point
    assert_eq!(python_float_repr(value), "2.5");
}

#[test]
fn the_non_finite_floats_are_written_with_pythons_spellings() {
    assert_eq!(
        dumps(&JsonLike::Array(vec![
            JsonLike::Float(f64::NAN),
            JsonLike::Float(f64::INFINITY),
            JsonLike::Float(f64::NEG_INFINITY),
        ])),
        "[NaN, Infinity, -Infinity]"
    );
}

// ---------------------------------------------------------------------------
// Writing — `encode_basestring` (`ensure_ascii=False`)
// ---------------------------------------------------------------------------

#[test]
fn only_quotes_backslash_and_the_c0_controls_are_escaped() {
    // Python: json.dumps('a"b\\c\nd\te\x07f\x1ff', ensure_ascii=False)
    assert_eq!(
        encode_basestring("a\"b\\c\nd\te\u{7}f\u{1f}f"),
        "\"a\\\"b\\\\c\\nd\\te\\u0007f\\u001ff\""
    );
}

#[test]
fn the_five_short_control_escapes_are_preferred_over_the_hex_form() {
    assert_eq!(encode_basestring("\u{8}\u{c}\n\r\t"), r#""\b\f\n\r\t""#);
}

#[test]
fn a_solidus_and_a_delete_character_are_not_escaped() {
    assert_eq!(encode_basestring("a/b\u{7f}"), "\"a/b\u{7f}\"");
}

#[test]
fn non_ascii_is_written_literally_because_ensure_ascii_is_false() {
    assert_eq!(encode_basestring("clé ✅ 😀"), "\"clé ✅ 😀\"");
}

// ---------------------------------------------------------------------------
// Writing — `json.dump(..., indent=4, ensure_ascii=False)`
// ---------------------------------------------------------------------------

#[test]
fn pretty_output_is_four_space_indented_and_has_no_trailing_newline() {
    assert_eq!(dumps_pretty(&obj(&[("a", int(1))])), "{\n    \"a\": 1\n}");
}

#[test]
fn empty_containers_stay_on_one_line_when_pretty_printing() {
    let value = obj(&[
        ("e", JsonLike::Object(IndexMap::new())),
        ("l", JsonLike::Array(vec![])),
        (
            "n",
            JsonLike::Array(vec![
                JsonLike::Array(vec![]),
                JsonLike::Object(IndexMap::new()),
            ]),
        ),
    ]);
    assert_eq!(
        dumps_pretty(&value),
        "{\n    \"e\": {},\n    \"l\": [],\n    \"n\": [\n        [],\n        {}\n    ]\n}"
    );
}

#[test]
fn a_tuple_is_written_as_a_json_array_just_as_python_writes_one() {
    assert_eq!(dumps(&JsonLike::Tuple(vec![int(1), int(2)])), "[1, 2]");
}

#[test]
fn an_unbounded_integer_is_written_with_every_digit() {
    let huge = "1606938044258990275541962092341162602522202993782792835301376"; // 2**200
    assert_eq!(dumps(&JsonLike::Int(BigInt::from_str(huge).unwrap())), huge);
}

#[test]
fn booleans_and_null_are_written_as_json_not_as_python() {
    assert_eq!(
        dumps(&JsonLike::Array(vec![
            JsonLike::Bool(true),
            JsonLike::Bool(false),
            JsonLike::Null
        ])),
        "[true, false, null]"
    );
}

#[test]
fn the_default_separators_are_pythons_when_no_indent_is_given() {
    assert_eq!(
        dumps(&obj(&[
            ("a", int(1)),
            ("b", JsonLike::Array(vec![int(2), int(3)]))
        ])),
        r#"{"a": 1, "b": [2, 3]}"#
    );
}

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

#[test]
fn a_document_of_every_readable_kind_round_trips_through_dump_and_load() {
    let text = r#"{
    "s": "héllo ✅",
    "i": 340282366920938463463374607431768211457,
    "f": 1e+16,
    "small": 1e-05,
    "b": [
        true,
        false,
        null
    ],
    "inf": Infinity,
    "ninf": -Infinity,
    "nested": {
        "empty": {},
        "list": []
    }
}"#;
    let value = loads(text).unwrap();
    assert_eq!(dumps_pretty(&value), text);
    assert_eq!(loads(&dumps_pretty(&value)).unwrap(), value);
}

#[test]
fn type_name_reports_the_python_type_of_each_variant() {
    assert_eq!(JsonLike::Null.type_name(), "NoneType");
    assert_eq!(JsonLike::Bool(true).type_name(), "bool");
    assert_eq!(int(1).type_name(), "int");
    assert_eq!(JsonLike::Float(1.0).type_name(), "float");
    assert_eq!(s("x").type_name(), "str");
    assert_eq!(JsonLike::Array(vec![]).type_name(), "list");
    assert_eq!(JsonLike::Tuple(vec![]).type_name(), "tuple");
    assert_eq!(JsonLike::Object(IndexMap::new()).type_name(), "dict");
}

#[test]
fn as_object_borrows_only_a_dict() {
    assert!(obj(&[]).as_object().is_some());
    assert!(JsonLike::Array(vec![]).as_object().is_none());
}

#[test]
fn a_leading_utf8_bom_has_cpythons_specific_diagnostic() {
    assert_eq!(
        loads("\u{feff}{}").unwrap_err(),
        ParseError {
            msg: "Unexpected UTF-8 BOM (decode using utf-8-sig)".to_string(),
            position: 0,
            line: 1,
            column: 1,
        }
    );
}

#[test]
fn deeply_nested_input_returns_an_error_without_aborting_the_process() {
    const CHILD: &str = "REROBOT_DEEP_JSON_READER_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let depth = 100_000;
        let text = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        assert!(loads(&text).is_err());
        return;
    }

    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "deeply_nested_input_returns_an_error_without_aborting_the_process",
        ])
        .env(CHILD, "1")
        .status()
        .unwrap();
    assert!(status.success(), "the parser child aborted: {status}");
}

#[test]
fn deeply_nested_output_is_written_without_recursive_stack_exhaustion() {
    const CHILD: &str = "REROBOT_DEEP_JSON_WRITER_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let depth = 100_000;
        let mut value = JsonLike::Int(BigInt::from(0));
        for _ in 0..depth {
            value = JsonLike::Array(vec![value]);
        }
        let encoded = dumps(&value);
        assert_eq!(encoded.len(), depth * 2 + 1);
        // Recursively dropping this synthetic value would test `Drop`, not JSON.
        std::mem::forget(value);
        return;
    }

    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "deeply_nested_output_is_written_without_recursive_stack_exhaustion",
        ])
        .env(CHILD, "1")
        .status()
        .unwrap();
    assert!(status.success(), "the writer child aborted: {status}");
}

#[test]
fn deeply_caller_built_value_destruction_remains_a_documented_recursive_boundary() {
    const CHILD: &str = "REROBOT_DEEP_DROP_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let mut value = JsonLike::Null;
        for _ in 0..100_000 {
            value = JsonLike::Array(vec![value]);
        }
        drop(value);
        return;
    }

    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "deeply_caller_built_value_destruction_remains_a_documented_recursive_boundary",
        ])
        .env(CHILD, "1")
        .status()
        .unwrap();
    assert!(
        !status.success(),
        "recursive destruction unexpectedly became stack-safe; update the documented boundary"
    );
}

#[test]
fn parser_complexity_budgets_return_errors_before_unbounded_allocation() {
    let oversized = " ".repeat(MAX_JSON_INPUT_BYTES + 1);
    assert!(loads(&oversized)
        .unwrap_err()
        .msg
        .starts_with("Rerobot JSON input byte limit exceeded"));

    let number = "1".repeat(MAX_JSON_NUMBER_CHARS + 1);
    assert_eq!(
        loads(&number).unwrap_err().msg,
        "Rerobot JSON number limit exceeded"
    );

    let string = format!("\"{}\"", "a".repeat(MAX_JSON_STRING_CHARS + 1));
    assert_eq!(
        loads(&string).unwrap_err().msg,
        "Rerobot JSON string limit exceeded"
    );
    let key = format!("{{\"{}\": 0}}", "k".repeat(MAX_JSON_STRING_CHARS + 1));
    assert_eq!(
        loads(&key).unwrap_err().msg,
        "Rerobot JSON string limit exceeded"
    );

    let at_depth = format!(
        "{}0{}",
        "[".repeat(MAX_JSON_NESTING_DEPTH),
        "]".repeat(MAX_JSON_NESTING_DEPTH)
    );
    assert!(loads(&at_depth).is_ok());
    let over_depth = format!("[{at_depth}]");
    assert_eq!(
        loads(&over_depth).unwrap_err().msg,
        "Rerobot JSON nesting limit exceeded"
    );

    let at_node_limit = format!("[{}]", vec!["0"; MAX_JSON_NODES - 1].join(","));
    assert!(loads(&at_node_limit).is_ok());
    let array = format!("[{}]", vec!["0"; MAX_JSON_NODES].join(","));
    assert_eq!(
        loads(&array).unwrap_err().msg,
        "Rerobot JSON node limit exceeded"
    );
}
