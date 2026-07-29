//! Behaviour parity tests for the pure parts of `lerobot.scripts.lerobot_info`.

use rerobot_core::sysinfo::{
    format_dict_for_markdown, parse_ffmpeg_version, FFMPEG_PARSE_FAILED, NOT_AVAILABLE,
};

#[test]
fn parses_a_realistic_ffmpeg_banner() {
    let out =
        "ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers\nbuilt with clang\n";
    assert_eq!(parse_ffmpeg_version(out), "7.1");
}

#[test]
fn parses_a_distro_suffixed_version() {
    let out = "ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers";
    assert_eq!(parse_ffmpeg_version(out), "6.1.1-3ubuntu5");
}

#[test]
fn parses_a_git_build_version() {
    let out = "ffmpeg version N-113518-g8ea1b0e5e8 Copyright (c) 2000-2024";
    assert_eq!(parse_ffmpeg_version(out), "N-113518-g8ea1b0e5e8");
}

#[test]
fn empty_output_reports_a_parse_failure() {
    assert_eq!(parse_ffmpeg_version(""), FFMPEG_PARSE_FAILED);
}

#[test]
fn a_first_line_with_fewer_than_three_tokens_reports_a_parse_failure() {
    assert_eq!(parse_ffmpeg_version("ffmpeg version"), FFMPEG_PARSE_FAILED);
    assert_eq!(
        parse_ffmpeg_version("ffmpeg\nversion 7.1 x"),
        FFMPEG_PARSE_FAILED
    );
    assert_eq!(
        parse_ffmpeg_version("\nffmpeg version 7.1"),
        FFMPEG_PARSE_FAILED
    );
}

#[test]
fn splitting_is_on_single_spaces_so_runs_produce_empty_tokens() {
    // Python `"a  b c".split(" ")` == ["a", "", "b", "c"], so index 2 is "b".
    assert_eq!(parse_ffmpeg_version("ffmpeg  version 7.1"), "version");
    assert_eq!(parse_ffmpeg_version("ffmpeg   version"), "");
}

#[test]
fn tabs_are_not_separators() {
    assert_eq!(
        parse_ffmpeg_version("ffmpeg\tversion\t7.1"),
        FFMPEG_PARSE_FAILED
    );
}

#[test]
fn carriage_returns_terminate_the_first_line_like_python_splitlines() {
    assert_eq!(
        parse_ffmpeg_version("ffmpeg version 7.1\r\nbuilt with"),
        "7.1"
    );
    assert_eq!(parse_ffmpeg_version("ffmpeg version 7.1\rnext"), "7.1");
}

#[test]
fn trailing_content_after_the_version_token_is_ignored() {
    assert_eq!(parse_ffmpeg_version("a b c d e f"), "c");
}

#[test]
fn markdown_formatting_matches_upstream() {
    let rendered = format_dict_for_markdown([("Platform", "macOS"), ("Python version", "3.12.0")]);
    assert_eq!(rendered, "- Platform: macOS\n- Python version: 3.12.0");
}

#[test]
fn markdown_formatting_of_no_pairs_is_the_empty_string() {
    assert_eq!(format_dict_for_markdown([]), "");
}

#[test]
fn markdown_formatting_has_no_trailing_newline() {
    let rendered = format_dict_for_markdown([("A", "1")]);
    assert_eq!(rendered, "- A: 1");
    assert!(!rendered.ends_with('\n'));
}

#[test]
fn markdown_formatting_preserves_the_not_available_sentinel() {
    let rendered = format_dict_for_markdown([("Transformers version", NOT_AVAILABLE)]);
    assert_eq!(rendered, "- Transformers version: N/A");
}
