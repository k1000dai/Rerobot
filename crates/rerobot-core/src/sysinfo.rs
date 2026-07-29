//! Port of `lerobot.scripts.lerobot_info` (the pure, dependency-free parts).

/// Upstream sentinel for "package/tool not found".
pub const NOT_AVAILABLE: &str = "N/A";

/// Upstream sentinel emitted when `ffmpeg -version` output cannot be parsed.
pub const FFMPEG_PARSE_FAILED: &str = "Installed (version parsing failed)";

/// First line of `text` under Python's `str.splitlines` rules.
///
/// Wider than Rust's `str::lines`, which only breaks on `\n`: Python also
/// breaks on `\r`, `\v`, `\f`, the C1 file/group/record separators, `NEL`, and
/// the Unicode line/paragraph separators. `parse_ffmpeg_version` observes the
/// difference on Windows-style `\r`-only output.
fn first_line(text: &str) -> Option<&str> {
    if text.is_empty() {
        return None;
    }
    const BREAKS: [char; 10] = [
        '\n',       // LINE FEED
        '\r',       // CARRIAGE RETURN
        '\u{0b}',   // LINE TABULATION
        '\u{0c}',   // FORM FEED
        '\u{1c}',   // FILE SEPARATOR
        '\u{1d}',   // GROUP SEPARATOR
        '\u{1e}',   // RECORD SEPARATOR
        '\u{85}',   // NEXT LINE
        '\u{2028}', // LINE SEPARATOR
        '\u{2029}', // PARAGRAPH SEPARATOR
    ];
    let end = text.find(BREAKS).unwrap_or(text.len());
    Some(&text[..end])
}

/// Extract the version token from `ffmpeg -version` stdout.
///
/// Port of `get_ffmpeg_version`'s parsing half: take the first line, split it on
/// single spaces, and return element `2`. A missing element maps to
/// [`FFMPEG_PARSE_FAILED`], matching upstream's `IndexError` branch.
///
/// ```
/// use rerobot_core::sysinfo::{parse_ffmpeg_version, FFMPEG_PARSE_FAILED};
///
/// let banner = "ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers";
/// assert_eq!(parse_ffmpeg_version(banner), "7.1");
/// assert_eq!(parse_ffmpeg_version(""), FFMPEG_PARSE_FAILED);
/// ```
pub fn parse_ffmpeg_version(stdout: &str) -> String {
    // Python's `split(" ")` keeps empty fields, unlike `str.split()`; Rust's
    // `split(' ')` matches it exactly.
    match first_line(stdout).and_then(|line| line.split(' ').nth(2)) {
        Some(version) => version.to_string(),
        None => FFMPEG_PARSE_FAILED.to_string(),
    }
}

/// Format key/value pairs as a markdown bullet list, port of
/// `format_dict_for_markdown`.
///
/// ```
/// use rerobot_core::sysinfo::format_dict_for_markdown;
///
/// let rendered = format_dict_for_markdown([("Platform", "macOS"), ("FFmpeg version", "7.1")]);
/// assert_eq!(rendered, "- Platform: macOS\n- FFmpeg version: 7.1");
/// ```
pub fn format_dict_for_markdown<'a, I>(pairs: I) -> String
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    pairs
        .into_iter()
        .map(|(key, value)| format!("- {key}: {value}"))
        .collect::<Vec<_>>()
        .join("\n")
}
