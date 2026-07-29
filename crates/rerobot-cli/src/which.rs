//! Port of the `shutil.which` call that `get_ffmpeg_version` makes before it
//! runs anything.
//!
//! Upstream's two `ffmpeg` failure modes are reported differently, and the split
//! is decided here rather than by the spawn: `shutil.which` returning `None` is
//! the only path to `N/A`, so name resolution has to be a separate step with
//! the same acceptance rule Python uses.
//!
//! The rule being ported is CPython **3.12**'s `shutil.which`, which differs
//! from older releases in three ways this module reproduces: the search path
//! falls back to `os.confstr("CS_PATH")` when `PATH` is unset, the Windows
//! current directory is inserted only when `NeedCurrentDirectoryForExePath`
//! says so, and a Windows name whose extension is *not* in `PATHEXT` is no
//! longer tried as itself. Only the `mode` CPython defaults to,
//! `os.F_OK | os.X_OK`, is modelled, because that is the only mode
//! `get_ffmpeg_version` uses.
//!
//! ```
//! use rerobot_cli::which::which;
//!
//! // Resolution is a lookup, not an execution: nothing is spawned.
//! assert!(which("this-command-does-not-exist-anywhere").is_none());
//! ```

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

/// Python's `shutil._WIN_DEFAULT_PATHEXT`, used when `PATHEXT` is unset or empty.
#[cfg(windows)]
const WIN_DEFAULT_PATHEXT: &str = ".COM;.EXE;.BAT;.CMD;.VBS;.JS;.WS;.MSC";

/// `os.defpath`, the last-resort search path when `PATH` is unset.
///
/// `posixpath.defpath` on Unix, `ntpath.defpath` on Windows — the values
/// CPython 3.12 ships.
#[cfg(not(windows))]
const DEFPATH: &str = "/bin:/usr/bin";
#[cfg(windows)]
const DEFPATH: &str = ".;C:\\bin";

/// The `os.pathsep` this target uses.
#[cfg(not(windows))]
const PATHSEP: char = ':';
#[cfg(windows)]
const PATHSEP: char = ';';

/// Resolve `program` against the process `PATH`, like `shutil.which(program)`.
///
/// An unset `PATH` falls back to the system default search path; an *empty*
/// one resolves nothing (CPython bpo-35755: `PATH=''` deliberately does not
/// fall back). See [`which_in`].
pub fn which(program: &str) -> Option<PathBuf> {
    which_in(program, std::env::var_os("PATH").as_deref())
}

/// [`which`] against an explicit search path, so callers can test it.
///
/// `search_path` is `None` for "no `PATH` in the environment at all" and
/// `Some("")` for an empty one. CPython distinguishes the two: the first falls
/// back to `os.confstr("CS_PATH")`, or to `os.defpath` where that is
/// unavailable, while the second returns `None` without searching anything.
pub fn which_in(program: &str, search_path: Option<&OsStr>) -> Option<PathBuf> {
    // `dirname, cmd = os.path.split(cmd)`: a name with a directory part is
    // looked up directly, and the *basename* is what gets looked up.
    let (dirname, cmd) = split_path(program);

    let directories: Vec<OsString> = if !dirname.is_empty() {
        vec![OsString::from(dirname)]
    } else {
        let owned;
        let search_path: &OsStr = match search_path {
            Some(path) => path,
            None => {
                owned = default_search_path();
                &owned
            }
        };
        if search_path.is_empty() {
            return None;
        }
        #[allow(unused_mut)]
        let mut dirs = split_pathsep(search_path);
        // On Windows the current directory is searched first, but only when
        // `NeedCurrentDirectoryForExePath` says it should be — the
        // `NoDefaultCurrentDirectoryInExePath` environment variable turns it
        // off. CPython 3.12 inserts unconditionally when that returns true,
        // without first checking whether the path already lists `.`.
        #[cfg(windows)]
        if win_path_needs_curdir(cmd) {
            dirs.insert(0, OsString::from("."));
        }
        dirs
    };

    let candidates = candidate_names(cmd);
    let mut seen: Vec<OsString> = Vec::with_capacity(directories.len());
    for directory in directories {
        // Python skips directories it has already searched, which also collapses
        // the repeated empty entries an adjacent-separator `PATH` produces.
        let key = normcase(&directory);
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        for candidate in &candidates {
            let path = Path::new(&directory).join(candidate);
            if is_executable_file(&path) {
                return Some(path);
            }
        }
    }
    None
}

/// `os.path.split`, which is not what `Path::parent` / `Path::file_name` do.
///
/// The differences that matter here: a trailing separator makes the whole name
/// a directory component with an empty basename (`"sh/"` is `("sh", "")`, not
/// `("", "sh")`), and a lone root stays a root rather than being stripped.
#[cfg(not(windows))]
fn split_path(p: &str) -> (&str, &str) {
    match p.rfind('/') {
        None => ("", p),
        Some(i) => {
            let (head, tail) = p.split_at(i + 1);
            let stripped = head.trim_end_matches('/');
            // `head.rstrip('/') or head`: an all-separator head is the root.
            (if stripped.is_empty() { head } else { stripped }, tail)
        }
    }
}

/// `ntpath.split`: both separators count, and a drive or UNC root is never
/// stripped off the head.
///
/// `\\?\`-prefixed device paths are treated as ordinary rooted paths rather
/// than given their own drive syntax; they are not names a `which` caller can
/// pass through this port's `&str` API without already knowing the answer.
#[cfg(windows)]
fn split_path(p: &str) -> (&str, &str) {
    let (root_len, rest) = split_root(p);
    let sep = |c: char| c == '\\' || c == '/';
    let cut = rest.rfind(sep).map_or(0, |i| i + 1);
    let (head, tail) = rest.split_at(cut);
    let stripped = head.trim_end_matches(sep);
    (&p[..root_len + stripped.len()], tail)
}

/// Length of the drive-and-root prefix of `p`, per `ntpath.splitroot`.
#[cfg(windows)]
fn split_root(p: &str) -> (usize, &str) {
    let normp: Vec<char> = p.chars().map(|c| if c == '/' { '\\' } else { c }).collect();
    let at = |i: usize| normp.get(i).copied();
    let len = if at(0) == Some('\\') {
        if at(1) == Some('\\') {
            // UNC or device root: `\\server\share`, `\\?\UNC\server\share`.
            let upper: String = normp.iter().take(8).collect::<String>().to_uppercase();
            let start = if upper == "\\\\?\\UNC\\" { 8 } else { 2 };
            match find_sep(&normp, start).and_then(|i| find_sep(&normp, i + 1)) {
                Some(index2) => index2 + 1,
                None => normp.len(),
            }
        } else {
            1
        }
    } else if at(1) == Some(':') {
        if at(2) == Some('\\') {
            3
        } else {
            2
        }
    } else {
        0
    };
    // `normp` is char-indexed; map back to a byte offset in `p`.
    let byte_len = p
        .char_indices()
        .nth(len)
        .map_or(p.len(), |(offset, _)| offset);
    (byte_len, &p[byte_len..])
}

#[cfg(windows)]
fn find_sep(chars: &[char], from: usize) -> Option<usize> {
    chars
        .iter()
        .skip(from)
        .position(|&c| c == '\\')
        .map(|i| i + from)
}

/// The search path CPython uses when `PATH` is absent from the environment.
///
/// Unix: `os.confstr("CS_PATH")`, falling back to `os.defpath` when `confstr`
/// or the `_CS_PATH` name is unavailable. Windows: `os.confstr` does not exist
/// there, so CPython's `except AttributeError` branch always fires and the
/// answer is `ntpath.defpath`.
#[cfg(windows)]
fn default_search_path() -> OsString {
    OsString::from(DEFPATH)
}

#[cfg(not(windows))]
fn default_search_path() -> OsString {
    confstr_cs_path().unwrap_or_else(|| OsString::from(DEFPATH))
}

/// `os.confstr("CS_PATH")`.
///
/// `confstr` reports the buffer size it needs when handed a null one, and
/// returns `0` both for "no such name" and for "the value is empty" — which is
/// exactly the pair of cases CPython's `except (AttributeError, ValueError)`
/// and its `if not path` fall through to `os.defpath` for.
#[cfg(not(windows))]
#[allow(unsafe_code)]
fn confstr_cs_path() -> Option<OsString> {
    use std::os::unix::ffi::OsStringExt;

    // SAFETY: the null-buffer/zero-length form of `confstr` is the documented
    // way to ask for the required size; it writes nothing.
    let needed = unsafe { libc::confstr(libc::_CS_PATH, std::ptr::null_mut(), 0) };
    if needed == 0 {
        return None;
    }
    let mut buffer = vec![0u8; needed];
    // SAFETY: `buffer` is `needed` bytes long and writable, which is the size
    // `confstr` just asked for.
    let written = unsafe {
        libc::confstr(
            libc::_CS_PATH,
            buffer.as_mut_ptr().cast::<libc::c_char>(),
            needed,
        )
    };
    if written == 0 || written > needed {
        return None;
    }
    // `written` counts the trailing NUL.
    buffer.truncate(written - 1);
    if buffer.is_empty() {
        return None;
    }
    Some(OsString::from_vec(buffer))
}

/// `path.split(os.pathsep)`, preserving empty entries.
///
/// Deliberately not `std::env::split_paths`, which additionally strips double
/// quotes on Windows. Python does no such thing, and an entry it would have
/// searched must not silently disappear.
#[cfg(not(windows))]
fn split_pathsep(path: &OsStr) -> Vec<OsString> {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    path.as_bytes()
        .split(|&b| b == PATHSEP as u8)
        .map(|part| OsString::from_vec(part.to_vec()))
        .collect()
}

#[cfg(windows)]
fn split_pathsep(path: &OsStr) -> Vec<OsString> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    let wide: Vec<u16> = path.encode_wide().collect();
    wide.split(|&unit| unit == PATHSEP as u16)
        .map(OsString::from_wide)
        .collect()
}

/// `shutil.which`'s `files` list: just the name on Unix.
#[cfg(not(windows))]
fn candidate_names(cmd: &str) -> Vec<String> {
    vec![cmd.to_string()]
}

/// `shutil.which`'s `files` list on Windows, from the live `PATHEXT`.
#[cfg(windows)]
fn candidate_names(cmd: &str) -> Vec<String> {
    // `os.getenv("PATHEXT") or _WIN_DEFAULT_PATHEXT`: an empty value is falsy.
    let source = match std::env::var("PATHEXT") {
        Ok(value) if !value.is_empty() => value,
        _ => WIN_DEFAULT_PATHEXT.to_string(),
    };
    pathext_candidates(cmd, &source)
}

/// The `PATHEXT` expansion, split out from the environment so it is testable.
///
/// CPython 3.12:
///
/// ```text
/// pathext = [ext.rstrip('.') for ext in pathext_source.split(os.pathsep) if ext]
/// files = [cmd + ext for ext in pathext]
/// if not (mode & os.X_OK) or any(normcmd.endswith(ext.upper()) for ext in pathext):
///     files.insert(0, cmd)
/// ```
///
/// `mode` always has `X_OK` here, so the bare name is tried first only when its
/// own extension is in `PATHEXT` — the cmd.exe rule.
#[cfg(windows)]
fn pathext_candidates(cmd: &str, pathext_source: &str) -> Vec<String> {
    // The emptiness filter runs before the `.`-stripping, so a lone "." entry
    // survives as an empty extension, exactly as upstream.
    let pathext: Vec<String> = pathext_source
        .split(PATHSEP)
        .filter(|ext| !ext.is_empty())
        .map(|ext| ext.trim_end_matches('.').to_string())
        .collect();
    let mut files: Vec<String> = pathext.iter().map(|ext| format!("{cmd}{ext}")).collect();
    let normcmd = cmd.to_uppercase();
    if pathext
        .iter()
        .any(|ext| normcmd.ends_with(&ext.to_uppercase()))
    {
        files.insert(0, cmd.to_string());
    }
    files
}

/// `_winapi.NeedCurrentDirectoryForExePath(cmd)`.
///
/// The Win32 call, not a reimplementation of it: its handling of
/// `NoDefaultCurrentDirectoryInExePath` is what CPython defers to, and guessing
/// at the details would be a parity claim this port cannot check. Declared
/// directly rather than through `windows-sys`, which would pull a large
/// generated dependency tree in for one stable entry point.
#[cfg(windows)]
#[allow(unsafe_code)]
fn win_path_needs_curdir(cmd: &str) -> bool {
    #[link(name = "kernel32")]
    extern "system" {
        fn NeedCurrentDirectoryForExePathW(exe_name: *const u16) -> i32;
    }

    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = OsStr::new(cmd)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    // SAFETY: `wide` is a NUL-terminated UTF-16 string that outlives the call,
    // and the callee only reads it.
    unsafe { NeedCurrentDirectoryForExePathW(wide.as_ptr()) != 0 }
}

/// `os.path.normcase`: case-folding on Windows, identity elsewhere.
#[cfg(not(windows))]
fn normcase(path: &OsStr) -> OsString {
    path.to_os_string()
}

/// `ntpath.normcase`: separators unified, then lowercased.
///
/// CPython lowercases with `LCMapStringEx(LOCALE_NAME_INVARIANT, ...)`, which
/// can disagree with Rust's Unicode `to_lowercase` on non-ASCII characters.
/// This value is only ever a key in the "already searched" set, so a
/// disagreement can at worst make a duplicated non-ASCII `PATH` entry be
/// searched twice; it cannot change which file is returned.
#[cfg(windows)]
fn normcase(path: &OsStr) -> OsString {
    OsString::from(path.to_string_lossy().replace('/', "\\").to_lowercase())
}

/// Python's `_access_check(name, os.F_OK | os.X_OK)`:
///
/// ```text
/// os.path.exists(fn) and os.access(fn, mode) and not os.path.isdir(fn)
/// ```
///
/// `os.access` asks the kernel via `access(2)`, which resolves the request
/// against the **real** uid and gid and applies the owner/group/other classes
/// in order. Reading the mode bits instead — as this port used to — accepts a
/// `0o011` file its owner cannot execute, and rejects files an ACL grants.
#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    // `os.path.exists` and `os.path.isdir` both follow symlinks and both answer
    // False on any error, which is what `fs::metadata` does.
    match std::fs::metadata(path) {
        Ok(meta) => !meta.is_dir() && access_x_ok(path),
        Err(_) => false,
    }
}

/// `os.access(path, os.F_OK | os.X_OK)`.
#[cfg(unix)]
#[allow(unsafe_code)]
fn access_x_ok(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;

    // A path with an interior NUL cannot name a file, and `access` could not be
    // handed one anyway.
    let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: `c_path` is a valid NUL-terminated string that outlives the call,
    // and `access` only reads it.
    unsafe { libc::access(c_path.as_ptr(), libc::F_OK | libc::X_OK) == 0 }
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    // `os.access(path, os.X_OK)` is equivalent to `F_OK` on Windows — the
    // execute bit has no meaning there — so existence and not-a-directory is
    // the whole check.
    match std::fs::metadata(path) {
        Ok(meta) => !meta.is_dir(),
        Err(_) => false,
    }
}

#[cfg(all(test, windows))]
mod windows_tests {
    //! Pure-logic tests for the Windows branches.
    //!
    //! These sit beside the code rather than in `tests/which.rs` so they can
    //! reach `pathext_candidates` without widening the public API, and so they
    //! can pin `PATHEXT` parsing without mutating a process-global environment
    //! variable that the rest of the suite reads concurrently.

    use super::{pathext_candidates, split_path, WIN_DEFAULT_PATHEXT};

    #[test]
    fn the_default_pathext_expands_a_bare_name_in_order() {
        assert_eq!(
            pathext_candidates("ffmpeg", WIN_DEFAULT_PATHEXT),
            vec![
                "ffmpeg.COM",
                "ffmpeg.EXE",
                "ffmpeg.BAT",
                "ffmpeg.CMD",
                "ffmpeg.VBS",
                "ffmpeg.JS",
                "ffmpeg.WS",
                "ffmpeg.MSC",
            ]
        );
    }

    #[test]
    fn a_name_whose_extension_is_in_pathext_is_tried_bare_first() {
        let files = pathext_candidates("ffmpeg.exe", WIN_DEFAULT_PATHEXT);
        assert_eq!(files[0], "ffmpeg.exe");
        assert_eq!(files[1], "ffmpeg.exe.COM");
    }

    #[test]
    fn a_name_whose_extension_is_not_in_pathext_is_never_tried_bare() {
        // The CPython 3.12 rule: with X_OK requested, only a PATHEXT extension
        // short-circuits to the direct match.
        let files = pathext_candidates("ffmpeg.bin", WIN_DEFAULT_PATHEXT);
        assert!(!files.contains(&"ffmpeg.bin".to_string()));
        assert_eq!(files[0], "ffmpeg.bin.COM");
    }

    #[test]
    fn empty_pathext_entries_are_dropped_and_trailing_dots_stripped() {
        // `[ext.rstrip('.') for ext in source.split(';') if ext]`.
        assert_eq!(
            pathext_candidates("x", ".EXE.;;.BAT"),
            vec!["x.EXE", "x.BAT"]
        );
    }

    #[test]
    fn a_lone_dot_entry_becomes_an_empty_extension_and_matches_bare() {
        // `"."` survives the emptiness filter and rstrips to `""`, so the bare
        // name both appears as an expansion and short-circuits.
        let files = pathext_candidates("x", ".");
        assert_eq!(files, vec!["x", "x"]);
    }

    #[test]
    fn split_path_matches_ntpath_split() {
        // Oracle: CPython 3.12 `ntpath.split` on each of these, which was run
        // on the development machine even though this module is not compiled
        // there. A stripped drive or UNC root would turn a rooted name into a
        // relative one and search the wrong directory.
        assert_eq!(split_path("ffmpeg"), ("", "ffmpeg"));
        assert_eq!(split_path(""), ("", ""));
        assert_eq!(split_path("bin\\ffmpeg"), ("bin", "ffmpeg"));
        assert_eq!(split_path("bin/ffmpeg"), ("bin", "ffmpeg"));
        assert_eq!(split_path("ffmpeg\\"), ("ffmpeg", ""));
        assert_eq!(split_path("a\\\\b\\"), ("a\\\\b", ""));
        assert_eq!(split_path("C:ffmpeg"), ("C:", "ffmpeg"));
        assert_eq!(split_path("C:"), ("C:", ""));
        assert_eq!(split_path("C:\\"), ("C:\\", ""));
        assert_eq!(split_path("C:\\bin\\ffmpeg"), ("C:\\bin", "ffmpeg"));
        assert_eq!(split_path("\\"), ("\\", ""));
        assert_eq!(split_path("\\\\host\\share"), ("\\\\host\\share", ""));
        // The share root keeps its trailing separator, so this is a directory
        // component and not a bare name.
        assert_eq!(
            split_path("\\\\host\\share\\ffmpeg"),
            ("\\\\host\\share\\", "ffmpeg")
        );
    }
}

#[cfg(all(test, not(windows)))]
mod unix_tests {
    use super::split_path;

    #[test]
    fn split_path_matches_posixpath_split() {
        // Oracle: CPython 3.12 `posixpath.split` on each of these.
        assert_eq!(split_path("ffmpeg"), ("", "ffmpeg"));
        assert_eq!(split_path(""), ("", ""));
        assert_eq!(split_path("a/b"), ("a", "b"));
        assert_eq!(split_path("./x"), (".", "x"));
        assert_eq!(split_path("ffmpeg/"), ("ffmpeg", ""));
        assert_eq!(split_path("/"), ("/", ""));
        assert_eq!(split_path("a//b/"), ("a//b", ""));
        assert_eq!(split_path("//x/y"), ("//x", "y"));
    }
}
