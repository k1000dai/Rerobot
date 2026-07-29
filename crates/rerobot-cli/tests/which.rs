//! Tests for the `shutil.which` port and for the `ffmpeg` probe built on it.
//!
//! Upstream `get_ffmpeg_version` resolves `ffmpeg` on `PATH` *first* and only
//! then runs it, and the two failures are reported differently: an unresolvable
//! name is `N/A`, while a resolved binary that cannot be run to completion is
//! `Installed (version parsing failed)`. These tests pin that split.

use rerobot_cli::info::{detect_ffmpeg_in, probe_ffmpeg_at, sys_info, Environment, FfmpegProbe};
use rerobot_cli::which::which_in;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

/// A throwaway directory, removed on drop. Kept here rather than pulled in as a
/// dev-dependency: the workspace ships no third-party test crates.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let unique = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rerobot-which-{}-{tag}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp dir is creatable");
        Self { path }
    }

    fn file(&self, name: &str, contents: &[u8]) -> PathBuf {
        let path = self.path.join(name);
        std::fs::write(&path, contents).expect("temp file is writable");
        path
    }

    /// A file with the execute bits set on Unix; a plain file elsewhere.
    fn executable(&self, name: &str, contents: &[u8]) -> PathBuf {
        let path = self.file(name, contents);
        set_executable(&path);
        path
    }

    fn subdir(&self, name: &str) -> PathBuf {
        let path = self.path.join(name);
        std::fs::create_dir_all(&path).expect("temp subdir is creatable");
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(unix)]
fn set_executable(path: &Path) {
    set_mode(path, 0o755);
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path).expect("metadata").permissions();
    perms.set_mode(mode);
    std::fs::set_permissions(path, perms).expect("chmod");
}

/// Whether `access(2)` would be answered for root.
///
/// `access(2)` consults the *real* uid, not the effective one, so `id -ru` is
/// the question that matches it. Shelling out keeps the test crate free of a
/// libc dependency it otherwise does not need.
#[cfg(unix)]
fn real_uid_is_root() -> bool {
    std::process::Command::new("id")
        .arg("-ru")
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim() == "0")
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) {}

fn path_list<I: IntoIterator<Item = PathBuf>>(dirs: I) -> OsString {
    std::env::join_paths(dirs).expect("joinable search path")
}

fn search(dirs: &OsString) -> Option<&OsStr> {
    Some(dirs.as_os_str())
}

// --- `shutil.which` ------------------------------------------------------

// CPython 3.12 never tries a bare Windows name when its extension is absent
// from PATHEXT. Generic lookup-order tests therefore create the platform's
// actual candidate spelling rather than accidentally asserting POSIX rules on
// Windows.
#[cfg(windows)]
const TEST_PROGRAM_FILE: &str = "ffmpeg.EXE";
#[cfg(not(windows))]
const TEST_PROGRAM_FILE: &str = "ffmpeg";

#[test]
fn an_executable_on_the_search_path_is_found() {
    let dir = TempDir::new("found");
    let exe = dir.executable(TEST_PROGRAM_FILE, b"#!/bin/sh\nexit 0\n");
    let path = path_list([dir.path.clone()]);
    assert_eq!(which_in("ffmpeg", search(&path)), Some(exe));
}

#[test]
fn an_absent_program_is_not_found() {
    let dir = TempDir::new("absent");
    let path = path_list([dir.path.clone()]);
    assert_eq!(which_in("ffmpeg", search(&path)), None);
}

#[test]
fn an_empty_search_path_never_matches() {
    // CPython bpo-35755: `PATH=''` does not fall back to `os.defpath`.
    assert_eq!(which_in("ffmpeg", Some(OsStr::new(""))), None);
}

#[test]
#[cfg(unix)]
fn an_unset_search_path_falls_back_to_the_system_default() {
    // CPython 3.12 `shutil.which`, when `os.environ["PATH"]` is absent:
    //
    //     try:      path = os.confstr("CS_PATH")
    //     except (AttributeError, ValueError):
    //                path = os.defpath
    //
    // Both name the system directories, so a POSIX-mandated utility resolves
    // even with no PATH in the environment at all. Returning `None` here — as
    // this port used to — reports a working `ffmpeg` install as missing for
    // any process started with a scrubbed environment.
    let found = which_in("sh", None).expect("`sh` resolves on the system default search path");
    assert!(found.is_absolute(), "{found:?}");
    assert_eq!(found.file_name().expect("a file name"), "sh");

    // The fallback is for *unset*, not for empty: that distinction is bpo-35755
    // and the test above.
    assert_eq!(which_in("sh", Some(OsStr::new(""))), None);
}

#[test]
#[cfg(unix)]
fn the_default_search_path_holds_the_system_directories() {
    // Every entry `os.confstr("CS_PATH")` / `os.defpath` can produce is an
    // absolute system directory, so a name that is not a system utility must
    // still not resolve through the fallback.
    assert_eq!(
        which_in("rerobot-not-a-real-system-utility", None),
        None,
        "the fallback must be the system path, not the current directory"
    );
}

#[test]
#[cfg(unix)]
fn executability_is_decided_by_the_kernel_not_by_the_raw_mode_bits() {
    // `_access_check` calls `os.access(name, os.F_OK | os.X_OK)`, which is
    // `access(2)`: the kernel applies the owner class first and stops there.
    // Mode 0o011 is executable for group and other but *not* for the owner, so
    // the raw test `mode & 0o111 != 0` says yes where `access(2)` says no.
    if real_uid_is_root() {
        // For root, `access(X_OK)` succeeds when *any* execute bit is set, so
        // the two rules agree and the test can prove nothing. CI's `gates` job
        // runs unprivileged on all three platforms.
        eprintln!("skipped: running as root, where access(2) ignores the owner class");
        return;
    }
    let dir = TempDir::new("owner-cannot-execute");
    let path = dir.file("ffmpeg", b"#!/bin/sh\nexit 0\n");
    set_mode(&path, 0o011);
    let search_path = path_list([dir.path.clone()]);
    assert_eq!(
        which_in("ffmpeg", search(&search_path)),
        None,
        "0o011 has execute bits set but is not executable by its owner"
    );

    // The same file, executable by its owner, does resolve — so the rejection
    // above is about the access rule, not about the file being unreadable.
    set_mode(&path, 0o100);
    assert_eq!(which_in("ffmpeg", search(&search_path)), Some(path));
}

#[test]
#[cfg(unix)]
fn a_trailing_separator_makes_the_whole_name_a_directory_component() {
    // `os.path.split("sh/")` is `("sh", "")`, so CPython looks for the empty
    // file name inside the directory `sh` and never consults the search path.
    // Rust's `Path::file_name` says `Some("sh")` instead, which would resolve
    // this to the `sh` on PATH — a different program from the one asked for.
    let dir = TempDir::new("trailing-sep");
    dir.executable("ffmpeg", b"#!/bin/sh\nexit 0\n");
    let path = path_list([dir.path.clone()]);
    assert_eq!(which_in("ffmpeg/", search(&path)), None);
}

#[test]
#[cfg(unix)]
fn a_non_executable_candidate_is_skipped() {
    // `_access_check` requires `os.access(name, os.X_OK)`.
    let dir = TempDir::new("noexec");
    dir.file("ffmpeg", b"not executable\n");
    let path = path_list([dir.path.clone()]);
    assert_eq!(which_in("ffmpeg", search(&path)), None);
}

#[test]
fn a_directory_named_like_the_program_is_skipped() {
    // `_access_check` requires `not os.path.isdir(name)`.
    let dir = TempDir::new("isdir");
    let sub = dir.subdir("ffmpeg");
    set_executable(&sub);
    let path = path_list([dir.path.clone()]);
    assert_eq!(which_in("ffmpeg", search(&path)), None);
}

#[test]
fn the_first_directory_on_the_search_path_wins() {
    let first = TempDir::new("first");
    let second = TempDir::new("second");
    let winner = first.executable(TEST_PROGRAM_FILE, b"#!/bin/sh\nexit 0\n");
    second.executable(TEST_PROGRAM_FILE, b"#!/bin/sh\nexit 0\n");
    let path = path_list([first.path.clone(), second.path.clone()]);
    assert_eq!(which_in("ffmpeg", search(&path)), Some(winner));
}

#[test]
fn a_directory_that_does_not_exist_is_skipped_not_fatal() {
    let dir = TempDir::new("skip");
    let exe = dir.executable(TEST_PROGRAM_FILE, b"#!/bin/sh\nexit 0\n");
    let path = path_list([dir.path.join("nope"), dir.path.clone()]);
    assert_eq!(which_in("ffmpeg", search(&path)), Some(exe));
}

#[test]
fn a_program_with_a_directory_component_ignores_the_search_path() {
    // `os.path.split(cmd)` with a non-empty dirname replaces the search path.
    let onpath = TempDir::new("onpath");
    let elsewhere = TempDir::new("elsewhere");
    onpath.executable(TEST_PROGRAM_FILE, b"#!/bin/sh\nexit 0\n");
    let direct = elsewhere.executable(TEST_PROGRAM_FILE, b"#!/bin/sh\nexit 0\n");
    let path = path_list([onpath.path.clone()]);

    let spelled = direct.to_str().expect("utf-8 temp path");
    assert_eq!(which_in(spelled, search(&path)), Some(direct.clone()));

    let missing = elsewhere.path.join("nope");
    assert_eq!(
        which_in(missing.to_str().expect("utf-8 temp path"), search(&path)),
        None,
        "an explicit path must not fall back to the search path"
    );
}

// --- Windows semantics ---------------------------------------------------
//
// These cannot run on the developer machine this port is written on. CI's
// `gates` job builds and runs the whole test suite on `windows-latest`, which
// is where they are actually exercised; see `docs/compatibility.md` for the
// precise list of behaviour that is CI-verified rather than locally verified.

#[test]
#[cfg(windows)]
fn a_bare_name_resolves_through_a_pathext_extension() {
    // `files = [cmd + ext for ext in pathext]`, so `ffmpeg` finds `ffmpeg.EXE`.
    let dir = TempDir::new("pathext");
    let exe = dir.executable("ffmpeg.EXE", b"MZ");
    let path = path_list([dir.path.clone()]);
    assert_eq!(which_in("ffmpeg", search(&path)), Some(exe));
}

#[test]
#[cfg(windows)]
fn a_name_already_ending_in_a_pathext_extension_matches_directly() {
    // CPython 3.12: the bare `cmd` is prepended to `files` if and only if its
    // extension is in PATHEXT, simulating cmd.exe.
    let dir = TempDir::new("pathext-direct");
    let exe = dir.executable("ffmpeg.exe", b"MZ");
    let path = path_list([dir.path.clone()]);
    assert_eq!(which_in("ffmpeg.exe", search(&path)), Some(exe));
}

#[test]
#[cfg(windows)]
fn a_name_with_an_extension_outside_pathext_is_not_matched_directly() {
    // The 3.12 rule, and the one an "unconditional direct match" port gets
    // wrong: with X_OK requested, `ffmpeg.bin` is *not* tried as itself, only
    // as `ffmpeg.bin.COM`, `ffmpeg.bin.EXE`, and so on.
    let dir = TempDir::new("pathext-foreign");
    dir.executable("ffmpeg.bin", b"MZ");
    let path = path_list([dir.path.clone()]);
    assert_eq!(which_in("ffmpeg.bin", search(&path)), None);

    // Adding the PATHEXT-suffixed form makes it resolvable.
    let exe = dir.executable("ffmpeg.bin.EXE", b"MZ");
    assert_eq!(which_in("ffmpeg.bin", search(&path)), Some(exe));
}

#[test]
#[cfg(windows)]
fn pathext_order_decides_which_extension_wins() {
    // `.COM` precedes `.EXE` in `_WIN_DEFAULT_PATHEXT`, so it is found first.
    let dir = TempDir::new("pathext-order");
    let com = dir.executable("ffmpeg.COM", b"MZ");
    dir.executable("ffmpeg.EXE", b"MZ");
    let path = path_list([dir.path.clone()]);
    assert_eq!(which_in("ffmpeg", search(&path)), Some(com));
}

#[test]
#[cfg(windows)]
fn a_name_with_a_directory_component_does_not_get_the_current_directory() {
    // The curdir insertion lives in the `else` branch of `if dirname:`, so an
    // explicit path is searched alone even on Windows.
    let dir = TempDir::new("dirname-no-curdir");
    let missing = dir.path.join("nope");
    let spelled = missing.to_str().expect("utf-8 temp path");
    let path = path_list([dir.path.clone()]);
    assert_eq!(which_in(spelled, search(&path)), None);
}

// --- the `ffmpeg` probe --------------------------------------------------

#[test]
fn a_probe_of_a_path_that_does_not_exist_is_a_run_failure_not_an_absence() {
    let dir = TempDir::new("gone");
    let probe = probe_ffmpeg_at(&dir.path.join("ffmpeg"));
    assert_eq!(probe, FfmpegProbe::Failed);
}

#[test]
#[cfg(unix)]
fn a_probe_of_a_file_that_cannot_be_executed_is_a_run_failure() {
    // Execute bits set (so `which` accepts it) but not a runnable image, so the
    // spawn itself fails with ENOEXEC. Upstream resolved the name already, so
    // this must not be reported as "absent".
    let dir = TempDir::new("enoexec");
    let exe = dir.executable("ffmpeg", b"\x00\x01\x02 not a program\n");
    assert_eq!(probe_ffmpeg_at(&exe), FfmpegProbe::Failed);
}

#[test]
#[cfg(unix)]
fn a_probe_of_a_file_without_execute_permission_is_a_run_failure() {
    let dir = TempDir::new("noperm");
    let file = dir.file("ffmpeg", b"#!/bin/sh\nexit 0\n");
    assert_eq!(probe_ffmpeg_at(&file), FfmpegProbe::Failed);
}

#[test]
#[cfg(unix)]
fn a_probe_of_a_binary_that_exits_nonzero_is_a_run_failure() {
    // Upstream passes `check=True`, so a non-zero exit raises
    // `CalledProcessError`, a `SubprocessError`.
    let dir = TempDir::new("nonzero");
    let exe = dir.executable("ffmpeg", b"#!/bin/sh\nexit 3\n");
    assert_eq!(probe_ffmpeg_at(&exe), FfmpegProbe::Failed);
}

#[test]
#[cfg(unix)]
fn a_probe_of_a_working_binary_captures_its_stdout() {
    let dir = TempDir::new("works");
    let exe = dir.executable(
        "ffmpeg",
        b"#!/bin/sh\necho 'ffmpeg version 7.1 Copyright (c) 2000-2024'\n",
    );
    match probe_ffmpeg_at(&exe) {
        FfmpegProbe::Ran(stdout) => assert!(stdout.starts_with("ffmpeg version 7.1 ")),
        other => panic!("expected the banner, got {other:?}"),
    }
}

// --- resolution and probing composed ------------------------------------

fn ffmpeg_value(probe: FfmpegProbe) -> String {
    let env = Environment {
        rerobot_version: "0.1.0".to_string(),
        upstream_version: "0.6.1".to_string(),
        platform: "test".to_string(),
        ffmpeg: probe,
    };
    sys_info(&env)
        .into_iter()
        .find(|(k, _)| k == "FFmpeg version")
        .expect("FFmpeg version key")
        .1
}

#[test]
fn an_unresolvable_ffmpeg_reports_not_available() {
    let dir = TempDir::new("resolve-none");
    let path = path_list([dir.path.clone()]);
    let probe = detect_ffmpeg_in(search(&path));
    assert_eq!(probe, FfmpegProbe::NotFound);
    assert_eq!(ffmpeg_value(probe), "N/A");
}

#[test]
#[cfg(unix)]
fn an_ffmpeg_that_resolves_but_cannot_run_reports_installed_not_not_available() {
    // The exact distinction finding 4 is about: resolution succeeded, execution
    // did not, so this is the `SubprocessError` branch, not the `which` branch.
    let dir = TempDir::new("resolve-broken");
    dir.executable("ffmpeg", b"\x00\x01\x02 not a program\n");
    let path = path_list([dir.path.clone()]);
    let probe = detect_ffmpeg_in(search(&path));
    assert_eq!(probe, FfmpegProbe::Failed);
    assert_eq!(ffmpeg_value(probe), "Installed (version parsing failed)");
}

#[test]
#[cfg(unix)]
fn a_non_executable_ffmpeg_on_the_path_is_absent_because_which_skips_it() {
    // `shutil.which` never returns it, so upstream reports `N/A` and never
    // reaches `subprocess.run`.
    let dir = TempDir::new("resolve-noexec");
    dir.file("ffmpeg", b"not executable\n");
    let path = path_list([dir.path.clone()]);
    assert_eq!(detect_ffmpeg_in(search(&path)), FfmpegProbe::NotFound);
}

#[test]
#[cfg(unix)]
fn a_resolvable_working_ffmpeg_reports_its_version() {
    let dir = TempDir::new("resolve-good");
    dir.executable(
        "ffmpeg",
        b"#!/bin/sh\necho 'ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023'\n",
    );
    let path = path_list([dir.path.clone()]);
    let probe = detect_ffmpeg_in(search(&path));
    assert_eq!(ffmpeg_value(probe), "6.1.1-3ubuntu5");
}
