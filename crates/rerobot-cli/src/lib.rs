#![doc = include_str!("../README.md")]
// `deny` rather than `forbid`: `which` needs three FFI calls that `std` does
// not wrap — `access(2)`, `confstr(3)` and `NeedCurrentDirectoryForExePath` —
// and each is opted in individually with `#[allow(unsafe_code)]` and a safety
// comment. Everything else in the crate is still unsafe-free.
#![deny(unsafe_code)]

pub mod info;
pub mod rollout;
pub mod so101;
pub mod train;
pub mod which;

use rerobot_compat::{entry_point, EntryPoint, UPSTREAM_PACKAGE, UPSTREAM_VERSION};

/// Exit status used by every not-yet-ported `lerobot-*` command.
pub const EXIT_UNSUPPORTED: i32 = 2;

/// Exit status used when an executable is asked for something it cannot parse.
pub const EXIT_USAGE: i32 = 64;

/// Version string of this workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Rerobot's public repository.
///
/// Every message a user can hit has to name something they can actually open.
/// Someone who ran `cargo install rerobot-cli` has no checkout, so a bare
/// `docs/compatibility.md` path resolves to nothing on their machine.
pub const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");

/// Stable URL of the compatibility boundary document.
pub const COMPATIBILITY_URL: &str = concat!(
    env!("CARGO_PKG_REPOSITORY"),
    "/blob/main/docs/compatibility.md"
);

/// What a run of [`dispatch`] decided to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    /// Text for stdout (no trailing newline).
    pub stdout: String,
    /// Text for stderr (no trailing newline).
    pub stderr: String,
    /// Process exit status.
    pub code: i32,
}

impl Outcome {
    fn ok(stdout: String) -> Self {
        Self {
            stdout,
            stderr: String::new(),
            code: 0,
        }
    }

    fn err(stderr: String, code: i32) -> Self {
        Self {
            stdout: String::new(),
            stderr,
            code,
        }
    }
}

fn unknown_command(name: &str) -> Outcome {
    Outcome::err(
        format!("{name}: not a {UPSTREAM_PACKAGE} {UPSTREAM_VERSION} console entry point"),
        EXIT_USAGE,
    )
}

/// `--help` text for `name`, including its compatibility status.
///
/// Panics if `name` is not an upstream entry point; use [`dispatch`] for the
/// non-panicking path.
pub fn help_text(name: &str) -> String {
    let e = entry_point(name).unwrap_or_else(|| panic!("{name} is not an upstream entry point"));
    let usage = if e.status.is_unsupported() {
        format!("Usage: {name} [ARGS...]    (all arguments are accepted and then rejected)")
    } else if name == "lerobot-train" {
        format!(
            "Usage: {name} [--help] [--version] --dataset.repo_id=ID --dataset.root=DIR \\\n\
                 \x20                   --output_dir=DIR --policy.type=act [OPTIONS...]"
        )
    } else if name == "lerobot-rollout" {
        format!("Usage: {name} [--help] [--version] --policy.path=DIR --dataset.root=DIR --steps=N")
    } else {
        format!("Usage: {name} [--help] [--version]")
    };
    // `lerobot-train` is the one command with an argument surface worth spelling
    // out, and spelling it out is what makes "refused, never ignored" checkable by
    // a user rather than only by a test.
    let extra = if name == "lerobot-train" {
        format!("\n\n{}", train::help_section())
    } else if name == "lerobot-rollout" {
        format!("\n\n{}", rollout::help_section())
    } else {
        String::new()
    };
    format!(
        "{name} {VERSION} (Rerobot)\n\
         \n\
         {summary}\n\
         \n\
         {usage}\n\
         \n\
         Compatibility status: {status}\n\
         Upstream: {package} {upstream_version} -- {target}\n\
         {note}\n\
         \n\
         Rerobot is a partial Rust port of Hugging Face LeRobot.\n\
         Full boundary: {compatibility_url}\n\
         Repository:    {repository}{extra}",
        summary = e.summary,
        status = e.status,
        package = UPSTREAM_PACKAGE,
        upstream_version = UPSTREAM_VERSION,
        target = e.target,
        note = e.note,
        compatibility_url = COMPATIBILITY_URL,
        repository = REPOSITORY,
    )
}

/// `--version` line for `name`.
pub fn version_line(name: &str) -> String {
    format!("{name} {VERSION} (Rerobot; targets {UPSTREAM_PACKAGE} {UPSTREAM_VERSION})")
}

/// The stable error emitted by a not-yet-ported command.
///
/// Always a single line so it stays greppable in CI logs.
pub fn unsupported_message(name: &str) -> String {
    let e: &EntryPoint =
        entry_point(name).unwrap_or_else(|| panic!("{name} is not an upstream entry point"));
    format!(
        "{name}: unsupported in Rerobot {VERSION}: not implemented ({status}); \
         upstream {package} {upstream_version} provides it as {target}; \
         run `{name} --help` or see {COMPATIBILITY_URL}",
        status = e.status,
        package = UPSTREAM_PACKAGE,
        upstream_version = UPSTREAM_VERSION,
        target = e.target,
    )
}

/// Handle `--help` / `--version` for any command, or report it unsupported.
///
/// `args` excludes the executable-name element at `argv` index zero. `--help`
/// and `--version` win over every other
/// argument so that help is always reachable.
pub fn dispatch(name: &str, args: &[String]) -> Outcome {
    let Some(e) = entry_point(name) else {
        return unknown_command(name);
    };

    if args.iter().any(|a| a == "--help" || a == "-h") {
        return Outcome::ok(help_text(name));
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        return Outcome::ok(version_line(name));
    }
    if e.status.is_unsupported() {
        return Outcome::err(unsupported_message(name), EXIT_UNSUPPORTED);
    }

    match name {
        "lerobot-info" => match args.first() {
            None => Outcome::ok(info::report(&info::Environment::detect())),
            Some(unexpected) => Outcome::err(
                format!("{name}: unrecognized argument {unexpected:?}; try `{name} --help`"),
                EXIT_USAGE,
            ),
        },
        "lerobot-train" => run_train(name, args),
        "lerobot-rollout" => run_rollout(name, args),
        // Unreachable while the supported commands are implemented below, but failing
        // loudly beats silently succeeding if the inventory and dispatcher diverge.
        other => Outcome::err(
            format!("{other}: marked supported but has no implementation"),
            EXIT_UNSUPPORTED,
        ),
    }
}

/// Parse `args` and train, or report exactly why not.
///
/// A usage problem exits [`EXIT_USAGE`]; an unsupported request or a run failure
/// exits [`EXIT_UNSUPPORTED`]. Both are non-zero, so a script cannot mistake
/// either for a completed run.
fn run_train(name: &str, args: &[String]) -> Outcome {
    let config = match train::parse(args) {
        Ok(config) => config,
        Err(error) => {
            let code = match error {
                train::ArgumentError::Unsupported { .. } => EXIT_UNSUPPORTED,
                _ => EXIT_USAGE,
            };
            return Outcome::err(format!("{name}: {error}"), code);
        }
    };

    // Progress is printed as it happens rather than collected, because a real run
    // is long and a transcript returned at the end would be useless during it.
    let mut lines = Vec::new();
    let result = rerobot_train::run::train(&config, &mut |line| {
        println!("{line}");
        lines.push(line.to_owned());
    });
    match result {
        Ok(outcome) => {
            let mut stdout = String::new();
            if let Some(last) = outcome.checkpoints.last() {
                stdout = format!("Checkpoint: {}", last.display());
            }
            Outcome::ok(stdout)
        }
        Err(error) => Outcome::err(format!("{name}: {error}"), EXIT_UNSUPPORTED),
    }
}

/// Parse `args` and run the local checkpoint-backed rollout.
fn run_rollout(name: &str, args: &[String]) -> Outcome {
    let config = match rollout::parse(args) {
        Ok(config) => config,
        Err(error) => {
            let code = match error {
                rollout::ArgumentError::Unsupported { .. } => EXIT_UNSUPPORTED,
                _ => EXIT_USAGE,
            };
            return Outcome::err(format!("{name}: {error}"), code);
        }
    };

    let result = rollout::run(&config, &mut |line| println!("{line}"));
    match result {
        Ok(()) => Outcome::ok(String::new()),
        Err(error) => Outcome::err(format!("{name}: {error}"), EXIT_UNSUPPORTED),
    }
}

/// Run [`dispatch`] against the real process argv and exit accordingly.
pub fn run(name: &str) -> ! {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let outcome = dispatch(name, &args);
    emit(&outcome);
    std::process::exit(outcome.code)
}

/// Write an [`Outcome`] to the process streams.
pub fn emit(outcome: &Outcome) {
    if !outcome.stdout.is_empty() {
        println!("{}", outcome.stdout);
    }
    if !outcome.stderr.is_empty() {
        eprintln!("{}", outcome.stderr);
    }
}
