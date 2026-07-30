# Contributing to Rerobot

Rerobot is a behaviour-compatible Rust port of [Hugging Face LeRobot][upstream].
The value of the project is entirely in *demonstrated* parity, so the rules below
are about evidence, not style.

[upstream]: https://github.com/huggingface/lerobot

## The one rule

**Never claim compatibility you have not demonstrated.**

Concretely:

* `Status::Implemented` in `rerobot-compat` is reserved for a unit whose whole
  observable behaviour is covered by tests in this workspace. Nothing carries it
  today. If your slice covers one module out of nineteen, the family is
  `partial`, and the note must say which module.
* A command that cannot do its job must fail. Silent success, a no-op, or a
  plausible-looking fake output is worse than a non-zero exit.
* Do not fake model inference, dataset contents, or hardware. If a slice needs
  any of those, it is not a pure-Rust slice yet.

`docs/compatibility.md` is hand-written, and its two status tables are checked
against `rerobot-compat` row for row — name, status, upstream target or module
count, and note — by `crates/rerobot-compat/tests/docs_consistency.rs`. If you
change a status, the doc and the inventory must move together or the build fails.
Prose outside those tables is not machine-checked, so do not write a claim there
that no test backs; there is a test asserting the file does not claim to be
generated, because nothing generates it.

## Adding a slice

1. **Read the upstream source and its tests.** Port observable behaviour, not
   names. Pin the upstream commit you read in your PR description. The current
   reference point is `lerobot` 0.6.1 at
   `f37be3edbee60f3a09a5183788b91eb19f0c07d1` in
   `crates/rerobot-compat/src/lib.rs`.

2. **Probe anything you are unsure about.** Upstream is Python; a lot of
   behaviour comes from the language rather than from LeRobot. Run the real
   thing (or the equivalent pure-Python snippet) instead of guessing. Several
   ported quirks in this repository — `deque(maxlen=0)` swallowing appends,
   `len(str)` counting code points, `str.split(" ")` keeping empty fields —
   were found that way, not read off the source.

3. **Write the tests first.** Define the API as signatures with
   `unimplemented!()` bodies so the tests compile and fail *at the function you
   intend to write*, rather than failing to compile. Cover, at minimum:
   malformed input, numerical boundaries, ordering, scalar widths where they are
   observable, and serialization round-trips.

4. **Record RED in `docs/red-green.md`.** Test names, the expected failure
   reason, and the command that turns them green. Summarise; do not paste raw
   transcripts.

5. **Implement, then update the inventory and the docs.**

6. **Run the gates below.** All of them.

## Reproducing quirks

If upstream does something surprising, reproduce it and document it in the
"Upstream quirks reproduced on purpose" section of `docs/compatibility.md`. Do
not fix it silently: a caller migrating from Python may depend on it, and a
divergence discovered in production is far more expensive than a documented one.

If a divergence is genuinely unavoidable — because Rust has no exceptions, no
dynamic typing, or no tensor runtime — add a row to the "Deliberate divergences"
table explaining *why the target language forces it*. "It was easier" is not a
reason.

## Gates

Run all of these before opening a pull request. CI runs the same set.

```shell
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo test --workspace --doc
cargo build --workspace --release
python3 tools/verify_packages.py

# The declared MSRV, on the toolchain `Cargo.toml` names. CI reads the version
# out of the manifest so the two cannot drift; do it by hand here.
cargo +1.85.0 build --workspace --all-features --locked
cargo +1.85.0 test --workspace --all-targets --all-features --locked
```

`rust-version` in the workspace `Cargo.toml` is a promise `cargo install`
enforces for users. Raise it only when a dependency or a language feature forces
the raise, and say which in the commit message — the current `1.85` comes from
`indexmap` 2.14.0 and `hashbrown` 0.17.1 in the lockfile, not from this source.

The package verifier runs `cargo package --workspace --locked --no-verify`,
extracts the exact `.crate` archives, patches their registry dependencies only
inside a temporary directory to the extracted sibling archives, and runs every
packaged test and doctest. It also asserts that every archive carries `LICENSE`
and `NOTICE`.

Cargo's built-in verifier alone cannot validate this workspace, and the reason is
worth stating precisely: **none of the four crates has been published**. A
normalized manifest resolves each path dependency from crates.io, so verifying
`rerobot-train` or `rerobot-cli` fails because there is no `rerobot-core 0.1.0`
on the registry to resolve to. That is a chicken-and-egg problem inherent to
releasing a set of interdependent crates for the first time, not a fault in the
archives — which is exactly what the verifier exists to demonstrate.

`cargo deny check` is also run in CI if you have `cargo-deny` installed locally;
see `deny.toml`.

## Conventions

* Stable Rust. No nightly features. Core and compatibility crates forbid
  `unsafe`; the CLI denies it globally and permits only the documented,
  platform-gated executable-lookup FFI blocks in `which.rs`.
* Dependencies are deliberately few. Adding one needs a justification in the PR
  description: what it replaces, and why hand-rolling is worse.
* Crate boundaries follow concerns, not upstream file layout. Do not add a crate
  per Python module.
* Executable names stay byte-identical to upstream's `[project.scripts]`.
* Public items are documented; `missing_docs` is a warning that CI escalates.
* Comments explain *why*, especially when the "why" is "because upstream does
  this". A comment that restates the code will be asked about in review.

## Test naming

Test names are sentences about behaviour, not about the function under test:
`frame_count_eviction_does_not_decrement_the_byte_accounting`, not `test_evict`.
When a test pins an upstream quirk, say so in the test body with a one-line
comment naming the Python construct responsible.
