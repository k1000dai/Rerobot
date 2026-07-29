# RED → GREEN log

Every test in this workspace was written before the code it exercises. This file
is the durable record of that: what failed, why it was *expected* to fail, and
the command that turned it green. It is a summary rather than a transcript so
that it stays reviewable in a diff.

Failure reasons fall into two kinds:

* **`not implemented`** — the API was written first as a signature with an
  `unimplemented!()` body, so the test compiled and failed at runtime pointing
  at the exact unimplemented function. This is the default and is preferred over
  a bare compile error, because it proves the test exercises the intended API.
* **assertion / IO failure** — the test failed on a real assertion against
  absent behaviour or an absent file.

## Cycle 1 — `rerobot-core`: the compatibility slice

**RED** — `cargo test -p rerobot-core --all-targets --no-fail-fast`

```
tests/action_interpolator.rs   29 tests   29 failed
tests/rename_processor.rs      23 tests   22 failed  (1 passed: REGISTRY_NAME const)
tests/ring_buffer.rs           21 tests   21 failed
tests/sysinfo.rs               13 tests   13 failed
tests/types.rs                 18 tests   15 failed  (3 passed: serde derives only)
```

| Test file | Expected failure reason | Unimplemented symbol |
| --- | --- | --- |
| `tests/action_interpolator.rs` (all 29) | `not implemented` | `ActionInterpolator::new` at `src/action_interpolator.rs:84` |
| `tests/ring_buffer.rs` (all 21) | `not implemented` | `estimate_frame_bytes`, `RolloutRingBuffer::{new,with_defaults}` |
| `tests/rename_processor.rs` (22) | `not implemented` | `RenameObservationsProcessorStep::{new,observation,get_config,transform_features}`, `rename_stats` |
| `tests/sysinfo.rs` (all 13) | `not implemented` | `parse_ffmpeg_version`, `format_dict_for_markdown` |
| `tests/types.rs` (15) | `not implemented` | `as_str`/`all`/`Display`/`FromStr` for the five str-enums and the two-field `PolicyFeature` wire shape |

Representative RED output:

```
---- second_action_produces_two_steps_at_2x stdout ----
thread 'second_action_produces_two_steps_at_2x' panicked at
crates/rerobot-core/src/action_interpolator.rs:84:9:
not implemented
```

The four tests that passed at RED did so only because they assert on a `const`
or on a `serde` derive, neither of which needed hand-written code. They are kept
because they still pin observable wire format.

**GREEN** — `cargo test -p rerobot-core --all-targets`

```
104 passed; 0 failed
```

## Cycle 2 — `rerobot-compat`: the upstream inventory

**RED** — `cargo test -p rerobot-compat --all-targets --no-fail-fast`

```
tests/inventory.rs             17 tests   10 failed  (7 passed: pinned consts)
```

| Test | Expected failure reason |
| --- | --- |
| `all_eighteen_entry_points_are_present_in_upstream_order` | assertion: `ENTRY_POINTS` was an empty slice, so the comparison against the 18 rows transcribed verbatim from upstream `pyproject.toml` failed |
| `module_families_cover_every_upstream_package` | assertion: `MODULE_FAMILIES` was empty |
| `status_slugs_are_stable`, `only_implemented_and_partial_are_supported` | `not implemented` — `Status::{as_str,is_unsupported}` |
| `lookup_finds_entry_points_and_rejects_unknown_names`, `module_family_lookup_rejects_unknown_names`, `module_counts_are_recorded`, `hardware_families_are_marked_hardware_gated` | `not implemented` — `entry_point`, `module_family` |
| `exactly_one_entry_point_is_runnable_in_this_milestone`, `partially_ported_families_are_exactly_the_ones_with_tests` | assertion: empty inventory produced empty lists |

The seven that passed at RED assert on the pinned upstream coordinates
(`UPSTREAM_VERSION`, `UPSTREAM_COMMIT`, …) and on vacuous-over-empty properties
such as "no entry point claims full implementation". They were kept because they
became load-bearing the moment the tables were filled in.

**GREEN** — `cargo test -p rerobot-compat --all-targets`

```
17 passed; 0 failed
```

## Cycle 3 — `rerobot-cli`: the 18 executables

**RED** — `cargo test -p rerobot-cli --all-targets --no-fail-fast`

```
tests/cli.rs                   17 tests   12 failed  (5 passed: built-binary existence)
tests/info.rs                  11 tests   11 failed
```

| Test | Expected failure reason |
| --- | --- |
| `tests/info.rs` (all 11) | `not implemented` — `info::{sys_info,report,Environment::detect}` |
| `dispatch_*`, `unsupported_message_*`, `hardware_gated_commands_say_so` | `not implemented` — `dispatch`, `help_text`, `version_line`, `unsupported_message` |
| `help_works_for_every_entry_point_and_states_its_status`, `short_help_flag_is_accepted`, `version_works_for_every_entry_point` | subprocess exited 101 (the binary panicked inside `unimplemented!()`) instead of 0 |
| `unported_commands_exit_nonzero_with_a_stable_error`, `unported_commands_stay_unsupported_even_with_arguments` | subprocess exited 101 (panic) instead of the contracted 2 |
| `lerobot_info_runs_end_to_end`, `lerobot_info_rejects_unknown_flags` | subprocess exited 101 instead of 0 / 64 |
| `seventeen_of_eighteen_commands_are_unsupported` | assertion: `ENTRY_POINTS` still empty at the time this cycle started |

Note the distinction the RED run made visible: a panicking stub exits `101`,
which is *not* the contracted unsupported status `2`. That is exactly the
"silently succeeds or fails in an unstable way" failure mode the milestone
forbids, and the tests pin against it.

Representative RED output:

```
---- unported_commands_stay_unsupported_even_with_arguments stdout ----
assertion `left == right` failed
  left: Some(101)
 right: Some(2)
```

**GREEN** — `cargo test -p rerobot-cli --all-targets`

```
28 passed; 0 failed
```

## Cycle 4 — documentation cannot drift from the inventory

**RED** — `cargo test -p rerobot-compat --test docs_consistency --no-fail-fast`

```
tests/docs_consistency.rs       5 tests    5 failed
```

All five failed with:

```
cannot read .../crates/rerobot-compat/../../docs/compatibility.md:
No such file or directory (os error 2)
```

`docs/compatibility.md` was then written to satisfy them: every entry point row
must name its upstream target and status, every module family row must name its
status, the pinned upstream version and commit must appear, every status label
must be explained, and no row may claim `implemented`.

**GREEN** — `cargo test -p rerobot-compat --test docs_consistency`

```
5 passed; 0 failed
```

## Cycle 5 — `ffmpeg` probe fidelity

Found while re-reading the implementation against upstream: `get_ffmpeg_version`
reaches `"N/A"` **only** through its `shutil.which` check. A binary that exists
but exits non-zero raises `CalledProcessError` under `check=True`, which is a
`subprocess.SubprocessError`, and is caught into
`"Installed (version parsing failed)"`. The first implementation collapsed both
into `N/A`, so it under-reported a broken `ffmpeg` install as a missing one.

**RED** — `cargo test -p rerobot-cli --test info --no-fail-fast`

```
error[E0432]: unresolved import `rerobot_cli::info::FfmpegProbe`
error[E0560]: struct `Environment` has no field named `ffmpeg`
```

This cycle is the exception to the `unimplemented!()` convention: the fix is a
*type* change (`Option<String>` -> a three-state `FfmpegProbe`), so the RED
signal is necessarily a compile error against the type that does not exist yet.

| New test | Pins |
| --- | --- |
| `ffmpeg_absent_from_path_reports_not_available` | `FfmpegProbe::NotFound` -> `N/A` |
| `ffmpeg_that_cannot_be_run_reports_the_parse_failed_sentinel_not_not_available` | `FfmpegProbe::Failed` -> `Installed (version parsing failed)` |
| `ffmpeg_that_runs_but_prints_nothing_reports_the_parse_failed_sentinel` | empty stdout takes the `IndexError` branch, not the `which` branch |

**GREEN** — `cargo test -p rerobot-cli --test info`

```
13 passed; 0 failed
```

## Cycle 6 — independent review findings

An independent review found the port's arithmetic, `PATH` handling, report shape,
and documentation claims wrong in specific, testable ways. Each was turned into
failing tests before anything was changed. Two things made a real behavioural
oracle available for the first time in this cycle:

* `torch` 2.13.0 in a throwaway venv, running the *pinned upstream*
  `ActionInterpolator` directly, for broadcasting and post-error state;
* CPython, for `int()` / `collections.deque(maxlen=…)` boundaries and unbounded
  integer accounting.

Every expected value quoted in the new tests came out of one of those two, not
out of reasoning about what they probably do.

**RED** — per target, all before any implementation change:

```
cargo test -p rerobot-core --test action_interpolator
  error[E0599] no variant named `NotBroadcastable` found for enum `InterpolatorError`   (x4)
  error[E0277] can't compare `usize` with `i64`                                          (x2)
  error[E0308] mismatched types                                                          (x2)
  -> could not compile (test "action_interpolator") due to 8 previous errors

cargo test -p rerobot-core --test ring_buffer
  error[E0599] no variant ... named `MaxLenNotRepresentable` found for `RingBufferError` (x3)
  error[E0599] no variant ... named `NanMaxLen` / `InfiniteMaxLen`                        (x2)
  error[E0277] can't compare `i64` with `u128`                                            (x3)
  error[E0277] can't compare `i64` with `i128`                                            (x1)
  error[E0308] mismatched types                                                           (x6)
  -> could not compile (test "ring_buffer") due to 15 previous errors

cargo test -p rerobot-cli --test which
  error[E0432] unresolved import `rerobot_cli::which`
  error[E0432] unresolved imports `rerobot_cli::info::detect_ffmpeg_in`,
                                  `rerobot_cli::info::probe_ffmpeg_at`

cargo test -p rerobot-cli --test cli
  error[E0432] unresolved imports `rerobot_cli::COMPATIBILITY_URL`, `rerobot_cli::REPOSITORY`

cargo test -p rerobot-cli --test info --no-fail-fast
  test result: FAILED. 12 passed; 6 failed
    the_report_has_exactly_the_upstream_keys_in_the_upstream_order
    the_report_adds_no_keys_of_its_own
    using_gpu_in_script_is_upstreams_fill_in_placeholder
    the_scripts_key_is_a_python_style_list_of_executable_names
    the_scripts_key_carries_no_compatibility_status
    the_lerobot_version_key_names_the_upstream_target_and_the_port_version

cargo test -p rerobot-compat --test docs_consistency --no-fail-fast
  test result: FAILED. 8 passed; 2 failed
    the_doc_does_not_claim_to_be_generated
    the_doc_says_exactly_which_parts_are_checked
```

Three of these are the type-change exception to the `unimplemented!()`
convention: `InterpolatorError::NotBroadcastable`, the four-variant
`RingBufferError`, the `u128`/`i128` accounting widths, and the `which` module
did not exist, so the only honest RED signal is a compile error against the API
the test demands.

| Finding | New tests | What was actually wrong |
| --- | --- | --- |
| Interpolator shapes | `a_length_one_previous_broadcasts_against_a_length_n_action`, `a_length_n_previous_broadcasts_against_a_length_one_action`, `broadcasting_updates_the_previous_action_to_the_new_shape`, the two empty-operand cases, `a_longer_action_against_an_empty_previous_is_not_broadcastable`, `an_empty_action_against_a_longer_previous_is_not_broadcastable`, `broadcast_error_message_matches_the_torch_runtime_error`, `three_step_broadcast_values_match_torch_bit_for_bit` | Unequal lengths were rejected outright. Upstream broadcasts length-1 against length-N *in both directions*, and `0` broadcasts against `1`. Verified against torch 2.13.0, which also supplied the exact `RuntimeError` wording. |
| Interpolator post-error state | `a_failed_add_clears_the_buffer_before_reporting_the_error`, `a_failed_add_leaves_the_previous_action_untouched`, `a_failed_add_does_not_make_a_previously_illegal_shape_legal` | Upstream assigns `self._buffer = []` *before* the arithmetic that raises, so the unconsumed tail is gone once the error surfaces, while `_prev` and `_idx` are untouched. Oracle: `buffer = [] idx = 1 prev = [0.0, 0.0, 0.0]` after the `RuntimeError`. |
| f32 scalar order | `f32_weight_is_narrowed_before_multiplication` | The old test used operands on which narrow-first and narrow-last agree, so it proved nothing. The new operands were found by search *because* the two orders disagree (`3.3924236` vs `3.3924246`), and torch produces the narrow-first value. |
| Multiplier width | `a_multiplier_at_two_to_the_sixty_three_is_stored_exactly`, `a_multiplier_far_beyond_every_machine_integer_is_stored_exactly`, `control_interval_is_exact_for_a_multiplier_past_i64`, `control_interval_fails_exactly_where_pythons_int_to_float_conversion_does`, and the two allocation-boundary tests | The first port narrowed through `usize`, and the first review fix still narrowed to `i64`. The final implementation stores a `BigInt`, preserving Python's signed arbitrary-precision constructor/getter domain; float conversion and allocation fail explicitly at the same operation boundaries instead of wrapping. |
| Ring buffer capacity errors | `nan_frame_capacity_is_rejected_like_pythons_int`, `infinite_frame_capacity_is_rejected_in_both_directions`, `frame_capacity_beyond_py_ssize_t_is_an_overflow_not_a_wrapped_length`, `frame_capacity_below_py_ssize_t_min_is_an_overflow_not_a_value_error`, `the_largest_representable_frame_capacity_is_accepted` | One `NotFinite` variant collapsed Python's `ValueError` (NaN) and `OverflowError` (infinity), and `frames as usize` truncated silently past `Py_ssize_t`. CPython's order — `int()`, then `PyLong_AsSsize_t`, then the non-negative check — is now reproduced, so `-1e30` is an `OverflowError` while `-30` is a `ValueError`. |
| Byte accounting overflow | `a_tensor_larger_than_i64_bytes_is_costed_exactly`, `the_maximal_tensor_estimate_is_exact`, `nbytes_beyond_i64_are_costed_exactly`, `the_byte_cap_is_exact_for_the_largest_megabyte_count`, `a_negative_byte_cap_evicts_before_every_append`, `a_frame_far_larger_than_the_cap_is_accounted_without_wrapping`, `repeated_appends_under_a_zero_frame_cap_accrue_exactly` | `(*numel as i64) * (*element_size as i64)` and `total += …` overflowed i64 (panic in debug, wrap in release), and `max_memory_mb.saturating_mul(1024 * 1024)` clamped a cap Python computes exactly. This cycle moved to `u128` totals and an `i128` cap, which left one residual difference: a running total saturating at `u128::MAX`. **Superseded** — the cycle below removed the width entirely, so there is no saturation left to characterise and the test that pinned it (`accrual_past_the_representable_domain_saturates_rather_than_wrapping`) was replaced by `accrual_across_frames_past_the_128_bit_boundary_is_exact`. The `i128` cap stayed, because it is exact for every cap Python can compute. |
| `ffmpeg` detection | all 18 in `tests/which.rs`, notably `a_non_executable_candidate_is_skipped`, `a_probe_of_a_file_that_cannot_be_executed_is_a_run_failure`, `a_probe_of_a_file_without_execute_permission_is_a_run_failure`, `an_ffmpeg_that_resolves_but_cannot_run_reports_installed_not_not_available`, `a_non_executable_ffmpeg_on_the_path_is_absent_because_which_skips_it` | A spawn error stood in for `shutil.which` returning `None`, so a resolvable-but-unrunnable `ffmpeg` was reported as absent. `shutil.which` is now ported (`X_OK`, not-a-directory, first match wins, directory-component short-circuit, `PATH=''`, Windows `PATHEXT`), and only its `None` reaches `N/A`. |
| `lerobot-info` shape | the six failing `tests/info.rs` tests above | The report had invented a `Rerobot version` key, dropped `Using GPU in script?`, and replaced the `lerobot scripts` value with `name=status` metadata. It is now upstream's 15 keys in upstream's order, with a Python-style list of names; status moved to `--help`. |
| Repository URL | `the_unsupported_message_points_at_a_resolvable_repository_url`, `help_points_at_a_resolvable_repository_url_not_only_a_local_path`, `the_repository_url_matches_the_published_package_metadata`, `every_executable_prints_a_repository_url_in_its_help` | Messages pointed at `docs/compatibility.md`, which resolves to nothing for anyone who installed with `cargo install`. |
| Docs self-claim | `the_doc_does_not_claim_to_be_generated`, `the_doc_says_exactly_which_parts_are_checked`, plus the strengthened `the_entry_point_table_is_the_inventory_row_for_row`, `the_module_family_table_is_the_inventory_row_for_row`, `the_doc_states_the_entry_point_count_from_the_inventory`, `the_doc_states_the_unsupported_entry_point_count_from_the_inventory` | The doc said it was "generated from" the inventory. Nothing generates it. The claim is now narrowed to exactly what the tests check — both tables row for row, including notes and module counts — and the tests were widened to make that narrower claim true. |

Representative RED output:

```
---- the_report_has_exactly_the_upstream_keys_in_the_upstream_order stdout ----
assertion `left == right` failed
  left: ["LeRobot version", "Platform", "Rerobot version", ...]
 right: ["LeRobot version", "Platform", "Python version", ...]
```

**GREEN** — `cargo test --workspace --all-targets --all-features`

```
214 passed; 0 failed
```

## Post-review RED → GREEN cycles

The independent reviews added three more vertical cycles after the 214-test
milestone above. The original total is retained there as historical evidence;
the final totals below include these review-driven tests.

| RED test surface | Expected RED reason | GREEN behavior |
| --- | --- | --- |
| `byte_count.rs`; adversarial `ring_buffer.rs` cases | fixed-width accounting saturated above `u128::MAX` | exact arbitrary-precision byte estimates and running totals, including multiple maximal values in one frame |
| `which.rs` unset-PATH/effective-access cases; Windows-gated cases | the first port did not follow CPython 3.12 fallback and platform lookup rules | `CS_PATH`/`os.defpath` fallback, kernel effective-access checks, and Windows `NeedCurrentDirectoryForExePath`/`PATHEXT` behavior |
| `types.rs` overflowing-shape case | non-upstream `PolicyFeature::numel` panicked in debug and wrapped in release | the convenience method was removed; the public type now contains exactly upstream's `type` and `shape` fields |

Focused RED runs failed for those intended missing behaviors before the
production changes. The final stable and MSRV runs below are the corresponding
whole-workspace GREEN evidence.

## Cycle 7 — the last two fixed widths standing in for a Python `int`

A further review found that removing `PolicyFeature::numel` had fixed the
*symptom* and left the *domain* wrong, and that the interpolator's multiplier
had the same defect one width up. Both fields model a Python `int`, which is
signed and unbounded; `Vec<usize>` is neither, and `i64` is bounded. Neither
gap is reachable through arithmetic the port performs, so only a test that
states the domain could fail.

**RED** — both before any production change:

```
cargo test -p rerobot-core --test types
  error[E0432] unresolved import `rerobot_core::types::BigInt`
  error[E0277] the trait bound `usize: Neg` is not satisfied
  -> could not compile (test "types") due to 2 previous errors

cargo test -p rerobot-core --test action_interpolator
  error[E0432] unresolved import `rerobot_core::BigInt`
  error[E0614] type `i64` cannot be dereferenced                                (x4)
  error[E0599] no method named `unwrap`/`unwrap_err` found for type `f64`       (x7)
  error[E0599] no variant named `MultiplierNotFloatRepresentable`               (x1)
  error[E0599] no variant named `BufferNotAllocatable`                          (x2)
  -> could not compile (test "action_interpolator") due to 15 previous errors
```

These are the type-change exception to the `unimplemented!()` convention, and
the first RED is the sharpest statement of the bug available: `shape: Vec<usize>`
cannot even *hold* the literal `-1`, so `PolicyFeature::new(FeatureType::State,
[-1, 0, 1])` does not compile. A test asserting the upstream domain could not be
written against the old type at all.

| Finding | New tests | What was actually wrong |
| --- | --- | --- |
| `PolicyFeature.shape` domain | `policy_feature_round_trips_a_negative_dimension_exactly`, `policy_feature_round_trips_dimensions_far_above_usize_max_exactly`, `policy_feature_shape_survives_a_thousand_digit_dimension`, `policy_feature_rejects_shape_entries_that_are_not_integers` | `Vec<usize>` is unsigned and machine-width, so it rejected `-1` — upstream's ordinary dynamic axis — and its JSON wire form depended on the target's pointer width. Now `Vec<BigInt>`, serialised as the bare decimal integer `json.dumps` writes. The wire values in the tests are literals rather than `usize::MAX`-derived, so the expected JSON is the same text on every target. |
| Shape wire exactness | the `2**128 + 1` and `10**999` cases above | Round-tripping an integer past `u64` through `serde_json` silently turns it into an `f64` unless the raw token is read. Both directions now go through `serde_json::value::RawValue`, so the decimal text is never rounded on the way in or out. `serde_json`'s `raw_value` feature is enabled for this; `arbitrary_precision` would also have worked but rewrites `Number` for every crate in the dependency graph. |
| Multiplier domain | `a_multiplier_at_two_to_the_sixty_three_is_stored_exactly`, `a_multiplier_far_beyond_every_machine_integer_is_stored_exactly`, `a_negative_multiplier_far_below_i64_min_is_rejected_carrying_its_exact_value`, `control_interval_is_exact_for_a_multiplier_past_i64` | `i64` truncates at `2**63`, a value `int` holds exactly, and the doc claimed "full width" for a width that is not full. Now `BigInt`; storage, the getter, `enabled` and the control interval are exact at every magnitude, and the "full width" claim is gone. |
| Multiplier operations that cannot be exact | `control_interval_fails_exactly_where_pythons_int_to_float_conversion_does`, `add_reports_a_multiplier_whose_buffer_cannot_be_allocated`, `add_reports_a_multiplier_that_no_machine_word_can_even_count_to`, `a_non_broadcastable_action_is_reported_before_the_allocation_is_attempted`, `a_huge_multiplier_still_passes_the_first_action_through` | Two operations need a finite Rust value and previously got one by narrowing. `get_control_interval` now returns `Result` and fails at exactly CPython's `int`-to-float boundary (`2**1023` converts, `2**1024` does not) instead of dividing by an infinity and reporting `0.0`; `add` returns `BufferNotAllocatable` rather than truncating the step count. The broadcast check still runs first, because upstream's `RuntimeError` comes from the first loop iteration. |

### `byte_count` — a refactor, not a RED cycle

`ByteCount` is now a newtype over `num_bigint::BigUint` instead of hand-written
base-2^64 limb arithmetic. Its behaviour is unchanged and it is stated here
plainly: **there was no RED for this one.** The four differential tests
(`product_agrees_with_biguint_over_a_deterministic_sweep`,
`a_running_total_agrees_with_biguint_over_a_deterministic_sweep`,
`comparison_and_u128_narrowing_agree_with_biguint`,
`saturating_sub_agrees_with_biguint_where_biguint_can_subtract_at_all`) were
written and run against the *old* implementation first, where they passed. That
is the point: they are the oracle that made replacing ~150 lines of unverified
carry and borrow code safe, and they were re-run unchanged against the new one.

They did fail twice on the way in, on their own self-checks rather than on the
type under test — the first sweeps never crossed the `u128` boundary and never
reached the `saturating_sub` clamp, and said so:

```
---- comparison_and_u128_narrowing_agree_with_biguint stdout ----
the sweep never produced a value past u128, so it proved nothing
---- saturating_sub_agrees_with_biguint_where_biguint_can_subtract_at_all stdout ----
the sweep never exercised the clamp
```

The generators were fixed (compare each product against a running total, and
subtract in both directions) until both branches were reached. The assertions
that caught this are kept in the tests.

## `NewLineTaskProcessorStep` development log

The processor was added one observable rule at a time with:

```text
cargo test -p rerobot-core --test newline_task_processor
```

The implementation session recorded the first test reaching the new API while
the implementation was deliberately absent:

```text
test complementary_data_without_a_task_key_is_returned_unchanged ... FAILED
panicked at crates/rerobot-core/src/processor/newline_task.rs:18:9:
not implemented
test result: FAILED. 0 passed; 1 failed
```

The same session recorded subsequent RED cycles against three distinct
incorrect partial implementations:

```text
left: String("pick up the cube")
right: String("pick up the cube\n")

left: String("pick up the cube\n\n")
right: String("pick up the cube\n")

left: Array [String("a\n"), Number(1)]
right: Array [String("a"), Number(1)]
```

The last failure motivated the upstream all-or-nothing list rule rather than a
per-element best effort. Separate cycles also recorded failures at the
intentionally unimplemented `get_config` and `transform_features` methods.
These excerpts are a development log, not independently reproducible proof of
history; the retained tests are the auditable current evidence. The suite has
24 cases covering strings, LF/CRLF, bare CR, Unicode, empty values, mixed and
nested lists, config identity, stateless lifecycle, feature value identity,
input independence, and insertion order.

## Final GREEN totals

`cargo test --workspace --all-targets --all-features`

| Target | Tests |
| --- | ---: |
| `rerobot-cli` library unit tests | 1 |
| `rerobot-core` `tests/action_interpolator.rs` | 50 |
| `rerobot-core` `tests/byte_count.rs` | 15 |
| `rerobot-core` `tests/ring_buffer.rs` | 37 |
| `rerobot-core` `tests/rename_processor.rs` | 23 |
| `rerobot-core` `tests/newline_task_processor.rs` | 24 |
| `rerobot-core` `tests/types.rs` | 20 |
| `rerobot-core` `tests/sysinfo.rs` | 13 |
| `rerobot-compat` `tests/inventory.rs` | 17 |
| `rerobot-compat` `tests/docs_consistency.rs` | 10 |
| `rerobot-cli` `tests/cli.rs` | 21 |
| `rerobot-cli` `tests/info.rs` | 18 |
| `rerobot-cli` `tests/which.rs` | 21 |
| **Total** | **270** |

The 18 `lerobot-*` binary targets contribute no unit tests; their behaviour is
covered by `tests/cli.rs`, which runs the built executables as subprocesses.
`tests/which.rs` reports 21 on a Unix runner: five of its cases are
`cfg(windows)` and are compiled and run only by the `gates` job on
`windows-latest`.

`cargo test --workspace --doc`

| Crate | Doctests |
| --- | ---: |
| `rerobot-core` (crate README + item docs) | 24 |
| `rerobot-compat` (crate README) | 2 |
| `rerobot-cli` (crate README + `which`) | 3 |
| **Total** | **29** |

182 of the 270 are the compatibility slice itself (`rerobot-core`), which is
where the milestone's parity claim lives.

## Whole-workspace gate

```
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo test --workspace --doc
cargo build --workspace --release
cargo package --workspace --allow-dirty
cargo +1.85.0 build --workspace --all-features --locked      # declared MSRV
cargo +1.85.0 test --workspace --all-targets --all-features --locked
```

The MSRV commands are the same ones the `msrv` CI job runs, except that CI reads
`1.85` out of `Cargo.toml` instead of hardcoding it, so the manifest and the
tested toolchain cannot drift. 1.85 is the floor the *locked* tree imposes:
`indexmap` 2.14.0 declares `rust-version = "1.85"` and `hashbrown` 0.17.1
declares `"1.85.0"`. The workspace's own code needs less than that, so the number
is a property of the dependency set, not of this source.
