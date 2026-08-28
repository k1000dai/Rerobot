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

## DAgger event state machine development log

The retained command for each focused cycle was:

```text
cargo test -p rerobot-core --test dagger
```

The implementation session first reached each deliberately absent API before
adding its minimum behavior. Representative recorded REDs included:

```text
test phase_values_are_the_upstream_member_values ... FAILED
panicked at crates/rerobot-core/src/rollout/dagger.rs:25:9:
not implemented

test reset_returns_the_machine_to_a_fresh_session ... FAILED
panicked at crates/rerobot-core/src/rollout/dagger.rs:167:9:
not implemented

test a_later_valid_request_overwrites_the_pending_one ... FAILED
test an_invalid_request_does_not_clear_a_valid_pending_one ... FAILED
test a_pending_request_invalidated_by_a_phase_change_is_consumed_and_yields_nothing ... FAILED
```

Those later failures separated three easy-to-conflate one-slot semantics:
valid requests overwrite, invalid requests do not clear, and consume takes the
slot before revalidating against the current phase. Poison-recovery tests live
inside the module because the lock is private. As with the newline excerpts,
this is a development log rather than independently reproducible proof of
history; the 28 retained DAgger tests are the auditable current evidence.

## DatasetInfo and local `meta/info.json`

The dataset metadata slice was developed in three focused cycles. The retained
tests were written against absent behavior before each implementation:

```text
dataset_json: JsonLike/parser/writer skeleton
    44 tests failed at `not implemented`

dataset_info: constants plus DatasetInfo skeleton
    30 behavior tests failed at `unimplemented!()`; 4 constant tests passed

dataset_io: load_json/write_json/load_info/write_info skeleton
    22 tests failed before the filesystem implementation
```

An independent differential sweep then exposed a real float-wire defect: 8 of
30,623 finite doubles had a different final digit from CPython because two
shortest decimals round-tripped and Rust chose upward where CPython chose the
even last digit. Two focused tie tests were observed RED before the exact
`BigInt` half-even implementation. The repaired formatter was then compared
against CPython 3.12.13 over 747,248 doubles, including 40,586 subnormals, with
zero disagreements. The fail-closed review then added four more observed REDs:
CPython's leading-BOM diagnostic, Unicode 15.0 `str` printability in unknown
field warnings, deeply nested parser input aborting a child process, and the
recursive writer doing the same. The reader now returns an explicit depth error,
and the writer uses an iterative work stack. A mixed-invalid input additionally
pins that upstream's `fps` post-init error precedes an unrelated typed `splits`
boundary error. The three retained suites are the auditable current evidence;
the historical RED excerpts are a development log.

## ACT policy configuration

`tests/act_config.rs` was compiled before `rerobot_core::policy` existed. The
focused RED failed with `E0433: could not find policy in rerobot_core`, at the
intended public import. Nine retained tests then drove defaults, exact validation
precedence/messages, feature validation, presets, lazy delta indices, ordered
checkpoint JSON, malformed/absent/unknown fields, and signed thousand-digit
integers. A tenth retained test pins the explicit non-finite float output error.
A second focused RED changed the compatibility inventory only after
the new policy test existed: `policies_are_partial_once_act_config_is_available`
observed `Unimplemented` where `Partial` was required.

The oracle was CPython 3.12.13 running upstream commit
`f37be3edbee60f3a09a5183788b91eb19f0c07d1`. It captured all four validation
failures and their precedence, zero/negative range behavior, feature acceptance,
the AdamW preset, and the full `config.json` shape. The retained tests are the
auditable evidence. A post-GREEN differential malformed-input probe found that
Draccus accepts numeric strings and booleans for integer fields; the focused
test was changed first and observed RED before the decoder reproduced that
coercion. A later bool-domain probe likewise found lowercase string coercion;
that focused assertion failed against serde's nominal bool decoder before the
Draccus-compatible decoder was added. The RED output above is a development log.
The final annotation audit found ACT's `int = False` dilation field; a focused
compile RED required the absent `PythonIntBool` type before the dual bool/BigInt
wire representation was implemented.

## ACT checkpoint boundary (Draccus)

An independent differential audit of the ACT slice against CPython 3.12.13 and
Draccus 0.10.0 found eight places where the port's accepted or emitted domain
was not upstream's. `tests/act_checkpoint.rs` was written first, against an API
that did not exist, and the focused RED was `E0432: unresolved import
`rerobot_core::policy::draccus`` plus 37 `E0599: no associated function named
`from_checkpoint_json`` — one per intended contract. `tests/types.rs` gained two
tests that failed against the shipped decoder: `2 failed` on
`policy_feature_rejects_unknown_fields_like_decode_dataclass` and
`policy_feature_shape_follows_draccus_decode_int`.

The oracle for every expectation was upstream itself, driven through
`draccus.parse(ACTConfig, path)` and `draccus.dump(config, stream, indent=4)`
under `draccus.config_type("json")` — the two calls `from_pretrained` and
`_save_pretrained` make. What it showed, and what the GREEN now reproduces:

1. `normalization_mapping` is `dict[str, NormalizationMode]`, so `{"BOGUS":
   "MIN_MAX"}` loads and round-trips. The port had narrowed the key to
   `FeatureType` and rejected it.
2. Draccus decodes every `str`-annotated field with `str(raw_value)`, so
   `"repo_id": 5` is `'5'`, `"license": true` is `'True'` and `"tags": [null]`
   is `['None']`. The port raised a serde type error on all of them.
3. `json.dump` writes `float.__repr__`, so the default `optimizer_lr` is
   `1e-05`; serde_json writes `0.00001`. The crate already had
   `dataset::json::python_float_repr` for the dataset slice and now uses it
   here too.
4. `int()` and `float()` run `_PyUnicode_TransformDecimalAndSpaceToASCII`
   first, so `"1_000.5"` and `"１００"` parse. Rust's parsers reject both.
5. `pretrained_path` is a `pathlib.Path`: `"a//b"` re-dumps as `"a/b"` and `""`
   as `"."`. The port stored the raw string.
6. `decode_dataclass` rejects an unknown key inside a nested `PolicyFeature`,
   which the port silently dropped, and `decode_int` coerces a shape entry the
   port refused.
7. `json.load` gives a duplicate object key Python `dict` semantics; serde
   raises `duplicate field`.
8. `json.load`/`json.dump` accept and emit bare `NaN`/`Infinity`, which
   `serde_json` cannot represent at all.

(7) and (8) are properties of the JSON layer rather than of any field, so they
are fixed by routing the checkpoint path through this crate's existing CPython
`json` port instead of through `serde_json`: `ActConfig::from_checkpoint_json`
and `to_checkpoint_json`. `dataset::json` gained `dumps_pretty_ascii` and
`encode_basestring_ascii` for that, because `draccus.dump` leaves CPython's
`ensure_ascii=True` default in place where `meta/info.json` does not.

Three assertions in the first draft of `act_checkpoint.rs` were wrong and were
corrected against the oracle rather than by loosening them: `ensure_ascii`
escapes non-ASCII instead of preserving it, `__post_init__` runs during
`draccus.parse` so a decode vector must leave `chunk_size >= n_action_steps`,
and a read/write cycle over upstream's own `config.json` is *not* the identity
upstream either — `replace_final_stride_with_dilation` widens from `false` to
`0` on the first read. That last one is pinned by
`checkpoint_round_trip_reproduces_upstreams_own_dilation_widening`.

The final checkpoint review found two more CPython numeric-string boundaries.
Focused RED runs rejected adjacent Mathematical Unicode digits (`𝟘𝟙`) and
ASCII edge whitespace (`"\t7\r\n"`). GREEN maps all 680 Unicode 15.0 `Nd`
characters to their decimal values and trims exactly the ASCII whitespace the
numeric parsers skip; retained tests cover every digit plus integer and float
edge-whitespace vectors.


## Cycle 8 — the first runnable training slice

Five pure modules in `rerobot-core`, a new `rerobot-train` crate, and
`lerobot-train`'s argument surface. Written in that order, because each layer's
tests had to be able to fail for its own reasons.

### Part 1 — the pure additions to `rerobot-core`

Signatures first with `unimplemented!()` bodies, so the tests compiled and failed
pointing at the exact function.

**RED** — `cargo test -p rerobot-core --test random --test dataset_delta --test
dataset_sampler --test dataset_stats --test policy_normalize --no-fail-fast`

```
tests/dataset_delta.rs     19 tests   19 failed
tests/dataset_sampler.rs   19 tests   19 failed
tests/dataset_stats.rs     11 tests   11 failed
tests/policy_normalize.rs  15 tests   15 failed
tests/random.rs            11 tests   11 failed
```

| Test file | Expected failure reason | Unimplemented symbol |
| --- | --- | --- |
| `tests/random.rs` (all 11) | `not implemented` | `mix64`, `SplitMix64::*`, `shuffled_permutation` |
| `tests/dataset_delta.rs` (all 19) | `not implemented` | `python_round_half_even`, `action_delta_timestamps`, `get_delta_indices`, `check_delta_timestamps`, `query_window` |
| `tests/dataset_sampler.rs` (all 19) | `not implemented` | `EpisodeAwareSampler::{new,frame_index,next_epoch}`, `compute_sampler_state` |
| `tests/dataset_stats.rs` (all 11) | `not implemented` | `load_stats`, `stats_from_value` |
| `tests/policy_normalize.rs` (all 15) | `not implemented` | `Normalizer::{new,normalize,unnormalize}` |

**GREEN** — 75 passed. Two of the tests were wrong rather than the code: the
tolerance arithmetic in `the_tolerance_is_measured_in_seconds_after_dividing_by_fps`
had assumed `0.101 * 10` deviates by exactly 1e-3 when binary64 makes it
1.0000000000000009e-3, and `episodes_are_concatenated_by_their_dataset_index_ranges`
had miscounted the frames of three contiguous episodes. Both were corrected in the
test with a comment recording why.

### Part 2 — `rerobot-train`

**GREEN** — `cargo test -p rerobot-train --all-targets`

```
tests/dataset.rs     22 passed
tests/model.rs       31 passed
tests/optimizer.rs   16 passed
tests/train.rs       32 passed
tests/goldens.rs     12 passed
```

Four defects were found by these tests rather than by inspection, and all four
would have produced a training loop that ran and reported plausible numbers:

| Defect | How it presented | Fix |
| --- | --- | --- |
| `candle_nn::ops::softmax_last_dim`'s backward pass does not reach its input | Forward pass and every loss value correct; `model.encoder_1d_feature_pos_embed.weight` and `model.decoder_pos_embed.weight` received **no gradient at all**, and every attention projection trained only through its value path | `candle_nn::ops::softmax`, the composed differentiable version. Pinned by `tests/model.rs::attention_logits_receive_gradients` and by the oracle |
| `state_dict()` returned handles that alias the live parameters | `Var::set` writes through, so a caller that snapshotted the weights, stepped, and compared saw no difference — and a test asserting "the step changed something" passed while asserting nothing | `state_dict()` returns detached deep copies, and says why |
| `where_cond` on an `f32` condition | The VAE encoder's key-padding mask failed at runtime: `unsupported dtype F32 for op where-cond` | the mask stays `u8` |
| `read_last_checkpoint` called `is_dir()` on `symlink_metadata` | `checkpoints/last` resolved as a file and the read failed with `Is a directory` | the symlink is read and re-anchored against the checkpoints directory |

A fifth finding was **not** a defect, and the investigation is recorded because
the conclusion matters: three tensors have an exactly-zero gradient on the first
step of a freshly initialized ACT
(`decoder.layers.0.{norm1.weight,self_attn.in_proj_weight,self_attn.out_proj.weight}`).
That is upstream's behaviour too — the decoder's input is `torch.zeros`,
`nn.MultiheadAttention` zero-initializes both biases, so the value stream is
identically zero and the attention output is constant in those weights. They train
from the second step. `tests/train.rs` pins the set in both directions and pins
that they move on step 2, so the exemption cannot excuse a real regression.

### Part 3 — the differential oracle

`tools/goldens/make_act_goldens.py` ran once against upstream at the pinned commit
and committed three files under
`crates/rerobot-train/tests/fixtures/goldens/`: the loss scalars and provenance as
JSON, `ACTPolicy.state_dict()` as safetensors, and the inputs, outputs, eleven
gradients and post-step parameters as safetensors. `tests/goldens.rs` reads them.
`cargo test` never runs Python.

**GREEN** — `cargo test -p rerobot-train --test goldens` → 12 passed, on the
first run for 9 of them. The three that failed did so for good reasons: two
because the oracle's batch had no padded action and the tests said so
(`the oracle batch has no padded action, so the mask is untested`), which moved the
oracle from frames `[0, 1]` to `[0, 3]`; and one because a fixed absolute floor of
1e-7 cannot judge a gradient tensor whose entries span six orders of magnitude,
which moved the comparison to `atol + rtol * |expected|` with `atol` tied to each
tensor's own scale.

Verified not vacuous by injecting three defects and confirming failure:

| Injected defect | Result |
| --- | --- |
| the packed `q` projection reads the `k` block | 5 of 12 fail; first predicted action -0.429 against the oracle's 0.055 |
| `LayerNorm`'s epsilon dropped | 4 of 12 fail |
| `softmax_last_dim` restored | 3 of 12 fail: both position embeddings receive no gradient |

### Part 4 — `lerobot-train`

**GREEN** — `cargo test -p rerobot-cli --test train_cli` → 25 passed. Two rounds
of test-driven correction on the way: `--optimizer.lr` was reported as an unknown
flag rather than as an unsupported one, which moved the refusal list to
prefix matching and then moved the whole check after the accepted flags so that
`--wandb.enable=false` stays honoured while `--wandb.project` is refused; and the
reload test reconstructed the run without its batch size, which is now read out of
the checkpoint's own `train_config.json`.

Finally, run for real, and checked in the direction the Rust tests cannot:

```
$ ./target/debug/lerobot-train --dataset.repo_id=rerobot/state_only_slice \
    --dataset.root=crates/rerobot-train/tests/fixtures/state_only \
    --output_dir=/tmp/rr-real-run/out --policy.type=act --steps=1 --batch_size=2 ...
step:1 loss:21.499 grdn:438.644 lr:1.0e-5
Checkpoint: /tmp/rr-real-run/out/checkpoints/000001

$ python tools/goldens/verify_checkpoint_upstream.py /tmp/rr-real-run/out/checkpoints/000001
2. model.safetensors -> load_state_dict(strict=True): <All keys matched successfully>
3. upstream forward pass on Rerobot's weights -> (2, 2, 2), all finite
```

## Cycle 9 — three independent reviews

Three reviewers audited the training slice independently and all three failed it.
Every material blocker below was fixed test-first: a focused regression test written
and run RED, then the root cause fixed and the test run GREEN.

### Upstream could not actually resume the checkpoint

**RED** — `optimizer_param_groups.json` omitted `decoupled_weight_decay`, which
`torch.optim.AdamW` records in every group. `Optimizer.load_state_dict` compares key
*sets*, so upstream's own loader raised:

```
ValueError: Dictionary keys do not match.
Expected: ... 'fused', 'decoupled_weight_decay', 'params'
got:      ... 'fused', 'params'
```

The verifier had missed this because it overlaid the saved group onto a fresh one and
never read `optimizer_state.safetensors` at all.

**GREEN** — the key is written where torch writes it, between `fused` and `params`,
and `tools/goldens/verify_checkpoint_upstream.py` now calls
`lerobot.optim.optimizers.load_optimizer_state` on the real directory, asserts that
61 parameters were restored with all three AdamW slots as tensors, and takes a step
with the restored optimizer. Stripping the key back out reproduces the error above
exactly, so the verifier is load-bearing rather than decorative.

### The checkpoint was not a deployable artifact

**RED** — `tests/train.rs::the_checkpoint_has_upstreams_directory_layout` failed with
`the checkpoint has no pretrained_model/policy_preprocessor.json`. Upstream's
`save_checkpoint` passes both processors, whose four artifacts carry the dataset
statistics the weights were trained against. Nothing else in the checkpoint records
them, so a policy loaded from one could not reproduce its own normalization.

**GREEN** — `rerobot_train::processor` writes all four, and `tests/processor.rs`
compares both JSON files **byte for byte** against output from upstream's own
`save_pretrained`, and both safetensors tensor for tensor. Matching required a
two-space JSON indent (`ProcessorPipeline` uses `indent=2` where a policy
`config.json` uses 4), which is now `rerobot_core::dataset::json::dumps_indent_ascii`.

### A truncating cast defeated the parser's fail-closed contract

**RED** — `--num_workers=4294967296` completed a full training run and exited 0. The
`u64 as u32` cast narrowed 2^32 to `0`, the one value `validate` accepts, so the run
trained while appearing to honour a worker count it does not implement.

**GREEN** — every integer flag goes through a checked `Value::as_integer::<T>`, which
reports the field, the value and the accepted range. No `as` cast remains on a parsed
value.

### A NaN in a flag produced a successful run

**RED** — `--policy.dropout=nan` exited 0 after `step:1 loss:NaN grdn:NaN lr:NaN`.
Worse than a NaN model, in fact: `NaN > 0.0` is `false`, so the comparison gating
dropout silently *disabled* it and the run trained a different configuration than the
one requested.

**GREEN** — two independent layers. `TrainConfig::validate_numeric_fields` refuses a
non-finite or out-of-range float before the dataset is opened, and
`TrainSession::step` refuses a non-finite loss, KL term, gradient norm or
post-update parameter norm — checked *before* the optimizer runs, so a poisoned
gradient cannot reach the weights.

### Attacker-controlled sizes were unbounded

**RED** — `tests/limits.rs` and `tests/parquet_budget.rs`, 31 tests. A `chunk_size` of
10^29 reached a `Vec` collection; `FeatureSpec::width` multiplied untrusted shape
dimensions with `product()` and mapped an out-of-range one to `0`; `batch_size` and
`steps` were passed straight to `Vec::with_capacity`; and the parquet reader's row
limit was applied *after* Arrow had decoded each batch, bounded nothing else, and was
per file rather than per dataset.

**GREEN** — `rerobot_train::limits` declares the whole budget in one auditable file
with the reasoning for each number, `checked_product`/`checked_mul`/`checked_add`
replace every untrusted multiplication (including `collate`'s
`frames * window * width` reservation, which `collate` bounds itself rather than
trusting its caller), and `ReadBudget` moves the parquet checks to the footer, before
any decode. `DatasetBudget` adds the three totals a per-file budget cannot bound — file
count, dataset rows and dataset-wide decoded values — because the episode table that
names the files is attacker-controlled too. Both budgets are injectable so the checks
are exercised without committing an enormous fixture, and one test each asserts the
default is the production constant.

The budget is bounded from both sides: static assertions require every limit to clear
upstream's own defaults, so it cannot be tightened into an outage.

Verified not vacuous by reverting three of the guards and confirming failure:

| Reverted | Result |
| --- | --- |
| `collate`'s batch-size bound | `a_batch_larger_than_the_budget_is_refused_by_collate_itself` fails |
| the dataset's file-count and row totals | 2 of 25 `limits.rs` tests fail |
| the optimizer loader's parameter-index check | `an_optimizer_state_naming_a_parameter_that_does_not_exist_is_refused` fails |

### Malformed episode metadata reached a panic

**RED** — `query_window` computed `ep_end - 1`, and
`tests/dataset_delta.rs::a_degenerate_episode_range_cannot_overflow_the_clamp` panicked
with `attempt to subtract with overflow` on `i64::MIN`. In release it would have
wrapped to `i64::MAX` and clamped the window onto unrelated frames.

**GREEN** — the clamp saturates, and `DatasetMetadata::validate_episodes` refuses
negative, inverted, overlapping, length-mismatched and past-the-end ranges. The tests
build genuinely malformed parquet with the same arrow stack the reader reads with.

### `checkpoints/last` could recursively delete a real directory

**RED** — `tests/checkpoint_safety.rs::a_real_directory_at_the_reserved_last_path_is_refused_not_deleted`
failed: a pre-existing tree at the reserved path was removed by `remove_dir_all`.

**GREEN** — the marker is only ever unlinked. A symlink is unlinked without being
followed, a regular file is removed, and a real directory is refused with an
explanation. Four further tests cover that the marker is still replaceable and that
a symlink's *target* survives replacement.

### A corrupt checkpoint loaded silently

**RED** — ten failures across `tests/checkpoint_safety.rs`. Model loading coerced any
dtype to `f32`; the RNG reader accepted extra tensors and took element zero of any
shape; the optimizer loader skipped keys it did not recognize and never checked a
parameter index, a moment's shape or dtype, or that a step count was finite.

**GREEN** — every one is now exact and fail-closed, and the load is atomic: the
optimizer validates every entry before installing any of it, so a rejected file leaves
the optimizer untouched.

### Documentation and CI

* The core subtotal in this file said 426 where its rows summed to 425. The tally is
  recomputed above and its rows are checked to sum to the stated total.
* `.github/workflows/ci.yml`'s pinning header named checkout `11d5960` / v4.4.0 while
  every job used `fbc6f39` / v5.
* The PATH-less `ffmpeg` smoke used `command -v ffmpeg` as its precondition, which a
  Homebrew or Nix install satisfies while `CS_PATH` knows nothing about it — so the
  step asserted a fallback that could not succeed. It now probes the fallback path
  itself and skips only when ffmpeg is genuinely not there.
* Strict rustdoc was not a gate. Doctests run the code in the docs and say nothing
  about whether the docs resolve, so `cargo doc` with `-D warnings` is now its own step.

**GREEN, whole workspace**

```
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features      # 795 tests
cargo test --workspace --doc                             # 56 doctests
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo deny check                                         # advisories, bans, licenses, sources
```

## Cycle 10 — two final reviews

Two more reviewers audited the tree after Cycle 9 and both failed it. What they found
was not the guards being absent but the guards being *incomplete*: each one checked the
property it was written for and left a neighbouring property of the same value
unchecked. Same method as before — the regression test first, run RED, then the fix.

### A checkpoint could still hold values no model can use

**RED** — `checkpoint_safety.rs`, three tests. Model loading validated every key, shape
and dtype and then accepted a `NaN` weight; a probe loaded such a checkpoint and
reported `model_nonfinite_accepted=true`. The optimizer loader accepted a state whose
step was `0.5`, whose step tensor was `[1]` rather than a scalar, whose moments held
`NaN`, and which covered only some of the parameters it was restoring.

**GREEN** — `ParameterStore::load` scans every element and names the offending index.
`AdamW::load_state_tensors` requires a rank-0 step (`[]`, which is what torch writes),
a finite non-negative *whole* number of steps, finite moments, and a state that is
either empty — a fresh run has none — or complete for every parameter, listing the
missing ones by index and name. Validation still completes before any of it is
installed.

### Every limit was individually respected and jointly absurd

**RED** — `limits.rs`. `dim_model` 2048, `dim_feedforward` 32768 and 128 layers is
inside every single bound, and each feed-forward weight is 268 MB — under the
per-tensor budget. The model is not: the allowed layer counts multiply that tensor
256 times over. `Initializer` also still used a bare `shape.iter().product()`, and
`TrainSession::new` never checked the `batch_size` a library caller handed it.

**GREEN** — `MAX_MODEL_BYTES` bounds the sum, accounted as each parameter and buffer is
created, so the refusal happens before the allocation rather than after. Shapes go
through `checked_product`, and `TrainSession::new` validates `batch_size` the way the
CLI already did. Two tests bound the budget from the other side: upstream's stock ACT
and the reduced configuration both fit.

### The parquet and metadata budgets did not bound the work

**RED** — `parquet_budget.rs` and `limits.rs`. Rows, columns and compressed bytes were
each bounded and their *product* was not: a thousand rows of a thousand wide columns is
inside all three and still a gibibyte of cells. And `meta/episodes/` is discovered by
walking a directory, so its size was bounded by what was on disk — every file read and
materialized before any invariant could be checked.

**GREEN** — `ReadBudget` gained `max_cells` and `max_decoded_bytes`, both read from the
footer before Arrow builds a reader, and the aggregate checks run before the per-column
ones. `MetadataBudget` bounds the tree's files, rows and decoded cells cumulatively,
with the file count enforced *during* the walk so a hostile tree is not enumerated in
full first.

### Two sources of truth about episodes were never compared

**RED** — `limits.rs`, nine tests. `validate_episodes` checked signs, spans and overlap
but accepted a duplicate `episode_index` — `episode_of` scans for the first match, so
one record becomes unreachable and the other's frames are clamped against the wrong
range, silently — accepted a non-contiguous numbering, and accepted ranges leaving a
frame owned by nobody. The frame rows carry their own `episode_index`, and nothing
compared it with the range it falls in.

**GREEN** — indices must be the exact contiguous range `0..len`, the ranges must tile
`0..total_frames` exactly, and `check_consistency` resolves every frame's absolute index
to an episode and refuses a disagreement, quoting both.

### The portable `last` marker had a window a symlink fits through

**RED** — the marker was unlinked and then written at the reserved path. A symlink
planted between those two operations is followed, and its target truncated.

**GREEN** — the contents go to a same-directory temporary file and are `rename`d into
place, so the reserved path is never opened for writing. Pinned by a 3000-iteration
test that races a symlink-planting thread against the writer and asserts the victim
file is never truncated; it fails on the unfixed code.

### Packaging and documentation

* `crates/rerobot-cli/README.md` said exactly one executable runs and that
  `lerobot-train` exits 2 as unimplemented. Two run, and a bare invocation is exit 64.
  That README ships inside `rerobot-cli-0.1.0.crate`, so the false claim would have
  been the crates.io page.
* All four `.crate` archives omitted `LICENSE` and `NOTICE` while the packaged READMEs
  told recipients to consult them, and the root `LICENSE` points at `NOTICE` for the
  LeRobot attribution. Every crate now ships byte-identical copies — real files rather
  than symlinks, which a Windows checkout without symlink support would turn into a
  text file holding a path — and `tools/verify_packages.py` asserts both are present
  and non-empty in each archive.
* CI ran the identical strict `cargo doc` gate twice; and both the verifier's docstring
  and `CONTRIBUTING.md` said `rerobot-core 0.1.0` was already on crates.io, where the
  live API returns 404. None of the four is published, which is precisely why the
  verifier patches the sibling dependencies inside the archive.

Verified not vacuous by reverting each new guard in turn:

| Reverted | Result |
| --- | --- |
| the model's finite-value scan | 3 of 32 `checkpoint_safety.rs` tests fail |
| the exact scalar step shape | `a_step_count_that_is_not_a_scalar_is_refused_even_at_one_element` fails |
| the integral step check | `a_fractional_step_count_is_refused` fails |
| the moment finiteness scan | `an_optimizer_moment_holding_a_non_finite_value_is_refused` fails |
| optimizer state completeness | `an_optimizer_state_covering_only_some_parameters_is_refused` fails |
| `MAX_MODEL_BYTES` | the test process is killed by the OOM killer — which is the outcome the budget exists to prevent |
| `TrainSession`'s `batch_size` check | `train_session_validates_the_batch_size_it_was_given` fails |
| the aggregate cell budget | 2 of 21 `parquet_budget.rs` tests fail |
| the aggregate decoded-byte budget | `an_aggregate_decoded_byte_estimate_above_the_budget_is_refused_from_the_footer` fails |
| the cumulative metadata row and value totals | 2 of 8 metadata-budget tests fail |
| the walk-time metadata file bound | `more_metadata_files_than_the_budget_are_refused_before_any_is_read` fails |
| the duplicate-index check | `two_episodes_sharing_an_index_are_refused` fails |
| the index contiguity check | `episode_indices_that_are_not_the_contiguous_range_are_refused` fails |
| the frame-domain coverage sweep | 3 of 3 `episode_ranges_that_*` tests fail |
| the frame/episode cross-check | `a_frame_whose_episode_index_disagrees_with_its_range_is_refused` fails |

Three of those reverts initially left their test *passing*, because a neighbouring check
produced a message the assertion also matched. Each assertion was tightened to name the
property under test — the numbering, the uncovered frames, the walk itself — and the
revert then failed as it should. An assertion loose enough to be satisfied by a
different guard is not evidence about this one.

## Cycle 11 — upstream checkpoint-frequency fix

Upstream moved from the pinned commit to
`1fe58f2d3afe0e7c46e86fee03de2e4122fbe9a1` during review. Most intervening
changes are outside this state-only ACT slice; one changed the shared training
contract: `save_freq <= 0` now disables periodic checkpoints while preserving
the final checkpoint.

**RED** — two executable tests ran before the implementation changed. The zero
case exited 2 with `save_freq must be positive`; the negative case exited 64
because the CLI narrowed the source `int` to `u64`.

**GREEN** — both complete two real optimization steps and write only
`checkpoints/000002`. `TrainConfig.save_freq` is now an arbitrary-precision
signed integer, so the modulo decision does not narrow the accepted source
domain.

## Cycle 12 — atomic checkpoint publication

**RED** — three writer tests compiled before `write_staged_directory` existed;
the focused run failed with `E0432: unresolved import`. They require an existing
destination to stay byte-for-byte unchanged, a symlink destination to leave its
target untouched, and an injected failure after the first staged write to leave
no final destination or temporary tree.

**GREEN** — checkpoint files are built in a unique temporary sibling and renamed
into place only after every write succeeds. Existing destinations and aliases are
refused before writing and checked again before publication.

The first Windows CI run then supplied a retained platform RED: the real two-step
CLI test failed replacing `checkpoints/last` with `Access is denied (os error 5)`.
Windows requires a directory symlink to be unlinked with `remove_dir`, while the
Unix implementation correctly uses `remove_file`. The platform-specific unlink
now handles both directory and file symlinks without following either target.

## Cycle 11 — an independent robustness review

One reviewer ran the slice against probes of its own rather than its tests, and found
one crash, one non-transactional write and three narrower defects. Same method: the
regression test first, run RED, then the fix.

### A zero-width feature reached `chunks(0)` and panicked

**RED** — `limits.rs`, five tests. Every bound in the budget is an *upper* one, and
zero passes all of them: `FeatureSpec::width` returned 0, `collate` built a
`[batch, window, 0]` tensor from it, and `Batch::normalized` divided the flat buffer
into rows with `slice::chunks(0)`, which panics. A panic is the outcome the budget
exists to prevent — no message, no exit code, no cleanup.

**GREEN** — an empty shape is refused where it is declared, in `FeatureSpec::width` and
in the policy's feature resolution, so it never reaches a tensor; `collate` and
`normalize_tensor` refuse a zero extent as well, because both are public and neither
may panic on data. The scalar shape `[]`, whose product is 1, still works — the
convention `math.prod(())` uses, and the one upstream relies on.

### `training_step.json` was read with defaults for malformed values

**RED** — `checkpoint_safety.rs`, four tests. `num_processes` of `"not a number"` and a
`batch_size` of `{"a": 1}` read back as `1` and `0`: `unwrap_or` discarded both the type
error and the range error, so a damaged file reported a run configuration the
checkpoint never had.

**GREEN** — absence and malformation are now different findings. Upstream's
`save_training_step` omits `num_processes` and `batch_size` when it has no value for
them, so *absence* keeps its documented default and a minimal upstream file still
reads. A value that is present and is a string, a float, an object, `null`, negative,
zero processes, or a batch size beyond this build's limit is refused by name.

### Concurrent marker writers shared one temporary

**RED** — `threads_writing_the_marker_at_once_do_not_collide_on_one_temporary`: 444 of
800 concurrent writes failed with `No such file or directory`. The temporary was named
after the process alone, so every thread of one process used the same path and one
thread's `rename` moved the file another was still writing.

**GREEN** — the name carries a per-writer counter as well as the process id, and the
temporary is claimed with `create_new` rather than assumed free, so two processes that
do collide retry instead of overwriting each other.

### The checkpoint write was already staged — but nothing proved it

**RED (negative control)** — `save_checkpoint` builds its eleven files in a sibling
staging directory and publishes them with one `rename`, and the three `run::tests`
unit tests covered refusal and cleanup. None of them could tell staged from unstaged:
with the staging bypassed, all three still passed.
`the_destination_is_never_visible_half_written` watches the destination while a real
save runs and fails on the unstaged version with *150 partial sightings*, the first
with all eleven files missing.

**GREEN** — the property is now pinned from both ends: a published checkpoint holds
exactly the eleven files the save wrote and nothing it inherited, an occupied
destination is refused with its contents untouched rather than merged into, and a save
that cannot write leaves neither the destination nor a staging directory behind.

### A Windows directory symlink is a directory to the filesystem API

**RED** — reasoned rather than executed; the host is darwin. `MoveFileExW` with
MOVEFILE_REPLACE_EXISTING refuses to replace a directory, and a directory symlink
carries `FILE_ATTRIBUTE_DIRECTORY`, so the portable marker could not replace the
symlink an earlier run left. `DeleteFileW` refuses one too, which is why unlinking
picked its call by trial and error.

**GREEN** — `unlink_symlink` reads the kind from the metadata (`is_symlink_dir`) and
calls `RemoveDirectoryW` or `DeleteFileW` accordingly, keeping the other as a fallback
for an entry that changed kind underneath it; neither follows the link. The portable
marker retries its `rename` once after unlinking a symlink it finds in the way — the
only case where the atomic path is given up, and only for a marker that was already a
link. Two tests cover it on every platform, skipping where the OS will not grant a
symlink; the `cfg(windows)` code is type-checked here with
`cargo check --target x86_64-pc-windows-msvc`, and CI's `windows-latest` job runs it.

**Not verified on this host:** the Windows behaviour itself. The two tests pass on
darwin, where `rename` over a symlink already works, so they demonstrate the shared
path is unchanged rather than that the Windows-specific one is fixed.

## Cycle 13 — what a real dataset and a real GPU found

The first run of the slice on something other than its own fixtures: ACT trained by
`lerobot-train` on ten episodes of `HuggingFaceVLA/libero`, on an RTX 5080, and the
checkpoint evaluated by upstream `lerobot` in the LIBERO simulator.

**These two were not found test-first.** Both were found by running the slice, and
the order was fix-then-test rather than the reverse. What is recorded below is the
RED that was demonstrated afterwards, by reverting each fix on its own and running
the new test against the unfixed code — so the tests are known to fail for the
reason claimed, which is the property the test-first rule exists to guarantee. The
honest summary is that the fixture suite could not have found either one: no
fixture declares a camera the way the Hub does, and no test ran more than a handful
of steps.

### A camera's declared shape was read channel-first whatever `names` said

`dataset_to_policy_features` picks the channel order from `names`, not from the
numbers:

```python
if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
    shape = (shape[2], shape[0], shape[1])
```

`policy_features` was a port of that function with those four lines missing, and
`image_shape` read `spec.shape` directly, so both took the declaration to be
`[channel, height, width]` unconditionally. Every LIBERO conversion published on
the Hub — `HuggingFaceVLA/libero`, `lerobot/libero_spatial_image` — declares
`[256, 256, 3]` with `["height", "width", "channel"]`, which the reader therefore
refused as a 256-channel image. The committed fixture is channel-first, so nothing
in the suite disagreed.

**RED** — `embedded_image.rs`, two tests, with
`FeatureSpec::policy_shape` reverted to returning `self.shape` unchanged:

```
a_camera_declared_height_width_channel_is_read_as_the_same_frames  FAILED
a_non_square_camera_proves_the_reorder_rather_than_the_shape_surviving  FAILED
  assertion `left == right` failed
    left: [24, 48, 3]
   right: [3, 24, 48]
```

The square case alone would not have been enough: at 32×32 a reorder and a no-op
produce the same three numbers. The non-square one is what makes the assertion
about the reorder rather than about the shape surviving.

**GREEN** — `FeatureSpec::policy_shape` ports the rule, and both `policy_features`
and the embedded-camera reader go through it, so the shape the policy config
records and the shape the decoder allocates against cannot disagree. Upstream
indexes `names[2]` unconditionally and raises `TypeError` on `"names": null`; this
treats a missing or short `names` as "already channel-first", which refuses less
rather than more.

`cargo test -p rerobot-train --test embedded_image`

### AdamW's stored moments kept the graph they were computed from

A gradient candle returns still carries the `BackpropOp` chain that produced it.
`exp_avg` and `exp_avg_sq` were computed from one and stored undetached, so each
step's moments pinned that step's whole forward pass — every ResNet and transformer
activation — and because step N+1's moments are computed from step N's, every step
of a run stayed reachable through the chain.

Measured on the GPU before the fix: **~800 MB per step**, independent of batch
size, `CUDA_ERROR_OUT_OF_MEMORY` after about 15 steps at batch 8, 4 and 2 alike.
After the fix, flat at 7.8 GB across 10 000 steps. The numbers the optimizer
produced were correct throughout; only the run's ability to finish was affected,
which is why 16 existing optimizer tests all passed.

**RED** — `optimizer.rs`, one test, with the two `detach()` calls removed:

```
the_stored_moments_do_not_keep_the_graph_they_were_computed_from  FAILED
  a stored AdamW moment is still attached to its backward graph
```

Getting this test to fail for the right reason took two attempts, and the first one
is the more useful record. Written against the file's existing
`step_with_gradient`, it passed *with the fix removed*: that helper's loss is linear
in the parameter, so its gradient is the coefficient tensor and carries no graph
for a moment to retain. A model's gradients are computed through its activations
and do point back at them. The test now builds `sum(p * p)`, whose gradient `2p` is
a tensor built from the parameter, and asserts `track_op()` on that gradient before
using it — so a future change that flattens the graph again cannot make the test
vacuous instead of failing.

**GREEN** — both moments are detached before being stored. `detach` shares the
storage and drops only the history, which is what `torch.optim` gets for free from
running under `no_grad`. `optim::any_moment_tracks_its_graph` exposes the property
because the moments are private and `track_op` is on the tensor.

`cargo test -p rerobot-train --test optimizer`

### Two limits and one refusal that were correct and still in the way

Not bugs, and recorded because a reader hitting them should know they were
deliberate.

* `MAX_DECODED_VALUES` was `1 << 28`, one gibibyte of `f32`. Ten LIBERO episodes
  read through two 256×256 cameras is 449 million scalars, so the smallest
  realistic two-camera dataset did not fit. Raised to `1 << 29`, which is the
  one-line change the refusal message itself points at.
* `fps` must be an integer, and every LIBERO dataset on the Hub writes `10.0`. Left
  as it is — it is the typed boundary the error message names, and `10.0` and `10`
  are the same value, so a caller can normalize it.
* ACT's default `pretrained_backbone_weights = "ResNet18_Weights.IMAGENET1K_V1"` is
  refused rather than silently ignored. Left as it is; the run passed `null` and
  trained the backbone from scratch.

### What the run demonstrated

The CUDA path in `tests/device_smoke.rs` had never been executed on NVIDIA
hardware. It has now:

```
cargo test -p rerobot-train --features cuda --test device_smoke
  a_cuda_session_puts_its_parameters_on_the_gpu ... ok
  the_cuda_path_runs_a_whole_step_on_the_gpu_and_writes_a_checkpoint_that_reloads ... ok
```

End to end, on `libero_spatial` task 2 ("pick up the black bowl from table center
and place it on the plate"), 10 training episodes, 10 000 steps at batch 8, loss
32.0 → 0.29:

| Checkpoint | Episodes | Success |
| --- | --- | --- |
| 200 steps | 1 | 0 % |
| 5 000 steps | 10 | 8 / 10 |
| 10 000 steps | 10 | 8 / 10 |
| 10 000 steps | 20 | 15 / 20 |

The checkpoint was loaded by upstream `ACTPolicy.from_pretrained` and
`make_pre_post_processors` without modification, and the rollouts were upstream's
`lerobot-eval`. This is one task evaluated on the task its own demonstrations came
from, so it is an in-distribution number and not a benchmark result; what it
demonstrates is that the Rust slice produces a policy upstream can load and that
drives a real MuJoCo robot to the goal. The 200-step checkpoint failing through the
identical path is what rules out the environment doing the work.

**Test count:** 851 → 862 distinct tests. The local deployment slice adds eight
new tests (`tests/deploy.rs` and `tests/rollout_cli.rs`); the naive sum of cargo's
`test result:` lines is 864 because `dataset_json.rs` intentionally re-executes
two cases.

## Final GREEN totals

`cargo test --workspace --all-targets --locked`

| Target | Tests |
| --- | ---: |
| `rerobot-cli` library unit tests | 1 |
| `rerobot-core` `rollout::dagger` unit tests | 2 |
| `rerobot-core` `tests/action_interpolator.rs` | 50 |
| `rerobot-core` `tests/act_config.rs` | 10 |
| `rerobot-core` `tests/act_checkpoint.rs` | 19 |
| `rerobot-core` `tests/byte_count.rs` | 15 |
| `rerobot-core` `tests/ring_buffer.rs` | 37 |
| `rerobot-core` `tests/rename_processor.rs` | 23 |
| `rerobot-core` `tests/newline_task_processor.rs` | 24 |
| `rerobot-core` `tests/dagger.rs` | 26 |
| `rerobot-core` `tests/dataset_info.rs` | 36 |
| `rerobot-core` `tests/dataset_io.rs` | 23 |
| `rerobot-core` `tests/dataset_json.rs` | 51 |
| `rerobot-core` `tests/types.rs` | 21 |
| `rerobot-core` `tests/sysinfo.rs` | 13 |
| `rerobot-core` `tests/random.rs` | 11 |
| `rerobot-core` `tests/dataset_delta.rs` | 20 |
| `rerobot-core` `tests/dataset_sampler.rs` | 19 |
| `rerobot-core` `tests/dataset_stats.rs` | 11 |
| `rerobot-core` `tests/policy_normalize.rs` | 15 |
| `rerobot-compat` `tests/inventory.rs` | 18 |
| `rerobot-compat` `tests/docs_consistency.rs` | 13 |
| `rerobot-train` `run` and deployment unit tests | 4 |
| `rerobot-train` `tests/dataset.rs` | 22 |
| `rerobot-train` `tests/deploy.rs` | 9 |
| `rerobot-train` `tests/device.rs` | 3 |
| `rerobot-train` `tests/device_smoke.rs` | 1 |
| `rerobot-train` `tests/embedded_image.rs` | 28 |
| `rerobot-train` `tests/model.rs` | 31 |
| `rerobot-train` `tests/optimizer.rs` | 17 |
| `rerobot-train` `tests/train.rs` | 39 |
| `rerobot-train` `tests/goldens.rs` | 12 |
| `rerobot-train` `tests/processor.rs` | 8 |
| `rerobot-train` `tests/limits.rs` | 53 |
| `rerobot-train` `tests/parquet_budget.rs` | 21 |
| `rerobot-train` `tests/checkpoint_safety.rs` | 45 |
| `rerobot-cli` `tests/cli.rs` | 21 |
| `rerobot-cli` `tests/info.rs` | 18 |
| `rerobot-cli` `tests/rollout_cli.rs` | 4 |
| `rerobot-cli` `tests/which.rs` | 21 |
| `rerobot-cli` `tests/train_cli.rs` | 35 |
| **Total** | **870** |

Summing the `test result:` lines cargo prints gives 872, not 870.
`tests/dataset_json.rs` re-executes its own harness twice to drive a case that
needs a fresh thread stack, and each re-execution prints a `1 passed; 50 filtered
out` line of its own. The table above counts distinct tests.

The 18 `lerobot-*` binary targets contribute no unit tests; their behaviour is
covered by `tests/cli.rs`, which runs the built executables as subprocesses.
`tests/which.rs` reports 21 on a Unix runner: five of its cases are
`cfg(windows)` and are compiled and run only by the `gates` job on
`windows-latest`.

`cargo test --workspace --doc`

| Crate | Doctests |
| --- | ---: |
| `rerobot-core` (crate README + item docs) | 49 |
| `rerobot-compat` (crate README) | 2 |
| `rerobot-cli` (crate README + `which`) | 3 |
| `rerobot-train` (crate README + `limits`) | 3 |
| **Total** | **57** |

The subtotals, which sum to the 870 above: 426 in `rerobot-core` (including its two
`rollout::dagger` unit tests), 313 in `rerobot-train`, 100 in `rerobot-cli` (including
its one library unit test) and 31 in `rerobot-compat`. `rerobot-core` is where the
pure-behaviour parity claim lives; of `rerobot-train`'s 313, `tests/goldens.rs` is
the only file whose expected values came from PyTorch rather than from upstream's
source, and `tests/processor.rs` is the only one comparing bytes against files
upstream's own writer produced.

The training slice's fixtures are committed and read offline. `cargo test` never
invokes Python: `tools/goldens/` holds the scripts that produced them, run once
against upstream at the pinned commit.

## Cycle 18 — local ACT deployment boundary

**RED** — before the deployment adapter existed, the two focused acceptance
commands were run against the last training-only tree:

```
cargo test -p rerobot-train --test deploy --locked
cargo test -p rerobot-cli --test rollout_cli --locked
```

Both exited non-zero: the first had no `rerobot_train::deploy` API to compile
against, and the second still reached the inventory's explicitly unsupported
rollout path. This was the intended boundary failure, not a fixture or test
harness error.

**GREEN** — after adding the checkpoint loader, feature normalization, ACT action
queue, temporal ensembler, finite dataset-backed rollout, CLI allow-list, and refusal
paths:

```
cargo test -p rerobot-train --test deploy --locked
9 passed; 0 failed

cargo test -p rerobot-cli --test rollout_cli --locked
4 passed; 0 failed
```

The tests train a reduced ACT checkpoint in the fixture dataset, load the actual
`safetensors` artifact, select finite actions through both queue and temporal-ensemble
paths, exercise the CLI as a subprocess, and prove that an oversized
arbitrary-precision integer is refused rather than silently narrowed. The accepted
boundary is intentionally local and hardware-independent; robot drivers,
environments, and video shards remain refused.

## Cycle 19 — Hub destination safety

**RED** — before tightening the Hub snapshot writer's destination boundary, the
new tests were run one at a time:

```
cargo test -p rerobot-train --test hub an_existing_empty_destination_is_rejected_without_being_removed -- --exact
exit 101

cargo test -p rerobot-train --test hub a_symlink_destination_is_rejected_even_when_its_target_is_complete -- --exact
exit 101
```

Both failures were the intended missing behaviour: the old implementation
removed an existing empty directory before downloading, and treated a symlink
to a complete snapshot as a cache hit. No production change was made before
these RED runs.

**GREEN** — `HubDownloader::download` now checks the destination with
`symlink_metadata`, preserves complete regular-directory cache hits, and
rejects existing incomplete directories, files, broken links, and symlink
aliases before making a request or creating a staging directory:

```
cargo test -p rerobot-train --test hub -- --nocapture
9 passed; 0 failed
```

The existing staging/rename and mid-download failure tests remain in the same
suite, so a failed transfer still leaves no final dataset or staging sibling.

## Cycle 20 — checkpoint-only ACT inference boundary

**RED** — before the checkpoint-only adapter existed, the new focused test was
run against the dataset-bound `InferenceSession` API:

```
cargo test -p rerobot-train --test deploy a_checkpoint_can_infer_from_a_caller_batch_without_opening_a_dataset --locked
exit 101 (compile failure: `InferenceSession::load_checkpoint` was not yet defined)
```

This was the intended API failure: the test could not compile because the
runtime-owned observation boundary did not exist. No production implementation
was present before the RED run.

**GREEN** — `InferenceSession::load_checkpoint` now loads the ACT config,
saved processor state, model weights, camera normalization choice, action queue,
and optional temporal ensembler without opening or resolving a dataset.
`select_action_on_batch` accepts one caller-owned raw `Batch`, applies the
checkpoint normalizer, preserves the checkpoint's frame index, and returns the
same finite action as the dataset-backed path for the identical fixture frame.
A checkpoint-only session reports no dataset and refuses dataset-indexed
rollouts rather than silently using a hidden fixture.

```
cargo test -p rerobot-train --test deploy a_checkpoint_can_infer_from_a_caller_batch_without_opening_a_dataset --locked
1 passed; 0 failed

cargo test -p rerobot-train --test deploy --locked
14 passed; 0 failed
```

The remaining boundary is deliberate: this API accepts already-acquired
Candle tensors, but it does not invent a simulator, camera driver, robot
transport, or Gymnasium environment.

The CPU/default feature gates passed locally with the locked dependency graph:

```
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
cargo test --workspace --doc --locked
RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps --locked
cargo build --workspace --release --locked
python3 tools/verify_packages.py                              # exact archive-only verification
cargo +1.85.0 build --workspace --locked
cargo +1.85.0 test --workspace --all-targets --locked
```

The required all-features attempt is a documented environment blocker on this
host, not a green result:

```
cargo test --workspace --all-targets --all-features --locked
error: failed to run custom build command for `cudarc v0.16.6`
Failed to execute `nvcc`: No such file or directory
```

The repository's CI deliberately uses the default feature set for the same reason:
`cuda` requires an NVIDIA toolkit at build time. The MSRV commands are the same
ones the `msrv` CI job runs, except that CI reads `1.85` out of `Cargo.toml`
instead of hardcoding it, so the manifest and tested toolchain cannot drift.
1.85 is the floor the *locked* tree imposes: `indexmap` 2.14.0 declares
`rust-version = "1.85"` and `hashbrown` 0.17.1 declares `"1.85.0"`. The
workspace's own code needs less than that, so the number is a property of the
dependency set, not of this source.

## Cycle 21 — per-camera normalization state is fail-closed

The pinned upstream writes visual statistics into the normalizer state with
`mean` and `std` together. `NormalizerProcessorStep._apply_transform` then
requires both for `MEAN_STD`; silently treating one missing entry as identity
would deploy a camera with a different input scale from the one used in training.

**RED** — the regression test was added before the loader change and run alone:

```
cargo test -p rerobot-train --test processor visual_processor_state_rejects_partial_camera_statistics --locked

FAILED: a visual feature with only one statistic must be refused: LoadedPolicyProcessors { ... camera_normalizations: {} }
```

The damaged preprocessor and postprocessor state each retained only
`observation.images.top.mean`. The loader accepted the checkpoint and returned
an empty camera-normalization map, which was the intended missing-behaviour
failure rather than a fixture or compilation error.

**GREEN** — `camera_normalizations_from_state` now distinguishes absent camera
statistics from a partial pair: absent `mean`/`std` remains the upstream
identity/no-statistics case, while exactly one is present returns a named
metadata error. Both complete pairs continue to round-trip through safetensors.

```
cargo test -p rerobot-train --test processor visual_processor_state_rejects_partial_camera_statistics --locked
1 passed; 0 failed

cargo test -p rerobot-train --test processor --locked
10 passed; 0 failed
```

## Cycle 22 — episode-filtered training rows

The pinned upstream constructs `LeRobotDataset(..., episodes=[...])` with a
filtered Arrow dataset, then builds an absolute-to-relative index map for the
sampler's delta-window lookups. The Rust reader previously accepted the
configuration field only at the sampler boundary: it still materialized the
whole dataset and `get(0)` meant absolute frame zero.

**RED** — the first dataset regression test was run before
`StateOnlyDataset::load_for_episodes` existed:

```
cargo test -p rerobot-train --test dataset selecting_episodes_compacts_relative_rows_but_keeps_absolute_delta_windows --no-default-features
error[E0599]: no associated function or constant named `load_for_episodes` found for struct `StateOnlyDataset`
```

After the first implementation, the end-to-end training regression exposed the
second boundary defect rather than passing vacuously: the sampler emitted
absolute `2`, `3` into a two-row filtered dataset and `get(3)` failed with
`frame 3 is out of range for a dataset of 2 frames`.

The metadata-count regression then failed because the implementation sized its
validity mask from the number of episode-table rows, warning for episode `1`
when `info.json` declared two episodes. The GREEN fix uses the declared
`total_episodes` without allocating a mask from attacker-controlled metadata.

The warning-order regression also failed on a real malformed copy: the missing
file error returned before the warning was emitted. The constructor now logs
before opening data files.

**GREEN** — selected rows now have compact relative indexing while retaining
absolute frame IDs for episode-clamped action windows. The sampler maps its
eligible absolute positions back to relative rows, the constructor warns about
out-of-range episode entries before later data reads, and the sampler preserves
upstream's subsequent out-of-range error. The training session uses that
filtered dataset for sampler length, checkpoint resume offsets, and real
updates.

```
cargo test -p rerobot-core --test dataset_sampler --no-default-features
21 passed; 0 failed
cargo test -p rerobot-train --test dataset --no-default-features
27 passed; 0 failed
cargo test -p rerobot-train --test train a_training_run_consumes_only_the_configured_episodes --no-default-features
1 passed; 0 failed
```

## Cycle 23 — fresh training from a saved config

The pinned upstream `lerobot-train` accepts `--config_path` as a configuration
source as well as for resume. Before this slice, Rerobot accepted the flag only
for `--resume=true`, so a valid locally-produced `train_config.json` could not
start a new run without repeating every dataset and policy option.

**RED** — the real executable regression was run before the config loader:

```
cargo test -p rerobot-cli --test train_cli a_saved_train_config_can_start_a_fresh_run_without_retyping_dataset_or_policy_flags -- --exact --nocapture
FAILED: --config_path is not supported in this slice: resuming or loading a run config needs the Draccus config loader, which is not ported
```

The test first trains the committed state-only fixture, then passes its actual
checkpoint `train_config.json` to a second process. The failure was a parser
boundary refusal, not a missing fixture or a failed model update.

**GREEN** — `TrainConfig::from_config_file` now reads the bounded native JSON
form, preserves arbitrary-precision integer fields, and rejects malformed,
wrong-type, oversized, or non-JSON documents through the checkpoint error path.
The CLI applies supported overrides, clears resume state, and runs the full ACT
training/checkpoint path. General YAML/Draccus files remain explicitly outside
this boundary.

```
cargo test -p rerobot-cli --test train_cli -- --nocapture
41 passed; 0 failed
cargo test -p rerobot-compat --test docs_consistency --locked
13 passed; 0 failed
```

## Cycle 24 — one-pass camera normalization for renamed training inputs

A saved observation rename is applied by the training processor only once. The
internal batch builder still selected camera tensors under their renamed keys
before `step_on` applied the saved mapping, which made chained aliases such as
`left -> top` and `top -> wrist` vulnerable to applying the mapping twice.

**RED** — the focused regression was added before the selector existed:

```
cargo test -p rerobot-train --test processor camera_normalization_is_selected_after_one_observation_rename
error[E0425]: cannot find function `camera_normalizations_for_input_images`
```

The compiler failure is at the intended API boundary; no runtime fixture or
unrelated test was involved.

**GREEN** — camera statistics are selected using the raw input key and its
single mapped destination, while the batch retains raw keys until the normal
observation rename. The focused regression includes a second mapping entry to
pin the one-pass rule.

```
cargo test -p rerobot-train --test processor camera_normalization_is_selected_after_one_observation_rename -- --exact
1 passed; 0 failed
cargo test -p rerobot-train --all-targets
all tests passed; 0 failed
```

## Cycle 25 — bounded policy-config reads during resume reconstruction

`TrainConfig::from_checkpoint_dir` now applies the same 16 MiB checkpoint-JSON
bound to the resolved `pretrained_model/config.json` that it already applied to
`train_config.json`. This rejects an oversized policy document before the JSON
parser or an unbounded `read_to_string` can materialize it.

**RED** — the regression first exercised the old unbounded read and reached the
JSON parser instead of the checkpoint reader's boundary:

```
cargo test -p rerobot-train --test train oversized_policy_config_is_rejected_before_unbounded_checkpoint_read --locked -- --exact
assertion `left == right` failed
left: .../config.json: Rerobot JSON input byte limit exceeded (16777216): line 1 column 1 (char 0)
right: .../config.json: config.json exceeds the 16777216-byte limit
```

**GREEN** — the shared bounded reader returns the checkpoint-specific error before
parsing:

```
cargo test -p rerobot-train --test train oversized_policy_config_is_rejected_before_unbounded_checkpoint_read --locked -- --exact
1 passed; 0 failed
```

## Cycle 26 — platform-native checkpoint test paths

The first CI run of commit `34da4bb` passed the macOS and Ubuntu gates but failed
Windows in the new oversized-policy-config regression. The implementation was
correct; the test expected a path built with a literal `/`, while the reader built
it from platform-native components.

**RED** — Windows CI reported:

```
left:  "...\\pretrained_model\\config.json: config.json exceeds the 16777216-byte limit"
right: "...\\pretrained_model/config.json: config.json exceeds the 16777216-byte limit"
test result: FAILED. 45 passed; 1 failed
```

**GREEN** — the test now uses `.join("pretrained_model").join("config.json")`,
matching the production path construction. The Windows failure was isolated to
that regression before any further implementation change.
