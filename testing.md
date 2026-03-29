# Forgetting Cycle Test Plan

## Purpose

This document is the implementation and release gate for issue `#44`:
`ForgettingService` scan, score, fade/prune, duplicate handling, and media
cleanup.

The rule for this branch is simple: write the expected behavior down first,
implement against it, then verify the branch against the same checklist before
opening the PR.

## Scope For This PR

Included:

- `forgetting/service.py`
- `ForgettingReport` and per-record decision entries
- config knobs for prune thresholds, fade thresholds, fade behavior, and
  procedural low-performance overrides
- semantic duplicate-cluster resolution for the forgetting cycle
- confirmed supersession forced-prune handling
- fade via store `replace()`
- prune via store `delete()` followed by owned-media deletion
- per-record forgetting events and cycle summary events
- automated tests for planning, execution, idempotency, and event payloads

Explicitly not included:

- LLM judging for duplicate or contradiction resolution
- transactional or cross-store atomic writes
- Chroma compaction/rebuild automation
- API/CLI endpoints for manually triggering the forgetting cycle
- benchmark claims beyond a local sanity check on duplicate-cluster handling

## Locked Design

### Planning model

The cycle operates in two phases:

1. Plan from a single snapshot:
   scan stores, detect duplicate clusters, resolve forced actions, compute decay
   scores with one shared `cycle_now`, and build the full decisions list.
2. Execute from the completed plan:
   apply fades, then prune records, then delete owned media for successfully
   pruned records, then emit the cycle summary event.

`dry_run=True` uses the same planning path but skips the mutation phase.

### Duplicate resolution

- Semantic likely-duplicate pairs are collapsed into connected components.
- Each component produces one survivor.
- The survivor is chosen by the deterministic chain:
  supersession state, importance, access count, recency (`created_at`), ID.
- All non-survivors are forced-pruned with reason `likely_duplicate`.
- The duplicate resolver must be swappable later, but this PR ships the
  deterministic default.

### Forced-action precedence

When multiple reasons could apply, the winning reason order is:

1. `superseded`
2. `likely_duplicate`
3. `low_performance`
4. `time_decay`

Confirmed supersession only forces prune when the successor record still exists.

### Bucket policy

Per memory type:

- score `< prune_threshold` -> `prune`
- prune_threshold `<= score < fade_threshold` -> `fade`
- score `>= fade_threshold` -> `keep`

Fade mutates `importance` with:

- `new_importance = max(FADE_FLOOR, old_importance * FADE_FACTOR)`

`0.0` remains reserved for confirmed supersession.

### Procedural low-performance policy

- If `wilson_score < PROCEDURAL_LOW_PERF_WILSON_THRESHOLD` and
  `total_outcomes >= PROCEDURAL_LOW_PERF_MIN_OUTCOMES`, forced-prune with reason
  `low_performance`.
- Otherwise low performance can still force `fade`, but does not force `prune`.

### Reporting and events

`ForgettingReport` must include:

- aggregate counters
- `by_type`
- `duplicates_flagged`
- `decisions`

Each decision entry must include:

- `record_id`
- `memory_type`
- `action`
- `reason` (`null` for retained records with no forgetting pressure)
- `score`
- `media_deleted`

Events:

- real run: per-record `memory.faded`, per-record `memory.pruned`,
  summary `forgetting.cycle_completed`
- dry run: summary `forgetting.cycle_dry_run` only

## Automated Test Matrix

### Planning and reporting

1. `run_cycle(dry_run=True)` returns a report with derived aggregate counters
   that exactly match the `decisions` list.
2. `dry_run=True` does not mutate any record, delete any record, or delete any
   media file.
3. All decisions in one cycle use the same `cycle_now` reference time.
4. The report includes keep, fade, and prune decisions with stable reasons, and
   keep decisions above the fade threshold carry `reason = null`.
5. Top-level counts are derived from decisions rather than maintained
   separately.

### Duplicate handling

1. A duplicate pair produces one survivor and one forced prune.
2. A transitive duplicate cluster (`A~B`, `B~C`) is resolved component-wise with
   exactly one survivor.
3. The survivor is chosen by importance before access count, access count before
   recency, recency before ID.
4. Duplicate resolution is deterministic regardless of pair iteration order.
5. Non-winning cluster members are reported as `likely_duplicate` prunes.

### Supersession handling

1. A semantic record with `superseded_by` and an existing successor is forced to
   prune even if its decay score would keep or fade it.
2. If the successor record is missing, supersession does not force prune and the
   record falls back to ordinary cycle logic.
3. Supersession reason outranks all other reasons in the report and emitted
   event payloads.

### Decay buckets and fading

1. Semantic records respect semantic prune and fade thresholds.
2. Episodic records respect episodic prune and fade thresholds.
3. Procedural records respect procedural prune and fade thresholds when no
   low-performance override applies.
4. Faded records persist with reduced importance using `FADE_FACTOR` and
   `FADE_FLOOR`.
5. Fading never sets importance to `0.0` unless the record was already
   superseded outside the fade path.

### Procedural low-performance overrides

1. Low Wilson score with insufficient outcomes does not force prune.
2. Low Wilson score with enough outcomes forces prune.
3. Low performance without prune-level evidence forces fade.
4. `low_performance` outranks ordinary `time_decay` when both apply.

### Execution semantics

1. Real runs apply fades through store `replace()` only after planning
   completes.
2. Real runs delete records through store `delete()` only after fade execution
   is complete.
3. Owned media is deleted only after the owning record is successfully deleted.
4. Missing records during execution are treated as skipped/idempotent cases, not
   counted as successful prunes.
5. Missing media files do not increment `media_deleted`.
6. Batched prune execution processes records in groups of 10.

### Event contract

1. `memory.faded` is emitted once per actual fade and includes record id,
   memory type, reason, old importance, and new importance.
2. `memory.pruned` is emitted once per actual prune and includes record id,
   memory type, reason, and media context.
3. `forgetting.cycle_completed` is emitted once per real run with the full
   report payload.
4. `forgetting.cycle_dry_run` is emitted once per dry run with the projected
   report payload.
5. Dry runs emit no per-record mutation events.

### Integration and regression coverage

1. Pruned records return `None` from `get_by_id`.
2. Faded records remain retrievable and show the updated importance.
3. Existing contradiction and decay tests still pass unchanged.
4. Existing store and event integration tests still pass unchanged.

## Manual Verification

These checks happen after automated tests pass.

1. Seed one semantic duplicate cluster and one superseded semantic record, run
   the cycle manually in Python, and inspect the returned report for one
   survivor per cluster and one immediate supersession prune.
2. Seed one media-backed episodic record that lands in prune and confirm the
   record disappears before the owned media file is removed.
3. Seed one procedural record that lands in fade and confirm its importance is
   halved but remains above `FADE_FLOOR` when applicable.
4. Run a dry cycle against the same dataset and confirm the report matches the
   real-run plan while the database and media files remain unchanged.

## API / Curl Checks

No new API surface is planned in this PR, so there are no required `curl`
acceptance checks for merge. If this branch grows an endpoint or admin trigger,
the API contract and `curl` cases must be added here before shipping.

## Formula Checks

1. Verify bucket boundaries exactly at prune and fade thresholds.
2. Verify fade math:
   `new_importance = max(FADE_FLOOR, old_importance * FADE_FACTOR)`.
3. Verify procedural prune override boundary at:
   `wilson_score < PROCEDURAL_LOW_PERF_WILSON_THRESHOLD` and
   `total_outcomes >= PROCEDURAL_LOW_PERF_MIN_OUTCOMES`.

## Benchmark / Sanity Checks

This PR does not claim production-scale optimization, but it should include one
local sanity check:

1. Duplicate-cluster planning on a moderate synthetic semantic dataset should
   complete without pathological order sensitivity or exploding decision counts.

If benchmark code is added, it should be documented in the PR description but
kept out of the merge gate unless it is deterministic offline.

## Exit Criteria

The PR can be opened only when all of the following are true:

1. `testing.md` still matches the shipped implementation.
2. New unit and integration tests for the forgetting cycle pass locally.
3. Existing relevant test suites still pass locally, or any unrelated
   pre-existing failures are explicitly called out.
4. Manual verification for dry run, fade, prune, duplicate resolution, and
   media cleanup is completed.
5. The PR description explains the locked behavior: component-wise duplicate
   resolution, staged execution, explicit fade bands, and swappable duplicate
   resolver design.
