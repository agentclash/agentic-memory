# Event Bus Phase 1 Test Plan

## Purpose

This document is the release gate for the event-bus work from issue `#4`.
Implementation starts only after the expected behavior, scope boundaries, and
verification steps are written down.

## Scope For This PR

This PR adds a synchronous, optional event bus to the current Phase 1 codebase.

Included:

- `events/bus.py` with `MemoryEvent` and `EventBus`
- `events/logger.py` with a console subscriber for debugging
- Event emission from `SemanticStore.store()`
- Event emission from `UnifiedRetriever.query()`
- CLI wiring so the demo path exercises the bus
- Automated tests for the bus and the event-emission contract

Explicitly not included:

- Asynchronous dispatch or background queues
- `EpisodicStore`, `ProceduralStore`, or `ForgettingService`
- Persistence or replay of events
- Wildcard subscriptions like `memory.*`

## Corrections To Issue Plan

The issue is directionally correct, but the implementation plan needs these
adjustments to match the repo and avoid over-promising:

1. Dispatch must be synchronous in this PR.
The issue text says subscribers run asynchronously, but the design constraints in
the same issue say Phase 1 is synchronous. This PR follows the design
constraints, not the earlier asynchronous wording.

2. The base store constructor change must stay optional and non-breaking.
`stores/base.py` is currently an abstract method interface only. If we touch it,
the change must remain a convenience layer for concrete stores, not a new hard
requirement for callers or future store implementations.

3. Event payloads must reflect current observable behavior only.
This repo has one real store today (`semantic`). Payloads and tests should be
written around current fields and current query stages, not future services.

4. Automated tests must not depend on the Gemini API.
All tests added in this PR need deterministic fakes/stubs so the PR can be
validated offline and repeatedly.

## Expected Behavior

### `memory.stored`

Emitted once after a semantic memory is successfully persisted.

Expected payload:

- `record_id`
- `memory_type`
- `content`
- `modality`
- `importance`

### `memory.retrieved`

Emitted once per `UnifiedRetriever.query()` call after raw retrieval fan-out is
complete and before ranking.

Expected payload:

- `query`
- `memory_types`
- `candidate_count`
- `top_similarity`

Behavior notes:

- `memory_types` must reflect the stores actually queried.
- `candidate_count` is the total number of raw candidates returned across all
  queried stores.
- `top_similarity` is the highest raw similarity across all candidates, or
  `None` when there are no candidates.

### `memory.ranked`

Emitted once per `UnifiedRetriever.query()` call after ranking and before the
method returns.

Expected payload:

- `query`
- `results` for the final `top_k` ranked results
- `weights`

Each result entry must include:

- `record_id`
- `content`
- `final_score`
- `raw_similarity`
- `recency_score`
- `importance_score`

### `memory.accessed`

Emitted once per returned record after access tracking is updated.

Expected payload:

- `record_id`
- `memory_type`
- `access_count`

Behavior notes:

- The emitted `access_count` must be the post-increment value.
- No `memory.accessed` event should fire for records that are not returned in the
  final ranked output.

## Automated Test Matrix

### Event bus core

1. Subscriber receives emitted event with correct `event_type`, timestamp, and
   data snapshot.
2. Multiple subscribers to the same event are all called.
3. Emitting an event with no subscribers is a no-op.
4. Subscribers only receive the event names they registered for.
5. Event data exposed to subscribers is isolated from later emitter-side
   mutations.

### Store integration

1. `SemanticStore.store()` emits exactly one `memory.stored` event on success.
2. The emitted payload matches the stored record metadata.
3. No event is emitted if persistence fails before the write completes.

### Retriever integration

1. `UnifiedRetriever.query()` emits `memory.retrieved` once with correct
   candidate summary.
2. `UnifiedRetriever.query()` emits `memory.ranked` once with the returned
   ranked results and configured weights.
3. `UnifiedRetriever.query()` emits one `memory.accessed` event per returned
   record after incrementing access count.
4. Querying with no hits still emits `memory.retrieved` and `memory.ranked`, but
   emits no `memory.accessed`.
5. Filtering with `memory_types` reports only the queried stores.

## Manual Verification

These checks happen only after automated tests pass:

1. Run `python demo/cli.py store "Python was created by Guido van Rossum"` and
   confirm the CLI still stores the record and logs `memory.stored`.
2. Run `python demo/cli.py query "Who created Python?"` and confirm event logs
   appear in this order:
   `memory.retrieved` -> `memory.ranked` -> one or more `memory.accessed`.
3. Confirm normal CLI output still appears after event logging and remains
   readable.

## Exit Criteria

The PR can be opened only when all of the following are true:

1. `testing.md` remains accurate to the shipped implementation.
2. New automated tests pass locally.
3. Existing automated tests still pass, or any pre-existing unrelated failures
   are identified explicitly.
4. Manual CLI verification is completed.
5. The PR description documents the issue-plan corrections, especially the
   synchronous Phase 1 dispatch choice.
