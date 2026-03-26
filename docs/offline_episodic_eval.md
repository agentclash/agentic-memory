# Offline Episodic-Memory Evaluation Harness

Issue `#8` adds a deterministic offline harness for episodic-memory behavior that
can run in CI or local development without provider APIs.

## What It Covers

`tests/test_offline_episodic_eval.py` runs fixed synthetic fixtures for:

- mixed semantic + episodic retrieval
- temporal recall over an exact time window
- session reconstruction with session separation
- recent-event lookup across sessions
- cross-modal retrieval using deterministic media-byte fixtures
- a negative temporal case where no episodic result should be returned

Every scenario asserts exact expected outputs. There is no probabilistic scoring
and no dependence on remote embedding providers.

## How To Run It

```bash
pytest tests/test_offline_episodic_eval.py
```

or as a standalone pass/fail harness:

```bash
python tests/test_offline_episodic_eval.py
```

## Why This Exists Before LongMemEval / LoCoMo

This harness is a regression gate for core memory mechanics, not a benchmark
replacement.

- The mixed-store scenario checks that semantic and episodic memories can
  compete in the same retrieval path before larger task-level evaluations are
  introduced.
- The temporal and recent-event scenarios exercise the exact chronology logic
  that later long-context benchmarks depend on.
- The session reconstruction scenario is a small deterministic analogue of
  dialogue-history and episode-boundary tasks that show up in LoCoMo-style evals.
- The cross-modal scenario verifies that the storage and retrieval plumbing can
  support media-backed episodes even when we are still using offline stubs.
- The negative scenario protects precision by asserting that the system can
  return no episodic answer when the requested time window is empty.

## Current Limits

- Fixtures are synthetic and small.
- Assertions are exact-match pass/fail checks, not benchmark scores.
- Cross-modal behavior is verified with a deterministic test stub, not a real
  multimodal model.

That tradeoff is intentional: this layer should fail fast on regressions in
memory mechanics before the project grows into larger LongMemEval or LoCoMo
style datasets and scoring pipelines.
