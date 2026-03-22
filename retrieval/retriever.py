from datetime import datetime, timezone

from stores.base import BaseStore
from retrieval.ranking import RankedResult, rank_results

CANDIDATE_MULTIPLIER = 3  # over-fetch from each store to give the reranker room


class UnifiedRetriever:
    """Queries across memory stores, applies weighted ranking, and tracks access.

    Currently wraps SemanticStore only. When episodic and procedural stores
    are added, they join the _stores dict and fan-out happens automatically.
    """

    def __init__(self, stores: dict[str, BaseStore]):
        self._stores = stores

    def query(
        self,
        text: str,
        top_k: int = 5,
        memory_types: list[str] | None = None,
        relevance_weight: float = 0.4,
        recency_weight: float = 0.3,
        importance_weight: float = 0.3,
    ) -> list[RankedResult]:
        # ── fan-out: query matching stores ──────────────────────────────────
        targets = self._stores
        if memory_types:
            targets = {k: v for k, v in self._stores.items() if k in memory_types}

        # Over-fetch so recency/importance can rescue items outside the
        # initial similarity slice.
        fetch_k = top_k * CANDIDATE_MULTIPLIER

        all_results = []
        for store in targets.values():
            all_results.extend(store.retrieve(text, top_k=fetch_k))

        # ── rank across all stores ──────────────────────────────────────────
        ranked = rank_results(
            all_results,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
        )

        final = ranked[:top_k]

        # ── update access tracking on returned records ──────────────────────
        now = datetime.now(timezone.utc)
        for r in final:
            store = self._stores.get(r.record.memory_type)
            if store:
                store.update_access(r.record.id)
                # Keep in-memory record in sync so callers see current values
                r.record.access_count += 1
                r.record.last_accessed_at = now

        return final
