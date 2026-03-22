from stores.base import BaseStore
from retrieval.ranking import RankedResult, rank_results


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
        # ── fan-out: query matching stores in parallel (sequential for now) ─
        targets = self._stores
        if memory_types:
            targets = {k: v for k, v in self._stores.items() if k in memory_types}

        all_results = []
        for store in targets.values():
            all_results.extend(store.retrieve(text, top_k=top_k))

        # ── rank across all stores ──────────────────────────────────────────
        ranked = rank_results(
            all_results,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
        )

        # ── update access tracking on returned records ──────────────────────
        for r in ranked[:top_k]:
            store = self._stores.get(r.record.memory_type)
            if store:
                store.update_access(r.record.id)

        return ranked[:top_k]
