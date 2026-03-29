from __future__ import annotations

from dataclasses import dataclass

from events.bus import EventBus
from models.semantic import SemanticMemory
from stores.semantic_store import SemanticStore


@dataclass(frozen=True, slots=True)
class ContradictionCandidate:
    record: SemanticMemory
    similarity: float


class ContradictionDetector:
    """Retrieve contradiction candidates and persist confirmed supersession links."""

    def __init__(self, store: SemanticStore, event_bus: EventBus | None = None):
        self._store = store
        self._event_bus = event_bus

    def find_potential_contradictions(
        self,
        new_record: SemanticMemory,
        threshold: float = 0.85,
        top_k: int = 5,
    ) -> list[ContradictionCandidate]:
        stored_record = self._store.get_by_id(new_record.id)
        if stored_record is None:
            raise ValueError(f"Semantic record '{new_record.id}' must be stored before contradiction lookup")
        if stored_record.embedding is None:
            raise ValueError(f"Semantic record '{new_record.id}' is missing an embedding")

        raw_results = self._store.retrieve_by_vector(stored_record.embedding, top_k=max(1, top_k + 1))
        candidates = [
            ContradictionCandidate(record=record, similarity=similarity)
            for record, similarity in raw_results
            if record.id != stored_record.id and similarity >= threshold
        ][:top_k]

        if candidates and self._event_bus is not None:
            self._event_bus.emit(
                "memory.contradiction_flagged",
                {
                    "record_id": stored_record.id,
                    "memory_type": stored_record.memory_type,
                    "threshold": threshold,
                    "candidate_count": len(candidates),
                    "candidate_ids": [candidate.record.id for candidate in candidates],
                    "top_similarity": candidates[0].similarity,
                },
            )
        return candidates

    def resolve_supersession(
        self,
        superseded_id: str,
        kept_id: str,
    ) -> None:
        superseded_record = self._store.get_by_id(superseded_id)
        if superseded_record is None:
            raise ValueError(f"Semantic record '{superseded_id}' does not exist")

        kept_record = self._store.get_by_id(kept_id)
        if kept_record is None:
            raise ValueError(f"Semantic record '{kept_id}' does not exist")

        superseded_record.importance = 0.0
        superseded_record.superseded_by = kept_id
        self._store.replace(superseded_record)

        # `supersedes` remains a single pointer for now, so the latest confirmed
        # supersession wins on the kept record. This preserves the canonical
        # immediate predecessor link, but it does not represent full chains.
        kept_record.supersedes = superseded_id
        self._store.replace(kept_record)

        if self._event_bus is not None:
            self._event_bus.emit(
                "memory.supersession_resolved",
                {
                    "superseded_id": superseded_id,
                    "kept_id": kept_id,
                    "memory_type": "semantic",
                },
            )

    def find_likely_duplicates_batch(
        self,
        threshold: float = 0.95,
    ) -> list[tuple[str, str, float]]:
        """Return likely duplicate semantic pairs.

        This performs one top-k vector query per stored record, so the current
        implementation scales linearly with the number of semantic memories.
        """
        pairs: list[tuple[str, str, float]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for record in self._store.get_all_records(include_embeddings=True):
            if record.embedding is None:
                continue

            matches = self._store.retrieve_by_vector(record.embedding, top_k=2)
            best_other = next((match for match in matches if match[0].id != record.id), None)
            if best_other is None:
                continue

            other, similarity = best_other
            if similarity < threshold:
                continue

            pair_ids = tuple(sorted((record.id, other.id)))
            if pair_ids in seen_pairs:
                continue

            seen_pairs.add(pair_ids)
            pairs.append((pair_ids[0], pair_ids[1], similarity))

        pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
        return pairs
