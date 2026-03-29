"""Verify the shared runtime factory isolates Chroma paths and config state."""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.semantic import SemanticMemory
from runtime import build_runtime
from tests.helpers import HashingEmbedder


def _make_chroma_dir(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=prefix)


def test_build_runtime_does_not_mutate_global_chroma_path():
    original_path = config.CHROMA_DB_PATH
    runtime_path = _make_chroma_dir("runtime_factory_primary_")

    try:
        runtime = build_runtime(
            chroma_path=runtime_path,
            embedder=HashingEmbedder(dimensions=config.EMBEDDING_DIMENSIONS),
        )
        runtime.semantic_store.store(SemanticMemory(content="Runtime-scoped fact"))
        records = runtime.semantic_store.get_all_records()

        assert config.CHROMA_DB_PATH == original_path
        assert len(records) == 1
        assert runtime.semantic_store.get_by_id(records[0].id) is not None
    finally:
        shutil.rmtree(runtime_path, ignore_errors=True)


def test_build_runtime_keeps_chroma_paths_isolated():
    original_path = config.CHROMA_DB_PATH
    first_path = _make_chroma_dir("runtime_factory_first_")
    second_path = _make_chroma_dir("runtime_factory_second_")

    try:
        first_runtime = build_runtime(
            chroma_path=first_path,
            embedder=HashingEmbedder(dimensions=config.EMBEDDING_DIMENSIONS),
        )
        second_runtime = build_runtime(
            chroma_path=second_path,
            embedder=HashingEmbedder(dimensions=config.EMBEDDING_DIMENSIONS),
        )

        first_runtime.semantic_store.store(SemanticMemory(content="Only in first runtime"))
        second_runtime.semantic_store.store(SemanticMemory(content="Only in second runtime"))

        first_records = {record.content for record in first_runtime.semantic_store.get_all_records()}
        second_records = {record.content for record in second_runtime.semantic_store.get_all_records()}

        assert first_records == {"Only in first runtime"}
        assert second_records == {"Only in second runtime"}
        assert config.CHROMA_DB_PATH == original_path
    finally:
        shutil.rmtree(first_path, ignore_errors=True)
        shutil.rmtree(second_path, ignore_errors=True)


def test_build_runtime_uses_custom_embedding_dimensions_for_vector_validation():
    runtime_path = _make_chroma_dir("runtime_factory_dimensions_")

    try:
        runtime = build_runtime(
            chroma_path=runtime_path,
            embedding_dimensions=8,
            embedder=HashingEmbedder(dimensions=8),
        )

        runtime.retriever.query_by_vector([0.0] * 8, top_k=1)

        try:
            runtime.retriever.query_by_vector([0.0] * 7, top_k=1)
            raise AssertionError("Expected dimension mismatch")
        except ValueError as exc:
            assert "Expected query vector dimension 8" in str(exc)
    finally:
        shutil.rmtree(runtime_path, ignore_errors=True)


def test_build_runtime_rejects_mismatched_embedder_and_dimension_override():
    runtime_path = _make_chroma_dir("runtime_factory_dimension_mismatch_")

    try:
        try:
            build_runtime(
                chroma_path=runtime_path,
                embedding_dimensions=16,
                embedder=HashingEmbedder(dimensions=8),
            )
            raise AssertionError("Expected mismatched dimensions to raise")
        except ValueError as exc:
            assert "Provided embedder dimensions do not match embedding_dimensions" in str(exc)
    finally:
        shutil.rmtree(runtime_path, ignore_errors=True)
