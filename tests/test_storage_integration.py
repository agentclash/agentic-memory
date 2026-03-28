"""Verify all three stores survive re-instantiation on one Chroma path."""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.episodic import EpisodicMemory
from models.procedural import ProceduralMemory
from models.semantic import SemanticMemory
from stores.episodic_store import EpisodicStore
from stores.procedural_store import ProceduralStore
from stores.semantic_store import SemanticStore
from tests.helpers import HashingEmbedder


def test_all_three_collections_survive_store_reinitialization():
    db_path = tempfile.mkdtemp(prefix="chroma_test_storage_integration_")
    original_db_path = config.CHROMA_DB_PATH
    config.CHROMA_DB_PATH = db_path

    try:
        semantic_store = SemanticStore(embedder=HashingEmbedder())
        episodic_store = EpisodicStore(embedder=HashingEmbedder())
        procedural_store = ProceduralStore(embedder=HashingEmbedder())

        semantic_id = semantic_store.store(SemanticMemory(content="Semantic fact about Docker"))
        episodic_id = episodic_store.store(
            EpisodicMemory(
                content="We debugged Docker networking in a session",
                session_id="session-docker",
            )
        )
        procedural_id = procedural_store.store(
            ProceduralMemory(
                content="Deploy with Docker Compose",
                steps=["Build the image", "Run docker compose up"],
            )
        )

        semantic_store = SemanticStore(embedder=HashingEmbedder())
        episodic_store = EpisodicStore(embedder=HashingEmbedder())
        procedural_store = ProceduralStore(embedder=HashingEmbedder())

        assert semantic_store.get_by_id(semantic_id) is not None
        assert episodic_store.get_by_id(episodic_id) is not None
        assert procedural_store.get_by_id(procedural_id) is not None
        print("  PASS  semantic, episodic, and procedural collections survive Chroma re-init")
    finally:
        config.CHROMA_DB_PATH = original_db_path
        shutil.rmtree(db_path, ignore_errors=True)
