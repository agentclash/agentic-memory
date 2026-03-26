import argparse
import mimetypes
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events import ConsoleLogger, EventBus
from models.episodic import EpisodicMemory
from models.semantic import SemanticMemory
from stores.episodic_store import EpisodicStore
from stores.semantic_store import SemanticStore
from retrieval.retriever import UnifiedRetriever

_DEFAULT_MIME_TYPES = {
    "audio": "audio/mpeg",
    "image": "image/png",
    "video": "video/mp4",
    "pdf": "application/pdf",
}


def _make_bus() -> EventBus:
    bus = EventBus()
    ConsoleLogger().register(bus)
    return bus


def _make_semantic_store(event_bus: EventBus | None = None) -> SemanticStore:
    return SemanticStore(event_bus=event_bus)


def _make_episodic_store(event_bus: EventBus | None = None) -> EpisodicStore:
    return EpisodicStore(event_bus=event_bus)


def _make_retriever(event_bus: EventBus | None = None) -> UnifiedRetriever:
    return UnifiedRetriever(
        stores={
            "semantic": _make_semantic_store(event_bus=event_bus),
            "episodic": _make_episodic_store(event_bus=event_bus),
        },
        event_bus=event_bus,
    )


def _guess_mime_type(path: str, modality: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or _DEFAULT_MIME_TYPES[modality]


def _default_episode_content(file_path: str, modality: str) -> str:
    return f"{modality} episode from {Path(file_path).name}"


def cmd_store(args):
    bus = _make_bus()
    store = _make_semantic_store(event_bus=bus)
    record = SemanticMemory(content=args.content)
    record_id = store.store(record)
    print(f"Stored [{record_id[:8]}]: {args.content}")


def cmd_store_episode(args):
    bus = _make_bus()
    store = _make_episodic_store(event_bus=bus)

    if args.text is not None:
        record = EpisodicMemory(
            content=args.text,
            session_id=args.session,
        )
    else:
        media_ref = os.path.abspath(args.file)
        content = args.content or _default_episode_content(media_ref, args.modality)
        record = EpisodicMemory(
            content=content,
            session_id=args.session,
            modality=args.modality,
            media_ref=media_ref,
            source_mime_type=_guess_mime_type(media_ref, args.modality),
        )

    record_id = store.store(record)
    print(f"Stored episode [{record_id[:8]}]: {record.content}")


def cmd_query(args):
    bus = _make_bus()
    retriever = _make_retriever(event_bus=bus)
    results = retriever.query(args.query, top_k=args.top_k)
    if not results:
        print("No results found.")
        return
    for rank, r in enumerate(results, 1):
        age = r.record.created_at.strftime("%Y-%m-%d %H:%M")
        print(
            f"  {rank}. [{r.final_score:.4f}] {r.record.content}\n"
            f"     type={r.record.memory_type}  stored={age}  "
            f"accessed={r.record.access_count}x  "
            f"sim={r.raw_similarity:.4f}  rec={r.recency_score:.4f}  imp={r.importance_score:.2f}"
        )


def cmd_recent(args):
    bus = _make_bus()
    retriever = _make_retriever(event_bus=bus)
    results = retriever.query_recent(args.n)
    if not results:
        print("No recent episodes found.")
        return

    for rank, record in enumerate(results, 1):
        age = record.created_at.strftime("%Y-%m-%d %H:%M")
        media_display = f"  media={record.media_ref}" if record.media_ref else ""
        print(
            f"  {rank}. {record.content}\n"
            f"     type={record.memory_type}  session={record.session_id}  "
            f"modality={record.modality}  stored={age}  accessed={record.access_count}x{media_display}"
        )


def main():
    parser = argparse.ArgumentParser(description="Agentic Memory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    store_p = sub.add_parser("store", help="Store a new memory")
    store_p.add_argument("content", type=str, help="Text content to store")

    query_p = sub.add_parser("query", help="Search memories by meaning")
    query_p.add_argument("query", type=str, help="Natural language query")
    query_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")

    episode_p = sub.add_parser("store-episode", help="Store a new episodic memory")
    episode_p.add_argument("--session", required=True, help="Session identifier")
    episode_src = episode_p.add_mutually_exclusive_group(required=True)
    episode_src.add_argument("--text", help="Text content for the episode")
    episode_src.add_argument("--file", help="Path to media file for the episode")
    episode_p.add_argument(
        "--modality",
        choices=["audio", "image", "video", "pdf"],
        help="Media modality for file-backed episodes",
    )
    episode_p.add_argument(
        "--content",
        help="Optional human-readable description for file-backed episodes",
    )

    recent_p = sub.add_parser("recent", help="Show recent episodic memories")
    recent_p.add_argument("n", type=int, help="Number of recent episodes to show")

    args = parser.parse_args()

    if args.command == "store":
        cmd_store(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "store-episode":
        if args.file and not args.modality:
            parser.error("--modality is required when using --file")
        cmd_store_episode(args)
    elif args.command == "recent":
        cmd_recent(args)


if __name__ == "__main__":
    main()
