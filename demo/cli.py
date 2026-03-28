import argparse
import mimetypes
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events import ConsoleLogger, EventBus
from models.episodic import EpisodicMemory
from models.procedural import ProceduralMemory
from models.semantic import SemanticMemory
from stores.episodic_store import EpisodicStore
from stores.media_store import MediaStore
from stores.procedural_store import ProceduralStore
from stores.semantic_store import SemanticStore
from retrieval.retriever import UnifiedRetriever
from utils.embeddings import GeminiEmbedder, TextEmbedder
import config

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
    return SemanticStore(event_bus=event_bus, media_store=_make_media_store())


def _make_episodic_store(event_bus: EventBus | None = None) -> EpisodicStore:
    return EpisodicStore(event_bus=event_bus)


def _make_procedural_store(event_bus: EventBus | None = None) -> ProceduralStore:
    return ProceduralStore(event_bus=event_bus, media_store=_make_media_store())


def _make_media_store() -> MediaStore:
    return MediaStore(config.MEDIA_STORAGE_PATH)


def _make_embedder() -> TextEmbedder:
    return GeminiEmbedder()


def _make_retriever(event_bus: EventBus | None = None) -> UnifiedRetriever:
    return UnifiedRetriever(
        stores={
            "semantic": _make_semantic_store(event_bus=event_bus),
            "episodic": _make_episodic_store(event_bus=event_bus),
            "procedural": _make_procedural_store(event_bus=event_bus),
        },
        event_bus=event_bus,
    )


def _guess_mime_type(path: str, modality: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    if guessed:
        return guessed
    if modality == "multimodal":
        media_type = _infer_media_type(path)
        return _DEFAULT_MIME_TYPES[media_type]
    return _DEFAULT_MIME_TYPES[modality]


def _default_episode_content(file_path: str, modality: str) -> str:
    return f"{modality} episode from {Path(file_path).name}"


def _exit_with_error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _infer_media_type(path: str) -> str:
    return MediaStore.resolve_media_type(path)


def _infer_file_contract(path: str) -> tuple[str, str]:
    media_type = _infer_media_type(path)
    modality = "multimodal" if media_type == "pdf" else media_type
    return modality, media_type


def _print_ranked_results(results) -> None:
    for rank, result in enumerate(results, 1):
        age = result.record.created_at.strftime("%Y-%m-%d %H:%M")
        media_bits = []
        if result.record.modality != "text":
            media_bits.append(f"modality={result.record.modality}")
        if result.record.media_type:
            media_bits.append(f"media_type={result.record.media_type}")
        if result.record.media_ref:
            media_bits.append(f"media={result.record.media_ref}")
        media_context = f"  {'  '.join(media_bits)}" if media_bits else ""
        print(
            f"  {rank}. [{result.final_score:.4f}] {result.record.content}\n"
            f"     type={result.record.memory_type}  stored={age}  "
            f"accessed={result.record.access_count}x  "
            f"sim={result.raw_similarity:.4f}  rec={result.recency_score:.4f}  "
            f"imp={result.importance_score:.2f}{media_context}"
        )


def _print_best_procedures(results) -> None:
    for rank, result in enumerate(results, 1):
        print(
            f"  {rank}. [{result['combined_score']:.4f}] {result['record'].content}\n"
            f"     similarity={result['similarity']:.4f}  "
            f"wilson={result['wilson_score']:.4f}  "
            f"success={result['record'].success_count}  "
            f"failure={result['record'].failure_count}"
        )


def _query_by_media(args, *, modality: str) -> None:
    bus = _make_bus()
    embedder = _make_embedder()
    retriever = _make_retriever(event_bus=bus)
    source_path = os.path.abspath(args.path)
    mime_type = _guess_mime_type(source_path, modality)
    if modality == "image":
        vector = embedder.embed_image(source_path, mime_type=mime_type)
    elif modality == "audio":
        vector = embedder.embed_audio(source_path, mime_type=mime_type)
    else:
        raise ValueError(f"Unsupported query modality: {modality}")

    results = retriever.query_by_vector(
        vector,
        top_k=args.top_k,
        memory_types=args.memory_types,
        metadata={"source_modality": modality},
    )
    if not results:
        print("No results found.")
        return
    _print_ranked_results(results)


def cmd_store(args):
    bus = _make_bus()
    store = _make_semantic_store(event_bus=bus)
    media_path = None
    modality = "text"
    if args.image:
        media_path = os.path.abspath(args.image)
        modality = "image"
    elif args.audio:
        media_path = os.path.abspath(args.audio)
        modality = "audio"

    record = SemanticMemory(
        content=args.content,
        modality=modality,
        media_ref=media_path,
        media_type=_infer_media_type(media_path) if media_path else None,
    )
    try:
        record_id = store.store(record)
    except (FileNotFoundError, ValueError) as exc:
        _exit_with_error(str(exc))
    print(f"Stored [{record_id[:8]}]: {args.content}")


def cmd_store_episode(args):
    bus = _make_bus()
    store = _make_episodic_store(event_bus=bus)
    media_store = _make_media_store() if args.text is None else None

    if args.text is not None:
        record = EpisodicMemory(
            content=args.text,
            session_id=args.session,
        )
    else:
        source_path = os.path.abspath(args.file)
        media_type = args.media_type
        if args.modality == "multimodal" and media_type is None:
            media_type = _infer_media_type(source_path)
        record = EpisodicMemory(
            content=args.content or _default_episode_content(source_path, args.modality),
            session_id=args.session,
            modality=args.modality,
            media_type=media_type,
            source_mime_type=_guess_mime_type(source_path, args.modality),
        )
        media_ref = media_store.store(source_path, record.id)
        record.media_ref = media_ref

    try:
        record_id = store.store(record)
    except Exception:
        if media_store is not None and record.media_ref:
            media_store.delete(record.media_ref)
        raise
    print(f"Stored episode [{record_id[:8]}]: {record.content}")


def cmd_query(args):
    bus = _make_bus()
    retriever = _make_retriever(event_bus=bus)
    results = retriever.query(args.query, top_k=args.top_k)
    if not results:
        print("No results found.")
        return
    _print_ranked_results(results)


def cmd_store_procedure(args):
    bus = _make_bus()
    store = _make_procedural_store(event_bus=bus)
    media_store = _make_media_store() if args.file else None

    modality = "text"
    media_type = None
    if args.file:
        source_path = os.path.abspath(args.file)
        inferred_modality, inferred_media_type = _infer_file_contract(source_path)
        modality = args.modality or inferred_modality
        media_type = args.media_type or inferred_media_type
        if args.modality and inferred_modality != args.modality and args.modality != "multimodal":
            _exit_with_error("uploaded file does not match the requested modality")

    record = ProceduralMemory(
        content=args.content,
        steps=args.steps,
        preconditions=args.preconditions or [],
        modality=modality,
        media_type=media_type,
        text_description=args.text_description,
    )
    if args.file:
        record.media_ref = media_store.store(os.path.abspath(args.file), record.id)

    try:
        record_id = store.store(record)
    except Exception:
        if media_store is not None and record.media_ref:
            media_store.delete(record.media_ref)
        raise
    print(f"Stored procedure [{record_id[:8]}]: {record.content}")


def cmd_record_outcome(args):
    bus = _make_bus()
    store = _make_procedural_store(event_bus=bus)
    success = bool(args.success)
    store.record_outcome(args.record_id, success)
    record = store.get_by_id(args.record_id)
    if record is None:
        print(f"Procedure not found: {args.record_id}")
        return
    print(
        f"Updated [{record.id[:8]}]: success={record.success_count} "
        f"failure={record.failure_count} wilson={record.wilson_score:.4f}"
    )


def cmd_best_procedure(args):
    bus = _make_bus()
    store = _make_procedural_store(event_bus=bus)
    results = store.get_best_procedure_matches(args.task, top_k=args.top_k)
    if not results:
        print("No procedures found.")
        return
    _print_best_procedures(
        [
            {
                "record": match.record,
                "similarity": match.similarity,
                "wilson_score": match.wilson_score,
                "combined_score": match.combined_score,
            }
            for match in results
        ]
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
    store_media = store_p.add_mutually_exclusive_group()
    store_media.add_argument("--image", help="Path to an image file for semantic storage")
    store_media.add_argument("--audio", help="Path to an audio file for semantic storage")

    query_p = sub.add_parser("query", help="Search memories by meaning")
    query_p.add_argument("query", type=str, help="Natural language query")
    query_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")

    query_image_p = sub.add_parser("query-by-image", help="Search memories using an image query")
    query_image_p.add_argument("path", help="Path to the image file")
    query_image_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    query_image_p.add_argument(
        "--memory-types",
        nargs="+",
        choices=["semantic", "episodic", "procedural"],
        help="Optional memory type filter",
    )

    query_audio_p = sub.add_parser("query-by-audio", help="Search memories using an audio query")
    query_audio_p.add_argument("path", help="Path to the audio file")
    query_audio_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    query_audio_p.add_argument(
        "--memory-types",
        nargs="+",
        choices=["semantic", "episodic", "procedural"],
        help="Optional memory type filter",
    )

    episode_p = sub.add_parser("store-episode", help="Store a new episodic memory")
    episode_p.add_argument("--session", required=True, help="Session identifier")
    episode_src = episode_p.add_mutually_exclusive_group(required=True)
    episode_src.add_argument("--text", help="Text content for the episode")
    episode_src.add_argument("--file", help="Path to media file for the episode")
    episode_p.add_argument(
        "--modality",
        choices=["audio", "image", "video", "pdf", "multimodal"],
        help="Media modality for file-backed episodes",
    )
    episode_p.add_argument(
        "--media-type",
        choices=["image", "audio", "video", "pdf"],
        help="Optional media type override for multimodal file-backed episodes",
    )
    episode_p.add_argument(
        "--content",
        help="Optional human-readable description for file-backed episodes",
    )

    procedure_p = sub.add_parser("store-procedure", help="Store a new procedural memory")
    procedure_p.add_argument("content", help="Task description for the procedure")
    procedure_p.add_argument("--steps", nargs="+", required=True, help="Ordered procedural steps")
    procedure_p.add_argument("--preconditions", nargs="+", help="Optional preconditions")
    procedure_p.add_argument("--file", help="Optional supporting media file")
    procedure_p.add_argument(
        "--modality",
        choices=["audio", "image", "video", "multimodal"],
        help="Optional modality override for file-backed procedures",
    )
    procedure_p.add_argument(
        "--media-type",
        choices=["image", "audio", "video", "pdf"],
        help="Optional media type override for multimodal file-backed procedures",
    )
    procedure_p.add_argument(
        "--text-description",
        help="Optional description of what the supporting media shows",
    )

    outcome_p = sub.add_parser("record-outcome", help="Record a procedural outcome")
    outcome_p.add_argument("record_id", help="Procedural memory id")
    outcome_group = outcome_p.add_mutually_exclusive_group(required=True)
    outcome_group.add_argument("--success", action="store_true", help="Record a successful outcome")
    outcome_group.add_argument("--failure", action="store_true", help="Record a failed outcome")

    best_p = sub.add_parser("best-procedure", help="Retrieve the best procedures for a task")
    best_p.add_argument("task", help="Task description")
    best_p.add_argument("-k", "--top-k", type=int, default=3, help="Number of results")

    recent_p = sub.add_parser("recent", help="Show recent episodic memories")
    recent_p.add_argument("n", type=int, help="Number of recent episodes to show")

    args = parser.parse_args()

    if args.command == "store":
        cmd_store(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "query-by-image":
        _query_by_media(args, modality="image")
    elif args.command == "query-by-audio":
        _query_by_media(args, modality="audio")
    elif args.command == "store-episode":
        if args.file and not args.modality:
            parser.error("--modality is required when using --file")
        if args.text and args.media_type:
            parser.error("--media-type is only valid when using --file")
        if args.modality != "multimodal" and args.media_type:
            parser.error("--media-type is only supported when --modality multimodal")
        cmd_store_episode(args)
    elif args.command == "store-procedure":
        if args.media_type and not args.file:
            parser.error("--media-type is only valid when using --file")
        if args.modality and not args.file:
            parser.error("--modality is only valid when using --file")
        if args.text_description and not args.file:
            parser.error("--text-description is only valid when using --file")
        if args.modality != "multimodal" and args.media_type:
            parser.error("--media-type is only supported when --modality multimodal")
        cmd_store_procedure(args)
    elif args.command == "record-outcome":
        args.success = not args.failure
        cmd_record_outcome(args)
    elif args.command == "best-procedure":
        cmd_best_procedure(args)
    elif args.command == "recent":
        cmd_recent(args)


if __name__ == "__main__":
    main()
