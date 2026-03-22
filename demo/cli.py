import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.semantic import SemanticMemory
from stores.semantic_store import SemanticStore
from retrieval.retriever import UnifiedRetriever


def _make_retriever() -> UnifiedRetriever:
    return UnifiedRetriever(stores={"semantic": SemanticStore()})


def cmd_store(args):
    store = SemanticStore()
    record = SemanticMemory(content=args.content)
    record_id = store.store(record)
    print(f"Stored [{record_id[:8]}]: {args.content}")


def cmd_query(args):
    retriever = _make_retriever()
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


def main():
    parser = argparse.ArgumentParser(description="Agentic Memory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    store_p = sub.add_parser("store", help="Store a new memory")
    store_p.add_argument("content", type=str, help="Text content to store")

    query_p = sub.add_parser("query", help="Search memories by meaning")
    query_p.add_argument("query", type=str, help="Natural language query")
    query_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if args.command == "store":
        cmd_store(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
