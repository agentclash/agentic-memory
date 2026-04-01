"""
LongMemEval harness for agentic-memory.

This script evaluates agentic-memory on the LongMemEval dataset by ingesting
timestamped history sessions into the memory runtime, retrieving relevant
memories for each benchmark question, generating an answer with an OpenAI
reader model, and optionally scoring that answer with an OpenAI judge model.

Usage:
    # Oracle smoke test (fastest)
    .venv/bin/python benchmarks/longmemeval_eval.py --dataset oracle --limit 5

    # MCP-backed run using Gemini Embedding 1
    .venv/bin/python benchmarks/longmemeval_eval.py --dataset oracle --backend mcp \
        --embedding-model gemini-embedding-001 --limit 20

    # Runtime-backed run with semantic fact extraction enabled
    .venv/bin/python benchmarks/longmemeval_eval.py --dataset s --backend runtime \
        --embedding-model gemini-embedding-001 --extract-facts --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv

from benchmarks.locomo_eval import MCPBackend, RuntimeBackend, generate_answer, judge_answer

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = DATA_DIR / "longmemeval_results"

DATASETS = {
    "oracle": {
        "filename": "longmemeval_oracle.json",
        "url": (
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
            "resolve/main/longmemeval_oracle.json"
        ),
    },
    "s": {
        "filename": "longmemeval_s_cleaned.json",
        "url": (
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
            "resolve/main/longmemeval_s_cleaned.json"
        ),
    },
    "m": {
        "filename": "longmemeval_m_cleaned.json",
        "url": (
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
            "resolve/main/longmemeval_m_cleaned.json"
        ),
    },
}

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
EXTRACT_SYSTEM = """\
You are extracting durable facts from one timestamped chat session.

Return ONLY a JSON array of strings. Each string should be a standalone fact
that is likely to matter later in a memory benchmark.

Rules:
- include names, preferences, plans, commitments, dates, locations, and updates
- include who the fact is about
- preserve temporal qualifiers when present
- skip greetings, filler, and obvious repetition
- if there are no durable facts, return []
"""

load_dotenv()


def ensure_dataset(dataset_name: str) -> Path:
    spec = DATASETS[dataset_name]
    path = DATA_DIR / spec["filename"]
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-sL", spec["url"], "-o", str(path)])
    return path


def load_dataset(dataset_name: str) -> list[dict[str, Any]]:
    with open(ensure_dataset(dataset_name)) as f:
        return json.load(f)


def parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    normalized = raw.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def iter_history_sessions(instance: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = []
    raw_ids = instance.get("haystack_session_ids") or []
    raw_dates = instance.get("haystack_dates") or []
    raw_sessions = instance.get("haystack_sessions") or []
    for index, turns in enumerate(raw_sessions):
        session_id = raw_ids[index] if index < len(raw_ids) else f"{instance['question_id']}_s{index + 1}"
        date = raw_dates[index] if index < len(raw_dates) else ""
        sessions.append(
            {
                "index": index,
                "session_id": str(session_id),
                "date": str(date),
                "timestamp": parse_timestamp(date),
                "turns": turns,
            }
        )
    sessions.sort(key=lambda session: (session["timestamp"] is None, session["timestamp"], session["index"]))
    return sessions


def format_session_text(session: dict[str, Any]) -> str:
    lines = []
    if session["date"]:
        lines.append(f"Session date: {session['date']}")
    lines.append(f"Session id: {session['session_id']}")
    lines.append("Transcript:")
    for turn in session["turns"]:
        role = str(turn.get("role", "unknown")).strip() or "unknown"
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        answer_marker = " [evidence]" if turn.get("has_answer") else ""
        lines.append(f"{role}{answer_marker}: {content}")
    return "\n".join(lines)


def count_turns(session: dict[str, Any]) -> int:
    return sum(1 for turn in session["turns"] if str(turn.get("content", "")).strip())


def normalize_answer(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    return lowered


def exact_match(generated: str, truth: str) -> int:
    return int(normalize_answer(generated) == normalize_answer(truth))


def extract_session_facts(
    client: OpenAI,
    model: str,
    session: dict[str, Any],
) -> list[str]:
    transcript = format_session_text(session)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        max_completion_tokens=500,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    text = raw
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    facts = []
    for fact in parsed:
        rendered = str(fact).strip()
        if not rendered:
            continue
        if session["date"]:
            rendered = f"[{session['date']}] {rendered}"
        facts.append(rendered)
    return facts


def output_paths(dataset_name: str, run_label: str) -> dict[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_name}_{run_label}"
    return {
        "hypotheses": RESULTS_DIR / f"{prefix}_hypotheses.jsonl",
        "results": RESULTS_DIR / f"{prefix}_results.json",
        "summary": RESULTS_DIR / f"{prefix}_summary.json",
    }


def build_run_label(backend_type: str, limit: int | None, judge_enabled: bool) -> str:
    parts = [backend_type, f"judge_{'on' if judge_enabled else 'off'}"]
    if limit is not None:
        parts.append(f"limit_{limit}")
    return "_".join(parts)


def _set_benchmark_embedding_env(embedding_model: str, embedding_dimensions: int) -> None:
    os.environ["EMBEDDING_MODEL"] = embedding_model
    os.environ["EMBEDDING_DIMENSIONS"] = str(embedding_dimensions)


async def run_longmemeval(
    *,
    dataset_name: str,
    backend_type: str,
    limit: int | None,
    top_k: int,
    embedding_model: str,
    embedding_dimensions: int,
    reader_model: str,
    judge_model: str | None,
    extract_facts_flag: bool,
    delay: float,
) -> dict[str, Any]:
    _set_benchmark_embedding_env(embedding_model, embedding_dimensions)

    dataset = load_dataset(dataset_name)
    if limit is not None:
        dataset = dataset[:limit]

    run_label = build_run_label(backend_type, limit, judge_model is not None)
    paths = output_paths(dataset_name, run_label)
    backend_cls = MCPBackend if backend_type == "mcp" else RuntimeBackend
    openai_client = OpenAI(timeout=60, max_retries=3)

    if paths["hypotheses"].exists():
        paths["hypotheses"].unlink()

    print("=== LongMemEval ===")
    print(f"  Dataset         : {dataset_name}")
    print(f"  Questions       : {len(dataset)}")
    print(f"  Backend         : {backend_type}")
    print(f"  Embedding model : {embedding_model}")
    print(f"  Dimensions      : {embedding_dimensions}")
    print(f"  Retrieval k     : {top_k}")
    print(f"  Fact extraction : {'on' if extract_facts_flag else 'off'}")
    print(f"  Reader model    : {reader_model}")
    print(f"  Judge model     : {judge_model or 'disabled'}")
    print()

    results_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_results: list[dict[str, Any]] = []
    total_sessions = 0
    total_facts = 0
    total_turns = 0
    total_retrieved = 0
    t0 = time.time()

    for index, instance in enumerate(dataset, 1):
        question_id = instance["question_id"]
        chroma_dir = tempfile.mkdtemp(prefix=f"longmemeval_{question_id}_")
        media_dir = tempfile.mkdtemp(prefix=f"longmemeval_media_{question_id}_")
        try:
            async with backend_cls(
                chroma_dir,
                media_dir,
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
            ) as backend:
                sessions = iter_history_sessions(instance)
                total_sessions += len(sessions)

                episode_records: list[dict[str, Any]] = []
                fact_contents: list[str] = []

                for session_idx, session in enumerate(sessions, 1):
                    session_text = format_session_text(session)
                    total_turns += count_turns(session)
                    episode_records.append(
                        {
                            "content": session_text,
                            "session_id": session["session_id"],
                            "turn_number": session_idx,
                            "participants": ["user", "assistant"],
                        }
                    )
                    if extract_facts_flag:
                        fact_contents.extend(
                            extract_session_facts(openai_client, reader_model, session)
                        )

                if isinstance(backend, RuntimeBackend):
                    backend.batch_store_episodes(episode_records)
                    if fact_contents:
                        backend.batch_store_facts(fact_contents)
                else:
                    for record in episode_records:
                        await backend.remember_episode(
                            session_id=record["session_id"],
                            text=record["content"],
                            turn_number=record["turn_number"],
                            participants=record["participants"],
                        )
                    for fact in fact_contents:
                        await backend.remember_fact(content=fact, category="extracted")
                total_facts += len(fact_contents)

                memories = await backend.recall_memories(instance["question"], top_k)
                total_retrieved += len(memories)
                generated = generate_answer(
                    openai_client,
                    reader_model,
                    instance["question"],
                    memories,
                )

                baseline_exact = exact_match(generated, str(instance["answer"]))
                if judge_model is not None:
                    judgment = judge_answer(
                        openai_client,
                        judge_model,
                        instance["question"],
                        generated,
                        str(instance["answer"]),
                    )
                else:
                    judgment = {
                        "correctness": float(baseline_exact),
                        "exact_match": baseline_exact,
                        "reasoning": "judge disabled; using normalized exact match only",
                    }

                result = {
                    "question_id": question_id,
                    "question_type": instance["question_type"],
                    "question": instance["question"],
                    "ground_truth": instance["answer"],
                    "generated_answer": generated,
                    "retrieved_count": len(memories),
                    "retrieved_memories": memories,
                    "exact_match_normalized": baseline_exact,
                    **judgment,
                }
                all_results.append(result)
                results_by_type[instance["question_type"]].append(result)

                with open(paths["hypotheses"], "a") as f:
                    f.write(
                        json.dumps(
                            {"question_id": question_id, "hypothesis": generated},
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

                pct = 100 * sum(r["correctness"] >= 0.5 for r in all_results) / len(all_results)
                print(
                    f"  [{index:>3}/{len(dataset)}] {question_id:<20} "
                    f"retrieved={len(memories):>2} exact={result['exact_match']:>1} "
                    f"score={result['correctness']:.2f} cumulative={pct:.1f}%"
                )

                if delay > 0:
                    time.sleep(delay)
        finally:
            shutil.rmtree(chroma_dir, ignore_errors=True)
            shutil.rmtree(media_dir, ignore_errors=True)

    elapsed = time.time() - t0
    per_type = {}
    correct = 0
    exact = 0
    for question_type, rows in sorted(results_by_type.items()):
        n = len(rows)
        num_correct = sum(r["correctness"] >= 0.5 for r in rows)
        num_exact = sum(r["exact_match"] for r in rows)
        avg_score = sum(r["correctness"] for r in rows) / n if n else 0.0
        per_type[question_type] = {
            "count": n,
            "accuracy": round(100 * num_correct / n, 2) if n else 0.0,
            "exact_match": round(100 * num_exact / n, 2) if n else 0.0,
            "avg_score": round(avg_score, 4),
        }
        correct += num_correct
        exact += num_exact

    overall_total = len(all_results)
    overall_accuracy = 100 * correct / overall_total if overall_total else 0.0
    overall_exact = 100 * exact / overall_total if overall_total else 0.0
    overall_avg = (
        sum(row["correctness"] for row in all_results) / overall_total
        if overall_total
        else 0.0
    )

    summary = {
        "benchmark": "LongMemEval",
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "dataset": dataset_name,
            "backend": backend_type,
            "limit": limit,
            "top_k": top_k,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
            "reader_model": reader_model,
            "judge_model": judge_model,
            "extract_facts": extract_facts_flag,
        },
        "summary": {
            "overall_accuracy": round(overall_accuracy, 2),
            "overall_exact_match": round(overall_exact, 2),
            "overall_avg_score": round(overall_avg, 4),
            "questions": overall_total,
            "sessions_ingested": total_sessions,
            "turns_ingested": total_turns,
            "facts_ingested": total_facts,
            "avg_retrieved": round(total_retrieved / overall_total, 2) if overall_total else 0.0,
            "by_question_type": per_type,
        },
        "results": all_results,
    }

    with open(paths["results"], "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(paths["summary"], "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 72)
    print("  LongMemEval RESULTS")
    print("=" * 72)
    for question_type, metrics in per_type.items():
        print(
            f"  {question_type:>24}: accuracy={metrics['accuracy']:5.1f}% "
            f"exact={metrics['exact_match']:5.1f}% avg={metrics['avg_score']:.3f} "
            f"n={metrics['count']}"
        )
    print()
    print(
        f"  {'OVERALL':>24}: accuracy={overall_accuracy:5.1f}% "
        f"exact={overall_exact:5.1f}% avg={overall_avg:.3f} n={overall_total}"
    )
    print(f"  Time elapsed           : {elapsed:.0f}s")
    print(f"  Sessions ingested      : {total_sessions}")
    print(f"  Facts ingested         : {total_facts}")
    print(f"  Hypotheses file        : {paths['hypotheses']}")
    print(f"  Summary file           : {paths['summary']}")
    print()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval harness for agentic-memory")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="oracle")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--backend", choices=["mcp", "runtime"], default="mcp")
    parser.add_argument("--embedding-model", default="gemini-embedding-001")
    parser.add_argument("--embedding-dimensions", type=int, default=768)
    parser.add_argument("--reader-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--extract-facts", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.environ.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if missing:
        print(f"Error: missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    asyncio.run(
        run_longmemeval(
            dataset_name=args.dataset,
            backend_type=args.backend,
            limit=args.limit,
            top_k=args.top_k,
            embedding_model=args.embedding_model,
            embedding_dimensions=args.embedding_dimensions,
            reader_model=args.reader_model,
            judge_model=None if args.skip_judge else args.judge_model,
            extract_facts_flag=args.extract_facts,
            delay=args.delay,
        )
    )


if __name__ == "__main__":
    main()
