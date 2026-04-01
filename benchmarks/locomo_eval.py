"""
LoCoMo benchmark harness for agentic-memory.

Evaluates long-term conversational memory recall quality using the LoCoMo
dataset (Maharana et al., ACL 2024).  Ingests multi-session conversations
as both episodic and semantic memories via the MCP server, then scores
retrieval-augmented QA against ground truth using an LLM judge.

Usage:
    # Full run (10 conversations, ~1 986 questions, ~$11)
    .venv/bin/python benchmarks/locomo_eval.py

    # Quick smoke test (1 conversation, ~$1.10)
    .venv/bin/python benchmarks/locomo_eval.py --conversations 1

    # Tune retrieval depth
    .venv/bin/python benchmarks/locomo_eval.py --top-k 15

    # Skip fact extraction (episodic-only, cheaper)
    .venv/bin/python benchmarks/locomo_eval.py --no-extract-facts

    # Direct runtime instead of MCP server (faster, same memory quality)
    .venv/bin/python benchmarks/locomo_eval.py --backend runtime

Requirements:
    - OPENAI_API_KEY in environment
    - GEMINI_API_KEY in environment (for embeddings)
    - benchmarks/data/locomo10.json (downloaded automatically on first run)

Estimated cost (full 10-conversation run):
    Gemini embeddings ............. ~$0.03
    OpenAI fact extraction ........ ~$1.50
    OpenAI answer generation ...... ~$4.50
    OpenAI judging ................ ~$5.00
    TOTAL ......................... ~$11

Reference scores (LoCoMo paper + community):
    Memori ..................... ~82%
    Engram (epi+sem+proc) ..... ~80%
    Mem0 ....................... ~76%
    Letta (MemGPT) ............. ~74%
    Full-context GPT-4 ........ ~72%
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}

DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "locomo10.json"
RESULTS_PATH = DATA_DIR / "locomo_results.json"

DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

DEFAULT_MODEL = "gpt-5.3-chat-latest"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def ensure_dataset() -> Path:
    if DATA_PATH.exists():
        return DATA_PATH
    print(f"Downloading LoCoMo dataset to {DATA_PATH} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-sL", DOWNLOAD_URL, "-o", str(DATA_PATH)])
    print(f"  Done ({DATA_PATH.stat().st_size / 1024:.0f} KB)")
    return DATA_PATH


def load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    resolved = path or ensure_dataset()
    with open(resolved) as f:
        return json.load(f)


def iter_sessions(
    conversation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Yield each session as {session_idx, session_id, date, turns}."""
    conv = conversation["conversation"]
    sample_id = conversation["sample_id"]
    sessions = []
    idx = 1
    while True:
        session_key = f"session_{idx}"
        date_key = f"session_{idx}_date_time"
        turns = conv.get(session_key)
        if turns is None and date_key not in conv:
            break
        if isinstance(turns, list) and len(turns) > 0:
            sessions.append(
                {
                    "session_idx": idx,
                    "session_id": f"{sample_id}_s{idx}",
                    "date": conv.get(date_key, ""),
                    "turns": turns,
                }
            )
        idx += 1
    return sessions


# ---------------------------------------------------------------------------
# Fact extraction (semantic memories from dialogue)
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """\
You are extracting durable facts from a conversation session.

Given a sequence of dialogue turns, extract the key facts, preferences,
biographical details, and relationships mentioned.  Return ONLY a JSON
array of strings, each string being one standalone fact.

Rules:
- Each fact must be self-contained (include who it is about)
- Include temporal anchors when mentioned (dates, durations, "last week")
- Include relationships between people
- Skip small talk, greetings, and filler
- Aim for 3-15 facts per session depending on content density
- If the session has no extractable facts, return []

Example output:
["Caroline identifies as queer", "Melanie has two children", "Caroline attended an LGBTQ support group on May 7 2023"]"""


def extract_facts(
    client: OpenAI,
    model: str,
    session: dict[str, Any],
) -> list[str]:
    """Use an LLM to extract durable facts from a dialogue session."""
    speaker_a = session.get("speaker_a", "Speaker A")
    speaker_b = session.get("speaker_b", "Speaker B")
    date = session.get("date", "")

    lines = []
    for turn in session["turns"]:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        lines.append(f"{speaker}: {text}")
    dialogue = "\n".join(lines)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Session date: {date}\n"
                    f"Participants: {speaker_a}, {speaker_b}\n\n"
                    f"Dialogue:\n{dialogue}"
                ),
            },
        ],
        max_completion_tokens=600,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()

    try:
        text = raw
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(f) for f in parsed if f]
        return []
    except (json.JSONDecodeError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Backend: MCP client over stdio
# ---------------------------------------------------------------------------


class MCPBackend:
    """Calls the MCP server over stdio for all memory operations."""

    def __init__(
        self,
        chroma_path: str,
        media_root: str,
        *,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
    ):
        self._chroma_path = chroma_path
        self._media_root = media_root
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._session = None

    async def __aenter__(self):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        env = {**os.environ}
        env["MEMORY_CHROMA_PATH"] = self._chroma_path
        env["MEMORY_MEDIA_DIR"] = self._media_root
        env["MEMORY_MCP_TRANSPORT"] = "stdio"
        if self._embedding_model:
            env["EMBEDDING_MODEL"] = self._embedding_model
        if self._embedding_dimensions is not None:
            env["EMBEDDING_DIMENSIONS"] = str(self._embedding_dimensions)

        self._stdio_ctx = stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=["-m", "mcp_server"],
                env=env,
                cwd=str(Path(__file__).parent.parent),
            )
        )
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._session_ctx:
                await self._session_ctx.__aexit__(exc_type, exc_val, exc_tb)
        except Exception:
            pass  # session cleanup can race with stdio teardown
        try:
            if self._stdio_ctx:
                await self._stdio_ctx.__aexit__(exc_type, exc_val, exc_tb)
        except Exception:
            pass  # stdio process cleanup is best-effort

    async def remember_episode(
        self,
        *,
        session_id: str,
        text: str,
        turn_number: int,
        participants: list[str],
    ) -> None:
        await self._session.call_tool(
            "remember_episode",
            {
                "session_id": session_id,
                "text": text,
                "turn_number": turn_number,
                "participants": participants,
            },
        )

    async def remember_fact(self, *, content: str, category: str = "general") -> None:
        await self._session.call_tool(
            "remember_fact",
            {"content": content, "category": category},
        )

    async def recall_memories(self, query: str, top_k: int) -> list[str]:
        result = await self._session.call_tool(
            "recall_memories",
            {"query": query, "top_k": top_k},
        )
        return self._extract_memories(result)

    @staticmethod
    def _extract_memories(result) -> list[str]:
        """Parse recall_memories result into a list of content strings.

        FastMCP with json_response=True returns the Pydantic model in two
        places: structuredContent (wrapped under a "result" key) and
        content[0].text (raw JSON).  We try both paths.
        """
        memories = []

        # Path 1: structuredContent -> result -> results
        if result.structuredContent:
            inner = result.structuredContent
            # FastMCP wraps the model under "result"
            if "result" in inner:
                inner = inner["result"]
            for item in inner.get("results", []):
                content = item.get("record", {}).get("content", "")
                if content:
                    memories.append(content)
            if memories:
                return memories

        # Path 2: text content block with JSON
        if result.content:
            for block in result.content:
                if hasattr(block, "text") and block.text:
                    try:
                        parsed = json.loads(block.text)
                        for item in parsed.get("results", []):
                            content = item.get("record", {}).get("content", "")
                            if content:
                                memories.append(content)
                    except (json.JSONDecodeError, AttributeError):
                        pass
        return memories


# ---------------------------------------------------------------------------
# Backend: direct runtime (no MCP transport)
# ---------------------------------------------------------------------------


EMBED_BATCH_SIZE = 100


class RuntimeBackend:
    """Calls the core runtime directly with batch embedding for speed."""

    def __init__(
        self,
        chroma_path: str,
        media_root: str,
        *,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
    ):
        self._chroma_path = chroma_path
        self._media_root = media_root
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._runtime = None

    async def __aenter__(self):
        from runtime import build_runtime

        if self._embedding_model:
            os.environ["EMBEDDING_MODEL"] = self._embedding_model
        if self._embedding_dimensions is not None:
            os.environ["EMBEDDING_DIMENSIONS"] = str(self._embedding_dimensions)
        self._runtime = build_runtime(
            chroma_path=self._chroma_path,
            media_root=self._media_root,
        )
        return self

    async def __aexit__(self, *exc):
        pass

    def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches of EMBED_BATCH_SIZE via a single API call each."""
        embedder = self._runtime.embedder
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            vectors = embedder._embed(batch, embedder._document_config())
            all_vectors.extend(vectors)
        return all_vectors

    def batch_store_episodes(
        self,
        records: list[dict[str, Any]],
    ) -> int:
        """Batch-embed and batch-insert episodic memories into ChromaDB."""
        from models.episodic import EpisodicMemory

        if not records:
            return 0

        episodes = [
            EpisodicMemory(
                content=r["content"],
                session_id=r["session_id"],
                turn_number=r["turn_number"],
                participants=r["participants"],
                importance=0.5,
            )
            for r in records
        ]

        texts = [e.content for e in episodes]
        embeddings = self._batch_embed(texts)

        store = self._runtime.episodic_store
        ids = [e.id for e in episodes]
        metadatas = [store._to_metadata(e) for e in episodes]
        store._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(episodes)

    def batch_store_facts(self, contents: list[str]) -> int:
        """Batch-embed and batch-insert semantic memories into ChromaDB."""
        from models.semantic import SemanticMemory

        if not contents:
            return 0

        facts = [SemanticMemory(content=c, category="extracted") for c in contents]
        texts = [f.content for f in facts]
        embeddings = self._batch_embed(texts)

        store = self._runtime.semantic_store
        ids = [f.id for f in facts]
        metadatas = [store._to_metadata(f) for f in facts]
        store._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(facts)

    async def recall_memories(self, query: str, top_k: int) -> list[str]:
        results = self._runtime.retriever.query(query, top_k=top_k)
        return [r.record.content for r in results]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


async def ingest_conversation(
    backend,
    conversation: dict[str, Any],
    openai_client: OpenAI,
    model: str,
    *,
    extract_facts_flag: bool = True,
) -> dict[str, int]:
    conv = conversation["conversation"]
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")

    sessions = iter_sessions(conversation)

    # Collect all episode records for batch embedding
    episode_records: list[dict[str, Any]] = []
    fact_contents: list[str] = []

    for session in sessions:
        session_id = session["session_id"]
        date = session["date"]

        for turn_num, turn in enumerate(session["turns"], 1):
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            dia_id = turn.get("dia_id", "")
            if not text.strip():
                continue

            content = f"[{date}] [{dia_id}] {speaker}: {text}"
            episode_records.append(
                {
                    "content": content,
                    "session_id": session_id,
                    "turn_number": turn_num,
                    "participants": [speaker_a, speaker_b],
                }
            )

        if extract_facts_flag:
            session_for_extraction = {
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "date": date,
                "turns": session["turns"],
            }
            facts = extract_facts(openai_client, model, session_for_extraction)
            for fact in facts:
                dated_fact = f"[{date}] {fact}" if date else fact
                fact_contents.append(dated_fact)

    # Batch embed + insert (the big speedup)
    if isinstance(backend, RuntimeBackend):
        num_episodes = backend.batch_store_episodes(episode_records)
        num_facts = backend.batch_store_facts(fact_contents)
    else:
        # MCP backend: fall back to individual calls
        num_episodes = 0
        for r in episode_records:
            await backend.remember_episode(
                session_id=r["session_id"],
                text=r["content"],
                turn_number=r["turn_number"],
                participants=r["participants"],
            )
            num_episodes += 1
        num_facts = 0
        for fact in fact_contents:
            await backend.remember_fact(content=fact, category="extracted")
            num_facts += 1

    return {
        "sessions": len(sessions),
        "episodes": num_episodes,
        "facts": num_facts,
    }


# ---------------------------------------------------------------------------
# QA evaluation
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = (
    "You are answering questions about past conversations using retrieved "
    "memory excerpts.  Answer concisely and factually based ONLY on the "
    "provided memories.  If the memories do not contain enough information, "
    "say 'I don't have enough information to answer this.'"
)

JUDGE_SYSTEM = """\
You are a strict factual evaluator.  Compare the generated answer against
the ground truth for the given question.

Score on two dimensions:
  correctness  -- float 0.0-1.0, how much of the ground-truth information
                  is present and accurate in the generated answer.
  exact_match  -- 0 or 1, whether the generated answer is essentially the
                  same answer as the ground truth.

Respond with ONLY a JSON object:
{"correctness": <float>, "exact_match": <0|1>, "reasoning": "<brief>"}"""


def generate_answer(
    client: OpenAI,
    model: str,
    question: str,
    memories: list[str],
) -> str:
    if not memories:
        return "No relevant memories found."

    numbered = "\n".join(f"[{i}] {m}" for i, m in enumerate(memories, 1))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Retrieved memories:\n{numbered}\n\n"
                    f"Question: {question}\nAnswer:"
                ),
            },
        ],
        max_completion_tokens=200,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def judge_answer(
    client: OpenAI,
    model: str,
    question: str,
    generated: str,
    ground_truth: str,
) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Ground truth: {ground_truth}\n"
                    f"Generated answer: {generated}"
                ),
            },
        ],
        max_completion_tokens=150,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()

    try:
        text = raw
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        parsed = json.loads(text)
        return {
            "correctness": float(parsed.get("correctness", 0.0)),
            "exact_match": int(parsed.get("exact_match", 0)),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except (json.JSONDecodeError, AttributeError, ValueError):
        return {
            "correctness": 0.0,
            "exact_match": 0,
            "reasoning": f"judge-parse-error: {raw[:120]}",
        }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(
    results_by_cat: dict[int, list[dict[str, Any]]],
    all_results: list[dict[str, Any]],
    config_info: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    print()
    print("=" * 72)
    print("  LoCoMo RESULTS")
    print("=" * 72)
    print()

    overall_correct = 0
    overall_exact = 0
    overall_total = 0
    category_summary: dict[str, dict[str, Any]] = {}

    for cat in sorted(results_by_cat):
        cat_results = results_by_cat[cat]
        name = CATEGORY_NAMES.get(cat, f"cat-{cat}")
        n = len(cat_results)
        correct = sum(1 for r in cat_results if r["correctness"] >= 0.5)
        exact = sum(r["exact_match"] for r in cat_results)
        avg = sum(r["correctness"] for r in cat_results) / n

        overall_correct += correct
        overall_exact += exact
        overall_total += n

        pct = 100 * correct / n
        exact_pct = 100 * exact / n
        print(
            f"  Cat {cat} ({name:>12}):  "
            f"accuracy={pct:5.1f}%  exact={exact_pct:5.1f}%  "
            f"avg_score={avg:.3f}  n={n}"
        )
        category_summary[name] = {
            "accuracy": round(pct, 2),
            "exact_match": round(exact_pct, 2),
            "avg_score": round(avg, 4),
            "count": n,
        }

    overall_acc = 100 * overall_correct / overall_total if overall_total else 0
    overall_ex = 100 * overall_exact / overall_total if overall_total else 0
    overall_avg = (
        sum(r["correctness"] for r in all_results) / len(all_results)
        if all_results
        else 0
    )

    print()
    print(
        f"  {'OVERALL':>22}:  "
        f"accuracy={overall_acc:5.1f}%  exact={overall_ex:5.1f}%  "
        f"avg_score={overall_avg:.3f}  n={overall_total}"
    )
    print(f"  Time elapsed: {elapsed:.0f}s")
    print()
    print("  Reference scores:")
    print("    Memori ..................... ~82%")
    print("    Engram (epi+sem+proc) ..... ~80%")
    print("    Mem0 ...................... ~76%")
    print("    Letta (MemGPT) ............ ~74%")
    print("    Full-context GPT-4 ....... ~72%")
    print()

    return {
        "benchmark": "LoCoMo",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "config": config_info,
        "summary": {
            "overall_accuracy": round(overall_acc, 2),
            "overall_exact_match": round(overall_ex, 2),
            "overall_avg_score": round(overall_avg, 4),
            "by_category": category_summary,
        },
        "results": all_results,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_benchmark(
    *,
    num_conversations: int = 10,
    top_k: int = 10,
    model: str = DEFAULT_MODEL,
    delay: float = 0.05,
    backend_type: str = "mcp",
    extract_facts_flag: bool = True,
) -> dict[str, Any]:
    dataset = load_dataset()
    conversations = dataset[: min(num_conversations, len(dataset))]

    total_qa = sum(len(c["qa"]) for c in conversations)
    print("=== LoCoMo Benchmark ===")
    print(f"  Conversations   : {len(conversations)}/{len(dataset)}")
    print(f"  Questions       : {total_qa}")
    print(f"  Model           : {model}")
    print(f"  Retrieval k     : {top_k}")
    print(f"  Backend         : {backend_type}")
    print(f"  Fact extraction : {'on' if extract_facts_flag else 'off'}")
    print(f"  Isolation       : per-conversation (separate memory store each)")
    print()

    backend_cls = MCPBackend if backend_type == "mcp" else RuntimeBackend
    openai_client = OpenAI(timeout=60, max_retries=3)

    results_by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    all_results: list[dict[str, Any]] = []
    errors = 0
    total_episodes = 0
    total_facts = 0
    t0 = time.time()

    for conv_idx, conv in enumerate(conversations):
        sample_id = conv["sample_id"]
        # Fresh isolated memory store per conversation
        chroma_dir = tempfile.mkdtemp(prefix=f"locomo_{sample_id}_")
        media_dir = tempfile.mkdtemp(prefix=f"locomo_media_{sample_id}_")

        try:
            async with backend_cls(chroma_dir, media_dir) as backend:
                # -- Ingest this conversation ------------------------------
                t_ingest = time.time()
                stats = await ingest_conversation(
                    backend,
                    conv,
                    openai_client,
                    model,
                    extract_facts_flag=extract_facts_flag,
                )
                total_episodes += stats["episodes"]
                total_facts += stats["facts"]
                ingest_secs = time.time() - t_ingest
                fact_str = f"  {stats['facts']:>3} facts" if extract_facts_flag else ""
                print(
                    f"  [{conv_idx + 1:>2}/{len(conversations)}] {sample_id:>8}  "
                    f"{stats['sessions']:>2} sessions  "
                    f"{stats['episodes']:>4} episodes"
                    f"{fact_str}  "
                    f"({ingest_secs:.0f}s)"
                )

                # -- Evaluate this conversation's questions ----------------
                qa_pairs: list[dict[str, Any]] = []
                for qa in conv["qa"]:
                    answer = qa.get("answer") or qa.get("adversarial_answer", "")
                    qa_pairs.append(
                        {
                            "question": qa["question"],
                            "answer": answer,
                            "category": qa["category"],
                            "evidence": qa["evidence"],
                            "sample_id": sample_id,
                        }
                    )

                conv_correct = 0
                for i, qa in enumerate(qa_pairs):
                    try:
                        memories = await backend.recall_memories(
                            qa["question"], top_k
                        )
                        generated = generate_answer(
                            openai_client, model, qa["question"], memories
                        )
                        judgment = judge_answer(
                            openai_client,
                            model,
                            qa["question"],
                            generated,
                            str(qa["answer"]),
                        )
                    except Exception as exc:
                        errors += 1
                        generated = f"ERROR: {exc}"
                        memories = []
                        judgment = {
                            "correctness": 0.0,
                            "exact_match": 0,
                            "reasoning": f"api-error: {exc}",
                        }

                    result = {
                        **qa,
                        "generated_answer": generated,
                        "retrieved_count": len(memories),
                        **judgment,
                    }
                    results_by_cat[qa["category"]].append(result)
                    all_results.append(result)
                    if result["correctness"] >= 0.5:
                        conv_correct += 1

                    if delay > 0:
                        time.sleep(delay)

                conv_acc = 100 * conv_correct / len(qa_pairs) if qa_pairs else 0
                overall_correct = sum(1 for r in all_results if r["correctness"] >= 0.5)
                overall_acc = 100 * overall_correct / len(all_results)
                print(
                    f"           {sample_id} QA: {conv_correct}/{len(qa_pairs)} "
                    f"({conv_acc:.1f}%)  "
                    f"cumulative: {overall_correct}/{len(all_results)} ({overall_acc:.1f}%)"
                    + (f"  errors={errors}" if errors else "")
                )

        finally:
            shutil.rmtree(chroma_dir, ignore_errors=True)
            shutil.rmtree(media_dir, ignore_errors=True)

    elapsed = time.time() - t0
    print()
    print(f"  Total: {total_episodes} episodes + {total_facts} facts across "
          f"{len(conversations)} conversations in {elapsed:.0f}s")

    # -- Report --------------------------------------------------------
    print()
    print("Phase: Report")

    config_info = {
        "num_conversations": len(conversations),
        "top_k": top_k,
        "model": model,
        "backend": backend_type,
        "extract_facts": extract_facts_flag,
        "isolation": "per-conversation",
        "total_episodes_ingested": total_episodes,
        "total_facts_ingested": total_facts,
        "total_questions": len(all_results),
        "errors": errors,
    }

    report = print_report(results_by_cat, all_results, config_info, elapsed)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Results saved to {RESULTS_PATH}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoCoMo benchmark for agentic-memory",
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=10,
        help="Number of conversations to evaluate (1-10, default: 10)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Memories retrieved per question (default: 10)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model for generation, extraction, and judging (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds between OpenAI API calls (default: 0.05)",
    )
    parser.add_argument(
        "--backend",
        choices=["mcp", "runtime"],
        default="mcp",
        help="Backend: 'mcp' spawns the MCP server over stdio, "
        "'runtime' calls the core directly (default: mcp)",
    )
    parser.add_argument(
        "--no-extract-facts",
        action="store_true",
        help="Skip fact extraction (episodic-only, cheaper but lower accuracy)",
    )
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
        run_benchmark(
            num_conversations=args.conversations,
            top_k=args.top_k,
            model=args.model,
            delay=args.delay,
            backend_type=args.backend,
            extract_facts_flag=not args.no_extract_facts,
        )
    )


if __name__ == "__main__":
    main()
