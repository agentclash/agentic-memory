# agentic-memory

A cognitive memory system for AI agents, grounded in the taxonomy from [Measuring Progress Toward AGI: A Cognitive Framework](research-docs/measuring-progress-toward-agi-a-cognitive-framework.pdf).

Most agent frameworks treat memory as a single vector store you dump context into. This project implements memory the way the cognitive science paper describes it: **separate stores for different memory types**, a **unified retriever** that queries across them, a **forgetting service** that prunes stale knowledge, and an **event bus** that lets future modules (working memory, learning, metacognition) subscribe without touching core code.

Built on **Gemini Embedding 2** for natively multimodal embeddings — text, images, audio, and video share a single vector space.

> Status: Phase 1 in progress. Semantic memory write path is functional. Read path and event bus next.

---

## Architecture

```mermaid
graph TB
    subgraph "Phase 1 — Memory System"
        CLI["demo/cli.py"]
        EMB["GeminiEmbedder"]
        SS["SemanticStore"]
        ES["EpisodicStore"]
        PS["ProceduralStore"]
        RET["UnifiedRetriever"]
        FGT["ForgettingService"]
        BUS["EventBus"]
        DB[(ChromaDB)]

        CLI --> SS
        CLI --> ES
        CLI --> PS
        SS --> EMB
        ES --> EMB
        PS --> EMB
        SS --> DB
        ES --> DB
        PS --> DB
        RET --> SS
        RET --> ES
        RET --> PS
        FGT --> SS
        FGT --> ES
        FGT --> PS
        SS -.->|emit| BUS
        ES -.->|emit| BUS
        PS -.->|emit| BUS
    end

    subgraph "Phase 2 — Working Memory + Learning"
        WM["WorkingMemory"]
        LRN["LearningModule"]
        WM -.->|subscribe| BUS
        LRN -.->|subscribe| BUS
    end

    subgraph "Phase 3 — Metacognition"
        META["MetacognitiveMonitor"]
        META -.->|subscribe| BUS
    end

    style SS fill:#2d6a4f,color:#fff
    style EMB fill:#2d6a4f,color:#fff
    style CLI fill:#2d6a4f,color:#fff
    style DB fill:#2d6a4f,color:#fff
    style ES fill:#555,color:#aaa
    style PS fill:#555,color:#aaa
    style RET fill:#555,color:#aaa
    style FGT fill:#555,color:#aaa
    style BUS fill:#555,color:#aaa
    style WM fill:#333,color:#666
    style LRN fill:#333,color:#666
    style META fill:#333,color:#666
```

Green = built. Grey = planned (Phase 1). Dark = future phases.

---

## Memory Types

The paper identifies distinct memory sub-types that behave differently — different decay rates, retrieval patterns, and update semantics. This project implements them as separate ChromaDB collections behind a shared interface.

```mermaid
classDiagram
    class MemoryRecord {
        +str content
        +str memory_type
        +str modality
        +str id
        +datetime created_at
        +float importance
        +list~float~ embedding
        +str media_ref
    }
    class SemanticMemory {
        +str category
        +float confidence
        +str supersedes
        +list~str~ related_ids
    }
    class EpisodicMemory {
        +str context
        +float emotional_valence
        +str session_id
    }
    class ProceduralMemory {
        +str trigger
        +list~str~ steps
        +int execution_count
    }
    MemoryRecord <|-- SemanticMemory
    MemoryRecord <|-- EpisodicMemory
    MemoryRecord <|-- ProceduralMemory
```

---

## Write Path

How a memory goes from raw content to a persisted vector:

```mermaid
sequenceDiagram
    participant C as CLI
    participant S as SemanticStore
    participant E as GeminiEmbedder
    participant G as Gemini API
    participant D as ChromaDB

    C->>S: store(SemanticMemory)
    S->>E: embed_text(content)
    E->>G: embed_content(text, 768 dims)
    G-->>E: [0.023, -0.41, 0.87, ...]
    E-->>S: embedding vector
    S->>D: add(id, embedding, metadata)
    D-->>S: persisted
    S-->>C: record_id
```

---

## Multimodal Embeddings

All modalities go through the same `GeminiEmbedder` and land in the same vector space. A text query can retrieve an image memory. An audio clip can be compared to text descriptions.

```mermaid
graph LR
    T["text"] -->|embed_text| EMB["GeminiEmbedder"]
    I["image bytes"] -->|embed_image| EMB
    A["audio bytes"] -->|embed_audio| EMB
    EMB --> V["768-dim vector space"]

    style V fill:#2d6a4f,color:#fff
```

The embedding model is [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/embeddings) — natively multimodal, mapping text, images, audio, and video into a single embedding space. Matryoshka support allows truncation from 3072 down to 768 dimensions with minimal accuracy loss.

---

## Experiment: Cross-Modal Emotion in Audio

We ran an experiment to test whether audio embeddings encode emotional tone or just acoustic structure. Four songs across three languages (English, Hindi, Arabic), embedded as raw bytes with no metadata passed to the model.

```mermaid
graph LR
    subgraph "Audio Pipeline"
        SONG["song.mp3"] -->|ffmpeg| CHUNKS["60s chunks"]
        CHUNKS -->|embed_audio| VECS["chunk vectors"]
        VECS -->|average| AV["song vector"]
    end

    subgraph "Probe Vectors"
        P1["grief / loss"]
        P2["melancholy"]
        P3["joy / happiness"]
        P4["peaceful / calm"]
        P5["tension / dread"]
        P6["neutral"]
    end

    AV -->|cosine similarity| SCORES["ranked scores"]
    P1 --> SCORES
    P2 --> SCORES
    P3 --> SCORES
```

**Results summary:**

| Song | Genre | Language | Top Match | Runner-up |
|---|---|---|---|---|
| Schindler's List Theme | Orchestral | Instrumental | melancholy (0.70) | grief (0.68) |
| Phir Se | Bollywood | Hindi | grief (0.69) | melancholy (0.69) |
| Rasputin | Disco-pop | English | joy (0.69) | tension (0.67) |
| Didi | Rai | Arabic/French | joy (0.66) | melancholy (0.63) |

The model correctly separated sad from happy songs, made nuanced distinctions within each cluster (melancholy vs. grief, joy vs. tension), and worked cross-lingually on raw audio bytes with no transcription.

Grief ranked last on both happy songs. Neutral ranked last on both sad songs. The largest winner-to-loser gap was 0.087 (Phir Se: grief vs. neutral).

Full methodology and analysis: [`experiments/audio_emotion_probe_results.md`](experiments/audio_emotion_probe_results.md)

**Architectural implication:** `emotional_valence` on episodic memories can be derived directly from the embedding — no separate sentiment analysis pipeline needed.

---

## Project Structure

```
agentic-memory/
├── config.py                  # API keys, model config, ChromaDB path
├── models/
│   ├── base.py                # MemoryRecord dataclass
│   └── semantic.py            # SemanticMemory (factual knowledge)
├── utils/
│   └── embeddings.py          # GeminiEmbedder — text, image, audio
├── stores/
│   ├── base.py                # Abstract BaseStore interface
│   └── semantic_store.py      # ChromaDB-backed semantic store
├── retrieval/                  # (planned) Unified cross-store retriever
├── events/                     # (planned) Event bus for store/retrieve signals
├── demo/
│   └── cli.py                 # CLI for storing and querying memories
├── experiments/
│   ├── audio_emotion_probe.py           # Cross-modal emotion probe script
│   └── audio_emotion_probe_results.md   # Full results and analysis
├── media/                      # Audio/image files (not committed)
└── research-docs/
    └── measuring-progress-toward-agi-a-cognitive-framework.pdf
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

System dependency for audio chunking:
```bash
# arch
sudo pacman -S ffmpeg

# ubuntu/debian
sudo apt install ffmpeg
```

---

## Usage

Store a fact:
```bash
python demo/cli.py store "Python was created by Guido van Rossum"
```

Run the audio emotion probe:
```bash
python experiments/audio_emotion_probe.py "song.mp3"
```

---

## Offline Evaluation

The repo now includes a deterministic offline episodic-memory evaluation harness
for fixed synthetic fixtures across mixed-store retrieval, temporal recall,
session reconstruction, recent-event lookup, and cross-modal media-backed
episodes.

Run it with:

```bash
pytest tests/test_offline_episodic_eval.py
```

or:

```bash
python tests/test_offline_episodic_eval.py
```

Benchmark mapping and rationale: [`docs/offline_episodic_eval.md`](docs/offline_episodic_eval.md)

---

## Theoretical Foundation

This project is built on the cognitive taxonomy from the DeepMind paper *Measuring Progress Toward AGI*. The paper distinguishes three faculties that most agent frameworks conflate:

- **Memory** — passive storage and retrieval (semantic facts, episodic events, procedural skills)
- **Working Memory** — active manipulation of information for a current goal (sits under Executive Functions, not Memory)
- **Learning** — acquisition and consolidation of new knowledge into long-term memory

The architecture implements these as separate systems. Phase 1 builds the memory stores. Phase 2 adds working memory (an active scratchpad) and a learning module (consolidation from working memory to long-term stores). Phase 3 adds metacognitive monitoring — the system's ability to assess confidence in its own retrieved context.

---

## License

MIT
