"""Offline tests for the multimodal Gemini embedder contract."""

import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDING_DIMENSIONS
from utils.embeddings import GeminiEmbedder


class RecordingGeminiEmbedder(GeminiEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.chunk_payloads = []

    def _make_text_part(self, text: str):
        return {"type": "text", "text": text}

    def _document_config(self):
        return "document-config"

    def _query_config_obj(self):
        return "query-config"

    def _make_bytes_part(self, data: bytes, mime_type: str):
        return {"type": "bytes", "mime_type": mime_type, "data": data}

    def _make_content(self, parts):
        return {"parts": parts}

    def _embed_raw(self, contents, config):
        self.calls.append((contents, config))
        payload = contents[0]
        if isinstance(payload, str):
            label = payload
        else:
            payload = payload["parts"][-1]
            if payload["type"] == "bytes":
                try:
                    label = payload["data"].decode("utf-8")
                except UnicodeDecodeError:
                    label = payload["mime_type"]
            else:
                label = payload["text"]
        self.chunk_payloads.append(label)

        vector = [0.0] * EMBEDDING_DIMENSIONS
        vector[0] = 3.0
        vector[1] = 4.0
        vector[2] = float(len(self.calls))
        return [vector]

    def _probe_duration_seconds(self, path: Path) -> float:
        if path.name.endswith(".long.mp3"):
            return 180.0
        if path.name.endswith(".long.mov"):
            return 180.0
        return 10.0

    def _require_binary(self, name: str) -> None:
        return None

    def _write_media_chunks(self, *, path: Path, temp_dir: Path, max_chunk_seconds: float, duration_seconds: float):
        chunk_paths = []
        for index in range(3):
            chunk_path = temp_dir / f"chunk-{index}{path.suffix}"
            chunk_path.write_bytes(f"chunk-{index}".encode("utf-8"))
            chunk_paths.append(chunk_path)
        return chunk_paths


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def test_embed_text_normalizes_to_unit_length():
    embedder = RecordingGeminiEmbedder()

    vector = embedder.embed_text("retrieval")

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert math.isclose(_norm(vector), 1.0, rel_tol=1e-9)


def test_embed_multimodal_aggregates_text_and_image_into_one_embedding():
    embedder = RecordingGeminiEmbedder()
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        handle.write(b"image-bytes")
        handle.flush()

        vector = embedder.embed_multimodal(
            text="deployment failure screenshot",
            image=handle.name,
            image_mime_type="image/png",
        )

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert len(embedder.calls) == 1
    parts = embedder.calls[0][0][0]["parts"]
    assert parts[0] == {"type": "text", "text": "deployment failure screenshot"}
    assert parts[1]["mime_type"] == "image/png"


def test_embed_audio_chunks_long_media_and_renormalizes_average():
    embedder = RecordingGeminiEmbedder()
    with tempfile.NamedTemporaryFile(suffix=".long.mp3") as handle:
        handle.write(b"long-audio")
        handle.flush()

        vector = embedder.embed_audio(handle.name, description="incident audio", mime_type="audio/mpeg")

    assert len(embedder.calls) == 3
    assert embedder.chunk_payloads == ["chunk-0", "chunk-1", "chunk-2"]
    assert len(vector) == EMBEDDING_DIMENSIONS
    assert math.isclose(_norm(vector), 1.0, rel_tol=1e-9)


def test_embed_video_chunks_long_media_and_renormalizes_average():
    embedder = RecordingGeminiEmbedder()
    with tempfile.NamedTemporaryFile(suffix=".long.mov") as handle:
        handle.write(b"long-video")
        handle.flush()

        vector = embedder.embed_video(handle.name, description="incident video", mime_type="video/quicktime")

    assert len(embedder.calls) == 3
    assert embedder.chunk_payloads == ["chunk-0", "chunk-1", "chunk-2"]
    assert len(vector) == EMBEDDING_DIMENSIONS
    assert math.isclose(_norm(vector), 1.0, rel_tol=1e-9)


def test_embed_multimodal_audio_chunks_with_text_and_image_context():
    embedder = RecordingGeminiEmbedder()
    with tempfile.NamedTemporaryFile(suffix=".png") as image_handle, tempfile.NamedTemporaryFile(
        suffix=".long.mp3"
    ) as audio_handle:
        image_handle.write(b"context-image")
        image_handle.flush()
        audio_handle.write(b"long-audio")
        audio_handle.flush()

        vector = embedder.embed_multimodal(
            text="deployment incident",
            image=image_handle.name,
            image_mime_type="image/png",
            audio=audio_handle.name,
            audio_mime_type="audio/mpeg",
        )

    assert len(embedder.calls) == 3
    first_parts = embedder.calls[0][0][0]["parts"]
    assert first_parts[0] == {"type": "text", "text": "deployment incident"}
    assert first_parts[1]["mime_type"] == "image/png"
    assert first_parts[2]["data"] == b"chunk-0"
    assert len(vector) == EMBEDDING_DIMENSIONS
    assert math.isclose(_norm(vector), 1.0, rel_tol=1e-9)


def test_embed_audio_requires_ffmpeg_when_chunking():
    embedder = RecordingGeminiEmbedder()
    embedder._require_binary = lambda name: (_ for _ in ()).throw(
        RuntimeError(f"{name} is required for audio/video chunking but was not found")
    )
    with tempfile.NamedTemporaryFile(suffix=".long.mp3") as handle:
        handle.write(b"long-audio")
        handle.flush()
        try:
            embedder.embed_audio(handle.name, mime_type="audio/mpeg")
            raise AssertionError("Expected missing ffmpeg to fail")
        except RuntimeError as exc:
            assert "ffmpeg" in str(exc)


def test_probe_duration_requires_ffprobe():
    embedder = RecordingGeminiEmbedder()
    embedder._probe_duration_seconds = GeminiEmbedder._probe_duration_seconds.__get__(
        embedder, RecordingGeminiEmbedder
    )
    embedder._require_binary = lambda name: (_ for _ in ()).throw(
        RuntimeError(f"{name} is required for audio/video chunking but was not found")
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3") as handle:
        handle.write(b"audio")
        handle.flush()
        try:
            embedder._probe_duration_seconds(Path(handle.name))
            raise AssertionError("Expected missing ffprobe to fail")
        except RuntimeError as exc:
            assert "ffprobe" in str(exc)


def test_embed_image_rejects_unsupported_mime():
    embedder = RecordingGeminiEmbedder()
    with tempfile.NamedTemporaryFile(suffix=".gif") as handle:
        handle.write(b"gif")
        handle.flush()
        try:
            embedder.embed_image(handle.name, mime_type="image/gif")
            raise AssertionError("Expected unsupported MIME to fail")
        except ValueError as exc:
            assert "Unsupported image MIME type" in str(exc)


def test_embed_bytes_rejects_unsupported_mime():
    embedder = RecordingGeminiEmbedder()
    try:
        embedder.embed_bytes(b"bytes", "application/octet-stream")
        raise AssertionError("Expected unsupported MIME to fail")
    except ValueError as exc:
        assert "Unsupported MIME type" in str(exc)


def test_chunking_rejects_runaway_chunk_counts():
    embedder = RecordingGeminiEmbedder()
    embedder._write_media_chunks = GeminiEmbedder._write_media_chunks.__get__(embedder, RecordingGeminiEmbedder)
    with tempfile.NamedTemporaryFile(suffix=".long.mp3") as handle, tempfile.TemporaryDirectory(
        prefix="embed_chunks_"
    ) as chunk_dir:
        handle.write(b"long-audio")
        handle.flush()
        try:
            embedder._write_media_chunks(
                path=Path(handle.name),
                temp_dir=Path(chunk_dir),
                max_chunk_seconds=1.0,
                duration_seconds=500.0,
            )
            raise AssertionError("Expected excessive chunk count to fail")
        except ValueError as exc:
            assert "too many chunks" in str(exc)


if __name__ == "__main__":
    test_embed_text_normalizes_to_unit_length()
    test_embed_multimodal_aggregates_text_and_image_into_one_embedding()
    test_embed_audio_chunks_long_media_and_renormalizes_average()
    test_embed_video_chunks_long_media_and_renormalizes_average()
    test_embed_multimodal_audio_chunks_with_text_and_image_context()
    test_embed_audio_requires_ffmpeg_when_chunking()
    test_probe_duration_requires_ffprobe()
    test_embed_image_rejects_unsupported_mime()
    test_embed_bytes_rejects_unsupported_mime()
    test_chunking_rejects_runaway_chunk_counts()
