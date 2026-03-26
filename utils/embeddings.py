from __future__ import annotations

import math
import mimetypes
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Protocol

from config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, GEMINI_API_KEY

_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}
_AUDIO_MIME_TYPES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave"}
_VIDEO_MIME_TYPES = {"video/mp4", "video/quicktime"}
_PDF_MIME_TYPES = {"application/pdf"}

_MEDIA_LIMIT_SECONDS = {
    "audio": 80.0,
    "video": 120.0,
}
_MAX_MEDIA_CHUNKS = 32

_DEFAULT_MIME_TYPES = {
    "image": "image/png",
    "audio": "audio/mpeg",
    "video": "video/mp4",
    "pdf": "application/pdf",
}

_SUPPORTED_MIME_TYPES = {
    "image": _IMAGE_MIME_TYPES,
    "audio": _AUDIO_MIME_TYPES,
    "video": _VIDEO_MIME_TYPES,
    "pdf": _PDF_MIME_TYPES,
}


class TextEmbedder(Protocol):
    def embed_text(self, text: str) -> list[float]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    """Converts text and local media into Gemini embedding vectors."""

    def __init__(self):
        self._genai = None
        self._client = None
        self._types = None
        self._doc_config = None
        self._query_config = None

    def embed_text(self, text: str) -> list[float]:
        return self._embed([text], self._document_config())[0]

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self._query_config_obj())[0]

    def embed_bytes(self, data: bytes, mime_type: str) -> list[float]:
        resolved_mime = self._validate_mime_type(mime_type)
        return self._embed_parts([self._make_bytes_part(data, resolved_mime)], self._document_config())

    def embed_image(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = None,
    ) -> list[float]:
        return self._embed_media(source, modality="image", description=description, mime_type=mime_type)

    def embed_audio(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = None,
    ) -> list[float]:
        return self._embed_media(source, modality="audio", description=description, mime_type=mime_type)

    def embed_video(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = None,
    ) -> list[float]:
        return self._embed_media(source, modality="video", description=description, mime_type=mime_type)

    def embed_pdf(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = None,
    ) -> list[float]:
        return self._embed_media(source, modality="pdf", description=description, mime_type=mime_type)

    def embed_multimodal(
        self,
        *,
        text: str | None = None,
        image: str | Path | bytes | None = None,
        audio: str | Path | bytes | None = None,
        video: str | Path | bytes | None = None,
        pdf: str | Path | bytes | None = None,
        image_mime_type: str | None = None,
        audio_mime_type: str | None = None,
        video_mime_type: str | None = None,
        pdf_mime_type: str | None = None,
    ) -> list[float]:
        """Return a single embedding across the provided parts.

        Text is appended as a normal content part. If audio or video is present,
        that modality still flows through the same chunking path as the dedicated
        media entrypoints, with the text/image/pdf parts carried along in
        `base_parts` for each chunk embedding.
        """
        base_parts = self._text_parts(text)
        if image is not None:
            base_parts.append(self._media_part(image, modality="image", resolved_mime=image_mime_type))
        if pdf is not None:
            base_parts.append(self._media_part(pdf, modality="pdf", resolved_mime=pdf_mime_type))

        chunked = []
        if audio is not None:
            chunked.append(("audio", audio, audio_mime_type))
        if video is not None:
            chunked.append(("video", video, video_mime_type))

        if len(chunked) > 1:
            raise ValueError("embed_multimodal supports at most one audio/video input per call")

        if chunked:
            modality, source, explicit_mime = chunked[0]
            return self._embed_media(
                source,
                modality=modality,
                mime_type=explicit_mime,
                base_parts=base_parts,
            )

        if not base_parts:
            raise ValueError("embed_multimodal requires at least one text or media input")
        return self._embed_parts(base_parts, self._document_config())

    def _embed_media(
        self,
        source: str | Path | bytes,
        *,
        modality: str,
        description: str | None = None,
        mime_type: str | None = None,
        base_parts: list[Any] | None = None,
    ) -> list[float]:
        parts = list(base_parts or [])
        parts.extend(self._text_parts(description))

        if isinstance(source, bytes):
            parts.append(
                self._media_part(
                    source,
                    modality=modality,
                    resolved_mime=self._resolve_inline_mime_type(modality, mime_type),
                )
            )
            return self._embed_parts(parts, self._document_config())

        path = Path(source)
        resolved_mime = self._resolve_mime_type(path, modality, mime_type)
        chunk_limit = _MEDIA_LIMIT_SECONDS.get(modality)
        if chunk_limit is None:
            parts.append(self._media_part(path, modality=modality, resolved_mime=resolved_mime))
            return self._embed_parts(parts, self._document_config())

        duration_seconds = self._probe_duration_seconds(path)
        if duration_seconds <= chunk_limit:
            parts.append(self._media_part(path, modality=modality, resolved_mime=resolved_mime))
            return self._embed_parts(parts, self._document_config())

        self._require_binary("ffmpeg")
        with tempfile.TemporaryDirectory(prefix=f"{modality}_embed_chunks_") as tmp_dir:
            chunk_paths = self._write_media_chunks(
                path=path,
                temp_dir=Path(tmp_dir),
                max_chunk_seconds=chunk_limit,
                duration_seconds=duration_seconds,
            )
            vectors = [
                self._embed_parts(
                    [*parts, self._media_part(chunk_path, modality=modality, resolved_mime=resolved_mime)],
                    self._document_config(),
                )
                for chunk_path in chunk_paths
            ]
        return self._average_vectors(vectors)

    def _text_parts(self, text: str | None) -> list[Any]:
        if text is None:
            return []
        return [self._make_text_part(text)]

    def _media_part(
        self,
        source: str | Path | bytes,
        *,
        modality: str,
        resolved_mime: str | None = None,
    ) -> Any:
        if isinstance(source, bytes):
            if resolved_mime is None:
                resolved_mime = self._resolve_inline_mime_type(modality, None)
            return self._make_bytes_part(source, resolved_mime)

        path = Path(source)
        if resolved_mime is None:
            resolved_mime = self._resolve_mime_type(path, modality, None)
        # The Gemini embeddings API accepts bytes/parts here rather than a streaming
        # file handle, so local paths are materialized into memory before upload.
        return self._make_bytes_part(self._read_media_bytes(path), resolved_mime)

    def _embed_parts(self, parts: list[Any], config: Any) -> list[float]:
        content = self._make_content(parts)
        return self._embed([content], config)[0]

    def _embed(self, contents: list[Any], config: Any | None = None) -> list[list[float]]:
        vectors = self._embed_raw(contents, config or self._document_config())
        return [self._normalize_vector(vector) for vector in vectors]

    def _embed_raw(self, contents: list[Any], config: Any) -> list[list[float]]:
        result = self._client_obj().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=contents,
            config=config,
        )
        return [list(embedding.values) for embedding in result.embeddings]

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Expected embedding dimension {EMBEDDING_DIMENSIONS}, got {len(vector)}"
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise ValueError("Embedding vector norm is zero")
        return [value / norm for value in vector]

    def _average_vectors(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            raise ValueError("Expected at least one chunk embedding")
        length = len(vectors[0])
        averaged = [0.0] * length
        for vector in vectors:
            if len(vector) != length:
                raise ValueError("Chunk embeddings must share the same dimensionality")
            for index, value in enumerate(vector):
                averaged[index] += value
        scale = 1.0 / len(vectors)
        return self._normalize_vector([value * scale for value in averaged])

    def _resolve_inline_mime_type(self, modality: str, mime_type: str | None) -> str:
        return self._validate_mime_type(mime_type or _DEFAULT_MIME_TYPES[modality], modality=modality)

    def _resolve_mime_type(self, path: Path, modality: str, mime_type: str | None) -> str:
        guessed_mime = mime_type or mimetypes.guess_type(path.name)[0] or _DEFAULT_MIME_TYPES[modality]
        return self._validate_mime_type(guessed_mime, modality=modality)

    def _validate_mime_type(self, mime_type: str, modality: str | None = None) -> str:
        if modality is not None:
            supported = _SUPPORTED_MIME_TYPES[modality]
            if mime_type not in supported:
                supported_str = ", ".join(sorted(supported))
                raise ValueError(
                    f"Unsupported {modality} MIME type '{mime_type}'. Supported: {supported_str}"
                )
            return mime_type

        for supported_modality, supported in _SUPPORTED_MIME_TYPES.items():
            if mime_type in supported:
                return self._validate_mime_type(mime_type, modality=supported_modality)

        raise ValueError(f"Unsupported MIME type '{mime_type}'")

    def _read_media_bytes(self, path: Path) -> bytes:
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")
        return path.read_bytes()

    def _probe_duration_seconds(self, path: Path) -> float:
        self._require_binary("ffprobe")
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffprobe is required for audio/video chunking but was not found") from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed for {path}: {(result.stderr or result.stdout).strip() or 'unknown error'}"
            )

        output = result.stdout.strip()
        try:
            return float(output)
        except ValueError as exc:
            raise RuntimeError(f"ffprobe returned an invalid duration for {path}: {output!r}") from exc

    def _write_media_chunks(
        self,
        *,
        path: Path,
        temp_dir: Path,
        max_chunk_seconds: float,
        duration_seconds: float,
    ) -> list[Path]:
        chunk_count = max(1, math.ceil(duration_seconds / max_chunk_seconds))
        if chunk_count > _MAX_MEDIA_CHUNKS:
            raise ValueError(
                "Refusing to embed media with too many chunks: "
                f"modality_limit_seconds={max_chunk_seconds} "
                f"duration_seconds={duration_seconds:.3f} "
                f"chunk_count={chunk_count} max_chunks={_MAX_MEDIA_CHUNKS} path={path}"
            )

        chunk_paths = []
        start_seconds = 0.0
        chunk_index = 0

        while start_seconds < duration_seconds:
            chunk_duration = min(max_chunk_seconds, duration_seconds - start_seconds)
            chunk_path = temp_dir / f"{path.stem}_chunk_{chunk_index}{path.suffix}"
            self._run_ffmpeg_chunk(
                path=path,
                target=chunk_path,
                start_seconds=start_seconds,
                duration_seconds=chunk_duration,
            )
            chunk_paths.append(chunk_path)
            start_seconds += max_chunk_seconds
            chunk_index += 1

        return chunk_paths

    def _run_ffmpeg_chunk(
        self,
        *,
        path: Path,
        target: Path,
        start_seconds: float,
        duration_seconds: float,
    ) -> None:
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start_seconds:.3f}",
                    "-t",
                    f"{duration_seconds:.3f}",
                    "-i",
                    str(path),
                    "-c",
                    "copy",
                    str(target),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for audio/video chunking but was not found") from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed while chunking {path}: {(result.stderr or result.stdout).strip() or 'unknown error'}"
            )
        if not target.exists():
            raise RuntimeError(f"ffmpeg did not produce expected chunk file: {target}")

    def _require_binary(self, name: str) -> None:
        if shutil.which(name) is None:
            raise RuntimeError(f"{name} is required for audio/video chunking but was not found")

    def _client_obj(self):
        if self._client is None:
            genai, _ = self._load_sdk()
            self._client = genai.Client(api_key=GEMINI_API_KEY)
        return self._client

    def _document_config(self):
        if self._doc_config is None:
            _, types = self._load_sdk()
            self._doc_config = types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSIONS,
                task_type="RETRIEVAL_DOCUMENT",
            )
        return self._doc_config

    def _query_config_obj(self):
        if self._query_config is None:
            _, types = self._load_sdk()
            self._query_config = types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSIONS,
                task_type="RETRIEVAL_QUERY",
            )
        return self._query_config

    def _make_bytes_part(self, data: bytes, mime_type: str):
        _, types = self._load_sdk()
        return types.Part.from_bytes(data=data, mime_type=mime_type)

    def _make_text_part(self, text: str):
        _, types = self._load_sdk()
        return types.Part.from_text(text=text)

    def _make_content(self, parts: list[Any]):
        _, types = self._load_sdk()
        return types.Content(parts=parts)

    def _load_sdk(self):
        if self._genai is not None and self._types is not None:
            return self._genai, self._types

        try:
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "google-genai is required to use GeminiEmbedder. Install dependencies from requirements.txt."
            ) from exc

        self._genai = genai
        self._types = types
        return self._genai, self._types
