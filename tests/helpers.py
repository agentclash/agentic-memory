import hashlib
import math
import re
import shutil
import tempfile
from pathlib import Path


class HashingEmbedder:
    """Deterministic offline embedder for tests."""

    def __init__(self, dimensions: int = 64):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_bytes(self, data: bytes, mime_type: str) -> list[float]:
        seed = hashlib.sha256(mime_type.encode("utf-8") + b":" + data).hexdigest()
        return self._embed(seed)

    def embed_image(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = "image/png",
    ) -> list[float]:
        return self._embed_media(source, mime_type or "image/png", description)

    def embed_audio(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = "audio/mpeg",
    ) -> list[float]:
        return self._embed_media(source, mime_type or "audio/mpeg", description)

    def embed_video(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = "video/mp4",
    ) -> list[float]:
        return self._embed_media(source, mime_type or "video/mp4", description)

    def embed_pdf(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = "application/pdf",
    ) -> list[float]:
        return self._embed_media(source, mime_type or "application/pdf", description)

    def embed_multimodal(
        self,
        *,
        text: str | None = None,
        image: str | Path | bytes | None = None,
        audio: str | Path | bytes | None = None,
        video: str | Path | bytes | None = None,
        pdf: str | Path | bytes | None = None,
        image_mime_type: str | None = "image/png",
        audio_mime_type: str | None = "audio/mpeg",
        video_mime_type: str | None = "video/mp4",
        pdf_mime_type: str | None = "application/pdf",
    ) -> list[float]:
        parts = []
        if text:
            parts.append(text)
        if image is not None:
            parts.append(self._media_seed(image, image_mime_type or "image/png"))
        if audio is not None:
            parts.append(self._media_seed(audio, audio_mime_type or "audio/mpeg"))
        if video is not None:
            parts.append(self._media_seed(video, video_mime_type or "video/mp4"))
        if pdf is not None:
            parts.append(self._media_seed(pdf, pdf_mime_type or "application/pdf"))
        return self._embed(" ".join(parts))

    def _embed_media(self, source: str | Path | bytes, mime_type: str, description: str | None) -> list[float]:
        seed = self._media_seed(source, mime_type)
        if description:
            seed = f"{description} {seed}"
        return self._embed(seed)

    def _media_seed(self, source: str | Path | bytes, mime_type: str) -> str:
        return hashlib.sha256(mime_type.encode("utf-8") + b":" + self._read_source(source)).hexdigest()

    def _read_source(self, source: str | Path | bytes) -> bytes:
        if isinstance(source, bytes):
            return source
        return Path(source).read_bytes()

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class DeterministicMultimodalEmbedder(HashingEmbedder):
    """Test stub that keeps media bytes and text queries in the same token space."""

    def embed_bytes(self, data: bytes, mime_type: str) -> list[float]:
        try:
            payload = data.decode("utf-8")
        except UnicodeDecodeError:
            payload = hashlib.sha256(data).hexdigest()
        return self._embed(f"{mime_type} {payload}")

    def _media_seed(self, source: str | Path | bytes, mime_type: str) -> str:
        data = self._read_source(source)
        try:
            payload = data.decode("utf-8")
        except UnicodeDecodeError:
            payload = hashlib.sha256(data).hexdigest()
        return f"{mime_type} {payload}"


def make_temp_chroma_dir(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
