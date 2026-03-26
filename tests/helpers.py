import hashlib
import math
import re
import shutil
import tempfile


class HashingEmbedder:
    """Deterministic offline embedder for tests."""

    def __init__(self, dimensions: int = 64):
        self._dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_bytes(self, data: bytes, mime_type: str) -> list[float]:
        seed = hashlib.sha256(mime_type.encode("utf-8") + b":" + data).hexdigest()
        return self._embed(seed)

    def embed_image(self, image_bytes: bytes, mime_type: str = "image/png") -> list[float]:
        return self.embed_bytes(image_bytes, mime_type)

    def embed_audio(self, audio_bytes: bytes, mime_type: str = "audio/mpeg") -> list[float]:
        return self.embed_bytes(audio_bytes, mime_type)

    def embed_video(self, video_bytes: bytes, mime_type: str = "video/mp4") -> list[float]:
        return self.embed_bytes(video_bytes, mime_type)

    def embed_pdf(self, pdf_bytes: bytes, mime_type: str = "application/pdf") -> list[float]:
        return self.embed_bytes(pdf_bytes, mime_type)

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


def make_temp_chroma_dir(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
