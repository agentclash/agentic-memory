from google import genai
from google.genai import types
from typing import Protocol

from config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS


class TextEmbedder(Protocol):
    def embed_text(self, text: str) -> list[float]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    """Converts content (text, image, audio) into embedding vectors using Gemini Embedding 2."""

    def __init__(self):
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._doc_config = types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
            task_type="RETRIEVAL_DOCUMENT",
        )
        self._query_config = types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
            task_type="RETRIEVAL_QUERY",
        )

    def _embed(self, contents: list, config: types.EmbedContentConfig | None = None) -> list[list[float]]:
        """Core call — returns one vector per item in contents."""
        result = self._client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=contents,
            config=config or self._doc_config,
        )
        return [e.values for e in result.embeddings]

    def embed_text(self, text: str) -> list[float]:
        """Embed text for storage (RETRIEVAL_DOCUMENT task type)."""
        return self._embed([text], self._doc_config)[0]

    def embed_query(self, text: str) -> list[float]:
        """Embed text for querying (RETRIEVAL_QUERY task type)."""
        return self._embed([text], self._query_config)[0]

    def embed_image(self, image_bytes: bytes, mime_type: str = "image/png") -> list[float]:
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        return self._embed([part])[0]

    def embed_audio(self, audio_bytes: bytes, mime_type: str = "audio/mpeg") -> list[float]:
        part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        return self._embed([part])[0]
