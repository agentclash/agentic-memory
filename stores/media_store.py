from __future__ import annotations

import mimetypes
import shutil
from pathlib import Path


_EXTENSION_DIRECTORIES = {
    ".png": "images",
    ".jpg": "images",
    ".jpeg": "images",
    ".webp": "images",
    ".gif": "images",
    ".bmp": "images",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".aac": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
    ".webm": "video",
    ".avi": "video",
    ".pdf": "documents",
}

_EXTENSION_MEDIA_TYPES = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".bmp": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".aac": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
    ".webm": "video",
    ".avi": "video",
    ".pdf": "pdf",
}


class MediaStore:
    """Owns durable local media files for memory records."""

    def __init__(self, base_path: str | Path):
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._base_path_resolved = self._base_path.resolve()

    def store(self, source_path: str | Path, memory_id: str) -> str:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Media file not found: {source}")
        if not source.is_file():
            raise ValueError(f"Media source is not a file: {source}")

        target = self._target_path(source.name, memory_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return str(target)

    def store_bytes(self, data: bytes, filename: str, memory_id: str) -> str:
        if not data:
            raise ValueError("Cannot store empty media payload")
        target = self._target_path(filename, memory_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    def retrieve(self, media_ref: str | Path) -> str:
        path = self._validate_owned(media_ref)
        if not path.exists():
            raise FileNotFoundError(f"Stored media not found: {path}")
        return str(path)

    def ensure_owned(self, media_ref: str | Path | None, memory_id: str) -> tuple[str | None, bool]:
        if not media_ref:
            return None, False
        if self.owns(media_ref):
            return str(media_ref), False

        owned_media_ref = self.store(media_ref, memory_id)
        return owned_media_ref, True

    def owns(self, media_ref: str | Path) -> bool:
        try:
            self._validate_owned(media_ref)
            return True
        except ValueError:
            return False

    @staticmethod
    def resolve_media_type(media_ref: str | Path, media_type: str | None = None) -> str:
        if media_type in {"image", "audio", "video", "pdf"}:
            return media_type

        path = Path(media_ref)
        suffix = path.suffix.lower()
        if suffix in _EXTENSION_MEDIA_TYPES:
            return _EXTENSION_MEDIA_TYPES[suffix]

        guessed_mime = mimetypes.guess_type(path.name)[0]
        if guessed_mime:
            if guessed_mime.startswith("image/"):
                return "image"
            if guessed_mime.startswith("audio/"):
                return "audio"
            if guessed_mime.startswith("video/"):
                return "video"
            if guessed_mime == "application/pdf":
                return "pdf"

        raise ValueError(f"Unsupported media file type for {path.name}")

    def delete(self, media_ref: str | Path) -> None:
        path = self._validate_owned(media_ref)
        if not path.exists():
            return
        if not path.is_file():
            raise ValueError(f"Stored media path is not a file: {path}")
        path.unlink()

    def _target_path(self, filename: str, memory_id: str) -> Path:
        suffix = Path(filename).suffix.lower()
        directory = self._media_directory(filename)
        return self._base_path / directory / f"{memory_id}{suffix}"

    def _validate_owned(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self._base_path_resolved):
            raise ValueError(f"Path is not under media store root: {path}")
        return resolved

    def _media_directory(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in _EXTENSION_DIRECTORIES:
            return _EXTENSION_DIRECTORIES[suffix]

        guessed_mime = mimetypes.guess_type(filename)[0]
        if guessed_mime:
            if guessed_mime.startswith("image/"):
                return "images"
            if guessed_mime.startswith("audio/"):
                return "audio"
            if guessed_mime.startswith("video/"):
                return "video"
            if guessed_mime == "application/pdf":
                return "documents"

        raise ValueError(f"Unsupported media file type for {filename}")
