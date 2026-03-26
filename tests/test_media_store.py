"""Verify app-owned media storage behavior."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stores.media_store import MediaStore


def make_source_file(suffix: str, data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="media_store_src_")
    os.close(fd)
    Path(path).write_bytes(data)
    return path


def test_store_copies_image_to_owned_directory():
    with tempfile.TemporaryDirectory(prefix="media_store_root_") as media_root:
        source = make_source_file(".png", b"image-bytes")
        try:
            store = MediaStore(media_root)
            stored = store.store(source, "memory-123")

            assert stored == str(Path(media_root) / "images" / "memory-123.png")
            assert Path(stored).read_bytes() == b"image-bytes"
        finally:
            Path(source).unlink(missing_ok=True)


def test_retrieve_validates_existence():
    with tempfile.TemporaryDirectory(prefix="media_store_root_") as media_root:
        store = MediaStore(media_root)
        target = Path(media_root) / "audio" / "memory-456.mp3"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"audio-bytes")

        assert store.retrieve(str(target)) == str(target)

        try:
            store.retrieve(str(Path(media_root) / "audio" / "missing.mp3"))
            raise AssertionError("Expected missing media to fail")
        except FileNotFoundError:
            pass


def test_store_bytes_writes_to_owned_directory():
    with tempfile.TemporaryDirectory(prefix="media_store_root_") as media_root:
        store = MediaStore(media_root)

        stored = store.store_bytes(b"pdf-bytes", "report.pdf", "memory-222")

        assert stored == str(Path(media_root) / "documents" / "memory-222.pdf")
        assert Path(stored).read_bytes() == b"pdf-bytes"


def test_retrieve_and_delete_reject_paths_outside_store_root():
    with tempfile.TemporaryDirectory(prefix="media_store_root_") as media_root:
        store = MediaStore(media_root)
        outside = make_source_file(".png", b"outside")
        try:
            try:
                store.retrieve(outside)
                raise AssertionError("Expected retrieve outside root to fail")
            except ValueError as exc:
                assert "not under media store root" in str(exc)

            try:
                store.delete(outside)
                raise AssertionError("Expected delete outside root to fail")
            except ValueError as exc:
                assert "not under media store root" in str(exc)
        finally:
            Path(outside).unlink(missing_ok=True)


def test_delete_removes_stored_file():
    with tempfile.TemporaryDirectory(prefix="media_store_root_") as media_root:
        source = make_source_file(".mp4", b"video-bytes")
        try:
            store = MediaStore(media_root)
            stored = store.store(source, "memory-789")
            assert Path(stored).exists()

            store.delete(stored)

            assert not Path(stored).exists()
        finally:
            Path(source).unlink(missing_ok=True)


if __name__ == "__main__":
    test_store_copies_image_to_owned_directory()
    test_retrieve_validates_existence()
    test_store_bytes_writes_to_owned_directory()
    test_retrieve_and_delete_reject_paths_outside_store_root()
    test_delete_removes_stored_file()
