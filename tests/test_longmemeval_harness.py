from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.longmemeval_eval import exact_match, format_session_text, iter_history_sessions


def test_iter_history_sessions_sorts_by_timestamp():
    instance = {
        "question_id": "q-1",
        "haystack_session_ids": ["s2", "s1"],
        "haystack_dates": ["2025-01-02T10:00:00Z", "2025-01-01T09:00:00Z"],
        "haystack_sessions": [
            [{"role": "user", "content": "second"}],
            [{"role": "assistant", "content": "first"}],
        ],
    }

    ordered = iter_history_sessions(instance)

    assert [session["session_id"] for session in ordered] == ["s1", "s2"]


def test_format_session_text_marks_evidence_turns():
    session = {
        "session_id": "sess-1",
        "date": "2025-03-30T10:00:00Z",
        "turns": [
            {"role": "user", "content": "I moved to Berlin."},
            {"role": "assistant", "content": "Noted.", "has_answer": True},
        ],
    }

    rendered = format_session_text(session)

    assert "Session date: 2025-03-30T10:00:00Z" in rendered
    assert "user: I moved to Berlin." in rendered
    assert "assistant [evidence]: Noted." in rendered


def test_exact_match_normalizes_whitespace_and_punctuation():
    assert exact_match("Berlin, Germany!", "berlin germany") == 1
    assert exact_match("Berlin", "Munich") == 0
