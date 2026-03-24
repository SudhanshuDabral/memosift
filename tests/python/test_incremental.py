# Tests for incremental compression (CompressionState + session incremental mode).
from __future__ import annotations

from memosift.core.state import CompressionState
from memosift.core.types import ContentType, MemoSiftMessage
from memosift.session import MemoSiftSession


# ── CompressionState unit tests ──────────────────────────────────────────────


def test_state_classification_cache():
    """Classification cache should store and retrieve content types."""
    state = CompressionState()
    state.cache_classification("def hello(): pass", ContentType.CODE_BLOCK)
    assert state.get_cached_classification("def hello(): pass") == ContentType.CODE_BLOCK
    assert state.get_cached_classification("different content") is None


def test_state_token_cache():
    """Token cache should store and retrieve counts."""
    state = CompressionState()
    state.cache_token_count("hello world", 3)
    assert state.get_cached_token_count("hello world") == 3
    assert state.get_cached_token_count("other") is None


def test_state_content_hash_tracking():
    """Content hashes should track seen content."""
    state = CompressionState()
    assert not state.has_content("test content")
    state.record_content_hash("test content", 0)
    assert state.has_content("test content")


def test_state_bump_sequence():
    """Sequence should increment on each bump."""
    state = CompressionState()
    assert state.sequence == 0
    assert state.bump_sequence() == 1
    assert state.bump_sequence() == 2
    assert state.sequence == 2


def test_state_output_hash():
    """Output hash should change when content changes."""
    state = CompressionState()
    state.set_output_hash(["hello", "world"])
    hash1 = state.output_hash
    assert len(hash1) == 32

    state.set_output_hash(["hello", "world", "!"])
    hash2 = state.output_hash
    assert hash1 != hash2


def test_state_idf_vocabulary():
    """IDF vocabulary should be a mutable dict."""
    state = CompressionState()
    assert state.idf_vocabulary == {}
    state.idf_vocabulary["hello"] = 1.5
    assert state.idf_vocabulary["hello"] == 1.5


# ── Session incremental mode ────────────────────────────────────────────────


def test_session_incremental_flag():
    """Session should create a CompressionState when incremental=True."""
    session = MemoSiftSession("general", incremental=True)
    assert session.incremental is True
    assert session.state is not None
    assert isinstance(session.state, CompressionState)


def test_session_non_incremental():
    """Session should not have a state when incremental=False."""
    session = MemoSiftSession("general")
    assert session.incremental is False
    assert session.state is None


async def test_incremental_compress_populates_state():
    """Compressing messages should populate the state's caches."""
    session = MemoSiftSession("general", incremental=True)
    messages = [
        MemoSiftMessage(role="system", content="You are a helpful assistant."),
        MemoSiftMessage(role="user", content="Hello"),
        MemoSiftMessage(role="assistant", content="Hi! How can I help you today?"),
        MemoSiftMessage(role="user", content="Tell me about Python"),
    ]

    compressed, report = await session.compress(messages)
    state = session.state
    assert state is not None
    assert state.sequence == 1
    assert len(state.output_hash) == 32


async def test_incremental_second_call_uses_cache():
    """Second compress call should benefit from cached state."""
    session = MemoSiftSession("general", incremental=True)
    messages = [
        MemoSiftMessage(role="system", content="You are a helpful assistant."),
        MemoSiftMessage(role="user", content="Hello"),
        MemoSiftMessage(role="assistant", content="Hi!"),
        MemoSiftMessage(role="user", content="What is Python?"),
    ]

    # First call populates caches.
    compressed1, report1 = await session.compress(messages)
    assert session.state is not None
    assert session.state.sequence == 1

    # Add a new message. Zone 2 messages from first call are already compressed.
    new_messages = list(compressed1) + [
        MemoSiftMessage(role="assistant", content="Python is a programming language."),
        MemoSiftMessage(role="user", content="Tell me more."),
    ]

    compressed2, report2 = await session.compress(new_messages)
    assert session.state.sequence == 2
    assert len(compressed2) >= 1


async def test_incremental_output_parity():
    """Incremental compression should produce same output as batch mode."""
    # Batch mode.
    batch_session = MemoSiftSession("general")
    messages = [
        MemoSiftMessage(role="system", content="You are a helpful assistant."),
        MemoSiftMessage(role="user", content="Hello"),
        MemoSiftMessage(role="assistant", content="Hi there!"),
        MemoSiftMessage(role="user", content="What is Python?"),
    ]
    batch_result, batch_report = await batch_session.compress(messages)

    # Incremental mode.
    inc_session = MemoSiftSession("general", incremental=True)
    inc_result, inc_report = await inc_session.compress(messages)

    # Same messages should produce same output.
    assert len(batch_result) == len(inc_result)
    for b, i in zip(batch_result, inc_result):
        assert b.role == i.role
        assert b.content == i.content
