# Tests for MemoSiftStream (Python).
from __future__ import annotations

from memosift.core.context_window import Pressure
from memosift.core.types import MemoSiftMessage
from memosift.stream import MemoSiftStream, StreamEvent


async def test_stream_creation():
    """Stream should create an incremental session."""
    stream = MemoSiftStream("coding", model="claude-sonnet-4-6")
    assert stream.session.incremental is True
    assert stream.session.state is not None
    assert stream.message_count == 0


async def test_stream_push_buffers_at_low_pressure():
    """Push should buffer messages when pressure is NONE."""
    # Without a model, pressure is always NONE.
    stream = MemoSiftStream("general")
    event = await stream.push(
        MemoSiftMessage(role="user", content="Hello")
    )
    assert event.action == "buffered"
    assert event.compressed is False
    assert stream.message_count == 1


async def test_stream_push_multiple_messages():
    """Multiple pushes should accumulate messages."""
    stream = MemoSiftStream("general")
    messages = [
        MemoSiftMessage(role="user", content="Hello"),
        MemoSiftMessage(role="assistant", content="Hi!"),
        MemoSiftMessage(role="user", content="What is Python?"),
    ]
    for msg in messages:
        await stream.push(msg)
    assert stream.message_count == 3


async def test_stream_flush():
    """Flush should force compression regardless of pressure."""
    stream = MemoSiftStream("general")
    await stream.push(MemoSiftMessage(role="system", content="You are helpful."))
    await stream.push(MemoSiftMessage(role="user", content="Hello"))
    await stream.push(MemoSiftMessage(role="assistant", content="Hi!"))
    await stream.push(MemoSiftMessage(role="user", content="Thanks"))

    event = await stream.flush()
    assert event.action == "compressed"
    assert event.compressed is True
    assert stream.message_count >= 1


async def test_stream_messages_property():
    """messages property should return a copy."""
    stream = MemoSiftStream("general")
    await stream.push(MemoSiftMessage(role="user", content="Hello"))
    msgs = stream.messages
    assert len(msgs) == 1
    # Should be a copy, not the internal list.
    msgs.append(MemoSiftMessage(role="user", content="extra"))
    assert stream.message_count == 1


async def test_stream_facts_empty():
    """Facts should be empty initially."""
    stream = MemoSiftStream("general")
    assert stream.facts == []


async def test_stream_flush_empty():
    """Flushing an empty stream should return buffered event."""
    stream = MemoSiftStream("general")
    event = await stream.flush()
    assert event.action == "buffered"
    assert event.compressed is False


async def test_stream_with_dict_messages():
    """Stream should work with dict messages (framework-native format)."""
    stream = MemoSiftStream("general")
    await stream.push({"role": "system", "content": "You are helpful."})
    await stream.push({"role": "user", "content": "Hello"})
    assert stream.message_count == 2

    event = await stream.flush()
    assert event.compressed is True


async def test_stream_event_dataclass():
    """StreamEvent should be a proper frozen dataclass."""
    event = StreamEvent(
        action="compressed",
        compressed=True,
        tokens_saved=100,
        pressure=Pressure.LOW,
    )
    assert event.action == "compressed"
    assert event.tokens_saved == 100
    assert event.pressure == Pressure.LOW
