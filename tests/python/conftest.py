# Shared fixtures for MemoSift tests.
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the memosift package is importable from the source tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "python" / "src"))

from memosift.config import MemoSiftConfig
from memosift.core.types import MemoSiftMessage, ToolCall, ToolCallFunction


SPEC_DIR = Path(__file__).resolve().parent.parent.parent / "spec"
TEST_VECTORS_DIR = SPEC_DIR / "test-vectors"


def load_test_vector(name: str) -> dict:
    """Load a test vector JSON file from spec/test-vectors/."""
    path = TEST_VECTORS_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def messages_from_dicts(data: list[dict]) -> list[MemoSiftMessage]:
    """Convert a list of raw dicts (from test vectors) to MemoSiftMessage objects."""
    return [MemoSiftMessage.from_dict(d) for d in data]


@pytest.fixture
def classify_vector() -> dict:
    """Load the classify-001 test vector."""
    return load_test_vector("classify-001.json")


@pytest.fixture
def dedup_vector() -> dict:
    """Load the dedup-001 test vector."""
    return load_test_vector("dedup-001.json")


@pytest.fixture
def compress_vector() -> dict:
    """Load the compress-001 test vector."""
    return load_test_vector("compress-001.json")


@pytest.fixture
def default_config() -> MemoSiftConfig:
    """A default MemoSiftConfig for testing."""
    return MemoSiftConfig()


@pytest.fixture
def sample_messages() -> list[MemoSiftMessage]:
    """A minimal message list for quick unit tests."""
    return [
        MemoSiftMessage(role="system", content="You are a helpful assistant."),
        MemoSiftMessage(role="user", content="Hello"),
        MemoSiftMessage(
            role="assistant",
            content="Let me check.",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    function=ToolCallFunction(name="read_file", arguments='{"path": "test.py"}'),
                )
            ],
        ),
        MemoSiftMessage(
            role="tool",
            content="def hello():\n    print('hello')\n",
            tool_call_id="tc1",
            name="read_file",
        ),
        MemoSiftMessage(role="assistant", content="Here's the file content."),
        MemoSiftMessage(role="user", content="Thanks, what does it do?"),
    ]
