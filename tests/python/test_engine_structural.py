# Tests for Engine C: Structural Compression.
from __future__ import annotations

import json

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.engines.structural import (
    _compress_json,
    _compress_code,
    structural_compress,
)
from memosift.core.types import (
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
)


def _make_segment(
    content: str,
    content_type: ContentType,
    policy: CompressionPolicy,
) -> ClassifiedMessage:
    return ClassifiedMessage(
        message=MemoSiftMessage(role="tool", content=content, tool_call_id="tc1"),
        content_type=content_type,
        policy=policy,
    )


class TestJsonCompression:
    """Tests for JSON schema-aware compression."""

    def test_json_array_truncated(self) -> None:
        data = {"users": [{"id": i, "name": f"User{i}"} for i in range(50)]}
        result = _compress_json(json.dumps(data), 5)
        parsed = json.loads(result)
        # Array should be truncated to 2 exemplars + annotation.
        assert len(parsed["users"]) == 3
        assert "50" in str(parsed["users"][-1])

    def test_json_scalar_values_preserved(self) -> None:
        data = {"name": "Alice", "age": 30, "active": True}
        result = _compress_json(json.dumps(data), 5)
        parsed = json.loads(result)
        assert parsed == data

    def test_json_small_array_preserved(self) -> None:
        data = {"items": [1, 2, 3]}
        result = _compress_json(json.dumps(data), 5)
        parsed = json.loads(result)
        assert parsed["items"] == [1, 2, 3]

    def test_nested_json_recursion(self) -> None:
        """Arrays nested 3 levels deep are all truncated."""
        data = {
            "level1": [
                {"level2": [{"level3": list(range(20))} for _ in range(10)]}
                for _ in range(10)
            ]
        }
        result = _compress_json(json.dumps(data), 5)
        parsed = json.loads(result)
        assert len(parsed["level1"]) == 3  # Truncated.

    def test_invalid_json_passthrough(self) -> None:
        result = _compress_json("{broken json", 5)
        assert result == "{broken json"


class TestCodeCompression:
    """Tests for code signature extraction."""

    def test_python_signatures_preserved(self) -> None:
        code = '''class AuthService:
    """Handles authentication."""
    def __init__(self, db: Database, secret: str):
        self._db = db
        self._secret = secret
        self._cache = {}

    def authenticate(self, username: str, password: str) -> bool:
        """Verify credentials."""
        user = self._db.find_user(username)
        if user and check_password(password, user.hash):
            return True
        return False
'''
        result = _compress_code(code, keep_signatures=True)
        assert "class AuthService" in result
        assert "def __init__" in result
        assert "def authenticate" in result
        # Bodies should be collapsed.
        assert "self._cache = {}" not in result

    def test_python_bodies_collapsed(self) -> None:
        code = '''def hello(name: str) -> str:
    """Say hello."""
    greeting = f"Hello, {name}!"
    print(greeting)
    return greeting
'''
        result = _compress_code(code, keep_signatures=True)
        assert "def hello" in result
        # Implementation details should be gone.
        assert "greeting = " not in result

    def test_javascript_function_detection(self) -> None:
        """JS/TS signatures detected by regex fallback."""
        code = '''export class AuthService extends BaseService {
  private cache: Map<string, User> = new Map();

  async authenticate(username: string, password: string): Promise<User | null> {
    const user = await this.db.findUser(username);
    return user;
  }

  createSession(user: User): string {
    return jwt.sign({ userId: user.id });
  }
}'''
        result = _compress_code(code, keep_signatures=True)
        assert "class AuthService" in result

    def test_keep_signatures_false(self) -> None:
        code = "def hello():\n    pass\n"
        result = _compress_code(code, keep_signatures=False)
        assert result == code  # No change when disabled.


class TestStructuralCompress:
    """Integration tests for the full structural engine."""

    def test_json_segment_compressed(self) -> None:
        data = {"items": list(range(20))}
        seg = _make_segment(
            json.dumps(data),
            ContentType.TOOL_RESULT_JSON,
            CompressionPolicy.STRUCTURAL,
        )
        result = structural_compress([seg], MemoSiftConfig())
        assert "20" in result[0].content  # Count annotation.

    def test_code_segment_compressed(self) -> None:
        code = '''def add(a: int, b: int) -> int:
    """Add two numbers."""
    result = a + b
    return result
'''
        seg = _make_segment(code, ContentType.CODE_BLOCK, CompressionPolicy.SIGNATURE)
        result = structural_compress([seg], MemoSiftConfig())
        assert "def add" in result[0].content
        assert "result = a + b" not in result[0].content

    def test_non_target_untouched(self) -> None:
        seg = _make_segment("keep me", ContentType.TOOL_RESULT_TEXT, CompressionPolicy.MODERATE)
        result = structural_compress([seg], MemoSiftConfig())
        assert result[0].content == "keep me"

    def test_empty_input(self) -> None:
        result = structural_compress([], MemoSiftConfig())
        assert result == []
