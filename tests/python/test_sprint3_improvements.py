# Tests for Sprint 3 improvements (v0.3): method name preservation,
# compression determinism, per-domain compression caps.
from __future__ import annotations

import pytest

from memosift.config import MemoSiftConfig
from memosift.core.engines.structural import _compress_code, _extract_called_names
from memosift.core.budget import _exceeds_domain_cap, _heuristic_count
from memosift.core.types import (
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
)


def _make_segment(
    content: str,
    content_type: ContentType = ContentType.TOOL_RESULT_TEXT,
    policy: CompressionPolicy = CompressionPolicy.MODERATE,
    original_index: int = 0,
) -> ClassifiedMessage:
    return ClassifiedMessage(
        message=MemoSiftMessage(
            role="tool",
            content=content,
            tool_call_id="tc1",
        ),
        content_type=content_type,
        policy=policy,
        original_index=original_index,
    )


# ── Item 11: Preserve Method Names ──────────────────────────────────────────


class TestPreserveMethodNames:
    def test_python_ast_emits_called_names(self) -> None:
        """AST compression should emit method calls as comments."""
        code = '''
def authenticate(user):
    """Authenticate the user."""
    session = create_session(user)
    validate_credentials(user.password)
    return session

def authorize(user, role):
    check_permissions(user, role)
    log_access(user)
'''
        result = _compress_code(code, keep_signatures=True)
        assert "# calls:" in result
        assert "create_session" in result or "validate_credentials" in result

    def test_python_ast_preserves_signatures(self) -> None:
        code = '''
def foo(x: int) -> str:
    return str(x)

class Bar:
    def baz(self):
        pass
'''
        result = _compress_code(code, keep_signatures=True)
        assert "def foo" in result
        assert "..." in result

    def test_js_regex_uses_braces(self) -> None:
        """JS/TS code should use { ... } not just ..."""
        code = '''
function authenticate(user) {
    const session = createSession(user);
    return session;
}

export class AuthService {
    login(credentials) {
        return this.validate(credentials);
    }
}
'''
        result = _compress_code(code, keep_signatures=True)
        assert "{ ... }" in result

    def test_extract_called_names(self) -> None:
        import ast
        code = '''
def example():
    foo()
    bar.baz()
    _private()
    x()
'''
        tree = ast.parse(code)
        func = tree.body[0]
        names = _extract_called_names(func)
        assert "foo" in names
        assert "baz" in names
        assert "_private" not in names  # Skip underscore-prefixed.
        # Single-char names may or may not be included (depends on length > 1).


# ── Item 14: Compression Determinism ────────────────────────────────────────


class TestCompressionDeterminism:
    def test_deterministic_seed_default(self) -> None:
        config = MemoSiftConfig()
        assert config.deterministic_seed == 42

    def test_deterministic_seed_none(self) -> None:
        config = MemoSiftConfig(deterministic_seed=None)
        assert config.deterministic_seed is None

    @pytest.mark.asyncio
    async def test_deterministic_output(self) -> None:
        """Same input produces identical output across runs."""
        import copy
        from memosift.core.pipeline import compress
        from memosift.core.types import AnchorLedger

        def make_messages():
            return [
                MemoSiftMessage(role="system", content="System prompt"),
                MemoSiftMessage(role="user", content="Fix the auth bug"),
                MemoSiftMessage(
                    role="assistant",
                    content="I'll check the authentication middleware for issues.",
                ),
                MemoSiftMessage(
                    role="tool",
                    content="Contents of src/auth.ts:\n" + "code " * 100,
                    tool_call_id="tc1",
                    name="read_file",
                ),
                MemoSiftMessage(role="user", content="What did you find?"),
            ]

        config = MemoSiftConfig(token_budget=500)

        # Fresh messages and ledger for each call.
        result1, _ = await compress(make_messages(), config=config, ledger=AnchorLedger())
        result2, _ = await compress(make_messages(), config=config, ledger=AnchorLedger())

        # Output should be identical.
        assert len(result1) == len(result2)
        for m1, m2 in zip(result1, result2):
            assert m1.content == m2.content
            assert m1.role == m2.role


# ── Item 15: Per-Domain Compression Caps ────────────────────────────────────


class TestPerDomainCaps:
    def test_code_block_protected_from_dropping(self) -> None:
        """CODE_BLOCK segments should be protected by domain cap."""
        seg = _make_segment(
            "function authenticate() { ... }",
            content_type=ContentType.CODE_BLOCK,
        )
        # Without original tokens, code blocks are always protected.
        assert _exceeds_domain_cap(seg) is True

    def test_error_trace_protected_from_dropping(self) -> None:
        seg = _make_segment(
            "TypeError: Cannot read 'userId'",
            content_type=ContentType.ERROR_TRACE,
        )
        assert _exceeds_domain_cap(seg) is True

    def test_text_not_capped(self) -> None:
        seg = _make_segment(
            "Some regular text content",
            content_type=ContentType.TOOL_RESULT_TEXT,
        )
        assert _exceeds_domain_cap(seg) is False

    def test_code_with_known_original_under_cap(self) -> None:
        """Code with known original tokens under 4x cap should be droppable."""
        seg = _make_segment(
            "x " * 100,  # ~100 tokens
            content_type=ContentType.CODE_BLOCK,
        )
        seg.message._memosift_original_tokens = 200  # 2x compression — under 4x cap.
        assert _exceeds_domain_cap(seg) is False

    def test_code_with_known_original_over_cap(self) -> None:
        """Code already compressed beyond 4x should be protected."""
        seg = _make_segment(
            "x " * 10,  # ~10 tokens
            content_type=ContentType.CODE_BLOCK,
        )
        seg.message._memosift_original_tokens = 500  # 50x compression — well over 4x cap.
        assert _exceeds_domain_cap(seg) is True

    @pytest.mark.asyncio
    async def test_budget_respects_domain_caps(self) -> None:
        """Budget enforcement should not drop code blocks."""
        from memosift.core.budget import enforce_budget
        from memosift.core.types import DependencyMap

        segments = [
            _make_segment(
                "Important code: function auth() { validate(); }",
                content_type=ContentType.CODE_BLOCK,
                policy=CompressionPolicy.SIGNATURE,
                original_index=0,
            ),
            _make_segment(
                "Verbose prose that can be dropped " * 20,
                content_type=ContentType.ASSISTANT_REASONING,
                policy=CompressionPolicy.AGGRESSIVE,
                original_index=1,
            ),
        ]
        # Set relevance so prose would normally be dropped first.
        segments[0].relevance_score = 0.1
        segments[1].relevance_score = 0.2

        config = MemoSiftConfig(token_budget=100)
        deps = DependencyMap()
        result = await enforce_budget(segments, config, deps)

        # Code block should survive (domain cap protection).
        code_survived = any(
            seg.content_type == ContentType.CODE_BLOCK for seg in result
        )
        assert code_survived
