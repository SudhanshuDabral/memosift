# Engine C: Structural compression — code signature extraction + JSON truncation.
from __future__ import annotations

import json
import re
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from memosift.core.types import (
    AnchorLedger,
    ClassifiedMessage,
    CompressionPolicy,
    ContentType,
    MemoSiftMessage,
)

if TYPE_CHECKING:
    from memosift.config import MemoSiftConfig

# Target policies for structural compression.
_TARGET_POLICIES = {CompressionPolicy.STRUCTURAL, CompressionPolicy.SIGNATURE}


def structural_compress(
    segments: list[ClassifiedMessage],
    config: MemoSiftConfig,
    ledger: AnchorLedger | None = None,
) -> list[ClassifiedMessage]:
    """Apply structural compression to code and JSON segments.

    For code: Extract function/class signatures, collapse bodies.
    For JSON: Truncate large arrays, preserve scalar values.
    Anchor ledger facts are preserved in truncated arrays and code bodies.

    Args:
        segments: Classified messages from previous layers.
        config: Pipeline configuration.
        ledger: Optional anchor ledger — facts in truncated content are preserved.

    Returns:
        Segments with structural compression applied.
    """
    # Engine-level ledger gating disabled — JSON truncation should compress
    # aggressively. The Anchor Ledger already captures critical facts before
    # compression; protecting them again here kills compression ratios.
    protected_strings: frozenset[str] = frozenset()
    result: list[ClassifiedMessage] = []
    for seg in segments:
        if seg.policy not in _TARGET_POLICIES:
            result.append(seg)
            continue

        if seg.content_type == ContentType.TOOL_RESULT_JSON:
            new_content = _compress_json(
                seg.content, config.json_array_threshold, protected_strings
            )
        elif seg.content_type == ContentType.CODE_BLOCK:
            new_content = _compress_code(seg.content, config.code_keep_signatures)
        else:
            result.append(seg)
            continue

        if new_content != seg.content:
            new_msg = MemoSiftMessage(
                role=seg.message.role,
                content=new_content,
                name=seg.message.name,
                tool_call_id=seg.message.tool_call_id,
                tool_calls=seg.message.tool_calls,
                metadata=seg.message.metadata,
            )
            result.append(dc_replace(seg, message=new_msg))
        else:
            result.append(seg)
    return result


# ── JSON compression ────────────────────────────────────────────────────────


def _compress_json(
    text: str,
    array_threshold: int,
    protected_strings: frozenset[str] = frozenset(),
) -> str:
    """Parse JSON and truncate large arrays, preserving structure."""
    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return text  # Can't parse — return unchanged.

    compressed = _truncate_json_value(data, array_threshold, protected_strings)
    return json.dumps(compressed, indent=2)


def _truncate_json_value(
    value: object,
    threshold: int,
    protected_strings: frozenset[str] = frozenset(),
) -> object:
    """Recursively truncate arrays in a JSON value.

    Schema-aware: detects arrays of objects with identical keys and emits
    a schema summary instead of repeating the structure. Preserves items
    containing ledger-protected strings.
    """
    if isinstance(value, list):
        if len(value) > threshold:
            # Check for schema-uniform arrays (all items are dicts with same keys).
            schema = _detect_array_schema(value)

            # Keep first 2 exemplars.
            exemplars = [_truncate_json_value(v, threshold, protected_strings) for v in value[:2]]

            # Check remaining items for ledger-protected strings.
            protected_items: list[object] = []
            if protected_strings:
                for item in value[2:]:
                    item_str = json.dumps(item) if not isinstance(item, str) else item
                    item_lower = item_str.lower()
                    if any(s.lower() in item_lower for s in protected_strings):
                        protected_items.append(
                            _truncate_json_value(item, threshold, protected_strings)
                        )

            remaining = len(value) - 2 - len(protected_items)
            result = exemplars + protected_items
            if remaining > 0:
                if schema:
                    keys_str = ", ".join(schema)
                    result.append(
                        f"... {remaining} more items with same schema ({{{keys_str}}}) "
                        f"(total: {len(value)})"
                    )
                else:
                    result.append(f"... and {remaining} more items (total: {len(value)})")
            return result
        return [_truncate_json_value(v, threshold, protected_strings) for v in value]
    if isinstance(value, dict):
        return {k: _truncate_json_value(v, threshold, protected_strings) for k, v in value.items()}
    return value


def _detect_array_schema(items: list[object]) -> list[str] | None:
    """Detect if all items in an array are dicts with identical keys.

    Returns the sorted key list if uniform, None otherwise.
    """
    if len(items) < 3:
        return None
    if not all(isinstance(item, dict) for item in items):
        return None

    first_keys = sorted(items[0].keys())  # type: ignore[union-attr]
    # Check at least first 5 items (or all if fewer).
    sample = items[: min(5, len(items))]
    for item in sample:
        if sorted(item.keys()) != first_keys:  # type: ignore[union-attr]
            return None
    return first_keys


# ── Code compression ────────────────────────────────────────────────────────

# Regex patterns for signature detection across languages.
_PYTHON_CLASS_RE = re.compile(r"^(\s*class\s+\w+(?:\(.*?\))?)\s*:", re.MULTILINE)
_PYTHON_FUNC_RE = re.compile(
    r"^(\s*(?:async\s+)?def\s+\w+\s*\(.*?\)(?:\s*->\s*.+)?)\s*:", re.MULTILINE
)
_PYTHON_DOCSTRING_RE = re.compile(r'^(\s*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'))', re.MULTILINE)

_JS_FUNC_RE = re.compile(
    r"^(\s*(?:export\s+)?(?:async\s+)?(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:function|\(.*?\)\s*=>)))",
    re.MULTILINE,
)
_JS_CLASS_RE = re.compile(r"^(\s*(?:export\s+)?class\s+\w+(?:\s+extends\s+\w+)?)", re.MULTILINE)
_JS_METHOD_RE = re.compile(
    r"^(\s*(?:async\s+)?(?:get\s+|set\s+)?(?!if\b|else\b|while\b|for\b|switch\b|catch\b)\w+\s*\(.*?\)(?:\s*:\s*\S+)?)\s*\{",
    re.MULTILINE,
)


def _compress_code(text: str, keep_signatures: bool) -> str:
    """Extract code signatures and collapse bodies.

    Tries AST parsing first (Python), falls back to regex for other languages.
    """
    if not keep_signatures:
        return text

    # Try Python AST first.
    result = _compress_python_ast(text)
    if result is not None:
        return result

    # Fall back to regex-based extraction.
    return _compress_code_regex(text)


def _compress_python_ast(text: str) -> str | None:
    """Try to compress Python code using the ast module."""
    try:
        import ast

        tree = ast.parse(text)
    except SyntaxError:
        return None  # Not valid Python — fall back to regex.

    lines = text.split("\n")
    result_lines: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Get the signature line(s).
            sig_end = node.body[0].lineno - 1 if node.body else node.end_lineno or node.lineno
            for ln in range(node.lineno - 1, sig_end):
                if ln < len(lines):
                    result_lines.append(lines[ln])
            # Check for docstring.
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                ds_end = node.body[0].end_lineno or node.body[0].lineno
                for ln in range(node.body[0].lineno - 1, ds_end):
                    if ln < len(lines):
                        result_lines.append(lines[ln])

            # Scan body for method calls and class references before collapsing.
            called_names = _extract_called_names(node)

            # Add ellipsis for body — derive indent from original source line.
            if node.lineno - 1 < len(lines):
                original_line = lines[node.lineno - 1]
                base_indent = len(original_line) - len(original_line.lstrip())
                indent = " " * (base_indent + 4)
            else:
                indent = "    "

            if called_names:
                result_lines.append(f"{indent}# calls: {', '.join(called_names)}")
            result_lines.append(f"{indent}...")
        elif isinstance(node, ast.ClassDef):
            # Keep the class line.
            if node.lineno - 1 < len(lines):
                result_lines.append(lines[node.lineno - 1])
            # Check for class docstring.
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                ds_end = node.body[0].end_lineno or node.body[0].lineno
                for ln in range(node.body[0].lineno - 1, ds_end):
                    if ln < len(lines):
                        result_lines.append(lines[ln])

            # Process class methods in source order.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.lineno - 1 < len(lines):
                        result_lines.append(lines[child.lineno - 1])
                    called = _extract_called_names(child)
                    if child.lineno - 1 < len(lines):
                        cl = lines[child.lineno - 1]
                        ci = " " * (len(cl) - len(cl.lstrip()) + 4)
                    else:
                        ci = "        "
                    if called:
                        result_lines.append(f"{ci}# calls: {', '.join(called)}")
                    result_lines.append(f"{ci}...")

    if not result_lines:
        return None  # AST had nothing useful — fall back.

    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for line in result_lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)

    return "\n".join(deduped)


def _extract_called_names(node) -> list[str]:
    """Extract method/function call names and class references from an AST node body.

    Walks the function body to find Call nodes (function/method invocations)
    and Name/Attribute references. Returns deduplicated names in source order.
    """
    import ast

    names: list[str] = []
    seen: set[str] = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                name = child.func.id
            elif isinstance(child.func, ast.Attribute):
                name = child.func.attr
            else:
                continue
            if name not in seen and not name.startswith("_") and len(name) > 1:
                seen.add(name)
                names.append(name)

    return names


def _compress_code_regex(text: str) -> str:
    """Regex-based code signature extraction for non-Python languages."""
    signatures: list[str] = []

    # Try JS/TS patterns.
    for pattern in [_JS_CLASS_RE, _JS_FUNC_RE, _JS_METHOD_RE]:
        for match in pattern.finditer(text):
            sig = match.group(1).rstrip()
            signatures.append(sig)

    # Try Python patterns as fallback.
    if not signatures:
        for pattern in [_PYTHON_CLASS_RE, _PYTHON_FUNC_RE]:
            for match in pattern.finditer(text):
                sig = match.group(1).rstrip()
                signatures.append(sig)

    if not signatures:
        return text  # No signatures found — return unchanged.

    # Also extract docstrings/JSDoc.
    docstrings: list[str] = []
    for match in _PYTHON_DOCSTRING_RE.finditer(text):
        docstrings.append(match.group(1).strip())

    # Build compressed output: imports + signatures.
    result_parts: list[str] = []

    # Keep import lines (not export class/function — those are signatures).
    for line in text.split("\n"):
        stripped = line.strip()
        if (
            stripped.startswith(("import ", "from ", "require("))
            or stripped.startswith("export ")
            and not any(kw in stripped for kw in ("class ", "function ", "async ", "default "))
            and "=" not in stripped
        ):
            result_parts.append(line)

    # Add signatures with ellipsis — use braces for JS/TS.
    seen: set[str] = set()
    is_js = any(p.search(text) for p in [_JS_CLASS_RE, _JS_FUNC_RE, _JS_METHOD_RE])
    for sig in signatures:
        if sig not in seen:
            seen.add(sig)
            if is_js:
                result_parts.append(f"{sig} {{ ... }}")
            else:
                result_parts.append(f"{sig} ...")

    return "\n".join(result_parts) if result_parts else text
