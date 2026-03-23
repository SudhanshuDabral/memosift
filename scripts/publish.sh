#!/usr/bin/env bash
# MemoSift — One-click publish to PyPI and npm.
#
# Usage:
#   ./scripts/publish.sh              # Publish both Python + TypeScript
#   ./scripts/publish.sh --python     # Publish Python only
#   ./scripts/publish.sh --typescript # Publish TypeScript only
#   ./scripts/publish.sh --dry-run    # Build and verify, don't upload
#
# Prerequisites:
#   - PYPI_TOKEN env var (or ~/.pypirc configured)
#   - npm login (run `npm whoami` to verify)
#   - Python: pip install build twine
#   - Node: npm installed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_DIR="$REPO_ROOT/python"
TS_DIR="$REPO_ROOT/typescript"

# ── Defaults ──────────────────────────────────────────────────────────────
PUBLISH_PYTHON=true
PUBLISH_TS=true
DRY_RUN=false

# ── Parse args ────────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --python)     PUBLISH_PYTHON=true; PUBLISH_TS=false ;;
    --typescript) PUBLISH_PYTHON=false; PUBLISH_TS=true ;;
    --dry-run)    DRY_RUN=true ;;
    --help|-h)
      echo "Usage: ./scripts/publish.sh [--python] [--typescript] [--dry-run]"
      exit 0 ;;
    *)
      echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# ── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

step()  { echo -e "\n${CYAN}▸ $1${NC}"; }
ok()    { echo -e "${GREEN}  ✓ $1${NC}"; }
warn()  { echo -e "${YELLOW}  ⚠ $1${NC}"; }
fail()  { echo -e "${RED}  ✗ $1${NC}"; exit 1; }

# ── Extract versions ──────────────────────────────────────────────────────
PY_VERSION=$(python -c "
import tomllib
with open('$PYTHON_DIR/pyproject.toml', 'rb') as f:
    print(tomllib.load(f)['project']['version'])
")
TS_VERSION=$(node -e "console.log(require('$TS_DIR/package.json').version)")

echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   MemoSift Publish — v$PY_VERSION          ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
  echo -e "${YELLOW}  DRY RUN — will build and verify but not upload${NC}"
fi

# ── Version consistency check ─────────────────────────────────────────────
step "Checking version consistency"
if [ "$PY_VERSION" != "$TS_VERSION" ]; then
  fail "Version mismatch! Python=$PY_VERSION, TypeScript=$TS_VERSION"
fi
ok "Both packages at v$PY_VERSION"

# ── Git status check ─────────────────────────────────────────────────────
step "Checking git status"
cd "$REPO_ROOT"
if [ -n "$(git status --porcelain)" ]; then
  warn "Uncommitted changes detected. Publish will proceed but consider committing first."
  git status --short
else
  ok "Working tree clean"
fi

# ── Run tests ─────────────────────────────────────────────────────────────
step "Running Python tests"
cd "$REPO_ROOT"
python -m pytest tests/python/ -x -q 2>&1 | tail -3
PY_TEST_EXIT=${PIPESTATUS[0]:-0}
if [ "$PY_TEST_EXIT" -ne 0 ]; then
  fail "Python tests failed — aborting publish"
fi
ok "Python tests passed"

step "Running TypeScript tests"
cd "$TS_DIR"
npm test 2>&1 | grep -E "Tests|passed|failed"
ok "TypeScript tests passed"

# ══════════════════════════════════════════════════════════════════════════
# PYTHON PUBLISH
# ══════════════════════════════════════════════════════════════════════════
if [ "$PUBLISH_PYTHON" = true ]; then
  step "Building Python package"
  cd "$PYTHON_DIR"
  rm -rf dist/ build/ src/*.egg-info
  python -m build 2>&1 | tail -5
  ok "Python build complete"

  step "Verifying Python package"
  twine check dist/* 2>&1 | tail -3
  ok "Package verification passed"

  # Show what we built.
  echo ""
  ls -lh dist/
  echo ""

  if [ "$DRY_RUN" = true ]; then
    ok "DRY RUN — skipping PyPI upload"
  else
    step "Publishing to PyPI"
    if [ -n "${PYPI_TOKEN:-}" ]; then
      twine upload -u __token__ -p "$PYPI_TOKEN" dist/*
    else
      twine upload dist/*
    fi
    ok "Published memosift==$PY_VERSION to PyPI"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
# TYPESCRIPT PUBLISH
# ══════════════════════════════════════════════════════════════════════════
if [ "$PUBLISH_TS" = true ]; then
  step "Building TypeScript package"
  cd "$TS_DIR"
  rm -rf dist/
  npm run build 2>&1 | tail -5
  ok "TypeScript build complete"

  step "Verifying npm package contents"
  npm pack --dry-run 2>&1 | tail -15
  echo ""

  # Verify README is included.
  if [ ! -f "$TS_DIR/README.md" ]; then
    fail "README.md missing in typescript/ — npm page will be blank"
  fi
  ok "README.md present"

  if [ "$DRY_RUN" = true ]; then
    ok "DRY RUN — skipping npm upload"
  else
    step "Publishing to npm"
    cd "$TS_DIR"
    npm publish --access public
    ok "Published memosift@$TS_VERSION to npm"
  fi
fi

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Publish complete — v$PY_VERSION          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""

if [ "$DRY_RUN" = false ]; then
  if [ "$PUBLISH_PYTHON" = true ]; then
    echo -e "  PyPI:  ${CYAN}https://pypi.org/project/memosift/$PY_VERSION/${NC}"
  fi
  if [ "$PUBLISH_TS" = true ]; then
    echo -e "  npm:   ${CYAN}https://www.npmjs.com/package/memosift/v/$TS_VERSION${NC}"
  fi
  echo ""
  echo -e "  ${YELLOW}Don't forget to create the GitHub release:${NC}"
  echo -e "  gh release create v$PY_VERSION --title \"v$PY_VERSION\" --notes-file CHANGELOG.md"
fi
