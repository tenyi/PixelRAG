#!/bin/bash
# One-liner setup: install pixelrag + register plugin with Claude Code
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Install pixelrag into an isolated env via uv
echo "Installing pixelrag..."
uv tool install --from "$REPO_DIR" pixelrag 2>/dev/null || \
    uv tool upgrade --from "$REPO_DIR" pixelrag

# Install playwright browser
echo "Installing Chromium..."
uvx playwright install chromium 2>/dev/null || true

echo ""
echo "Done. Start Claude Code with:"
echo "  claude --plugin-dir $SCRIPT_DIR"
echo ""
echo "Or register permanently:"
echo "  claude mcp add-json pixelbrowse '{}' # not needed, it's a skill-only plugin"
