# End-to-End Demo: Wikipedia → Search

Builds a visual search index from Wikipedia articles and queries it.

## Quick Start

```bash
cd pixelrag
uv run python demos/e2e/run.py
```

This will:
1. Start kiwix-serve with Simple English Wikipedia
2. Capture 100 article screenshots (~20s)
3. Chunk tiles into 1024px strips (~1s)
4. Embed chunks with Qwen3-VL on CPU (~5 min for 100 articles)
5. Build a FAISS index
6. Start a search API and run sample queries

## Prerequisites

- Simple English Wikipedia ZIM at `~/pixelrag-data/zim/wikipedia_en_simple.zim`
  (download: `curl -L https://download.kiwix.org/zim/wikipedia/wikipedia_en_simple_all_nopic_2026-05.zim -o ~/pixelrag-data/zim/wikipedia_en_simple.zim`)
- kiwix-serve binary at `.local/bin/kiwix-serve`

## Configuration

Edit `pixelrag.yaml` in this directory to change:
- Number of articles (`limit`)
- Embedding device (`cpu` or `cuda`)
- Output location
