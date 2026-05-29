# Ingest Demo: Heterogeneous Documents

Captures screenshots from a mix of URLs, PDFs, and HTML files in one call.

## Run

```bash
cd pixelrag
uv run python demos/render/run.py
```

## What it does

1. Fetches 3 Wikipedia articles (URLs)
2. Renders 2 local HTML files (created on the fly)
3. Produces tiled JPEG screenshots for all 5
4. Shows a summary of tiles, sizes, and timing
