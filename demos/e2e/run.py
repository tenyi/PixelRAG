#!/usr/bin/env python3
"""End-to-end demo: Wikipedia → visual search index → query.

Demonstrates the full PixelRAG pipeline via pixelrag index:
  source → ingest → chunk → embed → build index → serve → search

Run:
    cd pixelrag
    uv run python demos/e2e/run.py
    uv run python demos/e2e/run.py --limit 50
    uv run python demos/e2e/run.py --skip-build  # just serve existing index
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("e2e_demo")

DEFAULT_OUTPUT = Path(__file__).parent / "output"

SAMPLE_QUERIES = [
    "theory of relativity physics",
    "photosynthesis plants energy",
    "world war two history",
    "programming language computer",
    "solar system planets",
    "human brain neuroscience",
    "climate change global warming",
    "DNA genetics biology",
]


def search(query: str, port: int) -> list[dict]:
    body = json.dumps({"queries": [{"text": query}], "n_docs": 5}).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/search",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("results", [{}])[0].get("hits", [])


def _generate_html_report(results: list[dict], html_path: Path) -> None:
    """Generate an HTML page showing search results with tile images."""
    import base64

    rows = []
    for r in results:
        query = r["query"]
        rows.append(f'<h2>Q: "{query}"</h2>')
        for i, h in enumerate(r.get("hits", [])[:3]):
            url = h.get("url", "")
            title = (
                url.split("/")[-1].replace("_", " ") if url else f"#{h['article_id']}"
            )
            score = h["score"]
            tile_html = ""
            tile_path = h.get("_tile_path")
            if tile_path and Path(tile_path).exists():
                data = Path(tile_path).read_bytes()
                ext = Path(tile_path).suffix.lstrip(".")
                b64 = base64.b64encode(data).decode()
                tile_html = f'<img src="data:image/{ext};base64,{b64}" style="max-width:600px;border:1px solid #ddd;border-radius:4px;">'
            rows.append(f"""
            <div style="margin:1em 0;padding:1em;border:1px solid #222;border-radius:8px;background:#111;">
              <div style="color:#4a9eff;font-weight:600;">{i + 1}. {score:.3f} — {title}</div>
              {f'<div style="margin-top:0.5em;">{tile_html}</div>' if tile_html else ""}
            </div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PixelRAG E2E Results</title>
<style>body{{font-family:system-ui;background:#0a0a0a;color:#e0e0e0;max-width:800px;margin:2em auto;padding:0 1em;}}
h1{{color:#fff;}}h2{{color:#aaa;margin-top:2em;}}</style></head>
<body><h1>PixelRAG Search Results</h1>
{"".join(rows)}
</body></html>"""
    html_path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="PixelRAG E2E Demo")
    parser.add_argument("--limit", "-n", type=int, default=100)
    parser.add_argument(
        "--config", "-c", type=Path, default=Path(__file__).parent / "pixelrag.yaml"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--serve-port", type=int, default=31337)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--show-tiles",
        action="store_true",
        help="Display tile images in terminal (requires chafa)",
    )
    args = parser.parse_args()

    output = args.output.resolve()

    print("=" * 60)
    print("  PixelRAG End-to-End Demo")
    print("=" * 60)
    print(f"  Articles:  {args.limit}")
    print(f"  Device:    {args.device}")
    print(f"  Output:    {output}")
    print()

    # --- Build index ---
    if not args.skip_build:
        from pixelrag_index.config import load_config
        from pixelrag_index.pipelines import build

        config = load_config(str(args.config))
        if args.device:
            config.setdefault("embed", {})["device"] = args.device
        config["output"] = str(output)

        t0 = time.time()
        build(config, limit=args.limit)
        total_time = time.time() - t0
        print(f"\n  Pipeline completed in {total_time:.1f}s\n")

    # --- Serve + Search ---
    logger.info("Starting search API on :%d...", args.serve_port)
    env = os.environ.copy()
    env["PIXELRAG_INDEX_DIR"] = str(output)
    env["PIXELRAG_ARTICLES_JSON"] = str(output / "articles.json")
    serve_proc = subprocess.Popen(
        [sys.executable, "-m", "pixelrag_serve.api", "--port", str(args.serve_port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(
                    f"http://localhost:{args.serve_port}/health", timeout=1
                )
                break
            except Exception:
                time.sleep(2)
        else:
            raise TimeoutError("Search API failed to start")

        print("=" * 60)
        print("  Search Results")
        print("=" * 60)

        all_results = []
        for query in SAMPLE_QUERIES:
            hits = search(query, args.serve_port)
            all_results.append({"query": query, "hits": hits})
            print(f'\n  Q: "{query}"')
            if not hits:
                print("    (no results)")
                continue
            for i, h in enumerate(hits[:3]):
                url = h.get("url", "")
                score = h["score"]
                # Extract readable title from URL
                if url:
                    title = (
                        url.split("/")[-1]
                        .replace("_", " ")
                        .replace("%22", '"')
                        .replace("%20", " ")
                    )
                    title = urllib.parse.unquote(title)
                else:
                    title = f"#{h['article_id']}"
                print(f"    {i + 1}. {score:.3f}  {title}")
                # Collect tile path for HTML report
                if args.show_tiles:
                    aid = h["article_id"]
                    ti = h.get("tile_index", 0)
                    ci = h.get("chunk_index", 0)
                    for candidate in [
                        output
                        / "tiles"
                        / f"{aid}.png.tiles"
                        / f"chunk_{ti:04d}_{ci:02d}.png",
                        output / "tiles" / f"{aid}.png.tiles" / f"tile_{ti:04d}.jpg",
                    ]:
                        if candidate.exists():
                            h["_tile_path"] = str(candidate)
                            break

        # Generate HTML results page with tile images
        if args.show_tiles:
            html_path = output / "results.html"
            _generate_html_report(all_results, html_path)
            print(f"\n  Results with images: file://{html_path}")

        print()
        print(f"Search API: http://localhost:{args.serve_port}")
        print(
            f"Try: curl -X POST http://localhost:{args.serve_port}/search "
            f"-H 'Content-Type: application/json' "
            f'-d \'{{"queries": [{{"text": "your query"}}], "n_docs": 5}}\''
        )

    finally:
        serve_proc.terminate()


if __name__ == "__main__":
    main()
