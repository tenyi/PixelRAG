#!/usr/bin/env python3
"""Ingest demo: capture heterogeneous documents as tiled screenshots.

Demonstrates pixelshot rendering a mix of:
- Wikipedia article URLs (via CDP lean capture)
- Local HTML files (auto-detected, rendered via file:// URL)
- Could also handle PDFs (requires pdf2image)

Run:
    cd pixelrag
    uv run python demos/render/run.py
"""

import shutil
import time
from pathlib import Path

OUTPUT = Path("demos/render/output")

# --- Sample data ---

WIKI_URLS = [
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://en.wikipedia.org/wiki/Screenshot",
    "https://en.wikipedia.org/wiki/FAISS",
]

SAMPLE_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: Georgia, serif; max-width: 800px; margin: 2em auto; padding: 0 1em; line-height: 1.6; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #e2e2e2; padding-bottom: .3em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
.highlight {{ background: #fff3cd; padding: .2em .4em; border-radius: 3px; }}
</style></head>
<body>
<h1>{title}</h1>
<p>{body}</p>
{extra}
</body></html>"""


def create_sample_html(output_dir: Path) -> list[Path]:
    """Create sample HTML files to demonstrate local file ingestion."""
    html_dir = output_dir / "sample_html"
    html_dir.mkdir(parents=True, exist_ok=True)

    files = []

    # A simple article-style page
    p1 = html_dir / "visual_retrieval.html"
    p1.write_text(
        SAMPLE_HTML.format(
            title="Visual Document Retrieval",
            body=(
                "Visual document retrieval captures documents as images and uses "
                "vision-language models to embed them into a shared vector space. "
                "Unlike text-based retrieval which requires parsing, visual retrieval "
                "preserves <span class='highlight'>layout, tables, figures, and formatting</span> "
                "that text extraction often loses."
            ),
            extra="""
<h2>Comparison</h2>
<table>
<tr><th>Method</th><th>Preserves Layout</th><th>Handles Tables</th><th>Needs Parser</th></tr>
<tr><td>Text extraction</td><td>No</td><td>Partial</td><td>Yes</td></tr>
<tr><td>HTML rendering</td><td>Partial</td><td>Yes</td><td>Yes</td></tr>
<tr><td><b>Visual (screenshot)</b></td><td><b>Yes</b></td><td><b>Yes</b></td><td><b>No</b></td></tr>
</table>
""",
        )
    )
    files.append(p1)

    # A data-heavy page with tables
    p2 = html_dir / "benchmark_results.html"
    rows = "".join(
        f"<tr><td>Config {i}</td><td>{70 + i * 1.3:.1f}</td><td>{0.5 + i * 0.02:.2f}s</td><td>{'LoRA' if i % 2 else 'Base'}</td></tr>"
        for i in range(15)
    )
    p2.write_text(
        SAMPLE_HTML.format(
            title="PixelRAG Benchmark Results",
            body="Evaluation results across different configurations and model variants.",
            extra=f"""
<h2>SimpleQA Retrieval Scores</h2>
<table>
<tr><th>Configuration</th><th>Recall@1</th><th>Latency</th><th>Model</th></tr>
{rows}
</table>
""",
        )
    )
    files.append(p2)

    return files


def main() -> None:
    from pixelrag_render.render import render_file

    # Clean previous output
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    print("=" * 60)
    print("  PixelRAG Ingest Demo: Heterogeneous Documents")
    print("=" * 60)
    print()

    # --- Step 1: Create sample local HTML ---
    print("[1] Creating sample HTML files...")
    html_files = create_sample_html(OUTPUT)
    for f in html_files:
        print(f"    {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    print()

    tiles_dir = OUTPUT / "tiles"
    tiles_dir.mkdir()
    all_results: list[tuple[str, int, float]] = []

    # --- Step 2: Render Wikipedia URLs ---
    print(f"[2] Rendering {len(WIKI_URLS)} Wikipedia articles (CDP backend)...")
    t0 = time.time()
    from pixelrag_render.render import render_urls

    url_tiles = render_urls(WIKI_URLS, str(tiles_dir), backend="cdp", workers=3)
    elapsed = time.time() - t0
    for td in url_tiles:
        n = len(list(td.glob("tile_*")))
        name = td.name.replace(".png.tiles", "")
        all_results.append((f"URL: {name}", n, elapsed / len(WIKI_URLS)))
    print(f"    {len(url_tiles)} pages rendered in {elapsed:.1f}s")
    print()

    # --- Step 3: Render local HTML files ---
    print(f"[3] Rendering {len(html_files)} local HTML files...")
    for html_file in html_files:
        t0 = time.time()
        result = render_file(str(html_file), str(tiles_dir), backend="cdp")
        elapsed = time.time() - t0
        for td in result:
            n = len(list(Path(td).glob("tile_*")))
            all_results.append((f"HTML: {html_file.name}", n, elapsed))
    print(f"    {len(html_files)} files rendered")
    print()

    # --- Summary ---
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    total_tiles = 0
    for name, n_tiles, elapsed in all_results:
        total_tiles += n_tiles
        print(f"  {name:<45} {n_tiles:>3} tiles  {elapsed:.1f}s")
    print(f"  {'─' * 55}")
    print(f"  {'TOTAL':<45} {total_tiles:>3} tiles")
    print()

    # Show output structure
    print("Output structure:")
    for td in sorted(tiles_dir.iterdir()):
        if td.is_dir():
            tiles = list(td.glob("tile_*"))
            size = sum(t.stat().st_size for t in tiles) / 1024
            print(f"  {td.name}/")
            print(f"    {len(tiles)} tiles, {size:.0f} KB total")
    print()
    print(f"All output in: {tiles_dir}")


if __name__ == "__main__":
    main()
