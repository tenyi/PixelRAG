"""Render smoke test — the README's `pixelshot <page> --output ./tiles` path.

Renders a local HTML file through the default CDP backend (patched headless
Chrome, auto-downloaded and cached in CI) and asserts tile images are produced.
This is the one user-facing capability the README promises, so it is tested end
to end rather than mocked.
"""

from pathlib import Path

from pixelrag_render import render_file


def test_render_local_html_to_tiles(tmp_path):
    html = tmp_path / "page.html"
    html.write_text(
        "<html><body><h1>PixelRAG render smoke</h1>"
        + "<p>lorem ipsum dolor sit amet. </p>" * 60
        + "</body></html>"
    )
    out = tmp_path / "tiles"

    dirs = render_file(html, out)

    assert dirs, "render_file returned no tile directories"
    tile_dir = Path(dirs[0])
    tiles = sorted(tile_dir.glob("tile_*.jpg"))
    assert tiles, (
        f"no tile images produced in {tile_dir} "
        f"(contents: {[p.name for p in tile_dir.iterdir()]})"
    )
