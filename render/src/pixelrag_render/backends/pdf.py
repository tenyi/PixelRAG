"""PDF backend for pixelshot.

Renders PDF pages to JPEG tiles using pdf2image (poppler).

Requires: pdf2image>=1.16.0 (install pixelrag-render[pdf])
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pixelrag_render.backends.pdf")


def render_pdf(
    path: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 200,
    pages: Optional[list[int]] = None,
    quality: int = 85,
) -> list[Path]:
    """Render a PDF to JPEG tiles.

    Each page is written as ``{stem}.png.tiles/tile_NNNN.jpg`` with a
    ``tiles.json`` manifest alongside.

    Args:
        path: Path to the source PDF file.
        output_dir: Directory to write the tile subdirectory into.
        dpi: Resolution for rendering (default 200 gives ~1650×2200px for A4).
        pages: 1-based list of page numbers to render. ``None`` renders all pages.
        quality: JPEG quality 1-100 (default 85).

    Returns:
        List containing the single tile directory Path on success.

    Raises:
        ImportError: If pdf2image is not installed.
        FileNotFoundError: If the PDF file does not exist.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError as e:
        raise ImportError(
            "pdf2image is required for PDF rendering. "
            "Install with: pip install 'pixelrag-render[pdf]'"
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = path.stem
    tile_dir = output_dir / f"{stem}.png.tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Rendering PDF: %s (dpi=%d)", path, dpi)

    convert_kwargs: dict = {
        "pdf_path": str(path),
        "dpi": dpi,
        "fmt": "jpeg",
        "jpegopt": {"quality": quality, "progressive": True},
        "thread_count": 4,
    }
    if pages is not None:
        # pdf2image uses 1-based page numbers
        convert_kwargs["first_page"] = min(pages)
        convert_kwargs["last_page"] = max(pages)

    images = convert_from_path(**convert_kwargs)

    saved_tiles: list[str] = []
    for idx, img in enumerate(images):
        # If caller provided a sparse page list, skip pages not in the list
        if pages is not None:
            page_num = min(pages) + idx
            if page_num not in pages:
                continue

        tile_name = f"tile_{idx:04d}.jpg"
        tile_path = tile_dir / tile_name
        img.save(str(tile_path), "JPEG", quality=quality)
        saved_tiles.append(tile_name)
        logger.debug("  Page %d → %s (%dx%d)", idx, tile_name, *img.size)

    manifest = {
        "source": str(path),
        "dpi": dpi,
        "total_pages": len(saved_tiles),
        "tiles": saved_tiles,
        "complete": True,
    }
    with open(tile_dir / "tiles.json", "w") as f:
        json.dump(manifest, f)

    logger.info("PDF rendered: %d pages → %s", len(saved_tiles), tile_dir)
    return [tile_dir]
