"""pixelshot: Document to image tiles.

Renders web pages, PDFs, and local files as tiled screenshots.
"""

from .render import render_url, render_pdf, render_file

__all__ = ["render_url", "render_pdf", "render_file"]
