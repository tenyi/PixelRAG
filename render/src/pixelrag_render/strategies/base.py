"""Base interfaces for capture strategies.

Two layers:
- ChromeConnection: how to talk to Chrome (websocket vs Playwright)
- CaptureStrategy: how to capture tiles (sequential vs parallel)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TileCapture:
    """Raw capture result for one tile. Decoded later during verification."""

    image_bytes: bytes | None = None
    raw_file_path: str | None = None

    shot_ms: float = 0.0
    nav_ms: float = 0.0
    tile_index: int = 0
    clip_y: int = 0
    clip_h: int = 0


@dataclass
class ArticleCapture:
    """Raw capture results for all tiles of one article."""

    article_path: str
    tiles: list[TileCapture] = field(default_factory=list)
    page_height: int = 0
    n_tiles_expected: int = 0
    total_shot_ms: float = 0.0
    total_nav_ms: float = 0.0
    sem_wait_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


def article_url(article: dict) -> str:
    """Get navigate URL for an article. Supports both file:// and http://."""
    f = article["file"]
    return f if f.startswith("http") else f"file://{f}"


class ChromeConnection(Protocol):
    """Abstract connection to one Chrome process."""

    async def cdp(self, method: str, params: dict | None = None) -> dict:
        """Send a CDP command, wait for response."""
        ...

    async def close(self) -> None: ...


class CaptureStrategy(Protocol):
    """Interface for capture strategies."""

    @property
    def name(self) -> str: ...

    @property
    def fmt(self) -> str: ...

    async def setup(self) -> None: ...

    async def teardown(self) -> None: ...

    async def capture_articles(self, articles: list[dict]) -> list[ArticleCapture]: ...
