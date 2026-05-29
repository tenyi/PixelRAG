"""Screenshot capture strategies — core render capability, independent of benchmarking."""

from .base import CaptureStrategy, TileCapture, ArticleCapture, article_url
from .connection import (
    WebsocketConnection,
    PlaywrightConnection,
    launch_websocket,
    launch_playwright,
)
from .cdp_sequential import CDPSequentialStrategy
from .cdp_directclip import CDPDirectClipStrategy
from .cdp_pertile_imgwait import CDPPerTileImgWaitStrategy
from .cdp_oneshot import CDPOneShotStrategy

__all__ = [
    "CaptureStrategy",
    "TileCapture",
    "ArticleCapture",
    "article_url",
    "WebsocketConnection",
    "PlaywrightConnection",
    "launch_websocket",
    "launch_playwright",
    "CDPSequentialStrategy",
    "CDPDirectClipStrategy",
    "CDPPerTileImgWaitStrategy",
    "CDPOneShotStrategy",
]
