"""Async ZIM HTTP server using aiohttp. No GIL bottleneck.

Serves ZIM content via async HTTP with thread-pool for ZIM reads.
Much faster than Python ThreadingMixIn or single-threaded kiwix-serve.

Usage:
    python -m pixelrag_serve.zim_server_async --zim /path/to/wikipedia.zim --port 9454
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

from aiohttp import web
from libzim.reader import Archive


class ZimApp:
    def __init__(self, zim_path: str, workers: int = 32):
        self.archive = Archive(zim_path)
        self.book_name = Path(zim_path).stem
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.loop = None

    def _read_entry(self, entry_path: str) -> tuple[bytes, str] | None:
        """Read from ZIM (runs in thread pool)."""
        if not self.archive.has_entry_by_path(entry_path):
            return None
        entry = self.archive.get_entry_by_path(entry_path)
        item = entry.get_item()
        content = bytes(item.content)
        mimetype = item.mimetype.split(";")[0].strip()
        return content, mimetype

    async def handle(self, request: web.Request) -> web.Response:
        path = unquote(request.path)

        prefix = f"/content/{self.book_name}/"
        if path.startswith(prefix):
            entry_path = path[len(prefix) :]
        elif path.startswith("/"):
            entry_path = path[1:]
        else:
            entry_path = path

        if "?" in entry_path:
            entry_path = entry_path.split("?")[0]

        result = await self.loop.run_in_executor(
            self.pool, self._read_entry, entry_path
        )

        if result is None:
            return web.Response(status=404, text=f"Not found: {entry_path}")

        content, mimetype = result
        return web.Response(
            body=content,
            content_type=mimetype,
            headers={"Cache-Control": "public, max-age=3600"},
        )


def main():
    parser = argparse.ArgumentParser(description="Async ZIM HTTP server")
    parser.add_argument("--zim", required=True)
    parser.add_argument("--port", type=int, default=9454)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    app_state = ZimApp(args.zim, workers=args.workers)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", app_state.handle)

    async def on_startup(app):
        app_state.loop = asyncio.get_event_loop()

    app.on_startup.append(on_startup)

    print(
        f"Async ZIM server: http://{args.host}:{args.port}/content/{app_state.book_name}/"
    )
    print(f"ZIM: {args.zim} ({app_state.archive.article_count:,} articles)")
    print(f"Thread pool: {args.workers} workers")

    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
