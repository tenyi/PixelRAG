"""Direct websocket CDP backend for pixelshot.

No Playwright dependency — uses subprocess to launch Chrome and websockets
to communicate via CDP directly. ~35% faster than the Playwright-based cdp.py
backend due to eliminating the Node.js IPC layer.

Requirements: websockets, pillow (no playwright needed)

Usage:
    from pixelrag_render.backends.websocket import render_urls
    tile_dirs = render_urls(["https://example.com"], "./tiles", workers=4)
"""

import asyncio
import base64
import io
import json
import logging
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from PIL import Image

logger = logging.getLogger("pixelrag_render.backends.websocket")

VIEWPORT_W = 875
VIEWPORT_H = 1080

BROWSER_ARGS = [
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-background-networking",
    "--disable-features=Translate,MediaRouter,OptimizationHints",
    "--enable-gpu-rasterization",
    "--force-gpu-rasterization",
]


def _find_chrome() -> str:
    from ..chrome import find_chrome

    return find_chrome()


async def _connect_cdp(port: int, retries: int = 5, delay: float = 1.0):
    """Connect to Chrome's CDP websocket endpoint."""
    import websockets

    for attempt in range(retries):
        try:
            data = urllib.request.urlopen(
                f"http://localhost:{port}/json", timeout=3
            ).read()
            targets = json.loads(data)
            ws = await websockets.connect(
                targets[0]["webSocketDebuggerUrl"],
                open_timeout=10,
                max_size=50 * 1024 * 1024,
            )
            return ws
        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    raise ConnectionError(f"Failed to connect to Chrome on port {port}")


async def _cdp_send(ws, msg_id_ref: list, method: str, params: dict | None = None):
    """Send a CDP command and wait for its response."""
    msg_id_ref[0] += 1
    mid = msg_id_ref[0]
    msg = {"id": mid, "method": method}
    if params:
        msg["params"] = params
    await ws.send(json.dumps(msg))
    while True:
        r = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
        if r.get("id") == mid:
            if "error" in r:
                raise RuntimeError(f"CDP error: {r['error']}")
            return r.get("result", {})


async def capture_url(
    ws,
    msg_id_ref: list,
    url: str,
    tile_dir: Path,
    *,
    tile_h: int = 8192,
    quality: int = 85,
    viewport_w: int = VIEWPORT_W,
    image_format: str = "jpeg",
    from_surface: bool = True,
) -> int:
    """Capture a URL as tiled images via direct CDP websocket.

    Returns the number of tiles written.
    """
    tile_dir.mkdir(parents=True, exist_ok=True)

    await _cdp_send(ws, msg_id_ref, "Page.navigate", {"url": url})

    # Wait for fonts + layout to stabilize, return scrollHeight in one call
    result = await _cdp_send(
        ws,
        msg_id_ref,
        "Runtime.evaluate",
        {
            "expression": """new Promise(resolve => {
            document.fonts.ready.then(() => {
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        document.documentElement.style.scrollBehavior = 'auto';
                        const sh = document.documentElement.scrollHeight;
                        const body = document.body;
                        if (body) {
                            const bottom = Math.ceil(body.getBoundingClientRect().bottom);
                            resolve(Math.min(sh, Math.max(bottom, 1)));
                        } else {
                            resolve(sh);
                        }
                    });
                });
            });
        })""",
            "awaitPromise": True,
            "returnByValue": True,
        },
    )
    try:
        page_height = result["result"]["value"]
    except (KeyError, TypeError):
        page_height = tile_h

    tiles = []
    y = 0
    idx = 0

    while y < page_height:
        clip_h = min(tile_h, page_height - y)
        if clip_h <= 0:
            break

        params = {
            "format": image_format,
            "fromSurface": from_surface,
            "optimizeForSpeed": True,
            "clip": {
                "x": 0,
                "y": y,
                "width": viewport_w,
                "height": clip_h,
                "scale": 1,
            },
        }
        if image_format == "jpeg":
            params["quality"] = quality

        result = await _cdp_send(ws, msg_id_ref, "Page.captureScreenshot", params)

        img_bytes = base64.b64decode(result["data"])
        tile_path = (
            tile_dir / f"tile_{idx:04d}.{'jpg' if image_format == 'jpeg' else 'png'}"
        )

        if clip_h < tile_h:
            img = Image.open(io.BytesIO(img_bytes))
            w, h = img.size
            if h > clip_h:
                img = img.crop((0, 0, w, clip_h))
            img.save(
                tile_path, "JPEG" if image_format == "jpeg" else "PNG", quality=quality
            )
        else:
            tile_path.write_bytes(img_bytes)

        tiles.append(tile_path.name)
        idx += 1
        y += tile_h

    manifest = {
        "url": url,
        "page_height": page_height,
        "tiles": tiles,
        "complete": True,
    }
    with open(tile_dir / "tiles.json", "w") as f:
        json.dump(manifest, f)

    return len(tiles)


async def _worker(
    chrome_path: str,
    port: int,
    work_queue: asyncio.Queue,
    output_dir: Path,
    tile_height: int,
    quality: int,
    viewport_w: int,
    image_format: str,
    from_surface: bool,
    worker_id: int,
    stats: dict,
    results: list,
):
    """Async worker: owns a Chrome process, pulls URLs from queue."""
    proc = subprocess.Popen(
        [chrome_path, f"--remote-debugging-port={port}", "--headless"]
        + BROWSER_ARGS
        + ["about:blank"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        await asyncio.sleep(3)
        ws = await _connect_cdp(port)
        msg_id_ref = [0]

        await _cdp_send(ws, msg_id_ref, "Page.enable")
        await _cdp_send(
            ws,
            msg_id_ref,
            "Emulation.setDeviceMetricsOverride",
            {
                "width": viewport_w,
                "height": tile_height,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )

        while True:
            try:
                item = work_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            url = item["url"]
            stem = item["stem"]
            tile_dir = output_dir / f"{stem}.png.tiles"

            t0 = time.monotonic()
            try:
                n_tiles = await capture_url(
                    ws,
                    msg_id_ref,
                    url,
                    tile_dir,
                    tile_h=tile_height,
                    quality=quality,
                    viewport_w=viewport_w,
                    image_format=image_format,
                    from_surface=from_surface,
                )
                stats["done"] += 1
                elapsed = time.monotonic() - t0
                logger.info(
                    "[w%d] %s → %d tiles (%.1fs)", worker_id, url, n_tiles, elapsed
                )
                results.append(tile_dir)
            except Exception as e:
                stats["failed"] += 1
                logger.warning("[w%d] FAIL %s: %s", worker_id, url, str(e)[:200])

        await ws.close()
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _run_batch(
    urls: list[str],
    output_dir: Path,
    num_workers: int,
    tile_height: int,
    quality: int,
    viewport_w: int,
    image_format: str,
    from_surface: bool,
    stems: list[str] | None,
    chrome_path: str,
) -> list[Path]:
    work_queue: asyncio.Queue = asyncio.Queue()
    seen_stems: dict[str, int] = {}
    for i, url in enumerate(urls):
        if stems and i < len(stems):
            stem = str(stems[i])
        else:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            raw = (parsed.netloc + parsed.path).rstrip("/")
            stem = (
                raw.replace("/", "_")
                .replace(":", "_")
                .replace("?", "_")
                .replace("&", "_")
            )
            stem = stem[:200] or "page"
            count = seen_stems.get(stem, 0)
            seen_stems[stem] = count + 1
            if count > 0:
                stem = f"{stem}_{count}"
        work_queue.put_nowait({"url": url, "stem": stem})

    stats = {"done": 0, "failed": 0}
    results: list[Path] = []
    base_port = 9400

    actual_workers = min(num_workers, len(urls))
    workers = [
        _worker(
            chrome_path,
            base_port + wid,
            work_queue,
            output_dir,
            tile_height,
            quality,
            viewport_w,
            image_format,
            from_surface,
            wid,
            stats,
            results,
        )
        for wid in range(actual_workers)
    ]
    await asyncio.gather(*workers, return_exceptions=True)

    logger.info("Batch complete: done=%d failed=%d", stats["done"], stats["failed"])
    return results


def render_urls(
    urls: list[str],
    output_dir: str | Path,
    *,
    stems: list[str] | None = None,
    tile_height: int = 8192,
    quality: int = 85,
    viewport_width: int = VIEWPORT_W,
    workers: int = 4,
    image_format: str = "jpeg",
    from_surface: bool = True,
    chrome_path: str | None = None,
) -> list[Path]:
    """Render URLs to tiled images using direct CDP websocket.

    No Playwright dependency. Each worker launches its own Chrome process
    and communicates via CDP over websocket.

    Args:
        urls: URLs to capture.
        output_dir: Output directory for tile subdirectories.
        stems: Optional output directory name per URL.
        tile_height: Max tile height in pixels (default 8192).
        quality: JPEG quality 1-100 (default 85).
        viewport_width: Browser viewport width (default 875).
        workers: Number of parallel Chrome processes (default 4).
        image_format: 'jpeg' or 'png' (default 'jpeg').
        from_surface: CDP fromSurface param. True for batch (throughput),
                      False for serve (low latency). Default True.
        chrome_path: Path to Chrome binary. Auto-detected if None.

    Returns:
        List of Path objects for created tile directories.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not urls:
        return []

    chrome = chrome_path or _find_chrome()

    return asyncio.run(
        _run_batch(
            urls,
            output_dir,
            workers,
            tile_height,
            quality,
            viewport_width,
            image_format,
            from_surface,
            stems,
            chrome,
        )
    )
