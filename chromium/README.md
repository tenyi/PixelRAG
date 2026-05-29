# Chromium Screenshot Patches

Custom Chromium patches for high-throughput headless screenshot capture.
Adds three CDP parameters to `Page.captureScreenshot` and a helper method.

## Patches

**`screenshot-patches.diff`** — 222 lines, 5 files:

| Feature | Description |
|---------|-------------|
| `rawFilePath` | Write raw BGRA pixels to a file path (async ThreadPool). Bypasses PNG/JPEG encoding. |
| `directClip` | Capture clip region directly via `CopyFromSurface(src_rect)` without modifying viewport/emulation state. |
| `skipRedraw` | Lightweight `ForceRedrawWithCallback` → `CopyFromSurface`. Skips the full `GetSnapshotFromBrowser` presentation feedback wait. |

## Usage (CDP)

```javascript
// rawFilePath — write raw BGRA to /dev/shm (28MB for 875×8192)
await cdp("Page.captureScreenshot", {
    rawFilePath: "/dev/shm/tile.raw",
    fromSurface: true,
    optimizeForSpeed: true,
    clip: { x: 0, y: 0, width: 875, height: 8192, scale: 1 }
});

// directClip — capture region without emulation change
await cdp("Page.captureScreenshot", {
    directClip: true,
    clip: { x: 0, y: 1024, width: 875, height: 1024, scale: 1 }
});

// skipRedraw — ForceRedrawWithCallback then CopyFromSurface
await cdp("Page.captureScreenshot", {
    skipRedraw: true,
    rawFilePath: "/dev/shm/tile.raw",
    clip: { x: 0, y: 0, width: 875, height: 8192, scale: 1 }
});
```

## Raw file format

12-byte header + pixel data:
```
offset 0:  uint32 width
offset 4:  uint32 height
offset 8:  uint32 rowBytes (= width × 4)
offset 12: BGRA pixel data (rowBytes × height bytes)
```

Read in Python:
```python
import struct
from PIL import Image

data = open("tile.raw", "rb").read()
w, h, rb = struct.unpack_from("<III", data, 0)
img = Image.frombuffer("RGBA", (w, h), data[12:], "raw", "BGRA", rb, 1)
```

## Building

Requires Chromium source checkout. Tested on Chromium 150.0.7844.0.

```bash
# 1. Get Chromium source (if not already)
mkdir chromium && cd chromium
fetch --no-history chromium
cd src

# 2. Apply patches
git apply /path/to/screenshot-patches.diff

# 3. Configure build
mkdir -p out/Release
cat > out/Release/args.gn << 'EOF'
is_debug = false
is_official_build = true
is_component_build = false
symbol_level = 0
blink_symbol_level = 0
chrome_pgo_phase = 0
EOF

gn gen out/Release

# 4. Build
autoninja -C out/Release chrome
# ~20 min on 224 cores, ~2 hours on 16 cores
```

Build output: `out/Release/chrome` (~476MB)

## Compatibility

- Chromium 150.x (May 2026). May apply cleanly to nearby versions.
- `is_official_build=true` required for competitive performance (10x vs debug).
- All patches are in `content/browser/devtools/protocol/` (CDP layer) and
  `content/browser/renderer_host/` (widget host). No rendering engine changes.
