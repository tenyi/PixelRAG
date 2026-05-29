# PixelRAG Frontend Design Spec

## Overview

A modern web frontend for the PixelRAG visual retrieval engine, serving as both an academic paper companion demo and a functional API service. Built as a standalone Next.js application alongside the existing FastAPI backend.

## Goals

- Showcase visual retrieval quality with rich tile image display
- Provide interactive search (text + image queries) over the FAISS index
- Document the API with live try-it-out capability
- Look professional enough for paper/conference demos — not generic

## Non-Goals

- User authentication or multi-tenancy
- Index management or data ingestion UI
- Mobile-first design (desktop-first, responsive is fine)

## Architecture

### Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 15 (App Router) |
| Styling | Tailwind CSS 4 |
| Components | shadcn/ui |
| Animation | Framer Motion |
| Language | TypeScript |
| Backend | FastAPI (existing, unchanged except CORS) |

### Project Structure

```
web/                          ← new Next.js app
  src/
    app/
      page.tsx                ← search home
      docs/page.tsx           ← API reference
      status/page.tsx         ← index dashboard
      layout.tsx              ← shell (nav, theme provider)
    components/
      SearchBar.tsx           ← text input + image upload/drag-drop
      ResultGroup.tsx         ← article group with horizontal tile row
      TileCard.tsx            ← single tile result card
      Lightbox.tsx            ← fullscreen tile viewer with pan/zoom
      ComparePanel.tsx        ← side-by-side tile comparison
      ApiPlayground.tsx       ← try-it-live widget for /docs
      StatusCard.tsx          ← metric card for /status dashboard
    lib/
      api.ts                  ← typed fetch wrapper for all API endpoints
      types.ts                ← shared TypeScript types matching Pydantic models
  next.config.ts              ← rewrites /api/* → FastAPI
  tailwind.config.ts
  package.json

serve/                        ← existing (minimal changes)
  src/pixelrag_serve/api.py     ← add CORSMiddleware
```

### API Proxy

In development, `next.config.ts` rewrites `/api/*` to `http://localhost:30001/*` so the frontend can call the FastAPI backend without CORS issues. In production, CORS middleware on FastAPI allows the Next.js origin.

## Visual Design

### Color Palette

| Role | Value | Usage |
|------|-------|-------|
| Background | `#0c0c0c` | Page background |
| Surface | `#1a1a1a` | Cards, inputs, panels |
| Border | `#222222` | Card borders, dividers |
| Text primary | `#ffffff` | Headings, important text |
| Text secondary | `#888888` | Descriptions, metadata |
| Text muted | `#555555` | Labels, placeholders |
| Accent | `#6366f1` | Links, scores, CTAs, active states |
| Accent gradient | `#6366f1 → #8b5cf6` | Primary buttons |

### Typography

- **Inter** — UI text (body, labels, metadata)
- **Crimson Pro** — Branding headings (logo, page titles)
- **JetBrains Mono** — Code blocks, API examples, monospace data

### Design Principles

- Dark theme only (matches academic demo context, highlights tile images)
- Generous whitespace, no visual clutter
- Images are the hero — UI chrome stays minimal
- Subtle borders over drop shadows
- Micro-animations for state transitions (loading, lightbox open/close)

## Pages

### 1. Search Home (`/`)

The landing page and primary interface.

**Layout:**
- Centered logo + tagline at top: "PixelRAG — Visual retrieval over 15.7M Wikipedia tiles"
- Search bar below: text input with search button. Supports drag-and-drop or click-to-upload for image queries. Image preview shown inline when an image is attached.
- Mode chips below search bar: "Text query", "Image upload", "Drag & drop"
- Results appear below after search

**Search Controls (collapsible):**
- `n_docs` — number of results (default 10)
- `nprobe` — FAISS nprobe override
- `min_tile_height` — filter small/blank tiles
- `instruction` — custom embedding instruction
- Defaults are hidden; expand via "Advanced" toggle

**Results Display:**

Results are **grouped by article**. The API returns a flat ranked list of hits; the frontend groups them by `article_id`.

Each article group shows:
- Article title (derived from `url` field, decode the Wikipedia slug)
- External link to the Wikipedia article
- Tile count badge
- Horizontal scrollable row of tile cards

Each tile card shows:
- Tile image (loaded via `GET /tile?path=...`)
- Global rank badge (top-left corner, e.g. "#1")
- Cosine similarity score
- Tile height in pixels
- Tile position identifier (e.g. "tile 2:1" = tile_index 2, chunk_index 1)

**Status bar** between search bar and results:
- Result count
- Total latency
- Latency breakdown: measure client-side round-trip time (no backend changes needed; server-side encode/search breakdown is logged to stdout already)

### 2. API Documentation (`/docs`)

Custom-built API reference (not Swagger/ReDoc — those are functional but ugly and break visual consistency).

**Layout:**
- Left sidebar: endpoint list with HTTP method badges (POST green, GET blue)
- Guides section below endpoints: "Quick Start", "Python Client"
- Main content area: endpoint detail

**Each endpoint section:**
- Method + path + description
- Request body schema with syntax-highlighted JSON
- Response schema
- "Try It" playground: editable JSON input + Send button + response preview
- curl example

**Endpoints documented:**
- `POST /search` — primary search (text, image, or embedding queries)
- `GET /status` — index metadata and stats
- `GET /tile?path=...` — serve tile image by path
- `GET /health` — health check
- `POST /reconstruct` — reconstruct stored embeddings by vector_id

### 3. Index Dashboard (`/status`)

Displays data from `GET /status` in a visual dashboard.

**Metric cards (2×2 grid):**
- Total vectors (formatted: "15.7M")
- Embedding dimension
- Model name
- Index size (human-readable bytes)

**Additional info:**
- Index build timestamp
- Metadata size
- nlist / nprobe configuration
- Index and tiles directory paths

Auto-refreshes on page load. No polling needed (index stats are static during a session).

## Interactions

### Tile Lightbox

Click any tile card → full-screen overlay:
- Full-resolution tile image with pan and zoom (mouse wheel / pinch)
- Metadata sidebar: score, article title + link, tile position, tile height, y_offset
- Arrow keys or swipe to navigate between results (respects global rank order)
- Esc or click backdrop to close
- Animated open/close with Framer Motion

### Image Query

- Click the image upload area or drag-and-drop onto the search bar
- Shows image preview thumbnail inline in the search bar
- Sends base64-encoded image in the `queries[].image` field
- Can combine with text for multimodal query (text + image simultaneously)

### Side-by-Side Compare

- Checkbox or shift-click on tile cards to select 2+ tiles
- "Compare" button appears in a floating action bar
- Opens a comparison panel: selected tiles shown at equal width with scores overlaid
- Useful for evaluating retrieval quality on similar-looking results

### Search Controls

- Hidden by default behind an "Advanced" toggle
- Collapsible panel with labeled inputs for n_docs, nprobe, min_tile_height, instruction
- Changes take effect on next search
- URL query params reflect current settings (shareable search URLs)

## Backend Changes

Minimal changes to `serve/src/pixelrag_serve/api.py`:

1. **Add CORS middleware** — allow requests from Next.js dev server (`localhost:3000`) and production origin
2. **No other changes** — all existing endpoints remain as-is

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Dev & Deploy

### Development

```bash
# Terminal 1: FastAPI backend
pixelrag-serve --index-dir ./index --tiles-dir ./tiles --articles-json ./articles.json --device cuda

# Terminal 2: Next.js frontend
cd web && npm run dev
# Runs on localhost:3000, proxies /api/* → localhost:30001
```

### Production

```bash
# Build frontend
cd web && npm run build

# Run both
pixelrag-serve --device cuda --port 30001 &
cd web && npm start -- -p 3000
```

### next.config.ts Rewrites

```typescript
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: 'http://localhost:30001/:path*',
    },
  ];
},
```

## Scope & Milestones

### Phase 1: Core Search (MVP)

- Project scaffolding (Next.js + Tailwind + shadcn/ui)
- Search page with text query
- Result display with article grouping and tile images
- Tile lightbox with zoom
- CORS on FastAPI
- Navigation shell

### Phase 2: Full Features

- Image upload / drag-and-drop query
- Side-by-side comparison panel
- Advanced search controls
- API documentation page with try-it playground
- Index status dashboard

### Phase 3: Polish

- Loading states and skeleton screens
- Error handling and empty states
- Shareable search URLs (query params)
- Keyboard navigation (arrow keys in lightbox, Cmd+K for search focus)
- Performance optimization (image lazy loading, virtualized lists for large result sets)
