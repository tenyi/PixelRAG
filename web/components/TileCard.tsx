"use client"

import * as React from "react"
import { tileUrl } from "@/lib/api"
import type { Hit } from "@/lib/types"
import { cn } from "@/lib/utils"

interface TileCardProps {
  hit: Hit
  rank: number
  selected?: boolean
  onSelect?: (hit: Hit) => void
  onClick?: (hit: Hit) => void
}

export function TileCard({ hit, rank, selected, onSelect, onClick }: TileCardProps) {
  const [imgError, setImgError] = React.useState(false)
  const [imgLoaded, setImgLoaded] = React.useState(false)

  return (
    <div
      className={cn(
        "tile-card-glow group relative shrink-0 cursor-pointer overflow-hidden rounded-xl border bg-card transition-all",
        selected ? "border-primary ring-2 ring-primary/30" : "border-border/60"
      )}
      onClick={() => onClick?.(hit)}
    >
      {/* Rank badge */}
      <div className="absolute left-2 top-2 z-10 flex h-6 min-w-6 items-center justify-center rounded-full bg-primary px-1.5 text-xs font-bold text-primary-foreground">
        #{rank}
      </div>

      {/* Selection checkbox */}
      <div
        className={cn(
          "absolute right-2 top-2 z-10 flex h-5 w-5 items-center justify-center rounded border transition-opacity",
          selected
            ? "border-primary bg-primary opacity-100"
            : "border-muted-foreground/50 bg-background/80 opacity-0 group-hover:opacity-100"
        )}
        onClick={(e) => {
          e.stopPropagation()
          onSelect?.(hit)
        }}
      >
        {selected && (
          <svg
            className="h-3 w-3 text-primary-foreground"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={3}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
        )}
      </div>

      {/* Image */}
      <div className="relative h-48 w-72 bg-muted">
        {imgError ? (
          <div className="flex h-full w-full items-center justify-center text-xs text-muted-foreground">
            Failed to load tile
          </div>
        ) : (
          <>
            {!imgLoaded && (
              <div className="absolute inset-0 animate-pulse bg-muted" />
            )}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={tileUrl(hit)}
              alt={`Tile ${hit.tile_index} from article ${hit.article_id}`}
              className={cn(
                "h-full w-full object-cover object-top transition-opacity duration-300",
                imgLoaded ? "opacity-100" : "opacity-0"
              )}
              onLoad={() => setImgLoaded(true)}
              onError={() => setImgError(true)}
            />
          </>
        )}
      </div>

      {/* Metadata footer */}
      <div className="flex items-center justify-between border-t border-border/60 px-3 py-1.5 text-xs">
        <span className="score-badge font-mono font-semibold text-[var(--pixelrag-score)]">
          {hit.score.toFixed(3)}
        </span>
        <span className="text-muted-foreground">{hit.tile_height}px</span>
      </div>
    </div>
  )
}
