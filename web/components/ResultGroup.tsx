"use client"

import { ExternalLink } from "lucide-react"
import type { ArticleGroup, Hit } from "@/lib/types"
import { TileCard } from "@/components/TileCard"
import { Badge } from "@/components/ui/badge"

interface ResultGroupProps {
  group: ArticleGroup
  selectedHits: Set<number>
  onSelectHit: (hit: Hit) => void
  onClickHit: (hit: Hit) => void
}

export function ResultGroup({
  group,
  selectedHits,
  onSelectHit,
  onClickHit,
}: ResultGroupProps) {
  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-center gap-3">
        <h3 className="truncate text-sm font-semibold">{group.title}</h3>
        <Badge variant="secondary">{group.hits.length} tile{group.hits.length !== 1 ? "s" : ""}</Badge>
        <a
          href={group.url}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-auto flex shrink-0 items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          Open article
          <ExternalLink className="h-3 w-3" />
        </a>
      </div>

      {/* Horizontal scrollable row of tiles */}
      <div className="flex gap-3 overflow-x-auto pb-2">
        {group.hits.map((hit) => (
          <TileCard
            key={hit.vector_id}
            hit={hit}
            rank={hit.rank}
            selected={selectedHits.has(hit.vector_id)}
            onSelect={onSelectHit}
            onClick={onClickHit}
          />
        ))}
      </div>
    </div>
  )
}
