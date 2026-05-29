"use client"

import { motion } from "framer-motion"
import { X } from "lucide-react"
import { tileUrl } from "@/lib/api"
import type { Hit } from "@/lib/types"
import { Button } from "@/components/ui/button"

interface ComparePanelProps {
  hits: Hit[]
  allHits: Hit[]
  onClose: () => void
}

export function ComparePanel({ hits, allHits, onClose }: ComparePanelProps) {
  return (
    <motion.div
      initial={{ y: "100%" }}
      animate={{ y: 0 }}
      exit={{ y: "100%" }}
      transition={{ type: "spring", damping: 25, stiffness: 300 }}
      className="fixed inset-x-0 bottom-0 z-[90] flex max-h-[70vh] flex-col border-t border-border bg-card shadow-2xl"
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-6 py-3">
        <h3 className="text-sm font-semibold">
          Comparing {hits.length} tiles
        </h3>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Tiles */}
      <div className="flex-1 overflow-auto p-6">
        <div
          className="grid gap-6"
          style={{
            gridTemplateColumns: `repeat(${Math.min(hits.length, 4)}, 1fr)`,
          }}
        >
          {hits.map((hit) => {
            const globalRank =
              allHits.findIndex((h) => h.vector_id === hit.vector_id) + 1
            const slug = hit.url.split("/wiki/").pop() ?? ""
            const title =
              decodeURIComponent(slug).replace(/_/g, " ") ||
              `Article #${hit.article_id}`

            return (
              <div
                key={hit.vector_id}
                className="flex flex-col overflow-hidden rounded-lg border border-border"
              >
                {/* Image */}
                <div className="bg-muted">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={tileUrl(hit)}
                    alt={`Tile ${hit.tile_index}`}
                    className="w-full object-contain"
                  />
                </div>

                {/* Metadata */}
                <div className="space-y-1.5 border-t border-border p-3 text-xs">
                  <div className="truncate font-medium">{title}</div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Score</span>
                    <span className="font-mono font-semibold text-[var(--pixelrag-score)]">
                      {hit.score.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Rank</span>
                    <span className="font-mono">#{globalRank}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Tile height</span>
                    <span className="font-mono">{hit.tile_height}px</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Vector ID</span>
                    <span className="font-mono">{hit.vector_id}</span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </motion.div>
  )
}
