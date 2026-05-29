"use client"

import * as React from "react"
import { ChevronRight, ChevronDown } from "lucide-react"
import { Input } from "@/components/ui/input"
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible"

export interface SearchOptions {
  n_docs: number
  nprobe?: number
  min_tile_height?: number
  instruction?: string
}

interface SearchControlsProps {
  options: SearchOptions
  onChange: (options: SearchOptions) => void
}

export function SearchControls({ options, onChange }: SearchControlsProps) {
  const [open, setOpen] = React.useState(false)

  function update(patch: Partial<SearchOptions>) {
    onChange({ ...options, ...patch })
  }

  function handleNumber(
    field: "n_docs" | "nprobe" | "min_tile_height",
    value: string
  ) {
    if (value === "") {
      if (field === "n_docs") return // n_docs is required
      update({ [field]: undefined })
      return
    }
    const num = parseInt(value, 10)
    if (!isNaN(num) && num > 0) {
      update({ [field]: num })
    }
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="w-full max-w-2xl">
      <CollapsibleTrigger className="flex w-full items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground">
        {open ? (
          <ChevronDown className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
        Advanced
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 rounded-lg border border-border bg-card p-4 sm:grid-cols-4">
          <Field label="Results (n_docs)">
            <Input
              type="number"
              min={1}
              max={200}
              value={options.n_docs}
              onChange={(e) => handleNumber("n_docs", e.target.value)}
            />
          </Field>
          <Field label="nprobe">
            <Input
              type="number"
              min={1}
              placeholder="128"
              value={options.nprobe ?? ""}
              onChange={(e) => handleNumber("nprobe", e.target.value)}
            />
          </Field>
          <Field label="Min tile height (px)">
            <Input
              type="number"
              min={0}
              placeholder="No filter"
              value={options.min_tile_height ?? ""}
              onChange={(e) => handleNumber("min_tile_height", e.target.value)}
            />
          </Field>
          <Field label="Instruction" className="col-span-2 sm:col-span-4">
            <Input
              type="text"
              placeholder="Retrieve images or text relevant to the user's query."
              value={options.instruction ?? ""}
              onChange={(e) =>
                update({ instruction: e.target.value || undefined })
              }
            />
          </Field>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function Field({
  label,
  children,
  className,
}: {
  label: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={className}>
      <label className="mb-1 block text-xs font-medium text-muted-foreground">
        {label}
      </label>
      {children}
    </div>
  )
}
