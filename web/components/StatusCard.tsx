import * as React from "react"

interface StatusCardProps {
  label: string
  value: string
  sub?: string
  icon?: React.ReactNode
}

export function StatusCard({ label, value, sub, icon }: StatusCardProps) {
  return (
    <div className="status-card-accent rounded-xl border border-border/60 bg-card p-5">
      <div className="flex items-center gap-2">
        {icon && (
          <span className="text-primary/60">{icon}</span>
        )}
        <p className="text-[0.65rem] font-semibold uppercase tracking-wider text-muted-foreground">
          {label}
        </p>
      </div>
      <p className="mt-1.5 text-3xl font-bold tracking-tight truncate" title={value}>
        {value}
      </p>
      {sub && (
        <p className="mt-1 text-xs text-muted-foreground">{sub}</p>
      )}
    </div>
  )
}
