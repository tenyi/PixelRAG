const KEY = "pixelrag-search-history"
const MAX = 10

export function getHistory(): string[] {
  if (typeof window === "undefined") return []
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter((s): s is string => typeof s === "string").slice(0, MAX)
  } catch {
    return []
  }
}

export function addHistory(query: string): void {
  if (typeof window === "undefined") return
  const trimmed = query.trim()
  if (!trimmed) return
  try {
    const prev = getHistory()
    const deduped = prev.filter((s) => s !== trimmed)
    const next = [trimmed, ...deduped].slice(0, MAX)
    localStorage.setItem(KEY, JSON.stringify(next))
  } catch {
    // localStorage unavailable — silently ignore
  }
}

export function clearHistory(): void {
  if (typeof window === "undefined") return
  try {
    localStorage.removeItem(KEY)
  } catch {
    // silently ignore
  }
}
