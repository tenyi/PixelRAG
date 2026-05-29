const REPO_OWNER = "StarTrail-org"
const REPO_NAME = "PixelRAG"
const BRANCH = "main"

interface UptimeSite {
  name: string
  url: string
  slug: string
  status: "up" | "down" | "degraded"
  uptime: string
  uptimeDay: string
  uptimeWeek: string
  uptimeMonth: string
  uptimeYear: string
  time: number
  timeDay: number
  timeWeek: number
  timeMonth: number
  timeYear: number
  dailyMinutesDown: Record<string, number>
}

export type { UptimeSite }

export async function fetchUptimeSummary(): Promise<UptimeSite[]> {
  const url = `https://raw.githubusercontent.com/${REPO_OWNER}/${REPO_NAME}/${BRANCH}/history/summary.json`
  const res = await fetch(url, { cache: "no-store" })
  if (!res.ok) throw new Error(`Failed to fetch uptime data: ${res.status}`)
  return res.json()
}
