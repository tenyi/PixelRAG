"use client"

import * as React from "react"
import { Play, Copy, Check, Loader2, Terminal } from "lucide-react"
import { Button } from "@/components/ui/button"

export interface PlaygroundProps {
  method: "GET" | "POST"
  path: string
  curlPrefix: string
  defaultBody?: string
  defaultParams?: string
  buildPath?: (body: string, params: string) => string
}

export function ApiPlayground({
  method,
  path,
  curlPrefix,
  defaultBody,
  defaultParams,
  buildPath,
}: PlaygroundProps) {
  const [body, setBody] = React.useState(defaultBody ?? "")
  const [params, setParams] = React.useState(defaultParams ?? "")
  const [response, setResponse] = React.useState<string | null>(null)
  const [responseImage, setResponseImage] = React.useState<string | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [loading, setLoading] = React.useState(false)
  const [copied, setCopied] = React.useState(false)

  function getCurl(): string {
    if (method === "POST" && body) {
      const compact = body.replace(/\n\s*/g, " ").trim()
      return `${curlPrefix} \\\n  -d '${compact}'`
    }
    if (method === "GET" && defaultParams !== undefined) {
      const resolvedPath = buildPath ? buildPath(body, params) : path
      const host = typeof window !== "undefined" ? window.location.origin : "http://localhost:3000"
      return `curl "${host}/api${resolvedPath}"`
    }
    return curlPrefix
  }

  async function handleSend() {
    setLoading(true)
    setError(null)
    setResponse(null)
    setResponseImage(null)
    try {
      const resolvedPath = buildPath ? buildPath(body, params) : path
      const url = `/api${resolvedPath}`
      const init: RequestInit = {
        method,
        headers: method === "POST" ? { "Content-Type": "application/json" } : undefined,
        body: method === "POST" ? body : undefined,
      }
      const res = await fetch(url, init)
      if (!res.ok) setError(`HTTP ${res.status}`)
      const ct = res.headers.get("content-type") ?? ""
      if (ct.startsWith("image/")) {
        setResponseImage(URL.createObjectURL(await res.blob()))
      } else {
        const text = await res.text()
        try { setResponse(JSON.stringify(JSON.parse(text), null, 2)) }
        catch { setResponse(text) }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed")
    } finally {
      setLoading(false)
    }
  }

  function handleCopy() {
    navigator.clipboard.writeText(getCurl())
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="space-y-3">
      <div className="overflow-hidden rounded-xl border border-border/60 bg-card">
        {/* Toolbar */}
        <div className="flex items-center justify-between bg-secondary px-3 py-1.5">
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Terminal className="h-3 w-3" />
            curl
          </span>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost" size="sm"
              className="h-7 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground"
              onClick={handleCopy}
            >
              {copied ? <Check className="h-3 w-3 text-green-400" /> : <Copy className="h-3 w-3" />}
              {copied ? "Copied!" : "Copy"}
            </Button>
            <Button size="sm" className="h-7 gap-1.5 px-3 text-xs" onClick={handleSend} disabled={loading}>
              {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />}
              {loading ? "Running..." : "Run"}
            </Button>
          </div>
        </div>

        {/* curl body */}
        <div className="border-t border-border/40 p-4 font-mono text-xs leading-relaxed">
          {/* Prefix with syntax highlighting */}
          {highlightCurl(curlPrefix)}

          {/* POST — editable JSON body with syntax highlighting */}
          {method === "POST" && defaultBody !== undefined && (
            <>
              <span className="text-foreground/40">{" \\\n  -d '"}</span>
              <JsonEditor value={body} onChange={setBody} />
              <span className="text-foreground/40">{"'"}</span>
            </>
          )}

          {/* GET with editable params */}
          {method === "GET" && defaultParams !== undefined && (
            <input
              value={params}
              onChange={(e) => setParams(e.target.value)}
              spellCheck={false}
              className="mt-1 block w-full rounded-lg border-2 border-primary/30 bg-secondary px-3 py-2 font-mono text-xs text-foreground/90 shadow-[0_0_8px_rgba(139,105,67,0.08)] transition-colors focus:border-primary/60 focus:shadow-[0_0_12px_rgba(139,105,67,0.15)] focus:outline-none"
            />
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/5 px-4 py-3 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Response */}
      {(response || responseImage) && (
        <div className="overflow-hidden rounded-xl border border-border/60 bg-card">
          <div className="flex items-center gap-1.5 bg-secondary px-4 py-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
            <span className={error ? "text-red-400" : "text-green-400"}>●</span>
            Response
          </div>
          {responseImage ? (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img src={responseImage} alt="API response" className="max-h-96 border-t border-border/40 object-contain" />
          ) : (
            <pre className="max-h-80 overflow-auto border-t border-border/40 p-4 font-mono text-xs leading-relaxed text-foreground/80">
              {response}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

function highlightCurl(curl: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = []
  const regex = /\b(curl)\b|(-[XHIG]|--\w[\w-]*)|("(?:[^"\\]|\\.)*")|'([^']*)'|(https?:\/\/\S+)|(\\\n\s*)|(\s+)/g
  let match: RegExpExecArray | null
  let lastIndex = 0
  let key = 0
  while ((match = regex.exec(curl)) !== null) {
    if (match.index > lastIndex)
      nodes.push(<span key={key++} className="text-foreground/40">{curl.slice(lastIndex, match.index)}</span>)
    lastIndex = match.index + match[0].length
    if (match[1]) {
      nodes.push(<span key={key++} className="text-green-400">{match[1]}</span>)
    } else if (match[2]) {
      nodes.push(<span key={key++} className="text-amber-400">{match[2]}</span>)
    } else if (match[3]) {
      nodes.push(<span key={key++} className="text-purple-400">{match[3]}</span>)
    } else if (match[4] !== undefined) {
      nodes.push(<span key={key++} className="text-foreground/40">{"'"}</span>)
      nodes.push(<span key={key++} className="text-green-400/70">{match[4]}</span>)
      nodes.push(<span key={key++} className="text-foreground/40">{"'"}</span>)
    } else if (match[5]) {
      nodes.push(<span key={key++} className="text-cyan-400">{match[5]}</span>)
    } else {
      nodes.push(<span key={key++} className="text-foreground/30">{match[0]}</span>)
    }
  }
  if (lastIndex < curl.length)
    nodes.push(<span key={key++} className="text-foreground/40">{curl.slice(lastIndex)}</span>)
  return nodes
}

function highlightJson(code: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = []
  const regex = /("(?:[^"\\]|\\.)*")(\s*:)?|(\b(?:true|false|null)\b)|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([{}[\],:])|(\s+)/g
  let match: RegExpExecArray | null
  let lastIndex = 0
  let key = 0
  while ((match = regex.exec(code)) !== null) {
    if (match.index > lastIndex)
      nodes.push(<span key={key++} className="text-foreground/50">{code.slice(lastIndex, match.index)}</span>)
    lastIndex = match.index + match[0].length
    if (match[1] !== undefined) {
      if (match[2] !== undefined) {
        nodes.push(<span key={key++} className="text-foreground/90">{match[1]}</span>)
        nodes.push(<span key={key++} className="text-foreground/30">{match[2]}</span>)
      } else {
        nodes.push(<span key={key++} className="text-green-400">{match[1]}</span>)
      }
    } else if (match[3] !== undefined) {
      nodes.push(<span key={key++} className="text-amber-400">{match[3]}</span>)
    } else if (match[4] !== undefined) {
      nodes.push(<span key={key++} className="text-blue-400">{match[4]}</span>)
    } else if (match[5] !== undefined) {
      nodes.push(<span key={key++} className="text-foreground/25">{match[5]}</span>)
    } else if (match[6] !== undefined) {
      nodes.push(match[6])
    }
  }
  if (lastIndex < code.length)
    nodes.push(<span key={key++} className="text-foreground/50">{code.slice(lastIndex)}</span>)
  return nodes
}

function JsonEditor({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const preRef = React.useRef<HTMLPreElement>(null)

  function syncScroll() {
    if (textareaRef.current && preRef.current) {
      preRef.current.scrollTop = textareaRef.current.scrollTop
      preRef.current.scrollLeft = textareaRef.current.scrollLeft
    }
  }

  React.useEffect(() => {
    const el = textareaRef.current
    if (el) { el.style.height = "0"; el.style.height = `${el.scrollHeight}px` }
  }, [value])

  return (
    <div className="relative mt-1 rounded-lg border-2 border-primary/30 bg-secondary shadow-[0_0_8px_rgba(139,105,67,0.08)] transition-colors focus-within:border-primary/60 focus-within:shadow-[0_0_12px_rgba(139,105,67,0.15)] hover:border-primary/40">
      <pre
        ref={preRef}
        aria-hidden
        className="pointer-events-none overflow-hidden px-3 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words"
      >
        {highlightJson(value)}
        {"\n"}
      </pre>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onScroll={syncScroll}
        spellCheck={false}
        className="absolute inset-0 w-full resize-none overflow-hidden bg-transparent px-3 py-2 font-mono text-xs leading-relaxed text-transparent caret-primary focus:outline-none"
      />
    </div>
  )
}
