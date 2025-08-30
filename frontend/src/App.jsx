import { useState, useRef, useEffect } from 'react'
import Header from './components/Header'
import { API_BASE_URL, DEV_SIGNALS_ENABLED } from './config'
import Sparkline from './components/Sparkline'
import BiggestLosers from './components/BiggestLosers'
import SignalsDev from './components/SignalsDev'

function App() {
  const [symbols, setSymbols] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [quotes, setQuotes] = useState({})
  const [charts, setCharts] = useState({})
  const [ogReady, setOgReady] = useState({})
  const ogPollRef = useRef(null)
  const resultsRef = useRef(null)
  useEffect(() => {
    return () => { if (ogPollRef.current) clearInterval(ogPollRef.current) }
  }, [])

  const handleLoserClick = (symbol) => {
    // Set the clicked symbol in the input and trigger analysis
    setSymbols(symbol)
    setError('')
    onAnalyze(symbol)
  }

  const onAnalyze = async (symbolsInput = null) => {
    const candidate = typeof symbolsInput === 'string' ? symbolsInput : symbols
    const input = (candidate || '').trim()
    if (!input) {
      setError('Please enter at least one stock symbol')
      return
    }
    // Kick off OG image generation in the background for all input symbols (async queue)
    try {
      const warmSymbols = Array.from(new Set(
        input
          .split(/[\s,]+/)
          .map((s) => s.trim().toUpperCase())
          .filter(Boolean)
      ))
      setOgReady((prev) => {
        const next = { ...prev }
        warmSymbols.forEach((s) => { next[s] = false })
        return next
      })
      ;(async () => {
        await Promise.allSettled(
          warmSymbols.map((s) =>
            fetch(`${API_BASE_URL}/og-image/warm/${encodeURIComponent(s)}/async`, { method: 'POST' })
          )
        )
      })()
      // Begin polling OG status until ready for each symbol
      if (ogPollRef.current) clearInterval(ogPollRef.current)
      ogPollRef.current = setInterval(async () => {
        try {
          const checks = await Promise.all(
            warmSymbols.map(async (s) => {
              const r = await fetch(`${API_BASE_URL}/og-image/status/${encodeURIComponent(s)}`)
              if (!r.ok) return [s, false]
              const j = await r.json()
              return [s, !!j.ready]
            })
          )
          setOgReady((prev) => {
            const next = { ...prev }
            checks.forEach(([s, ready]) => { if (ready) next[s] = true })
            return next
          })
        } catch {}
      }, 1200)
    } catch {}
    setLoading(true)
    setError('')
    setResults([])
    try {
      // Submit async analyze job
      const response = await fetch(`${API_BASE_URL}/analyze/async`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: input, days: 7, max_results: 8, tone: 'humorous' })
      })
      if (!response.ok) {
        const txt = await response.text()
        throw new Error(`${response.status}: ${txt}`)
      }
      const { job_id } = await response.json()
      // Poll job until completed or failed
      const started = Date.now()
      const timeoutMs = 90_000
      let done = false
      while (!done) {
        const jr = await fetch(`${API_BASE_URL}/jobs/${encodeURIComponent(job_id)}`)
        if (!jr.ok) throw new Error('job status check failed')
        const js = await jr.json()
        if (js.status === 'completed') {
          const analysisResults = (js.result?.results) || []
          setResults(analysisResults)
          // Fetch quotes and mini charts in parallel for each symbol
          const syms = analysisResults.map(r => r.symbol)
          await Promise.all([
            (async () => {
              const entries = await Promise.all(syms.map(async (s) => {
                try {
                  const r = await fetch(`${API_BASE_URL}/quote/${encodeURIComponent(s)}`)
                  if (!r.ok) throw new Error('quote failed')
                  return [s, await r.json()]
                } catch {
                  return [s, null]
                }
              }))
              setQuotes(Object.fromEntries(entries))
            })(),
            (async () => {
              const entries = await Promise.all(syms.map(async (s) => {
                try {
                  const r = await fetch(`${API_BASE_URL}/chart/${encodeURIComponent(s)}?range=5d&interval=1d`)
                  if (!r.ok) throw new Error('chart failed')
                  return [s, await r.json()]
                } catch {
                  return [s, null]
                }
              }))
              setCharts(Object.fromEntries(entries))
            })()
          ])
          // Wait for OG images for these symbols before rendering analysis
          try {
            const startOgWait = Date.now()
            const ogTimeout = 20000 // 20s cap
            // quick helper to check readiness via API to avoid race on state updates
            const checkReady = async () => {
              const pairs = await Promise.all(syms.map(async (s) => {
                try {
                  const rr = await fetch(`${API_BASE_URL}/og-image/status/${encodeURIComponent(s)}`)
                  if (!rr.ok) return [s, false]
                  const jj = await rr.json()
                  return [s, !!jj.ready]
                } catch { return [s, false] }
              }))
              return pairs
            }
            let pairs = await checkReady()
            let allReady = pairs.every(([, ready]) => ready)
            while (!allReady && (Date.now() - startOgWait) < ogTimeout) {
              await new Promise((res) => setTimeout(res, 800))
              pairs = await checkReady()
              allReady = pairs.every(([, ready]) => ready)
            }
            if (allReady) {
              // Ensure UI flips to images immediately by setting ogReady for these symbols
              setOgReady((prev) => {
                const next = { ...prev }
                pairs.forEach(([s, ready]) => { if (ready) next[s] = true })
                return next
              })
              if (ogPollRef.current) { clearInterval(ogPollRef.current); ogPollRef.current = null }
            }
          } catch {}
          // Now render analysis results
          setResults(analysisResults)
          done = true
          break
        } else if (js.status === 'failed') {
          throw new Error(js.error || 'analysis failed')
        }
        if (Date.now() - started > timeoutMs) {
          throw new Error('Request timed out, please try again')
        }
        await new Promise((res) => setTimeout(res, 1200))
      }
    } catch (e) {
      setError(e.message || 'Something went sideways')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (resultsRef.current && results.length > 0) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [results])

  return (
    <div className="bg-gray-900 text-white min-h-screen font-sans p-8">
      <div className="max-w-3xl mx-auto">
        <Header />

        <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8 overflow-hidden">
          <label className="block text-sm text-gray-300 mb-2">Enter stock symbols (comma or space separated)</label>
          <div className="flex flex-col sm:flex-row gap-3">
            <input
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              placeholder="AAPL, TSLA, NVDA"
              onKeyDown={(e) => { if (e.key === 'Enter') onAnalyze() }}
              className="w-full sm:flex-1 min-w-0 p-3 bg-gray-700 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-200 placeholder-gray-400"
            />
            <button
              onClick={() => onAnalyze()}
              disabled={loading}
              className="w-full sm:w-auto whitespace-nowrap px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-600 font-semibold"
            >
              {loading ? 'Investigating…' : 'Why did it drop?'}
            </button>
          </div>
          {error && <p className="text-red-400 mt-3">{error}</p>}
        </div>

        {results.length === 0 && <BiggestLosers onStockClick={handleLoserClick} />}

        <div ref={resultsRef} className="space-y-6">
          {results.map((r) => (
            <div key={r.symbol} className="bg-gray-800 p-6 rounded-lg shadow-lg">
              

              {/* OG preview image from backend - render only when ready to avoid heavy sync work */}
              <div className="mb-4">
                {ogReady[r.symbol] ? (
                  <img
                    src={`${API_BASE_URL}/og-image/${encodeURIComponent(r.symbol)}.png`}
                    alt={`${r.symbol} preview`}
                    className="w-full rounded-md border border-gray-700"
                    loading="lazy"
                  />
                ) : (
                  <div className="w-full h-48 rounded-md border border-gray-700 bg-gray-700/50 animate-pulse" />
                )}
              </div>

              <div className="prose prose-invert max-w-none whitespace-pre-wrap leading-relaxed">
                {r.summary}
              </div>
              {r.sources?.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-sm uppercase tracking-wide text-gray-400 mb-2">Sources</h3>
                  <ul className="list-disc list-inside space-y-1 text-blue-300">
                    {r.sources.map((s, i) => (
                      <li key={i}>
                        <a href={s.url} target="_blank" rel="noreferrer" className="hover:underline">
                          {s.title || s.url}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
        {DEV_SIGNALS_ENABLED && (
        <SignalsDev />
      )}
      </div>

    </div>
  )
}

export default App
