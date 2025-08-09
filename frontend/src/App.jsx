import { useState } from 'react'
import Header from './components/Header'
import { API_BASE_URL } from './config'
import Sparkline from './components/Sparkline'

function App() {
  const [symbols, setSymbols] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [quotes, setQuotes] = useState({})
  const [charts, setCharts] = useState({})

  const onAnalyze = async () => {
    const input = symbols.trim()
    if (!input) {
      setError('Please enter at least one stock symbol')
      return
    }
    setLoading(true)
    setError('')
    setResults([])
    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: input, days: 7, max_results: 8, tone: 'humorous' })
      })
      if (!response.ok) {
        const txt = await response.text()
        throw new Error(`${response.status}: ${txt}`)
      }
      const data = await response.json()
      const analysisResults = data.results || []
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
              const r = await fetch(`${API_BASE_URL}/chart/${encodeURIComponent(s)}?range=1mo&interval=1d`)
              if (!r.ok) throw new Error('chart failed')
              return [s, await r.json()]
            } catch {
              return [s, null]
            }
          }))
          setCharts(Object.fromEntries(entries))
        })()
      ])
    } catch (e) {
      setError(e.message || 'Something went sideways')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-gray-900 text-white min-h-screen font-sans p-8">
      <div className="max-w-3xl mx-auto">
        <Header />

        <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8">
          <label className="block text-sm text-gray-300 mb-2">Enter stock symbols (comma or space separated)</label>
          <div className="flex gap-3">
            <input
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              placeholder="AAPL, TSLA, NVDA"
              onKeyDown={(e) => { if (e.key === 'Enter') onAnalyze() }}
              className="flex-1 p-3 bg-gray-700 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-200 placeholder-gray-400"
            />
            <button
              onClick={onAnalyze}
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-600 font-semibold"
            >
              {loading ? 'Investigating…' : 'Why did it drop?'}
            </button>
          </div>
          {error && <p className="text-red-400 mt-3">{error}</p>}
        </div>

        <div className="space-y-6">
          {results.map((r) => (
            <div key={r.symbol} className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-4">
                  <h2 className="text-2xl font-bold">{r.symbol}</h2>
                  {quotes[r.symbol] && (
                    <div className="text-sm text-gray-300">
                      <span className="font-semibold mr-2">{quotes[r.symbol]?.price?.toFixed?.(2)} {quotes[r.symbol]?.currency || ''}</span>
                      {typeof quotes[r.symbol]?.change === 'number' && (
                        <span className={quotes[r.symbol].change >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {quotes[r.symbol].change >= 0 ? '+' : ''}{quotes[r.symbol].change.toFixed(2)} ({(quotes[r.symbol]?.change_percent || 0).toFixed(2)}%)
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <div>
                  {charts[r.symbol]?.closes && charts[r.symbol]?.closes.length > 1 && (
                    <Sparkline data={charts[r.symbol].closes} width={200} height={48} />
                  )}
                </div>
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
      </div>
    </div>
  )
}

export default App
