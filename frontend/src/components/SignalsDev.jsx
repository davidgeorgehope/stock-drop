import React, { useEffect, useState } from 'react';
import { API_BASE_URL } from '../config';

const DEFAULT_UNIVERSE = ['AAPL','TSLA','NVDA','AMD','AMZN','MSFT','META','GOOGL','NFLX','JPM','BA','DIS'];

export default function SignalsDev() {
  const [symbols, setSymbols] = useState(DEFAULT_UNIVERSE.join(', '));
  const [top, setTop] = useState(8);
  const [loadingTop, setLoadingTop] = useState(false);
  const [loadingScan, setLoadingScan] = useState(false);
  const [loadingPromote, setLoadingPromote] = useState(false);
  const [error, setError] = useState('');
  const [rows, setRows] = useState([]);
  const [threshold, setThreshold] = useState(-0.5);
  const [sinceMins, setSinceMins] = useState(120);
  const [includeNews, setIncludeNews] = useState(true);
  const [selected, setSelected] = useState(null);
  const [news, setNews] = useState(null);
  const [newsLoading, setNewsLoading] = useState(false);
  const [newsError, setNewsError] = useState('');

  const runScan = async () => {
    setLoadingScan(true);
    setError('');
    setRows([]);
    try {
      const t0 = performance.now();
      const body = {
        symbols: symbols,
        top: top,
        include_news: includeNews,
      };
      console.log('[scan] POST /premium/oversold/scan', body);
      const r = await fetch(`${API_BASE_URL}/premium/oversold/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      const data = await r.json();
      console.log(`[scan] done in ${(performance.now()-t0).toFixed(0)}ms`, data);
      setRows(data.candidates || []);
    } catch (e) {
      setError(e.message || 'scan failed');
    } finally {
      setLoadingScan(false);
    }
  };

  const fetchTop = async () => {
    setLoadingTop(true);
    setError('');
    setRows([]);
    try {
      const t0 = performance.now();
      const params = new URLSearchParams({ limit: String(top), since_minutes: String(sinceMins), include_news: String(includeNews) });
      console.log('[top] GET /premium/oversold/top', Object.fromEntries(params));
      const r = await fetch(`${API_BASE_URL}/premium/oversold/top?${params.toString()}`);
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      const data = await r.json();
      console.log(`[top] done in ${(performance.now()-t0).toFixed(0)}ms`, data);
      setRows(data.candidates || []);
    } catch (e) {
      setError(e.message || 'fetch top failed');
    } finally {
      setLoadingTop(false);
    }
  };

  const promoteTop = async () => {
    setLoadingPromote(true);
    setError('');
    try {
      const r = await fetch(`${API_BASE_URL}/premium/oversold/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ top, threshold, cooldown_minutes: 1440 })
      });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      await r.json();
      await fetchTop();
    } catch (e) {
      setError(e.message || 'promote failed');
    } finally {
      setLoadingPromote(false);
    }
  };

  const openNews = async (symbol) => {
    setSelected(symbol);
    setNews(null);
    setNewsError('');
    setNewsLoading(true);
    try {
      const r = await fetch(`${API_BASE_URL}/premium/oversold/${encodeURIComponent(symbol)}/news`);
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      const data = await r.json();
      setNews(data);
    } catch (e) {
      setNewsError(e.message || 'news failed');
    } finally {
      setNewsLoading(false);
    }
  };

  useEffect(() => {
    // Prepopulate symbols with current interesting losers if available
    const prefillSymbolsFromLosers = async () => {
      try {
        const r = await fetch(`${API_BASE_URL}/interesting-losers?top=12`);
        if (!r.ok) return;
        const data = await r.json();
        const losers = (data && Array.isArray(data.losers)) ? data.losers : [];
        const tickers = losers.map(l => l.symbol).filter(Boolean);
        if (tickers.length > 0) {
          const defaultStr = DEFAULT_UNIVERSE.join(', ');
          // Only overwrite if the user hasn't typed anything different yet
          setSymbols(prev => (prev === defaultStr ? tickers.join(', ') : prev));
        }
      } catch (_) {
        // best-effort only; ignore errors
      }
    };

    prefillSymbolsFromLosers();
    fetchTop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-fetch when toggling includeNews to update columns
  useEffect(() => {
    // Only refetch if we've already pulled something before
    if (!loadingTop && !loadingScan && !loadingPromote) {
      fetchTop();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [includeNews]);

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8">
      <h2 className="text-xl font-bold mb-4">Dev: Oversold Scanner</h2>
      <div className="flex flex-col gap-3 md:flex-row md:flex-wrap md:items-end">
        <div className="flex-1">
          <label className="block text-sm text-gray-300 mb-2">Symbols (comma/space)</label>
          <input
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            className="w-full p-3 bg-gray-700 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-200 placeholder-gray-400"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-2">Top N</label>
          <input
            type="number"
            value={top}
            onChange={(e) => setTop(parseInt(e.target.value || '0', 10))}
            className="w-24 p-3 bg-gray-700 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-200 placeholder-gray-400"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-2">Since (min)</label>
          <input
            type="number"
            value={sinceMins}
            onChange={(e) => setSinceMins(parseInt(e.target.value || '0', 10))}
            className="w-28 p-3 bg-gray-700 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-200 placeholder-gray-400"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-2">Threshold</label>
          <input
            type="number"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-28 p-3 bg-gray-700 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-200 placeholder-gray-400"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-2">Include news</label>
          <input
            type="checkbox"
            checked={includeNews}
            onChange={(e) => setIncludeNews(e.target.checked)}
            className="h-5 w-5"
          />
        </div>
        <div className="flex flex-wrap gap-2 w-full md:w-auto">
          <button
            onClick={fetchTop}
            disabled={loadingTop || loadingScan || loadingPromote}
            className="w-full md:w-auto px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:bg-gray-600 font-semibold"
          >
            {loadingTop ? 'Loading…' : 'Pull recent'}
          </button>
          <button
            onClick={runScan}
            disabled={loadingTop || loadingScan || loadingPromote}
            className="w-full md:w-auto px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-600 font-semibold"
          >
            {loadingScan ? 'Scanning…' : 'Scan'}
          </button>
          <button
            onClick={promoteTop}
            disabled={loadingTop || loadingScan || loadingPromote}
            className="w-full md:w-auto px-4 py-2 bg-emerald-600 text-white rounded-md hover:bg-emerald-700 disabled:bg-gray-600 font-semibold"
          >
            {loadingPromote ? 'Promoting…' : 'Promote top'}
          </button>
        </div>
      </div>
      {error && <p className="text-red-400 mt-3">{error}</p>}
      <div className="mt-6 overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="text-gray-400">
              <th className="py-2 pr-4 text-left">Symbol</th>
              <th className="py-2 pr-4 text-right">Oversold</th>
              <th className="py-2 pr-4 text-right">News</th>
              <th className="py-2 pr-4 text-right">Blended</th>
              <th className="py-2 pr-4 text-right">1d</th>
              <th className="py-2 pr-4 text-right">3d</th>
              <th className="py-2 pr-4 text-right">Gap</th>
              <th className="py-2 pr-4 text-right">Vol x20d</th>
              <th className="py-2 pr-4 text-right">TR%</th>
              <th className="py-2 pr-4 text-right">Z(close)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.symbol} className="border-t border-gray-700 hover:bg-gray-700/40 cursor-pointer" onClick={() => openNews(r.symbol)}>
                <td className="py-2 pr-4 font-semibold text-left">{r.symbol}</td>
                <td className="py-2 pr-4 text-red-300 text-right tabular-nums">{(r.metrics.oversold_score ?? 0).toFixed(3)}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{r.metrics.news_score != null ? r.metrics.news_score.toFixed(2) : '—'}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{r.metrics.blended_score != null ? r.metrics.blended_score.toFixed(3) : '—'}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{r.metrics.return_1d != null ? (r.metrics.return_1d*100).toFixed(2)+'%' : '—'}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{r.metrics.return_3d != null ? (r.metrics.return_3d*100).toFixed(2)+'%' : '—'}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{r.metrics.gap_pct != null ? (r.metrics.gap_pct*100).toFixed(2)+'%' : '—'}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{r.metrics.volume_ratio_20d != null ? r.metrics.volume_ratio_20d.toFixed(2)+'x' : '—'}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{r.metrics.true_range_pct != null ? (r.metrics.true_range_pct*100).toFixed(2)+'%' : '—'}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{r.metrics.zscore_close != null ? r.metrics.zscore_close.toFixed(2) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (
        <div className="mt-6 rounded-md border border-gray-700 bg-gray-900 p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">{selected} — News overview</h3>
            <button className="text-gray-400 hover:text-gray-200" onClick={() => { setSelected(null); setNews(null); }}>Close</button>
          </div>
          {newsLoading && <div className="text-gray-400 mt-3">Loading news…</div>}
          {newsError && <div className="text-red-400 mt-3">{newsError}</div>}
          {news && (
            <div className="mt-3 space-y-3">
              <div className="flex flex-wrap gap-2">
                {(() => {
                  try {
                    const parsed = JSON.parse(news.llm || '{}');
                    const chips = [];
                    if (parsed.event_type) chips.push({ label: parsed.event_type, color: 'bg-blue-700' });
                    if (parsed.severity) chips.push({ label: parsed.severity, color: 'bg-red-700' });
                    if (parsed.time_horizon) chips.push({ label: parsed.time_horizon, color: 'bg-purple-700' });
                    if (parsed.credibility) chips.push({ label: parsed.credibility, color: 'bg-emerald-700' });
                    if (parsed.company_specific !== undefined) chips.push({ label: parsed.company_specific ? 'company' : 'macro/sector', color: 'bg-gray-700' });
                    return chips.map((c, idx) => (
                      <span key={idx} className={`px-2 py-1 text-xs rounded ${c.color}`}>{c.label}</span>
                    ));
                  } catch {
                    return null;
                  }
                })()}
              </div>
              <div>
                <h4 className="text-sm text-gray-400 mb-2">Headlines</h4>
                <ul className="list-disc list-inside space-y-1 text-blue-300">
                  {(news.headlines || []).slice(0,5).map((h, i) => (
                    <li key={i}><a href={h.url} target="_blank" rel="noreferrer" className="hover:underline">{h.title || h.url}</a></li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
