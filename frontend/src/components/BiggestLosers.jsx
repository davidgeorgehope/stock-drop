import { useState, useEffect } from 'react'
import { API_BASE_URL } from '../config'

function BiggestLosers({ onStockClick }) {
  const [losers, setLosers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lastUpdated, setLastUpdated] = useState('')

  useEffect(() => {
    const fetchLosers = async () => {
      try {
        setLoading(true)
        const response = await fetch(`${API_BASE_URL}/interesting-losers?candidates=300&top=12`)
        if (!response.ok) {
          throw new Error(`Failed to fetch biggest losers: ${response.status}`)
        }
        const data = await response.json()
        setLosers(data.losers || [])
        setLastUpdated(data.last_updated || '')
      } catch (err) {
        setError(err.message || 'Failed to load biggest losers')
        console.error('Error fetching biggest losers:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchLosers()
    // EOD data: refresh once per hour
    const interval = setInterval(fetchLosers, 60 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8">
        <h2 className="text-xl font-bold text-white mb-4">Most Interesting Losers (EOD) 📉</h2>
        <div className="flex justify-center items-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-gray-300">Loading biggest losers...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8">
        <h2 className="text-xl font-bold text-white mb-4">Most Interesting Losers (EOD) 📉</h2>
        <div className="text-red-400 text-center py-4">
          {error}
        </div>
      </div>
    )
  }

  if (losers.length === 0) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8">
        <h2 className="text-xl font-bold text-white mb-4">Most Interesting Losers (EOD) 📉</h2>
        <div className="text-gray-400 text-center py-4">
          No losers found - everyone's winning today! 🎉
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8">
      <h2 className="text-xl font-bold text-white mb-4">Most Interesting Losers (EOD) 📉</h2>
      <p className="text-gray-400 text-sm mb-4">Curated by recent headlines and catalysts. Click to analyze.</p>
      
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
        {losers.map((stock) => (
          <button
            key={stock.symbol}
            onClick={() => onStockClick(stock.symbol)}
            className="bg-gray-700 hover:bg-gray-600 transition-colors duration-200 p-3 rounded-md border border-gray-600 hover:border-red-400 group"
          >
            <div className="text-white font-semibold text-sm mb-1">
              {stock.symbol}
            </div>
            {stock.change_percent && (
              <div className="text-red-400 font-medium text-xs">
                {stock.change_percent.toFixed(2)}%
              </div>
            )}
            {stock.reason && (
              <div className="text-gray-400 text-xs mt-1 line-clamp-3 text-left">
                {stock.reason}
              </div>
            )}
          </button>
        ))}
      </div>
      
      <div className="mt-4 text-xs text-gray-500 text-center">
        {lastUpdated && (
          <div className="mb-1">Last updated: {new Date(lastUpdated).toLocaleString()}</div>
        )}
        <div>
          Cached daily • Refreshes shortly after market close
        </div>
      </div>
    </div>
  )
}

export default BiggestLosers
