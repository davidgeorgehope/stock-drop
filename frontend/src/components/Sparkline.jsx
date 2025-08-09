import React from 'react'

export default function Sparkline({ data = [], width = 180, height = 48, stroke = '#60a5fa' }) {
  const padding = 4
  const w = width - padding * 2
  const h = height - padding * 2
  const values = data.filter((v) => typeof v === 'number')
  if (values.length === 0) {
    return <svg width={width} height={height}></svg>
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const stepX = values.length > 1 ? w / (values.length - 1) : 0
  const points = values.map((v, i) => {
    const x = padding + i * stepX
    const y = padding + (h - ((v - min) / range) * h)
    return `${x},${y}`
  })
  const d = `M ${points[0]} L ${points.slice(1).join(' ')}`
  const last = values[values.length - 1]
  const first = values[0]
  const up = last >= first
  const strokeColor = up ? '#34d399' : '#f87171'

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <path d={d} fill="none" stroke={strokeColor || stroke} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}


