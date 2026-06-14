import React from 'react'
import { EquityPoint } from '../services/api'

interface Props {
  data: EquityPoint[]
  height?: number
  window?: '30d' | '90d' | 'all'
  onWindowChange?: (w: '30d' | '90d' | 'all') => void
}

const PAD_LEFT = 60
const PAD_RIGHT = 16
const PAD_TOP = 20
const PAD_BOTTOM = 40

function smooth3(arr: number[]): number[] {
  if (arr.length < 3) return arr
  return arr.map((v, i) => {
    if (i === 0) return (v + arr[1]) / 2
    if (i === arr.length - 1) return (v + arr[i - 1]) / 2
    return (arr[i - 1] + v + arr[i + 1]) / 3
  })
}

function fmtDate(iso: string): string {
  try {
    const d = new Date(iso)
    return `${d.getMonth() + 1}/${d.getDate()}`
  } catch {
    return ''
  }
}

function fmtDateFull(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
  } catch {
    return iso
  }
}

const WINDOWS: Array<{ key: '30d' | '90d' | 'all'; label: string }> = [
  { key: '30d', label: '30D' },
  { key: '90d', label: '90D' },
  { key: 'all', label: 'ALL' },
]

const EquityChart: React.FC<Props> = ({
  data,
  height = 200,
  window: win = 'all',
  onWindowChange,
}) => {
  const svgRef = React.useRef<SVGSVGElement>(null)
  const containerRef = React.useRef<HTMLDivElement>(null)
  const [svgWidth, setSvgWidth] = React.useState(600)
  const [hoverIdx, setHoverIdx] = React.useState<number | null>(null)
  const [tooltipPos, setTooltipPos] = React.useState<{ x: number; y: number }>({ x: 0, y: 0 })

  // ResizeObserver to track container width
  React.useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        setSvgWidth(entry.contentRect.width || 600)
      }
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  const chartW = svgWidth - PAD_LEFT - PAD_RIGHT
  const chartH = height - PAD_TOP - PAD_BOTTOM

  // Memoize all path/axis computations
  const computed = React.useMemo(() => {
    if (!data || data.length === 0) return null

    const rawEquity = data.map(d => d.equity)
    const rawBalance = data.map(d => d.balance)
    const smoothedEquity = smooth3(rawEquity)

    const allVals = [...smoothedEquity, ...rawBalance]
    const minV = Math.min(...allVals)
    const maxV = Math.max(...allVals)
    const range = maxV - minV || 1

    const toX = (i: number) => PAD_LEFT + (i / Math.max(data.length - 1, 1)) * chartW
    const toY = (v: number) => PAD_TOP + chartH - ((v - minV) / range) * chartH

    // Equity line path + area fill
    const equityPoints = smoothedEquity.map((v, i) => ({ x: toX(i), y: toY(v) }))
    const equityPath = equityPoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
    const areaPath =
      equityPath +
      ` L${toX(data.length - 1).toFixed(1)},${(PAD_TOP + chartH).toFixed(1)} L${PAD_LEFT},${(PAD_TOP + chartH).toFixed(1)} Z`

    // Balance line path
    const balancePath = rawBalance
      .map((v, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`)
      .join(' ')

    // Peak line + drawdown fill
    let peak = smoothedEquity[0]
    const peakLine: number[] = []
    for (let i = 0; i < smoothedEquity.length; i++) {
      peak = Math.max(peak, smoothedEquity[i])
      peakLine.push(peak)
    }

    // Drawdown fill segments (red area between equity and peak when below peak)
    const drawdownSegments: string[] = []
    let inDD = false
    let ddPath = ''
    for (let i = 0; i < data.length; i++) {
      const eq = smoothedEquity[i]
      const pk = peakLine[i]
      if (pk > eq + 0.001) {
        if (!inDD) {
          inDD = true
          ddPath = `M${toX(i).toFixed(1)},${toY(eq).toFixed(1)}`
        } else {
          ddPath += ` L${toX(i).toFixed(1)},${toY(eq).toFixed(1)}`
        }
      } else {
        if (inDD) {
          // close segment: trace back along peak
          for (let j = i - 1; j >= 0; j--) {
            if (peakLine[j] > smoothedEquity[j] + 0.001) {
              ddPath += ` L${toX(j).toFixed(1)},${toY(peakLine[j]).toFixed(1)}`
            } else {
              ddPath += ` L${toX(j).toFixed(1)},${toY(peakLine[j]).toFixed(1)}`
              break
            }
          }
          ddPath += ' Z'
          drawdownSegments.push(ddPath)
          inDD = false
          ddPath = ''
        }
      }
    }
    if (inDD && ddPath) {
      const last = data.length - 1
      for (let j = last; j >= 0; j--) {
        ddPath += ` L${toX(j).toFixed(1)},${toY(peakLine[j]).toFixed(1)}`
        if (peakLine[j] <= smoothedEquity[j] + 0.001) break
      }
      ddPath += ' Z'
      drawdownSegments.push(ddPath)
    }

    // Y-axis gridlines: 4 lines
    const gridLines = [0, 1, 2, 3].map(i => {
      const fraction = i / 3
      const val = minV + fraction * range
      const y = PAD_TOP + chartH - fraction * chartH
      return { y: y.toFixed(1), val }
    })

    // X-axis ticks: smart spacing
    const maxTicks = Math.max(2, Math.floor(chartW / 80))
    const step = Math.max(1, Math.round(data.length / maxTicks))
    const xTicks = data
      .map((d, i) => ({ i, label: fmtDate(d.ts) }))
      .filter((_, i) => i % step === 0 || i === data.length - 1)

    // Summary stats
    const currentEquity = smoothedEquity[smoothedEquity.length - 1]
    const peakEquity = Math.max(...smoothedEquity)
    const startEquity = smoothedEquity[0]
    const maxDD = Math.max(...data.map(d => d.drawdown_pct))

    return {
      equityPath,
      areaPath,
      balancePath,
      drawdownSegments,
      gridLines,
      xTicks,
      equityPoints,
      toX,
      toY,
      currentEquity,
      peakEquity,
      startEquity,
      maxDD,
      minV,
      maxV,
    }
  }, [data, chartW, chartH])

  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!computed || !svgRef.current || !data.length) return
      const rect = svgRef.current.getBoundingClientRect()
      const mouseX = e.clientX - rect.left - PAD_LEFT
      const fraction = Math.max(0, Math.min(1, mouseX / chartW))
      const idx = Math.round(fraction * (data.length - 1))
      setHoverIdx(idx)
      setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
    },
    [computed, chartW, data.length],
  )

  const handleMouseLeave = React.useCallback(() => {
    setHoverIdx(null)
  }, [])

  const hoverPoint = hoverIdx != null && computed ? computed.equityPoints[hoverIdx] : null
  const hoverData = hoverIdx != null && data[hoverIdx] ? data[hoverIdx] : null

  return (
    <div className="agit-equity-chart" ref={containerRef}>
      {/* Title */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h3 className="agit-panel-title" style={{ margin: 0 }}>
          Equity Curve
        </h3>
        {/* Time window toggles */}
        <div className="agit-equity-toggles">
          {WINDOWS.map(w => (
            <button
              key={w.key}
              className={`agit-equity-toggle${win === w.key ? ' active' : ''}`}
              onClick={() => onWindowChange?.(w.key)}
            >
              {w.label}
            </button>
          ))}
        </div>
      </div>

      {/* Summary stats */}
      {computed && (
        <div className="agit-equity-summary">
          <div className="agit-equity-stat">
            <span className="agit-equity-stat-label">Start</span>
            <span className="agit-equity-stat-value">${computed.startEquity.toFixed(2)}</span>
          </div>
          <div className="agit-equity-stat">
            <span className="agit-equity-stat-label">Peak</span>
            <span className="agit-equity-stat-value" style={{ color: 'var(--cyan)' }}>
              ${computed.peakEquity.toFixed(2)}
            </span>
          </div>
          <div className="agit-equity-stat">
            <span className="agit-equity-stat-label">Current</span>
            <span
              className="agit-equity-stat-value"
              style={{
                color:
                  computed.currentEquity > computed.startEquity
                    ? 'var(--green)'
                    : computed.currentEquity < computed.startEquity
                      ? 'var(--red)'
                      : 'var(--text)',
              }}
            >
              ${computed.currentEquity.toFixed(2)}
            </span>
          </div>
          <div className="agit-equity-stat">
            <span className="agit-equity-stat-label">Max DD</span>
            <span
              className="agit-equity-stat-value"
              style={{ color: computed.maxDD > 10 ? 'var(--red)' : computed.maxDD > 5 ? 'var(--amber)' : 'var(--green)' }}
            >
              {computed.maxDD.toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* SVG Chart */}
      <div style={{ position: 'relative' }}>
        <svg
          ref={svgRef}
          width="100%"
          height={height}
          viewBox={`0 0 ${svgWidth} ${height}`}
          preserveAspectRatio="none"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ display: 'block', cursor: 'crosshair' }}
        >
          <defs>
            {/* Equity area gradient */}
            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00f0ff" stopOpacity="0.18" />
              <stop offset="100%" stopColor="#00f0ff" stopOpacity="0.01" />
            </linearGradient>
          </defs>

          {computed ? (
            <>
              {/* Grid lines */}
              {computed.gridLines.map((gl, i) => (
                <g key={i}>
                  <line
                    x1={PAD_LEFT}
                    y1={gl.y}
                    x2={svgWidth - PAD_RIGHT}
                    y2={gl.y}
                    stroke="rgba(255,255,255,0.05)"
                    strokeWidth={1}
                  />
                  <text
                    x={PAD_LEFT - 6}
                    y={parseFloat(gl.y) + 4}
                    textAnchor="end"
                    fill="var(--dim)"
                    fontSize={9}
                    fontFamily="IBM Plex Mono, monospace"
                  >
                    ${gl.val.toFixed(0)}
                  </text>
                </g>
              ))}

              {/* X-axis ticks */}
              {computed.xTicks.map(tick => (
                <g key={tick.i}>
                  <line
                    x1={computed.toX(tick.i)}
                    y1={PAD_TOP + chartH}
                    x2={computed.toX(tick.i)}
                    y2={PAD_TOP + chartH + 4}
                    stroke="rgba(255,255,255,0.12)"
                    strokeWidth={1}
                  />
                  <text
                    x={computed.toX(tick.i)}
                    y={PAD_TOP + chartH + 16}
                    textAnchor="middle"
                    fill="var(--dim)"
                    fontSize={9}
                    fontFamily="IBM Plex Mono, monospace"
                  >
                    {tick.label}
                  </text>
                </g>
              ))}

              {/* Drawdown fills */}
              {computed.drawdownSegments.map((seg, i) => (
                <path key={i} d={seg} fill="rgba(255,51,102,0.10)" stroke="none" />
              ))}

              {/* Equity area fill */}
              <path d={computed.areaPath} fill="url(#equityGrad)" stroke="none" />

              {/* Balance line (dashed gray) */}
              <path
                d={computed.balancePath}
                fill="none"
                stroke="#4a6078"
                strokeWidth={1}
                strokeDasharray="4 3"
              />

              {/* Equity line (cyan) */}
              <path
                d={computed.equityPath}
                fill="none"
                stroke="#00f0ff"
                strokeWidth={1.5}
                strokeLinejoin="round"
                strokeLinecap="round"
              />

              {/* Hover crosshair */}
              {hoverPoint && (
                <line
                  x1={hoverPoint.x}
                  y1={PAD_TOP}
                  x2={hoverPoint.x}
                  y2={PAD_TOP + chartH}
                  stroke="rgba(0,240,255,0.35)"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                />
              )}

              {/* Hover dot */}
              {hoverPoint && (
                <circle
                  cx={hoverPoint.x}
                  cy={hoverPoint.y}
                  r={4}
                  fill="#00f0ff"
                  stroke="rgba(4,8,16,0.9)"
                  strokeWidth={2}
                />
              )}
            </>
          ) : (
            <text
              x={svgWidth / 2}
              y={height / 2}
              textAnchor="middle"
              fill="var(--dim)"
              fontSize={11}
              fontFamily="IBM Plex Mono, monospace"
            >
              No data
            </text>
          )}
        </svg>

        {/* Tooltip card */}
        {hoverData && hoverPoint && (
          <div
            className="agit-equity-tooltip-card"
            style={{
              left: Math.min(tooltipPos.x + 12, svgWidth - 160),
              top: Math.max(tooltipPos.y - 60, 0),
            }}
          >
            <div style={{ color: 'var(--dim)', marginBottom: 4 }}>{fmtDateFull(hoverData.ts)}</div>
            <div style={{ display: 'flex', gap: 16 }}>
              <div>
                <div style={{ color: 'var(--dim)', fontSize: '0.6rem' }}>EQUITY</div>
                <div style={{ color: 'var(--cyan)', fontWeight: 700 }}>
                  ${hoverData.equity.toFixed(2)}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--dim)', fontSize: '0.6rem' }}>BALANCE</div>
                <div style={{ color: 'var(--muted)' }}>${hoverData.balance.toFixed(2)}</div>
              </div>
              <div>
                <div style={{ color: 'var(--dim)', fontSize: '0.6rem' }}>DRAWDOWN</div>
                <div
                  style={{
                    color: hoverData.drawdown_pct > 5 ? 'var(--red)' : 'var(--green)',
                    fontWeight: 700,
                  }}
                >
                  {hoverData.drawdown_pct.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default EquityChart
