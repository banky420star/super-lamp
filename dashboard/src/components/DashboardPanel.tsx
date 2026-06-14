import React from 'react'
import { LaneBStatus, DecisionRecord } from '../types'
import { TradeRecord, fetchTrades, fetchEquityCurve, EquityResponse } from '../services/api'
import EquityChart from './EquityChart'

interface Props {
  status: LaneBStatus
}

const colors = {
  bg: '#0d1726', panelBg: 'rgba(13,23,38,0.92)', border: 'rgba(255,255,255,0.08)',
  text: '#eef5ff', muted: '#97a9c6', green: '#39d98a', amber: '#f3bb4a', red: '#ff7b8f', cyan: '#5ad7ff',
}

const panelStyle: React.CSSProperties = {
  background: colors.panelBg, border: `1px solid ${colors.border}`,
  borderRadius: 10, padding: 16,
}

function fmtMoney(v: number | undefined | null): string {
  if (v == null || isNaN(v)) return '--'
  return v.toFixed(2)
}

const DashboardPanel: React.FC<Props> = ({ status }) => {
  const [equityCurve, setEquityCurve] = React.useState<EquityResponse | null>(null)
  const [recentTrades, setRecentTrades] = React.useState<TradeRecord[]>([])
  const [equityWindow, setEquityWindow] = React.useState<'30d' | '90d' | 'all'>('all')

  React.useEffect(() => {
    const loadData = async () => {
      const [t, e] = await Promise.all([
        fetchTrades(5, 0).catch(() => ({ trades: [] as TradeRecord[], total: 0 })),
        fetchEquityCurve(equityWindow).catch(() => null),
      ])
      setRecentTrades(t.trades)
      if (e) setEquityCurve(e)
    }
    loadData()
    const id = setInterval(loadData, 10_000)
    return () => clearInterval(id)
  }, [equityWindow])

  const a = status.account
  const p = status.position
  const t = status.trading
  const m = status.model
  const diag = status.diag
  const decisions = status.recent_decisions ?? []

  return (
    <div style={{ background: colors.bg, color: colors.text, padding: 20 }}>
      {/* KPI Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 24 }}>
        {[
          { label: 'Equity', value: `$${fmtMoney(a.equity)}`, color: t.drawdown_pct > 5 ? colors.red : a.equity >= a.balance ? colors.green : colors.amber },
          { label: 'Balance', value: `$${fmtMoney(a.balance)}` },
          { label: 'Drawdown', value: `${t.drawdown_pct.toFixed(1)}%`, color: t.drawdown_pct > 5 ? colors.red : t.drawdown_pct > 2 ? colors.amber : colors.green },
          { label: 'Trades', value: t.trades_taken.toString() },
          { label: 'Position', value: p.active ? `${p.type} ${p.volume}` : 'FLAT', color: p.active ? (p.type === 'BUY' ? colors.green : colors.red) : colors.muted },
          { label: 'PnL', value: p.profit !== 0 ? `$${fmtMoney(p.profit)}` : '--', color: p.profit > 0 ? colors.green : p.profit < 0 ? colors.red : colors.text },
          { label: 'Bars', value: t.bar_count.toString() },
          { label: 'Action', value: diag.action, color: diag.action === 'LONG' ? colors.green : diag.action === 'SHORT' ? colors.red : colors.muted },
        ].map((kpi) => (
          <div key={kpi.label} style={{ ...panelStyle, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span style={{ fontSize: 11, fontWeight: 500, color: colors.muted, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{kpi.label}</span>
            <span style={{ fontSize: 22, fontWeight: 700, color: (kpi as any).color ?? colors.text }}>{kpi.value}</span>
          </div>
        ))}
      </div>

      {/* Status Row */}
      <div style={{ ...panelStyle, marginBottom: 24, display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap' }}>
        {[
          { label: 'Bot', value: status.running ? 'RUNNING' : 'STOPPED', color: status.running ? colors.green : colors.red },
          { label: 'Demo', value: a.trade_mode.toUpperCase(), color: a.trade_mode === 'demo' ? colors.amber : colors.cyan },
          { label: 'Model', value: m.loaded ? m.name : 'NOT LOADED', color: m.loaded ? colors.green : colors.red },
          { label: 'Trend', value: t.dry_run ? 'DRY-RUN' : 'LIVE', color: t.dry_run ? colors.amber : colors.cyan },
          { label: 'TG', value: status.telegram.enabled ? 'ON' : 'OFF', color: status.telegram.enabled ? colors.green : colors.muted },
          { label: 'Spread', value: `${diag.spread_pts}pts`, color: diag.spread_pts > 5 ? colors.amber : colors.green },
          { label: 'Risk', value: `${(t.risk_pct * 100).toFixed(1)}%`, color: t.risk_pct > 0.03 ? colors.red : colors.green },
          { label: 'Max DD', value: `${t.max_dd_pct.toFixed(1)}%` },
        ].map((item) => (
          <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 10, color: colors.muted, textTransform: 'uppercase', letterSpacing: 0.5 }}>{item.label}</span>
            <span style={{ fontSize: 13, fontWeight: 700, color: (item as any).color ?? colors.text }}>{item.value}</span>
          </div>
        ))}
      </div>

      {/* Equity Chart */}
      <div style={{ ...panelStyle, marginBottom: 24 }}>
        <EquityChart data={equityCurve?.points ?? []} height={200} />
      </div>

      {/* Recent Decisions */}
      <div style={{ ...panelStyle, marginBottom: 24 }}>
        <h3 style={{ margin: '0 0 12px 0', fontSize: 14, color: colors.cyan, fontWeight: 600 }}>Recent Decisions</h3>
        {decisions.length === 0 ? (
          <div style={{ color: colors.muted, fontSize: 13 }}>No decisions logged yet.</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  {['Time', 'Action', 'Equity', 'DD%', 'Spread'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '6px 10px', borderBottom: `1px solid ${colors.border}`, color: colors.muted, fontWeight: 600, fontSize: 11, whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {decisions.map((d, idx) => (
                  <tr key={idx} style={{ background: idx % 2 === 0 ? 'transparent' : 'rgba(90,215,255,0.02)' }}>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{d.ts}</td>
                    <td style={{ padding: '6px 10px', color: d.action === 'LONG' ? colors.green : d.action === 'SHORT' ? colors.red : colors.muted, fontWeight: 700 }}>{d.action}</td>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>${d.equity.toFixed(2)}</td>
                    <td style={{ padding: '6px 10px', color: d.drawdown_pct > 5 ? colors.red : colors.text }}>{d.drawdown_pct.toFixed(1)}%</td>
                    <td style={{ padding: '6px 10px' }}>{d.spread_pts}pts</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Recent Trades */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 12px 0', fontSize: 14, color: colors.cyan, fontWeight: 600 }}>Recent Trades</h3>
        {recentTrades.length === 0 ? (
          <div style={{ color: colors.muted, fontSize: 13 }}>No trades yet.</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  {['Ticket', 'Side', 'Vol', 'Open', 'Close', 'PnL', 'Held'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '6px 10px', borderBottom: `1px solid ${colors.border}`, color: colors.muted, fontWeight: 600, fontSize: 11, whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recentTrades.map((t) => (
                  <tr key={t.ticket}>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{t.ticket}</td>
                    <td style={{ padding: '6px 10px', color: t.side === 'BUY' ? colors.green : colors.red, fontWeight: 600 }}>{t.side}</td>
                    <td style={{ padding: '6px 10px' }}>{t.volume}</td>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{t.time?.slice(11, 19) ?? '--'}</td>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{t.time?.slice(11, 19) ?? '--'}</td>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', color: t.profit > 0 ? colors.green : t.profit < 0 ? colors.red : colors.text, fontWeight: 700 }}>
                      {t.profit >= 0 ? '+' : ''}{t.profit.toFixed(2)}
                    </td>
                    <td style={{ padding: '6px 10px', fontSize: 11 }}>{t.hold_minutes ?? '--'}m</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default DashboardPanel
