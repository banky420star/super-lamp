import React from 'react'
import { BotInfo, LaneBStatus } from './types'
import { fetchBots, fetchBotStatus, createBotWS } from './services/api'
import DashboardPanel from './components/DashboardPanel'
import './styles.css'

function LiveClock() {
  const [time, setTime] = React.useState(new Date())
  React.useEffect(() => { const id = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(id) }, [])
  const h = String(time.getUTCHours()).padStart(2, '0')
  const m = String(time.getUTCMinutes()).padStart(2, '0')
  const s = String(time.getUTCSeconds()).padStart(2, '0')
  return (
    <div className="agit-clock">
      <span className="agit-clock-label">UTC</span>
      <span>{h}</span><span style={{ animation: 'blink 1s step-end infinite' }}>:</span>
      <span>{m}</span><span style={{ animation: 'blink 1s step-end infinite' }}>:</span>
      <span>{s}</span>
    </div>
  )
}

const LoadSteps = [
  'INITIALIZING LANE B KERNEL...',
  'MOUNTING FEATURE PIPELINE...',
  'LOADING LSTM WEIGHTS...',
  'CONNECTING MT5 BRIDGE...',
  'HANDSHAKING BROKER...',
  'FETCHING MARKET DATA...',
  'BUILDING FEATURE BUFFER...',
  'WARMING LSTM SEQUENCE MEMORY...',
  'FREEZING NORMALIZATION...',
  'SYSTEM ONLINE',
]

function LoadingScreen() {
  const [step, setStep] = React.useState(0)
  const [progress, setProgress] = React.useState(0)
  React.useEffect(() => {
    const targets = Array.from({ length: LoadSteps.length }, (_, i) => Math.round(((i + 1) / LoadSteps.length) * 100))
    let i = 0
    const advance = () => { if (i >= targets.length) return; setProgress(targets[i]); setStep(i); i++ }
    advance()
    const id = setInterval(advance, 250)
    return () => clearInterval(id)
  }, [])
  return (
    <div className="agit-loading">
      <div className="agit-loading-core"><div className="agit-loading-dot" /></div>
      <div className="agit-loading-content">
        <div className="agit-loading-title">LANE B</div>
        <div className="agit-loading-sub">LSTM PPO TRADING BOT</div>
        <div className="agit-loading-progress" style={{ position: 'relative' }}>
          <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(90deg, #009db0, #00f0ff, #ff00a0)', boxShadow: '0 0 8px rgba(0,240,255,0.4)', width: progress + '%', transition: 'width 0.45s cubic-bezier(0.4,0,0.2,1)', borderRadius: 1 }} />
        </div>
        <div className="agit-loading-status">
          <span style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: 'var(--dim)', letterSpacing: '0.1em' }}>{LoadSteps[step]}</span>
        </div>
      </div>
    </div>
  )
}

function positionBadge(pos: string | undefined, alive: boolean): React.ReactNode {
  const color = pos === 'LONG' ? '#39d98a' : pos === 'SHORT' ? '#ff7b8f' : '#7a94b0'
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 10, fontWeight: 700, color, textTransform: 'uppercase' }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: alive ? color : '#3a4050', boxShadow: alive ? '0 0 6px ' + color : 'none', flexShrink: 0 }} />
      {pos ?? 'FLAT'}
    </span>
  )
}

const App: React.FC = () => {
  const [bots, setBots] = React.useState<BotInfo[]>([])
  const [selected, setSelected] = React.useState<string>('')
  const [status, setStatus] = React.useState<LaneBStatus | null>(null)
  const [wsConnected, setWsConnected] = React.useState(false)
  const [loading, setLoading] = React.useState(true)

  // Initial bot discovery
  React.useEffect(() => {
    fetchBots().then((list) => {
      setBots(list)
      setLoading(false)
      if (list.length > 0 && !selected) setSelected(list[0].symbol)
    }).catch(() => setLoading(false))
  }, [])

  // WebSocket for real-time bot list updates (with 30s fetch fallback)
  React.useEffect(() => {
    const destroyWS = createBotWS((data) => {
      if (data.bots) setBots(data.bots)
    }, setWsConnected)
    // Fallback: refresh bot list every 30s in case WS disconnects
    const fallback = setInterval(() => {
      fetchBots().then(setBots).catch(() => {})
    }, 30000)
    return () => { destroyWS(); clearInterval(fallback) }
  }, [])

  // Poll per-symbol status (WebSocket keeps bots list fresh; fetch keeps selected symbol detailed)
  React.useEffect(() => {
    if (!selected) return
    fetchBotStatus(selected).then(setStatus).catch(() => {})
    const poll = setInterval(() => {
      fetchBotStatus(selected).then(setStatus).catch(() => {})
    }, 10000)
    return () => clearInterval(poll)
  }, [selected])

  if (loading) return <LoadingScreen />

  return (
    <div className="agit-shell">
      <div className="scanlines" />

      {/* Top nav bar */}
      <nav className="agit-nav" style={{ padding: '8px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid rgba(90,215,255,0.08)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="agit-nav-mark" />
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#5ad7ff', textTransform: 'uppercase', letterSpacing: 1 }}>LANE B</div>
            <div style={{ fontSize: 10, color: '#7a94b0' }}>LSTM PPO Trading Bot</div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* WS connection dot */}
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 10, color: wsConnected ? '#39d98a' : '#ff7b8f' }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: wsConnected ? '#39d98a' : '#ff7b8f', boxShadow: wsConnected ? '0 0 4px #39d98a' : 'none', flexShrink: 0 }} />
            {wsConnected ? 'WS' : 'polling'}
          </span>
          <LiveClock />
          {status?.trading?.dry_run
            ? <span style={{ background: '#f3bb4a', color: '#000', padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 700, textTransform: 'uppercase' }}>DRY-RUN</span>
            : <span style={{ background: '#39d98a', color: '#000', padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 700, textTransform: 'uppercase' }}>LIVE</span>
          }
          {status && (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11, color: status.running ? '#39d98a' : '#ff7b8f' }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: status.running ? '#39d98a' : '#ff7b8f', boxShadow: status.running ? '0 0 6px #39d98a' : 'none', flexShrink: 0 }} />
              {status.running ? 'PID ' + status.pid : 'OFF'}
            </span>
          )}
        </div>
      </nav>

      {/* Bot tab bar */}
      {bots.length > 1 && (
        <div style={{ display: 'flex', gap: 2, padding: '6px 16px 0', background: 'rgba(0,0,0,0.3)', borderBottom: '1px solid rgba(90,215,255,0.06)' }}>
          {bots.map((bot) => {
            const isActive = bot.symbol === selected
            return (
              <button
                key={bot.symbol}
                onClick={() => setSelected(bot.symbol)}
                style={{
                  padding: '6px 14px', border: 'none',
                  borderBottom: isActive ? '2px solid #5ad7ff' : '2px solid transparent',
                  background: isActive ? 'rgba(90,215,255,0.08)' : 'transparent',
                  color: isActive ? '#e0f0ff' : '#7a94b0',
                  fontSize: 12, fontWeight: isActive ? 700 : 500,
                  cursor: 'pointer', fontFamily: 'var(--mono)',
                  textTransform: 'uppercase', letterSpacing: '0.08em',
                  transition: 'all 0.2s', display: 'flex', alignItems: 'center', gap: 8,
                }}
              >
                {bot.symbol}
                {positionBadge(bot.current_position, bot.alive)}
              </button>
            )
          })}
        </div>
      )}

      {/* Main content */}
      <main className="agit-main animate-in">
        {status
          ? <DashboardPanel status={status} />
          : selected
            ? <div style={{ textAlign: 'center', padding: 60, color: '#7a94b0' }}>Loading {selected}...</div>
            : <div style={{ textAlign: 'center', padding: 60, color: '#7a94b0' }}>No bots discovered. Start a Lane B bot to see data.</div>
        }
      </main>
    </div>
  )
}

export default App
