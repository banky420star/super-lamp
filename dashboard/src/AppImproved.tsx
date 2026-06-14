/**
 * Chain Gambler Dashboard - Main Application
 *
 * A comprehensive trading system dashboard with 17 tabs covering:
 * - Trading activity and performance
 * - Model training and evaluation
 * - Risk management and safety controls
 * - System health and diagnostics
 *
 * ARCHITECTURE:
 * - React 18 with TypeScript
 * - Real-time updates via WebSocket + polling fallback
 * - Dark theme with cyan/magenta accents
 * - Responsive grid layouts
 *
 * DATA FLOW:
 * 1. App mounts → LoadingScreen displayed
 * 2. fetchStatus() called → Gets system state from API
 * 3. WebSocket connection established for real-time updates
 * 4. Polling fallback every 10s if WebSocket fails
 * 5. Status updates trigger re-renders of active tab
 */
import React from 'react'
import { StatusPayload, PatternRecord, SystemHeaderState } from './types'
import {
  fetchStatus,
  fetchPatterns,
  fetchPerf,
  fetchPPODiagnostics,
  fetchLSTMExplanations,
  fetchLanes,
  fetchScenarios,
  fetchEconomicCalendar,
  fetchSystemHeader,
  fetchPipelineStages,
  fetchModelBrains,
  fetchTrainingLanes,
  fetchRegistry,
  fetchPromotionGates,
  fetchDemoCanary,
  fetchTradeCoroner,
  fetchPatternsVerified,
  fetchPerpetualImprovement,
  fetchAgentsOperational,
  fetchSafety,
  fetchEvidence,
  createStatusWS,
  PPODiagnostics,
  LSTMExplanation,
  LaneStatus,
  RegimesResponse,
  EconomicEvent,
} from './services/api'

import SystemCommandBar from './components/SystemCommandBar'
import OverviewPanel from './components/OverviewPanel'
import PipelinePanel from './components/PipelinePanel'
import ModelBrainsPanel from './components/ModelBrainsPanelFixed'
import TrainingPanel from './components/TrainingPanelFixed'
import RegistryPanel from './components/RegistryPanel'
import PromotionGatesPanel from './components/PromotionGatesPanel'
import DemoCanaryPanel from './components/DemoCanaryPanel'
import TradesPanel from './components/TradesPanelImproved'
import TradeCoronerPanel from './components/TradeCoronerPanel'
import PatternsPanel from './components/PatternsPanel'
import PerpetualPanel from './components/PerpetualPanel'
import AgentsPanel from './components/AgentsPanel'
import SafetyPanel from './components/SafetyPanel'
import EvidenceLockerPanel from './components/EvidenceLockerPanel'
import SettingsPanel from './components/SettingsPanel'
import DashboardPanel from './components/DashboardPanelImproved'
import DecisionExecutionPanel from './components/DecisionExecutionPanel'
import HelpTooltip from './components/HelpTooltip'

/* ─── Navigation tabs with descriptions ─── */
type TabId =
  | 'overview' | 'pipeline' | 'model_brains' | 'training' | 'registry'
  | 'promotion_gates' | 'demo_canary' | 'trades' | 'trade_coroner'
  | 'patterns' | 'perpetual' | 'agents' | 'safety' | 'evidence'
  | 'settings' | 'legacy_dashboard' | 'decision_execution'

interface TabInfo {
  id: TabId
  label: string
  description: string
}

const TABS: TabInfo[] = [
  { id: 'trades', label: 'Trades', description: 'Live positions, PnL, equity curve, and trade history' },
  { id: 'model_brains', label: 'Model Brains', description: 'AI model status and diagnostics' },
  { id: 'pipeline', label: 'Pipeline', description: 'Training pipeline stages and progress' },
  { id: 'training', label: 'Training', description: 'Active training cycles and controls' },
  { id: 'registry', label: 'Registry', description: 'Champion/canary model library' },
  { id: 'promotion_gates', label: 'Promotion Gates', description: 'Model evaluation criteria' },
  { id: 'demo_canary', label: 'Demo Canary', description: 'Canary model testing in demo mode' },
  { id: 'trade_coroner', label: 'Trade Coroner', description: 'Trade failure analysis and forensics' },
  { id: 'patterns', label: 'Patterns', description: 'Detected price patterns and regimes' },
  { id: 'perpetual', label: 'Perpetual Improvement', description: 'Continuous learning metrics' },
  { id: 'agents', label: 'Agents', description: 'Autonomous agent swarm status' },
  { id: 'evidence', label: 'Evidence Locker', description: 'Audit logs and compliance records' },
  { id: 'settings', label: 'Settings', description: 'System controls and configuration' },
  { id: 'overview', label: 'System Truth', description: 'Real-time system health dashboard' },
  { id: 'safety', label: 'Safety Lock', description: 'Risk controls and emergency stops' },
  { id: 'decision_execution', label: 'Decision+Exec', description: 'Decision to execution flow' },
]

/* ─── Boot sequence messages ─── */
const LOAD_STEPS = [
  'INITIALIZING KERNEL...',
  'MOUNTING FILESYSTEMS...',
  'LOADING NEURAL WEIGHTS...',
  'CALIBRATING RISK ENGINE...',
  'CONNECTING MT5 BRIDGE...',
  'HANDSHAKING BROKER...',
  'LOADING MARKET DATA...',
  'BUILDING FEATURE SPACE...',
  'WARMING LSTM SEQUENCE MEMORY...',
  'WARMING RAINFOREST REGIME DETECTOR...',
  'WARMING PPO POLICY NETWORK...',
  'DREAMER WORLD MODEL: STUBBED...',
  'SYNCHRONIZING AGENT SWARM...',
  'VERIFYING SAFETY GATES...',
  'RUNNING TELEMETRY CHECK...',
  'LOADING CHAMPION REGISTRY...',
  'CHECKING DEMO CANARY STATUS...',
  'REAL MONEY: LOCKED',
  'ARMING EXECUTION GATES...',
  'SYSTEM ONLINE',
]

/**
 * LoadingScreen - Animated boot sequence
 * Shows during initial data fetch to give the impression of a complex system starting up
 */
function LoadingScreen() {
  const [step, setStep] = React.useState(0)
  const [progress, setProgress] = React.useState(0)

  React.useEffect(() => {
    const targets = Array.from({ length: LOAD_STEPS.length }, (_, i) =>
      Math.round(((i + 1) / LOAD_STEPS.length) * 100)
    )
    let i = 0
    const advance = () => {
      if (i >= targets.length) return
      setProgress(targets[i])
      setStep(i)
      i++
    }
    advance()
    const id = setInterval(advance, 280)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="agit-loading">
      <div className="agit-loading-sparks" aria-hidden="true">
        {Array.from({ length: 12 }).map((_, i) => (
          <span
            key={i}
            style={{
              '--x': `${10 + Math.random() * 80}%`,
              '--y': `${20 + Math.random() * 60}%`,
              '--d': `${3 + Math.random() * 3}s`,
              '--delay': `${Math.random() * 2}s`,
            } as React.CSSProperties}
          />
        ))}
      </div>
      <div className="agit-loading-core">
        <div className="agit-loading-dot" />
      </div>
      <div className="agit-loading-content">
        <div className="agit-loading-title">CHAIN GAMBLER</div>
        <div className="agit-loading-sub">AUTONOMOUS TRADING STACK</div>
        <div className="agit-loading-progress" style={{ position: 'relative' }}>
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: 'linear-gradient(90deg, #009db0, #00f0ff, #ff00a0)',
              boxShadow: '0 0 8px rgba(0,240,255,0.4)',
              width: `${progress}%`,
              transition: 'width 0.45s cubic-bezier(0.4,0,0.2,1)',
              borderRadius: 1,
            }}
          />
        </div>
        <div className="agit-loading-status">
          <span
            style={{
              fontFamily: 'var(--mono)',
              fontSize: '0.65rem',
              color: 'var(--dim)',
              letterSpacing: '0.1em',
            }}
          >
            {LOAD_STEPS[step] ?? LOAD_STEPS[LOAD_STEPS.length - 1]}
          </span>
        </div>
      </div>
    </div>
  )
}

/**
 * LiveClock - UTC time display
 * Traders often reference UTC for market open/close times
 */
function LiveClock() {
  const [time, setTime] = React.useState(new Date())
  React.useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  const h = String(time.getUTCHours()).padStart(2, '0')
  const m = String(time.getUTCMinutes()).padStart(2, '0')
  const s = String(time.getUTCSeconds()).padStart(2, '0')
  return (
    <div className="agit-clock">
      <span className="agit-clock-label">UTC</span>
      <span>{h}</span>
      <span style={{ animation: 'blink 1s step-end infinite' }}>:</span>
      <span>{m}</span>
      <span style={{ animation: 'blink 1s step-end infinite' }}>:</span>
      <span>{s}</span>
    </div>
  )
}

/* ─── Full pipeline state ─── */
interface PipelineState {
  status: StatusPayload | null
  patterns: PatternRecord[]
  perf: any
  ppoDiag: PPODiagnostics | null
  lstmExpl: Record<string, LSTMExplanation>
  lanes: LaneStatus[]
  scenarios: RegimesResponse
  calendar: EconomicEvent[]
  systemHeader: SystemHeaderState | null
}

const EMPTY_PIPELINE: PipelineState = {
  status: null,
  patterns: [],
  perf: null,
  ppoDiag: null,
  lstmExpl: {},
  lanes: [],
  scenarios: { regimes: {} },
  calendar: [],
  systemHeader: null,
}

/**
 * Main Application Component
 */
const AppImproved: React.FC = () => {
  const [pipe, setPipe] = React.useState<PipelineState>(EMPTY_PIPELINE)
  const [activeTab, setActiveTab] = React.useState<TabId>('trades')
  const wsConnectedRef = React.useRef(false)
  const [wsConnected, setWsConnected] = React.useState(false)

  /* ─── Full pipeline refresh (non-status endpoints) ─── */
  const refreshSideData = React.useCallback(async () => {
    const [
      patterns,
      perf,
      ppoDiag,
      lstmExpl,
      lanesRes,
      scenarios,
      calendar,
      systemHeader,
    ] = await Promise.allSettled([
      fetchPatterns(),
      fetchPerf(),
      fetchPPODiagnostics(),
      fetchLSTMExplanations(),
      fetchLanes(),
      fetchScenarios(),
      fetchEconomicCalendar(7),
      fetchSystemHeader(),
    ])

    setPipe((prev) => {
      const next = { ...prev }
      if (patterns.status === 'fulfilled') next.patterns = patterns.value ?? []
      if (perf.status === 'fulfilled' && perf.value) next.perf = perf.value
      if (ppoDiag.status === 'fulfilled') next.ppoDiag = ppoDiag.value
      if (lstmExpl.status === 'fulfilled') next.lstmExpl = lstmExpl.value ?? {}
      if (lanesRes.status === 'fulfilled') next.lanes = lanesRes.value?.lanes ?? []
      if (scenarios.status === 'fulfilled') next.scenarios = scenarios.value ?? { regimes: {} }
      if (calendar.status === 'fulfilled') next.calendar = calendar.value ?? []
      if (systemHeader.status === 'fulfilled') next.systemHeader = systemHeader.value ?? null
      return next
    })
  }, [])

  /* ─── WebSocket + polling setup ─── */
  React.useEffect(() => {
    const initialRefresh = async () => {
      const statusResult = await fetchStatus().catch(() => null)
      if (statusResult) {
        setPipe((prev) => ({ ...prev, status: statusResult }))
      }
      await refreshSideData()
    }
    initialRefresh()

    const destroyWS = createStatusWS(
      (data) => {
        setPipe((prev) => ({ ...prev, status: data }))
      },
      (connected) => {
        wsConnectedRef.current = connected
        setWsConnected(connected)
      }
    )

    const poll = async () => {
      if (!wsConnectedRef.current) {
        const statusResult = await fetchStatus().catch(() => null)
        if (statusResult) {
          setPipe((prev) => ({ ...prev, status: statusResult }))
        }
      }
      await refreshSideData()
    }

    const interval = setInterval(poll, 10_000)

    return () => {
      destroyWS()
      clearInterval(interval)
    }
  }, [refreshSideData])

  /* Sync pattern library from live status */
  React.useEffect(() => {
    const lib = pipe.status?.training?.pattern_library
    if (!lib) return
    const records: PatternRecord[] = Object.entries(lib)
      .map(([pattern_name, payload]) => ({ pattern_name, ...(payload || {}) }))
      .sort((a, b) => new Date(b.discovered_at || 0).getTime() - new Date(a.discovered_at || 0).getTime())
    setPipe((prev) => ({ ...prev, patterns: records.length > 0 ? records : prev.patterns }))
  }, [pipe.status])

  const refreshAll = React.useCallback(async () => {
    const statusResult = await fetchStatus().catch(() => null)
    if (statusResult) {
      setPipe((prev) => ({ ...prev, status: statusResult }))
    }
    await refreshSideData()
  }, [refreshSideData])

  if (!pipe.status) return <LoadingScreen />

  const { status, patterns, perf, ppoDiag, lstmExpl, lanes, scenarios, calendar, systemHeader } = pipe
  const halted = status?.risk?.halt

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':        return <OverviewPanel status={status!} header={systemHeader} />
      case 'pipeline':        return <PipelinePanel />
      case 'model_brains':    return <ModelBrainsPanel />
      case 'training':        return <TrainingPanel status={status!} />
      case 'registry':        return <RegistryPanel />
      case 'promotion_gates': return <PromotionGatesPanel />
      case 'demo_canary':     return <DemoCanaryPanel />
      case 'trades':          return <TradesPanel calendar={calendar} />
      case 'trade_coroner':   return <TradeCoronerPanel />
      case 'patterns':        return <PatternsPanel patterns={patterns} status={status!} lstmExpl={lstmExpl} />
      case 'perpetual':       return <PerpetualPanel />
      case 'agents':          return <AgentsPanel />
      case 'safety':          return <SafetyPanel />
      case 'evidence':        return <EvidenceLockerPanel />
      case 'settings':        return <SettingsPanel status={status!} />
      case 'legacy_dashboard':return <DashboardPanel status={status!} />
      case 'decision_execution': return <DecisionExecutionPanel />
      default: return null
    }
  }

  return (
    <div className="agit-shell">
      <div className="scanlines" />

      <SystemCommandBar header={systemHeader} />

      <nav className="agit-nav">
        <div className="agit-nav-brand">
          <div className="agit-nav-mark" />
          <div>
            <div className="agit-nav-title">CHAIN GAMBLER</div>
            <div className="agit-nav-subtitle">Autonomous Trading Stack</div>
          </div>
        </div>

        <div className="agit-nav-tabs">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`agit-nav-tab${activeTab === tab.id ? ' active' : ''}`}
              title={tab.description}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="agit-nav-status">
          <LiveClock />
          {halted ? (
            <span className="agit-pill agit-pill-halt">HALTED</span>
          ) : (
            <span className="agit-pill agit-pill-live">LIVE</span>
          )}
          {pipe.status?.system?.real_money_locked && (
            <span className="agit-pill" style={{ background: 'var(--red)', color: '#fff', marginLeft: 6 }}>
              LOCKED
            </span>
          )}
          {pipe.status?.tests?.status === 'failing' && (
            <span className="agit-pill" style={{ background: 'var(--amber)', color: '#000', marginLeft: 6 }}>
              TESTS FAIL
            </span>
          )}
          {pipe.status?.account?.telemetry_valid === false && (
            <span className="agit-pill" style={{ background: 'var(--amber)', color: '#000', marginLeft: 6 }}>
              TELEMETRY INVALID
            </span>
          )}
          <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 4,
            fontSize: 11, color: wsConnected ? 'var(--green)' : 'var(--amber)',
            marginLeft: 6,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: wsConnected ? 'var(--green)' : 'var(--amber)',
              boxShadow: wsConnected ? '0 0 6px var(--green)' : 'none',
              flexShrink: 0,
            }} />
            {wsConnected ? 'WS' : 'POLL'}
          </span>
        </div>
      </nav>

      <main className="agit-main animate-in">
        {renderContent()}
      </main>
    </div>
  )
}

export default AppImproved
