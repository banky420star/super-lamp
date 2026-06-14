import {
  AgentStatus,
  AgentOperationalStatus,
  DemoCanaryState,
  EvidenceArtifact,
  ModelBrains,
  ModelBundle,
  PatternRecord,
  PatternVerification,
  PerpetualImprovementState,
  PipelineStage,
  PromotionGateItem,
  SafetyState,
  StatusPayload,
  SystemHeaderState,
  TradeCoronerState,
  TrainingLaneCard,
} from '../types'

const BASE = '' // relative — Vite proxy or same-origin in production

export async function fetchStatus(): Promise<StatusPayload> {
  const r = await fetch(`${BASE}/api/status`, { cache: 'no-store' })
  return r.ok ? r.json() : {}
}

/* ─── Training Status (alias — same payload as fetchStatus) ─── */
export async function fetchTrainingStatus(): Promise<StatusPayload> {
  return fetchStatus()
}

/* ─── Agent Team Status ─── */

export function extractAgentStatus(status: StatusPayload): AgentStatus[] {
  const now = new Date()
  const ts = (d?: string | number | null) => {
    if (!d) return now.toISOString()
    try {
      return new Date(d).toISOString()
    } catch {
      return now.toISOString()
    }
  }
  const fmtTime = (d?: string | number | null) => {
    const s = ts(d)
    return s.split('T')[1]?.replace('Z', '').slice(0, 8) ?? '--:--:--'
  }
  const patterns = status.training?.pattern_library
  const patternCount = patterns ? Object.keys(patterns).length : 0
  const logs = status.logs ?? {}
  const activeModels = status.active_models ?? {}
  const modelCount = Array.isArray(activeModels) ? activeModels.length : Object.keys(activeModels).length

  const agents: AgentStatus[] = [
    {
      id: 'data_feed',
      name: 'Data Feed Agent',
      role: 'Ingestion & Normalization',
      status: status.server?.running ? 'Online' : 'Error',
      lastActivity: `Updated ${fmtTime(status.server?.pids?.length ? now.toISOString() : undefined)}`,
      metric: `Server pids: ${status.server?.pids?.length ?? 0}`,
      logs: logs.data_feed ?? 'Polling market data streams...',
    },
    {
      id: 'pattern_detector',
      name: 'Pattern Detector',
      role: 'Regime & Pattern Recognition',
      status: patternCount > 0 ? 'Online' : 'Idle',
      lastActivity: `Scanned ${fmtTime()}`,
      metric: `Patterns found: ${patternCount}`,
      logs: logs.pattern ?? 'Scanning price action for regime shifts...',
    },
    {
      id: 'risk_guardian',
      name: 'Risk Guardian',
      role: 'Exposure & Drawdown Control',
      status: status.risk?.halt ? 'Error' : 'Online',
      lastActivity: `Check ${fmtTime()}`,
      metric: `Drawdown: ${(status.risk as any)?.max_drawdown_pct ?? status.risk?.drawdownPct ?? '--'}%`,
      logs: logs.risk ?? (status.risk?.halt ? 'HALT TRIGGERED — exposure locked' : 'Risk metrics within tolerance.'),
    },
    {
      id: 'lstm_brain',
      name: 'LSTM Brain',
      role: 'Sequence Forecasting',
      status: status.training?.lstm_running ? 'Training' : 'Idle',
      lastActivity: status.training?.visual?.lstm?.state
        ? `State: ${status.training.visual.lstm.state}`
        : `Idle ${fmtTime()}`,
      metric: `Queue: ${status.training?.visual?.lstm?.queue?.length ?? 0}`,
      logs: logs.lstm ?? 'LSTM pipeline awaiting next symbol batch.',
    },
    {
      id: 'ppo_brain',
      name: 'PPO Brain',
      role: 'Policy Gradient Optimization',
      status: status.training?.drl_running ? 'Training' : 'Idle',
      lastActivity: status.training?.visual?.ppo?.state
        ? `State: ${status.training.visual.ppo.state}`
        : `Idle ${fmtTime()}`,
      metric: `Progress: ${(status.training?.visual?.ppo?.progress_pct ?? 0).toFixed(0)}%`,
      logs: logs.ppo ?? 'PPO worker standby.',
    },
    {
      id: 'dreamer',
      name: 'Dreamer',
      role: 'World Model & Planning',
      status: status.training?.dreamer_running ? 'Training' : 'Idle',
      lastActivity: status.training?.visual?.dreamer?.state
        ? `State: ${status.training.visual.dreamer.state}`
        : `Idle ${fmtTime()}`,
      metric: `Progress: ${(status.training?.visual?.dreamer?.progress_pct ?? 0).toFixed(0)}%`,
      logs: logs.dreamer ?? 'Dreamer imagination module idle.',
    },
    {
      id: 'trade_executor',
      name: 'Trade Executor',
      role: 'Order Routing & Execution',
      status: (status.account?.open_positions ?? 0) > 0 ? 'Online' : 'Idle',
      lastActivity: `Positions: ${status.account?.open_positions ?? 0}`,
      metric: `Open trades: ${status.account?.open_positions ?? 0}`,
      logs: logs.executor ?? 'Executor listening for champion signals.',
    },
    {
      id: 'news_sentiment',
      name: 'News Sentiment',
      role: 'NLP & Event Scoring',
      status: 'Idle',
      lastActivity: `Idle ${fmtTime()}`,
      metric: 'NLP: awaiting key/data',
      logs: logs.sentiment ?? 'News feed scanner inactive — no API key.',
    },
    {
      id: 'backtest_engine',
      name: 'Backtest Engine',
      role: 'Strategy Validation',
      status: status.training?.cycle_running ? 'Training' : 'Idle',
      lastActivity: status.training?.cycle_running ? 'Cycle active' : `Idle ${fmtTime()}`,
      metric: `Symbols: ${status.training?.configured_symbols?.length ?? 0}`,
      logs: logs.backtest ?? 'Backtest queue cleared.',
    },
    {
      id: 'champion_evaluator',
      name: 'Champion Evaluator',
      role: 'Model Registry & Canary Gates',
      status: modelCount > 0 ? 'Online' : 'Idle',
      lastActivity: `Registry: ${modelCount} models`,
      metric: `Models: ${modelCount}`,
      logs: logs.champion ?? (modelCount > 0 ? 'Canary gate monitoring active.' : 'No champion models registered.'),
    },
    {
      id: 'perpetual_optimizer',
      name: 'Perpetual Optimizer',
      role: 'Hyper-parameter Search',
      status: (status.training?.lstm_running || status.training?.drl_running || status.training?.dreamer_running) ? 'Training' : 'Idle',
      lastActivity: `Heartbeat ${fmtTime(status.training?.cycle_heartbeat?.ts)}`,
      metric: `Active lanes: ${status.training?.lane_summary?.actionable_symbols ?? 0}`,
      logs: logs.optimizer ?? 'Hyper-parameter sweep paused.',
    },
  ]

  return agents
}

export async function fetchAgentStatus(): Promise<AgentStatus[]> {
  const status = await fetchStatus()
  return extractAgentStatus(status)
}

export async function fetchPatterns(): Promise<PatternRecord[]> {
  const r = await fetch(`${BASE}/api/patterns`)
  if (!r.ok) return []
  const data = await r.json()
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.patterns)) return data.patterns
  return []
}

export async function fetchPerf(): Promise<any> {
  const r = await fetch(`${BASE}/api/perf`)
  return r.ok ? r.json() : null
}

export async function controlAction(action: string, payload?: Record<string, any>): Promise<any> {
  const r = await fetch(`${BASE}/api/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, ...payload }),
  })
  return r.json()
}

export function createStatusWS(
  onMessage: (data: StatusPayload) => void,
  onStateChange?: (connected: boolean) => void
): () => void {
  let ws: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let destroyed = false
  let backoff = 1000
  let failures = 0
  const MAX_FAILURES = 3

  function connect() {
    if (destroyed || failures >= MAX_FAILURES) return
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(`${protocol}//${location.host}/ws/status`)

    ws.onopen = () => {
      failures = 0
      backoff = 1000
      onStateChange?.(true)
    }

    ws.onmessage = (e) => {
      try {
        onMessage(JSON.parse(e.data))
      } catch {
        /* ignore parse errors */
      }
    }

    ws.onclose = () => {
      onStateChange?.(false)
      if (!destroyed && failures < MAX_FAILURES) {
        failures++
        reconnectTimer = setTimeout(connect, Math.min(backoff, 30000))
        backoff = Math.min(backoff * 2, 30000)
      }
    }

    ws.onerror = () => {
      ws?.close()
    }
  }

  connect()

  return () => {
    destroyed = true
    if (reconnectTimer) clearTimeout(reconnectTimer)
    ws?.close()
  }
}

/* ─── PPO Diagnostics API ─── */

export interface PPODiagnostics {
  ppo_loaded: boolean
  obs_shape: number[] | null
  action_shape: number[] | null
  is_canary: boolean
  device: string
  champion_path: string
  canary_path: string
  model_version: string
  last_actions: Record<string, any>
}

export async function fetchPPODiagnostics(): Promise<PPODiagnostics | null> {
  const r = await fetch(`${BASE}/api/ppo_diagnostics`, { cache: 'no-store' })
  return r.ok ? r.json() : null
}

/* ─── LSTM Explanations API ─── */

export interface LSTMExplanation {
  regime: string
  confidence: number
  top_indicators: Array<{ indicator: string; importance: number }>
  cached_at: number | null
}

export async function fetchLSTMExplanations(): Promise<Record<string, LSTMExplanation>> {
  const r = await fetch(`${BASE}/api/lstm_explanations`, { cache: 'no-store' })
  if (!r.ok) return {}
  const data = await r.json()
  return data?.symbols ?? {}
}

/* ─── Learning Pipeline API ─── */

export interface LearningStatus {
  canary: {
    active: boolean
    path: string | null
    version: string | null
    scorecard: Record<string, any>
  }
  champion: {
    path: string | null
    version: string | null
    scorecard: Record<string, any>
  }
  candidates: Array<{
    version: string
    path: string
    win_rate: number | null
    loss: number | null
    saved_at: string | null
    type: string | null
  }>
  training_schedule: {
    enabled: boolean
    interval_sec: number
    auto_canary: boolean
  }
  learning_log: any
}

export async function fetchLearning(): Promise<LearningStatus | null> {
  const r = await fetch(`${BASE}/api/learning`, { cache: 'no-store' })
  return r.ok ? r.json() : null
}

/* ─── Scenarios / Regime API ─── */

export interface RegimeStats {
  total_decisions: number
  buy_count: number
  sell_count: number
  hold_count: number
  avg_confidence: number
  avg_exposure: number
  symbols: string[]
}

export interface RegimesResponse {
  regimes: Record<string, RegimeStats>
}

export async function fetchScenarios(): Promise<RegimesResponse> {
  const r = await fetch(`${BASE}/api/regimes`, { cache: 'no-store' })
  return r.ok ? r.json() : { regimes: {} }
}

/* ─── Lanes API ─── */

export interface LaneStatus {
  symbol: string
  champion: string
  canary: string | null
  action: string
  exposure: number
  confidence: number
  volatility: string
  reason: string
  can_trade: boolean
  is_canary: boolean
  last_decision_at: number | null
  recent_decisions: number
}

export async function fetchLanes(): Promise<{ lanes: LaneStatus[] }> {
  const r = await fetch(`${BASE}/api/lanes`, { cache: 'no-store' })
  return r.ok ? r.json() : { lanes: [] }
}

/* ─── Trade History API ─── */

export interface Trade {
  ticket: number
  symbol: string
  side: string
  volume: number
  open_time: string | null
  close_time: string | null
  open_price: number
  close_price: number
  profit: number
  comment: string
  hold_minutes: number | null
  magic: number | null
  bot_lane: string
  model: string
  action_type: string
  outcome: string // "win" | "loss" | "breakeven"
}

export interface TradesResponse {
  trades: Trade[]
  total: number
  limit: number
  offset: number
}

export interface TradeSummary {
  overall: {
    total_trades: number
    wins: number
    losses: number
    win_rate: number
    total_pnl: number
    avg_profit: number
    avg_loss: number
    profit_factor: number | string
    avg_hold_minutes: number
    max_loss_streak: number
  }
  by_symbol: Record<string, any>
}

export async function fetchTrades(params: Record<string, string> = {}): Promise<TradesResponse> {
  const qs = new URLSearchParams(params).toString()
  const r = await fetch(`${BASE}/api/trades?${qs}`, { cache: 'no-store' })
  return r.ok ? r.json() : { trades: [], total: 0, limit: 50, offset: 0 }
}

export async function fetchTradesSummary(params: Record<string, string> = {}): Promise<TradeSummary> {
  const qs = new URLSearchParams(params).toString()
  const r = await fetch(`${BASE}/api/trades/summary?${qs}`, { cache: 'no-store' })
  return r.ok ? r.json() : { overall: {} as any, by_symbol: {} }
}

/* ─── Equity Curve API ─── */

export interface EquityPoint {
  ts: string
  equity: number
  balance: number
  drawdown_pct: number
}

export interface EquityCurveResponse {
  points: EquityPoint[]
  summary: {
    start_equity: number
    current_equity: number
    peak_equity: number
    max_drawdown_pct: number
    total_trades: number
  }
}

export async function fetchEquityCurve(window: '30d' | '90d' | 'all' = 'all'): Promise<EquityCurveResponse> {
  const r = await fetch(`${BASE}/api/equity_curve?window=${window}`, { cache: 'no-store' })
  return r.ok
    ? r.json()
    : { points: [], summary: { start_equity: 0, current_equity: 0, peak_equity: 0, max_drawdown_pct: 0, total_trades: 0 } }
}

/* ─── Rainforest Pattern Detector API ─── */

export interface RainforestSymbolData {
  regime: string
  confidence: number
  feature_importances: Record<string, number>
  top_patterns: Array<{ pattern: string; freq: number }>
  n_trees?: number
  trained_at?: string
}

export interface RainforestResponse {
  trained_at: string | null
  n_trees: number
  per_symbol: Record<string, RainforestSymbolData>
  error?: string
}

export async function fetchRainforest(): Promise<RainforestResponse> {
  const r = await fetch(`${BASE}/api/patterns/rainforest`, { cache: 'no-store' })
  return r.ok ? r.json() : { trained_at: null, n_trees: 0, per_symbol: {} }
}

/* ─── Economic Calendar API (from MT5) ─── */

export interface EconomicEvent {
  country: string
  country_name: string
  currency: string
  name: string
  event_id: string
  time: string // ISO 8601
  importance: number // 0=low, 1=medium, 2=high
  importance_label: string
  actual?: string | null
  forecast?: string | null
  previous?: string | null
}

export async function fetchEconomicCalendar(daysAhead = 7): Promise<EconomicEvent[]> {
  const r = await fetch(`${BASE}/api/economic_calendar?days=${daysAhead}`, { cache: 'no-store' })
  if (!r.ok) return []
  const data = await r.json()
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.events)) return data.events
  return []
}

/* ─── Trading Mode & MT5 Login API ─── */

export interface MT5LoginResponse {
  success: boolean
  login?: number
  server?: string
  name?: string | null
  balance?: number | null
  equity?: number | null
  error?: string
}

export async function setTradingMode(mode: 'paper' | 'live'): Promise<{ success: boolean; mode?: string; error?: string }> {
  const r = await fetch(`${BASE}/api/mode`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode }),
  })
  return r.json()
}

export async function mt5Login(login: number, password: string, server: string): Promise<MT5LoginResponse> {
  const r = await fetch(`${BASE}/api/mt5_login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ login, password, server }),
  })
  return r.json()
}

export async function resetPaperAccount(balance = 100_000): Promise<{ success: boolean; balance?: number; error?: string }> {
  const r = await fetch(`${BASE}/api/paper_reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ balance }),
  })
  return r.json()
}

/* ═══════════════════════════════════════════════════════════════
   NEW MISSION CONTROL API ENDPOINTS
   ═══════════════════════════════════════════════════════════════ */

export async function fetchSystemHeader(): Promise<SystemHeaderState | null> {
  const r = await fetch(`${BASE}/api/system_header`, { cache: 'no-store' })
  return r.ok ? r.json() : null
}

export async function fetchPipelineStages(): Promise<PipelineStage[]> {
  const r = await fetch(`${BASE}/api/pipeline/stages`, { cache: 'no-store' })
  return r.ok ? r.json() : []
}

export async function fetchModelBrains(): Promise<ModelBrains | null> {
  const r = await fetch(`${BASE}/api/model_brains`, { cache: 'no-store' })
  return r.ok ? r.json() : null
}

export async function fetchTrainingLanes(): Promise<TrainingLaneCard[]> {
  const r = await fetch(`${BASE}/api/training/lanes`, { cache: 'no-store' })
  return r.ok ? r.json() : []
}

export async function fetchRegistry(): Promise<ModelBundle[]> {
  const r = await fetch(`${BASE}/api/registry`, { cache: 'no-store' })
  return r.ok ? r.json() : []
}

export async function fetchPromotionGates(): Promise<PromotionGateItem[]> {
  const r = await fetch(`${BASE}/api/promotion_gates`, { cache: 'no-store' })
  return r.ok ? r.json() : []
}

export async function fetchDemoCanary(): Promise<DemoCanaryState | null> {
  const r = await fetch(`${BASE}/api/demo_canary`, { cache: 'no-store' })
  return r.ok ? r.json() : null
}

export async function fetchTradeCoroner(): Promise<TradeCoronerState | null> {
  const r = await fetch(`${BASE}/api/trades/coroner`, { cache: 'no-store' })
  return r.ok ? r.json() : null
}

export async function fetchPatternsVerified(): Promise<PatternVerification[]> {
  const r = await fetch(`${BASE}/api/patterns/verified`, { cache: 'no-store' })
  return r.ok ? r.json() : []
}

export async function fetchPerpetualImprovement(): Promise<PerpetualImprovementState | null> {
  const r = await fetch(`${BASE}/api/perpetual_improvement`, { cache: 'no-store' })
  return r.ok ? r.json() : null
}

export async function fetchAgentsOperational(): Promise<AgentOperationalStatus[]> {
  const r = await fetch(`${BASE}/api/agents/status`, { cache: 'no-store' })
  return r.ok ? r.json() : []
}

export async function fetchSafety(): Promise<SafetyState | null> {
  const r = await fetch(`${BASE}/api/safety`, { cache: 'no-store' })
  return r.ok ? r.json() : null
}

export async function fetchEvidence(): Promise<EvidenceArtifact[]> {
  const r = await fetch(`${BASE}/api/evidence`, { cache: 'no-store' })
  return r.ok ? r.json() : []
}
