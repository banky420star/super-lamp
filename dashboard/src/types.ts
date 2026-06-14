/* Lane B Dashboard  Multi-Symbol Types */

/** Bot entry from /api/lane_b/bots */
export interface BotInfo {
  symbol: string
  pid: number | null
  alive: boolean
  current_position: 'LONG' | 'SHORT' | 'FLAT'
  bar_count: number
  trades_taken: number
  model_name: string
  drawdown_pct: number
}

export interface BotsResponse {
  bots: BotInfo[]
}

export interface LaneBStatus {
  symbol: string
  running: boolean
  pid: number | null
  account: {
    login: number | null
    server: string | null
    balance: number
    equity: number
    free_margin: number
    currency: string
    trade_mode: 'demo' | 'real' | 'unknown'
  }
  position: {
    active: boolean
    type: 'BUY' | 'SELL' | null
    volume: number
    open_price: number
    sl: number
    profit: number
    ticket: number
  }
  model: {
    loaded: boolean
    path: string
    name: string
  }
  trading: {
    symbol: string
    dry_run: boolean
    risk_pct: number
    max_dd_pct: number
    sl_atr: number
    interval_sec: number
    current_position: 'LONG' | 'SHORT' | 'FLAT'
    last_action: number | null
    bar_count: number
    trades_taken: number
    peak_equity: number
    start_equity: number
    drawdown_pct: number
  }
  diag: {
    action: string
    spread_pts: number
  }
  recent_decisions: DecisionRecord[]
}

export interface DecisionRecord {
  ts: string
  action: 'LONG' | 'FLAT' | 'SHORT'
  equity: number
  drawdown_pct: number
  spread_pts: number
  dry_run: boolean
}

export interface TradeRecord {
  ticket: number
  symbol: string
  side: 'BUY' | 'SELL'
  volume: number
  time: string | null
  open_price: number
  close_price: number
  profit: number
  comment: string
  hold_minutes: number | null
}

export interface TradesResponse {
  trades: TradeRecord[]
  total: number
}

export interface EquityPoint {
  ts: string
  equity: number
  balance: number
  drawdown_pct: number
}

export interface EquityResponse {
  points: EquityPoint[]
  summary: {
    start_equity: number
    current_equity: number
    peak_equity: number
    max_drawdown_pct: number
    total_trades: number
  }
}

export interface ControlResponse {
  success: boolean
  message: string
  symbol?: string
}
