/* Lane B Dashboard  Multi-Symbol API Layer */

import {
  LaneBStatus,
  BotInfo,
  BotsResponse,
  TradesResponse,
  EquityResponse,
  ControlResponse,
} from '../types'

const BASE = '/api'

/** Discover all running Lane B bots */
export async function fetchBots(): Promise<BotInfo[]> {
  const r = await fetch(BASE + '/lane_b/bots', { cache: 'no-store' })
  if (!r.ok) return []
  const data: BotsResponse = await r.json()
  return data.bots ?? []
}

/** Fetch per-symbol bot status */
export async function fetchBotStatus(symbol: string): Promise<LaneBStatus> {
  const r = await fetch(BASE + '/lane_b/' + symbol + '/status', { cache: 'no-store' })
  if (!r.ok) return emptyStatus(symbol)
  return r.json()
}

/** Fetch trade history for a symbol */
export async function fetchTrades(symbol: string, limit = 50, offset = 0): Promise<TradesResponse> {
  const r = await fetch(BASE + '/lane_b/' + symbol + '/trades?limit=' + limit + '&offset=' + offset, { cache: 'no-store' })
  if (!r.ok) return { trades: [], total: 0 }
  return r.json()
}

/** Fetch equity curve for a symbol */
export async function fetchEquityCurve(symbol: string, window: '30d' | '90d' | 'all' = 'all'): Promise<EquityResponse> {
  const r = await fetch(BASE + '/lane_b/' + symbol + '/equity?window=' + window, { cache: 'no-store' })
  if (!r.ok) return { points: [], summary: { start_equity: 0, current_equity: 0, peak_equity: 0, max_drawdown_pct: 0, total_trades: 0 } }
  return r.json()
}

/** Start/stop a bot for a symbol */
export async function controlAction(symbol: string, action: string): Promise<ControlResponse> {
  const r = await fetch(BASE + '/lane_b/' + symbol + '/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  })
  return r.ok ? r.json() : { success: false, message: 'API unreachable' }
}

/** WebSocket connection that receives real-time combined status. */
export function createBotWS(
  onMessage: (data: { bots?: BotInfo[]; position?: LaneBStatus['position']; account?: LaneBStatus['account'] }) => void,
  onStateChange?: (connected: boolean) => void
): () => void {
  let ws: WebSocket | null = null
  let destroyed = false

  function connect() {
    if (destroyed) return
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = protocol + '//' + location.host + '/ws/lane_b'
    ws = new WebSocket(url)

    ws.onopen = () => onStateChange?.(true)
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        onMessage(data)
      } catch { /* ignore malformed frames */ }
    }
    ws.onclose = () => {
      onStateChange?.(false)
      if (!destroyed) setTimeout(connect, 3000)
    }
    ws.onerror = () => ws?.close()
  }

  connect()
  return () => { destroyed = true; ws?.close() }
}

function emptyStatus(symbol: string): LaneBStatus {
  return {
    symbol,
    running: false, pid: null,
    account: { login: null, server: null, balance: 0, equity: 0, free_margin: 0, currency: 'USD', trade_mode: 'unknown' },
    position: { active: false, type: null, volume: 0, open_price: 0, sl: 0, profit: 0, ticket: 0 },
    model: { loaded: false, path: '', name: '' },
    trading: { symbol, dry_run: true, risk_pct: 0, max_dd_pct: 0, sl_atr: 0, interval_sec: 60, current_position: 'FLAT', last_action: null, bar_count: 0, trades_taken: 0, peak_equity: 0, start_equity: 0, drawdown_pct: 0 },
    diag: { action: 'unknown', spread_pts: 0 },
    recent_decisions: [],
  }
}
