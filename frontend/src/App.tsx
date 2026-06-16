import React, { useState, useEffect, useCallback } from "react";

type StatusData = {
  mode?: string; can_trade?: boolean; uptime_seconds?: number;
  symbols?: string[];
  champion?: Record<string, {path?:string;model_id?:string;bundle_id?:string}>;
  canary?: Record<string, {path?:string;model_id?:string;bundle_id?:string}>;
  risk?: {
    equity?:number; balance?:number; floating_pnl?:number; daily_pnl?:number;
    drawdown_pct?:number; daily_trades?:number; max_daily_trades?:number;
    halted?:boolean; halt_reason?:string; max_daily_loss_pct?:number; max_drawdown_pct?:number;
  };
};

type Trade = { id?:string; symbol?:string; side?:string; pnl?:number; entry_price?:number; exit_price?:number; exit_time?:string; };
type Summary = { total_trades?:number; wins?:number; losses?:number; total_pnl?:number; profit_factor?:number; };

const fmt = (n: number | undefined | null, d = 2) => n != null ? n.toFixed(d) : "\u2014";
const fmtPct = (n: number | undefined | null) => n != null ? (n * 100).toFixed(2) + "%" : "\u2014";

function MetricCard({ title, value, sub, cls }: { title: string; value: string; sub?: string; cls?: string }) {
  return (<div className="card"><h3>{title}</h3><div className={"value " + (cls || "")}>{value}</div>{sub && <div className="sub">{sub}</div>}</div>);
}

function StatusBadge({ mode }: { mode?: string }) {
  if (mode === "LIVE") return <span className="badge badge-live">LIVE</span>;
  if (mode === "DEMO") return <span className="badge badge-demo">DEMO</span>;
  return <span className="badge badge-off">OFF</span>;
}

export default function App() {
  const [status, setStatus] = useState<StatusData>({});
  const [trades, setTrades] = useState<Trade[]>([]);
  const [summary, setSummary] = useState<Summary>({});
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchData = useCallback(async () => {
    try {
      const [sRes, tRes, sumRes] = await Promise.all([
        fetch("/api/status").then(r => r.json()).catch(() => ({})),
        fetch("/api/trades?limit=50").then(r => r.json()).catch(() => ({ trades: [] })),
        fetch("/api/trades/summary").then(r => r.json()).catch(() => ({})),
      ]);
      setStatus(sRes); setTrades(tRes.trades || tRes || []); setSummary(sumRes);
      setLastUpdate(new Date());
    } catch (e) { console.error("Fetch error:", e); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchData(); const iv = setInterval(fetchData, 15000); return () => clearInterval(iv); }, [fetchData]);

  const risk = status.risk || {};
  const symbols = status.symbols || [];
  const pnlCls = (risk.daily_pnl || 0) >= 0 ? "positive" : "negative";
  const floatCls = (risk.floating_pnl || 0) >= 0 ? "positive" : "negative";

  const stages = ["MT5 Data Intake","Feature Factory","LSTM Training","PPO Training","Model Bundle","Backtest Court","Walk Forward","Baseline Comparison","Promotion Gates","Demo Canary","Champion Promotion"];

  return (
    <div className="app">
      <div className="header">
        <h1>Chain Gambler AGI</h1>
        <div className="status">
          <StatusBadge mode={status.mode} />
          <span style={{ fontSize: 13, color: "var(--text2)" }}>Updated {lastUpdate.toLocaleTimeString()}</span>
          <button className="refresh-btn" onClick={fetchData}>Refresh</button>
        </div>
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: 80 }}>
          <div className="spinner" />
          <p style={{ marginTop: 12, color: "var(--text2)" }}>Loading dashboard...</p>
        </div>
      ) : (<>
        <div className="grid grid-4">
          <MetricCard title="Equity" value={"$" + fmt(risk.equity)} sub={"Balance: $" + fmt(risk.balance)} />
          <MetricCard title="Daily P&L" value={"$" + fmt(risk.daily_pnl)} cls={pnlCls} />
          <MetricCard title="Floating P&L" value={"$" + fmt(risk.floating_pnl)} cls={floatCls} sub={"Drawdown: " + fmtPct(risk.drawdown_pct)} />
          <MetricCard title="Trades Today" value={(risk.daily_trades || 0) + " / " + (risk.max_daily_trades || "\u2014")} sub={risk.halted ? "HALTED: " + risk.halt_reason : "Trading Active"} />
        </div>

        <div className="grid grid-2">
          <div className="card">
            <h3>Champion Models</h3>
            {symbols.length > 0 ? symbols.map(sym => {
              const ch = status.champion?.[sym];
              return (<div className="model-card" key={sym}>
                <div><div className="name">{sym}</div><div className="id">{ch?.model_id || ch?.bundle_id || "No champion"}</div></div>
                <span className={"badge " + (ch?.model_id ? "badge-live" : "badge-off")}>{ch?.model_id ? "Active" : "None"}</span>
              </div>);
            }) : <p style={{ color: "var(--text2)", fontSize: 14 }}>No symbols configured</p>}
          </div>
          <div className="card">
            <h3>Canary Models</h3>
            {symbols.length > 0 ? symbols.map(sym => {
              const cn = status.canary?.[sym];
              return (<div className="model-card" key={sym}>
                <div><div className="name">{sym}</div><div className="id">{cn?.model_id || cn?.bundle_id || "No canary"}</div></div>
                <span className={"badge " + (cn?.model_id ? "badge-demo" : "badge-off")}>{cn?.model_id ? "Testing" : "None"}</span>
              </div>);
            }) : <p style={{ color: "var(--text2)", fontSize: 14 }}>No symbols configured</p>}
          </div>
        </div>

        <div className="grid grid-3" style={{ marginTop: 20 }}>
          <div className="card">
            <h3>Pipeline Status</h3>
            {stages.map((s, i) => (<div className="pipeline-step" key={i}><div className="dot dot-green" /><span style={{ fontSize: 14 }}>{s}</span></div>))}
          </div>
          <div className="card">
            <h3>Trade Summary</h3>
            <div style={{ display: "grid", gap: 12 }}>
              <div><span style={{ color: "var(--text2)", fontSize: 13 }}>Total Trades</span><div style={{ fontSize: 22, fontWeight: 700 }}>{summary.total_trades || 0}</div></div>
              <div><span style={{ color: "var(--text2)", fontSize: 13 }}>Win Rate</span><div style={{ fontSize: 22, fontWeight: 700 }}>{summary.total_trades ? (((summary.wins || 0) / summary.total_trades) * 100).toFixed(1) + "%" : "\u2014"}</div></div>
              <div><span style={{ color: "var(--text2)", fontSize: 13 }}>Profit Factor</span><div style={{ fontSize: 22, fontWeight: 700 }}>{fmt(summary.profit_factor)}</div></div>
              <div><span style={{ color: "var(--text2)", fontSize: 13 }}>Total P&L</span><div style={{ fontSize: 22, fontWeight: 700 }} className={(summary.total_pnl || 0) >= 0 ? "positive" : "negative"}>${fmt(summary.total_pnl)}</div></div>
            </div>
          </div>
          <div className="card">
            <h3>Risk Limits</h3>
            <div style={{ display: "grid", gap: 12 }}>
              <div><span style={{ color: "var(--text2)", fontSize: 13 }}>Max Daily Loss</span><div style={{ fontSize: 18, fontWeight: 600 }}>{fmtPct(risk.max_daily_loss_pct)}</div></div>
              <div><span style={{ color: "var(--text2)", fontSize: 13 }}>Max Drawdown</span><div style={{ fontSize: 18, fontWeight: 600 }}>{fmtPct(risk.max_drawdown_pct)}</div></div>
              <div><span style={{ color: "var(--text2)", fontSize: 13 }}>Current Drawdown</span><div style={{ fontSize: 18, fontWeight: 600 }} className={(risk.drawdown_pct || 0) < 0.05 ? "positive" : "negative"}>{fmtPct(risk.drawdown_pct)}</div></div>
              <div><span style={{ color: "var(--text2)", fontSize: 13 }}>Can Trade</span><div style={{ fontSize: 18, fontWeight: 600 }} className={status.can_trade ? "positive" : "negative"}>{status.can_trade ? "Yes" : "No"}</div></div>
            </div>
          </div>
        </div>

        <div className="card" style={{ marginTop: 20 }}>
          <h3>Recent Trades</h3>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Time</th></tr></thead>
              <tbody>
                {trades.length > 0 ? trades.slice(0, 20).map((t, i) => (
                  <tr key={t.id || i}>
                    <td style={{ fontWeight: 600 }}>{t.symbol || "\u2014"}</td>
                    <td><span className={t.side === "buy" ? "positive" : "negative"}>{(t.side || "\u2014").toUpperCase()}</span></td>
                    <td>{fmt(t.entry_price, 5)}</td>
                    <td>{fmt(t.exit_price, 5)}</td>
                    <td className={(t.pnl || 0) >= 0 ? "positive" : "negative"}>${fmt(t.pnl)}</td>
                    <td style={{ color: "var(--text2)", fontSize: 13 }}>{t.exit_time ? new Date(t.exit_time).toLocaleString() : "\u2014"}</td>
                  </tr>
                )) : <tr><td colSpan={6} style={{ textAlign: "center", color: "var(--text2)", padding: 24 }}>No trades recorded yet</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </>)}
      <div className="footer">Chain Gambler AGI &mdash; Autonomous Trading Pipeline</div>
    </div>
  );
}
