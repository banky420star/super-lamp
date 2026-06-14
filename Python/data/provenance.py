"""
Provenance — Dataset lineage and approval tracking.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Provenance:
    dataset_id: str = ""
    symbol: str = ""
    timeframe: str = ""
    source: str = ""
    broker: str = ""
    start: str = ""
    end: str = ""
    rows: int = 0
    missing_candles: int = 0
    duplicate_timestamps: int = 0
    spread_included: bool = False
    commission_model: str = ""
    slippage_model: str = ""
    timezone_checked: bool = False
    leakage_checked: bool = False
    approved_for_training: bool = False
    approved_for_champion_training: bool = False
    dataset_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    meta: dict[str, Any] = field(default_factory=dict)

    def compute_hash(
        self,
        df=None,
        records: list[dict] | None = None,
        file_path: str | None = None,
    ) -> str:
        """Compute SHA-256 hash of dataset content."""
        hasher = hashlib.sha256()
        if df is not None and not df.empty:
            try:
                import pandas as pd
                hasher.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
            except Exception:
                hasher.update(str(df.to_dict()).encode())
        elif records:
            for r in sorted(records, key=lambda x: json.dumps(x, sort_keys=True)):
                hasher.update(json.dumps(r, sort_keys=True, default=str).encode())
        elif file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        self.dataset_hash = hasher.hexdigest()
        return self.dataset_hash

    def approve_for_training(
        self,
        min_rows: int = 1000,
        require_spread: bool = True,
        require_timezone: bool = True,
        require_leakage: bool = True,
        max_missing_ratio: float = 0.05,
        max_dupe_ratio: float = 0.01,
    ) -> bool:
        """Approve dataset for standard training if criteria met."""
        if self.rows < min_rows:
            return False
        if require_spread and not self.spread_included:
            return False
        if require_timezone and not self.timezone_checked:
            return False
        if require_leakage and not self.leakage_checked:
            return False
        if self.rows and (self.missing_candles / self.rows) > max_missing_ratio:
            return False
        if self.rows and (self.duplicate_timestamps / self.rows) > max_dupe_ratio:
            return False
        self.approved_for_training = True
        return True

    def approve_for_champion_training(
        self,
        min_rows: int = 5000,
        max_missing_ratio: float = 0.01,
        max_dupe_ratio: float = 0.005,
    ) -> bool:
        """Approve dataset for champion (production-grade) training."""
        if self.rows < min_rows:
            return False
        if not self.approved_for_training:
            return False
        if self.rows and (self.missing_candles / self.rows) > max_missing_ratio:
            return False
        if self.rows and (self.duplicate_timestamps / self.rows) > max_dupe_ratio:
            return False
        self.approved_for_champion_training = True
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source": self.source,
            "broker": self.broker,
            "start": self.start,
            "end": self.end,
            "rows": self.rows,
            "missing_candles": self.missing_candles,
            "duplicate_timestamps": self.duplicate_timestamps,
            "spread_included": self.spread_included,
            "commission_model": self.commission_model,
            "slippage_model": self.slippage_model,
            "timezone_checked": self.timezone_checked,
            "leakage_checked": self.leakage_checked,
            "approved_for_training": self.approved_for_training,
            "approved_for_champion_training": self.approved_for_champion_training,
            "dataset_hash": self.dataset_hash,
            "created_at": self.created_at,
            "meta": self.meta,
        }


# === Data Foundation (appended) ===
import csv
import time as _time
from pathlib import Path as _Path
DATA_FOUNDATION_DIR = _Path(__file__).resolve().parent.parent.parent / "runtime" / "data_foundation"
DATA_FOUNDATION_DIR.mkdir(parents=True, exist_ok=True)
_CSV_HEADERS = { "market_bars": ["ts", "symbol", "timeframe", "open", "high", "low", "close", "volume", "dataset_hash", "source"], "decisions": ["decision_id", "ts", "symbol", "model_id", "dataset_hash", "regime", "action", "confidence", "side", "lots", "linked_run_id", "meta_json"], "trades": ["trade_id", "decision_id", "symbol", "entry_ts", "exit_ts", "entry_price", "exit_price", "pnl", "pnl_pct", "bars_held", "regime", "model_id", "dataset_hash", "outcome"], "equity": ["ts", "run_id", "equity", "balance", "drawdown_pct", "open_positions", "dataset_hash"], "training_runs": ["run_id", "ts", "symbol", "lane", "steps", "model_id", "dataset_hash", "feature_version", "train_start", "train_end", "metrics_json", "passed_gates"], "feature_audits": ["audit_id", "ts", "symbol", "feature_version", "n_features", "mean_json", "std_json", "min_json", "max_json", "nonzero_pct_json", "nan_count_json", "corr_return_json", "dead_cols", "leakage_issues", "duplicates", "trend_momentum_ok", "patterns_dead", "cross_asset_ok", "passed", "details_json"], "regime_logs": ["ts", "symbol", "regime_score", "zone", "h1_bar_time", "model_used", "decision_id", "dataset_hash"], }
def _ensure_csv(name):
    p = DATA_FOUNDATION_DIR / f"{name}.csv"
    if not p.exists():
        with p.open("w", newline="", encoding="utf-8") as f: csv.writer(f).writerow(_CSV_HEADERS[name])
    return p
def _append_csv(name, row):
    p=_ensure_csv(name); headers=_CSV_HEADERS[name]
    with p.open("a", newline="", encoding="utf-8") as f: csv.DictWriter(f, fieldnames=headers).writerow({h: row.get(h,"") for h in headers})
def compute_dataset_hash(df=None, records=None):
    prov=Provenance()
    if df is not None: prov.compute_hash(df=df)
    elif records: prov.compute_hash(records=records)
    return prov.dataset_hash or __import__("hashlib").sha256(str(_time.time()).encode()).hexdigest()[:16]
class DataFoundation:
    def __init__(self):
        for k in _CSV_HEADERS: _ensure_csv(k)
    def record_market_bars(self, df, symbol, timeframe, dataset_hash, source="mt5"):
        if df is None or getattr(df,"empty",False): return dataset_hash
        ts = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        try:
            for _,row in list(df.head(1).iterrows())+list(df.tail(1).iterrows()):
                _append_csv("market_bars", {"ts":ts,"symbol":symbol,"timeframe":timeframe,"open":float(row.get("open",row.get("Open",0))),"high":float(row.get("high",row.get("High",0))),"low":float(row.get("low",row.get("Low",0))),"close":float(row.get("close",row.get("Close",0))),"volume":float(row.get("volume",row.get("tick_volume",0))),"dataset_hash":dataset_hash,"source":source})
        except: pass
        return dataset_hash
    def record_training_run(self, run_id, symbol, lane, steps, model_id, dataset_hash, feature_version="", metrics=None, passed=False):
        _append_csv("training_runs", {"run_id":run_id,"ts":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),"symbol":symbol,"lane":lane,"steps":int(steps),"model_id":model_id,"dataset_hash":dataset_hash,"feature_version":feature_version,"train_start":"","train_end":"","metrics_json":__import__("json").dumps(metrics or {},default=str),"passed_gates":"1" if passed else "0"})
    def record_model_dataset(self, model_id, dataset_hash, metadata=None):
        _append_csv("training_runs", {"run_id":f"model-link-{str(model_id)[:8]}","ts":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),"symbol":(metadata or {}).get("symbol",""),"lane":"registry","steps":0,"model_id":str(model_id),"dataset_hash":dataset_hash,"feature_version":(metadata or {}).get("feature_version",""),"metrics_json":__import__("json").dumps({"link":True,**(metadata or {})},default=str),"passed_gates":"1"})
    def record_decision(self, decision_id, symbol, model_id, dataset_hash, regime, action, confidence=0.0, side="", lots=0.0, linked_run_id="", meta=None):
        _append_csv("decisions", {"decision_id":decision_id,"ts":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),"symbol":symbol,"model_id":model_id or "unknown","dataset_hash":dataset_hash or "","regime":regime or "unknown","action":str(action),"confidence":float(confidence),"side":side,"lots":float(lots),"linked_run_id":linked_run_id,"meta_json":__import__("json").dumps(meta or {},default=str)})
    def record_trade(self, trade_id, decision_id, symbol, entry_ts, exit_ts, entry_price, exit_price, pnl, pnl_pct, bars_held, regime, model_id, dataset_hash, outcome=""):
        _append_csv("trades", {"trade_id":trade_id,"decision_id":decision_id,"symbol":symbol,"entry_ts":entry_ts,"exit_ts":exit_ts,"entry_price":float(entry_price),"exit_price":float(exit_price),"pnl":float(pnl),"pnl_pct":float(pnl_pct),"bars_held":int(bars_held),"regime":regime,"model_id":model_id,"dataset_hash":dataset_hash,"outcome":outcome})
    def record_equity(self, run_id, equity, balance=0.0, drawdown_pct=0.0, open_pos=0, dataset_hash=""):
        _append_csv("equity", {"ts":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),"run_id":run_id,"equity":float(equity),"balance":float(balance),"drawdown_pct":float(drawdown_pct),"open_positions":int(open_pos),"dataset_hash":dataset_hash})
    def record_feature_audit(self, symbol, feature_version, n_features, stats, dead_cols, leakage, duplicates, trend_ok, patterns_dead, cross_ok, passed, details=None):
        audit_id = f"audit_{int(_time.time()*1000)}"
        _append_csv("feature_audits", {"audit_id":audit_id,"ts":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),"symbol":symbol,"feature_version":feature_version,"n_features":int(n_features),"mean_json":__import__("json").dumps(stats.get("mean",{}),default=str),"std_json":__import__("json").dumps(stats.get("std",{}),default=str),"min_json":__import__("json").dumps(stats.get("min",{}),default=str),"max_json":__import__("json").dumps(stats.get("max",{}),default=str),"nonzero_pct_json":__import__("json").dumps(stats.get("nonzero_pct",{}),default=str),"nan_count_json":__import__("json").dumps(stats.get("nan_count",{}),default=str),"corr_return_json":__import__("json").dumps(stats.get("corr_return",{}),default=str),"dead_cols":__import__("json").dumps(dead_cols),"leakage_issues":__import__("json").dumps(leakage),"duplicates":int(duplicates),"trend_momentum_ok":"1" if trend_ok else "0","patterns_dead":"1" if patterns_dead else "0","cross_asset_ok":"1" if cross_ok else "0","passed":"1" if passed else "0","details_json":__import__("json").dumps(details or {},default=str)})
        return audit_id
    def record_regime_log(self, symbol, regime_score, zone, h1_bar_time="", model_used="", decision_id="", dataset_hash=""):
        _append_csv("regime_logs", {"ts":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),"symbol":symbol,"regime_score":float(regime_score),"zone":zone,"h1_bar_time":h1_bar_time,"model_used":model_used,"decision_id":decision_id,"dataset_hash":dataset_hash})
    def get_recent(self, name, limit=50):
        p=DATA_FOUNDATION_DIR/f"{name}.csv"
        if not p.exists(): return []
        try:
            with p.open("r",encoding="utf-8") as f: return list(__import__("csv").DictReader(f))[-limit:]
        except: return []
    def summary(self): return {name: len(self.get_recent(name,1000)) for name in _CSV_HEADERS}
data_foundation = DataFoundation()
