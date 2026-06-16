import atexit
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOCK_DIR = os.path.join(PROJECT_ROOT, ".tmp")
LOCK_PATH = os.path.join(LOCK_DIR, "champion_cycle.lock")

from Python.model_evaluator import evaluate_candidate_vs_champion
from Python.model_registry import ModelRegistry
from Python.config_utils import DEFAULT_TRADING_SYMBOLS, load_project_config, parse_symbol_list, resolve_trading_symbols
from alerts.telegram_alerts import TelegramAlerter


def _resolve_cfg_value(v):
    if isinstance(v, str) and v.startswith("ENV:"):
        return os.environ.get(v.split(":", 1)[1])
    return v


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _resolve_cycle_symbols(cfg: dict) -> list[str]:
    env_symbols = os.environ.get("AGI_CYCLE_SYMBOLS")
    if env_symbols:
        return parse_symbol_list(env_symbols)

    env_symbol = os.environ.get("AGI_CYCLE_SYMBOL")
    if env_symbol:
        return parse_symbol_list([env_symbol])

    return resolve_trading_symbols(cfg, fallback=DEFAULT_TRADING_SYMBOLS)


def _build_alerter(cfg: dict):
    tel = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
    token = os.environ.get("TELEGRAM_TOKEN") or _resolve_cfg_value(tel.get("token"))
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _resolve_cfg_value(tel.get("chat_id"))
    if not token or not chat_id:
        return TelegramAlerter(None, None)
    return TelegramAlerter(token, str(chat_id))


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_single_instance_lock() -> None:
    os.makedirs(LOCK_DIR, exist_ok=True)

    if os.path.exists(LOCK_PATH):
        existing_pid = None
        try:
            with open(LOCK_PATH, "r", encoding="utf-8") as handle:
                existing_pid = int((handle.read() or "0").strip())
        except Exception:
            existing_pid = None
        if existing_pid and _pid_exists(existing_pid):
            raise RuntimeError(f"champion_cycle is already running with pid={existing_pid}")
        try:
            os.remove(LOCK_PATH)
        except Exception as exc:
            raise RuntimeError(f"Could not clear stale champion_cycle lock: {exc}") from exc

    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode("utf-8"))
    os.close(fd)

    def _cleanup_lock():
        try:
            if os.path.exists(LOCK_PATH):
                with open(LOCK_PATH, "r", encoding="utf-8") as handle:
                    raw = handle.read().strip()
                if raw == str(os.getpid()):
                    os.remove(LOCK_PATH)
        except Exception:
            pass

    atexit.register(_cleanup_lock)


def _has_lstm_artifact(symbol: str) -> bool:
    return Path(PROJECT_ROOT, "models", "per_symbol", f"lstm_{symbol}.pt").exists()


def _has_dreamer_artifact(symbol: str) -> bool:
    return Path(PROJECT_ROOT, "models", "dreamer", f"dreamer_{symbol}.pt").exists()


def _pending_symbols(symbols: list[str], checker) -> list[str]:
    return [symbol for symbol in symbols if not checker(symbol)]


def _latest_candidate(reg: ModelRegistry, symbol: str | None = None):
    root = reg.candidates_dir
    dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not dirs:
        return None
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if not symbol:
        return dirs[0]

    safe = symbol.replace("/", "_")
    for d in dirs:
        scorecard = os.path.join(d, "scorecard.json")
        if not os.path.exists(scorecard):
            continue
        try:
            with open(scorecard, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            if str(meta.get("symbol", "")).replace("/", "_") == safe:
                return d
        except Exception:
            continue
    return None


def _gates_from_cfg(cfg: dict) -> dict:
    ev = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    return {
        "max_drawdown": float(ev.get("max_drawdown", 0.10)),
        "min_sharpe": float(ev.get("min_sharpe", 0.30)),
        "min_return": float(ev.get("min_return", ev.get("min_expected_payoff", 0.015))),
        "score_margin": float(ev.get("score_margin", 0.30)),
        "min_steps_per_symbol": int(ev.get("min_steps_per_symbol", 600)),
        "min_pass_rate": float(ev.get("min_pass_rate", 0.80)),
        "return_margin": float(ev.get("return_margin", 0.0)),
        "sharpe_margin": float(ev.get("sharpe_margin", 0.05)),
        "drawdown_margin": float(ev.get("drawdown_margin", 0.0)),
        "forward_windows": ev.get("forward_windows", []),
        "min_forward_win_rate": float(ev.get("min_forward_win_rate", 0.67)),
    }


def _write_cycle_report(report: dict, name: str = "champion_cycle_last_report.json"):
    path = Path(PROJECT_ROOT) / "logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _run_symbol_simulation(symbols: list[str]) -> dict[str, str]:
    """Run trading simulations to select best symbol/timeframe combinations.

    Returns dict mapping symbol to recommended timeframe.
    """
    logger.info(f"Running symbol simulations for {symbols}")

    try:
        # Import simulation logic
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.run_symbol_simulations import calculate_micro_account_results, generate_recommendations

        all_results = []
        timeframes = ["5m", "15m", "1h"]

        for symbol in symbols:
            for tf in timeframes:
                result = calculate_micro_account_results(symbol, tf)
                all_results.append(result)

        recommendations = generate_recommendations(all_results)

        # Get best timeframe per symbol
        best_timeframes = {}
        for symbol in symbols:
            symbol_recs = [r for r in recommendations if r["symbol"] == symbol and r["recommendation"] in ("TRADE", "TEST")]
            if symbol_recs:
                # Sort by score
                best = max(symbol_recs, key=lambda x: x.get("score", 0))
                best_timeframes[symbol] = best["timeframe"]
                logger.info(f"Simulation recommends {symbol} on {best['timeframe']}: {best['reason']}")

        return best_timeframes
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return {}


def _run_train_drl(symbol: str | None = None):
    """Run enhanced DRL training with per-symbol metrics and multi-timeframe optimization."""
    env = os.environ.copy()
    if symbol:
        env["AGI_DRL_SYMBOL"] = symbol

    # Check if enhanced training is enabled
    enhanced_enabled = os.environ.get("AGI_ENHANCED_TRAINING", "1") == "1"
    timeframe_opt = os.environ.get("AGI_TIMEFRAME_OPTIMIZATION", "1") == "1"
    per_symbol_metrics = os.environ.get("AGI_PER_SYMBOL_METRICS", "1") == "1"

    if enhanced_enabled:
        # Use enhanced training script
        cmd = [
            sys.executable, "training/enhanced_train_drl.py",
            "--per-symbol-metrics",
        ]
        if timeframe_opt:
            cmd.append("--timeframe-opt")
        if symbol:
            cmd.extend(["--symbols", symbol])
        logger.info(f"Running enhanced DRL training: {' '.join(cmd)}")
        subprocess.check_call(cmd, cwd=PROJECT_ROOT, env=env)
    else:
        # Fall back to standard training
        subprocess.check_call([sys.executable, "training/train_drl.py"], cwd=PROJECT_ROOT, env=env)


def _run_train_lstm(symbols: list[str]):
    env = os.environ.copy()
    env["AGI_LSTM_SYMBOLS"] = ",".join(symbols)
    subprocess.check_call([sys.executable, "training/train_lstm.py"], cwd=PROJECT_ROOT, env=env)


def _run_train_lstm_symbol(symbol: str):
    env = os.environ.copy()
    env["AGI_LSTM_SYMBOLS"] = str(symbol)
    subprocess.check_call([sys.executable, "training/train_lstm.py"], cwd=PROJECT_ROOT, env=env)


def _dreamer_cfg(cfg: dict) -> dict:
    drl_cfg = cfg.get("drl", {}) if isinstance(cfg, dict) else {}
    raw = drl_cfg.get("dreamer", {})
    return raw if isinstance(raw, dict) else {}


def _run_train_dreamer(cfg: dict, symbols: list[str]):
    dreamer_cfg = _dreamer_cfg(cfg)
    if not str(dreamer_cfg.get("enabled", False)).lower() == "true":
        return False
    if not str(dreamer_cfg.get("train_in_cycle", False)).lower() == "true":
        return False

    env = os.environ.copy()
    env["AGI_DREAMER_SYMBOLS"] = ",".join(symbols)
    if dreamer_cfg.get("steps") is not None:
        env["AGI_DREAMER_STEPS"] = str(int(dreamer_cfg.get("steps")))
    if dreamer_cfg.get("window") is not None:
        env["AGI_DREAMER_WINDOW"] = str(int(dreamer_cfg.get("window")))
    if dreamer_cfg.get("feature_version"):
        env["AGI_DREAMER_FEATURE_VERSION"] = str(dreamer_cfg.get("feature_version"))
    logger.info(f"Cycle step: train Dreamer for symbols={symbols}")
    subprocess.check_call([sys.executable, "training/train_dreamer.py"], cwd=PROJECT_ROOT, env=env)
    return True


def _run_train_dreamer_symbol(cfg: dict, symbol: str):
    dreamer_cfg = _dreamer_cfg(cfg)
    if not str(dreamer_cfg.get("enabled", False)).lower() == "true":
        return False
    if not str(dreamer_cfg.get("train_in_cycle", False)).lower() == "true":
        return False

    env = os.environ.copy()
    env["AGI_DREAMER_SYMBOL"] = str(symbol)
    if dreamer_cfg.get("steps") is not None:
        env["AGI_DREAMER_STEPS"] = str(int(dreamer_cfg.get("steps")))
    if dreamer_cfg.get("window") is not None:
        env["AGI_DREAMER_WINDOW"] = str(int(dreamer_cfg.get("window")))
    if dreamer_cfg.get("feature_version"):
        env["AGI_DREAMER_FEATURE_VERSION"] = str(dreamer_cfg.get("feature_version"))
    logger.info(f"Cycle step: train Dreamer for symbol={symbol}")
    subprocess.check_call([sys.executable, "training/train_dreamer.py"], cwd=PROJECT_ROOT, env=env)
    return True


def _parallel_enabled(cfg: dict) -> bool:
    cycle_cfg = cfg.get("cycle", {}) if isinstance(cfg.get("cycle", {}), dict) else {}
    return _env_bool("AGI_CYCLE_PARALLEL_SYMBOLS", bool(cycle_cfg.get("parallel_symbols", True)))


def _max_parallel_workers(cfg: dict, total: int) -> int:
    cycle_cfg = cfg.get("cycle", {}) if isinstance(cfg.get("cycle", {}), dict) else {}
    raw = os.environ.get("AGI_CYCLE_MAX_PARALLEL") or cycle_cfg.get("max_parallel_symbols")
    try:
        limit = int(raw)
    except Exception:
        limit = int(total)
    return max(1, min(int(total), limit))


def _run_symbol_jobs(label: str, symbols: list[str], cfg: dict, fn):
    if not symbols:
        return
    if not _parallel_enabled(cfg) or len(symbols) <= 1:
        for symbol in symbols:
            logger.info(f"{label} start | symbol={symbol}")
            fn(symbol)
            logger.info(f"{label} complete | symbol={symbol}")
        return

    max_workers = _max_parallel_workers(cfg, len(symbols))
    logger.info(f"{label} parallel start | symbols={symbols} | workers={max_workers}")
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(fn, symbol): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(future_map):
            symbol = future_map[future]
            try:
                future.result()
                logger.info(f"{label} complete | symbol={symbol}")
            except Exception as exc:
                logger.exception(f"{label} failed | symbol={symbol} | error={exc}")
                failures.append((symbol, exc))
    if failures:
        failed_symbols = ", ".join(symbol for symbol, _ in failures)
        raise RuntimeError(f"{label} failed for symbols: {failed_symbols}")


def main():
    _acquire_single_instance_lock()
    cfg = load_project_config(PROJECT_ROOT, live_mode=False)
    alerter = _build_alerter(cfg)

    trading_cfg = cfg.get("trading", {})
    drl_cfg = cfg.get("drl", {})
    cycle_cfg = cfg.get("cycle", {}) if isinstance(cfg.get("cycle", {}), dict) else {}

    symbols = _resolve_cycle_symbols(cfg)
    eval_period = str(drl_cfg.get("eval_period", "120d"))
    eval_interval = str(drl_cfg.get("interval", trading_cfg.get("timeframe", "M5")))
    reward_cfg = drl_cfg.get("reward", {}) if isinstance(drl_cfg.get("reward", {}), dict) else {}
    reward_weights = reward_cfg.get("weights", {}) if isinstance(reward_cfg.get("weights", {}), dict) else {}

    per_symbol = bool(drl_cfg.get("per_symbol", True))
    gates = _gates_from_cfg(cfg)
    skip_completed = _env_bool("AGI_CYCLE_SKIP_COMPLETED", bool(cycle_cfg.get("skip_completed_training", True)))
    force_skip_lstm = _env_bool("AGI_CYCLE_SKIP_LSTM", bool(cycle_cfg.get("skip_lstm", False)))
    force_skip_dreamer = _env_bool("AGI_CYCLE_SKIP_DREAMER", bool(cycle_cfg.get("skip_dreamer", False)))
    lstm_pending = _pending_symbols(symbols, _has_lstm_artifact) if skip_completed else list(symbols)
    dreamer_pending = _pending_symbols(symbols, _has_dreamer_artifact) if skip_completed else list(symbols)
    skip_lstm = force_skip_lstm or (skip_completed and not lstm_pending)
    skip_dreamer = force_skip_dreamer or (skip_completed and not dreamer_pending)

    if skip_lstm:
        logger.warning(f"Skipping LSTM; completed artifacts already exist for symbols={symbols}")
    else:
        logger.info(f"Cycle start: train LSTM per symbol | pending={lstm_pending}")
        _run_symbol_jobs("LSTM", lstm_pending, cfg, _run_train_lstm_symbol)

    if skip_dreamer:
        logger.warning(f"Skipping Dreamer; completed artifacts already exist for symbols={symbols}")
        dreamer_trained = False
    else:
        logger.info(f"Cycle step: train Dreamer for pending symbols={dreamer_pending}")
        _run_symbol_jobs("Dreamer", dreamer_pending, cfg, lambda symbol: _run_train_dreamer_symbol(cfg, symbol))
        dreamer_trained = True

    # Run symbol simulations to optimize timeframe selection
    logger.info(f"Cycle step: running symbol simulations for {symbols}")
    try:
        best_timeframes = _run_symbol_simulation(symbols)
        logger.info(f"Simulation results: {best_timeframes}")
    except Exception as e:
        logger.error(f"Simulation step failed: {e}")
        best_timeframes = {}

    reg = ModelRegistry()
    cycle_report = {
        "mode": "per_symbol" if per_symbol else "global",
        "symbols": [],
        "eval_period": eval_period,
        "eval_interval": eval_interval,
        "gates": gates,
        "skip_completed_training": skip_completed,
        "lstm_pending": lstm_pending,
        "dreamer_pending": dreamer_pending,
        "skip_lstm": skip_lstm,
        "skip_dreamer": skip_dreamer,
        "dreamer_trained": bool(dreamer_trained),
        "simulation_results": best_timeframes,
    }

    try:
        if per_symbol:
            # Train PPO per symbol; failures are caught inside _run_symbol_jobs (parallel path)
            # but in serial mode an exception would propagate. Wrap individually so one bad
            # symbol does not abort the rest of the cycle.
            ppo_failures = []
            for symbol in symbols:
                try:
                    logger.info(f"PPO start | symbol={symbol}")
                    _run_train_drl(symbol=symbol)
                    logger.info(f"PPO complete | symbol={symbol}")
                except Exception as exc:
                    logger.exception(f"PPO training failed | symbol={symbol} | error={exc}")
                    ppo_failures.append((symbol, exc))

            for symbol in symbols:
                # Skip evaluation for symbols whose training already failed.
                if any(s == symbol for s, _ in ppo_failures):
                    logger.warning(f"Skipping evaluation for {symbol} due to PPO training failure")
                    cycle_report["symbols"].append(
                        {
                            "symbol": symbol,
                            "candidate": None,
                            "champion": None,
                            "wins": False,
                            "passes_thresholds": False,
                            "pass_rate": 0.0,
                            "per_symbol_gates": [],
                            "forward_windows": [],
                            "error": f"PPO training failed: {dict(ppo_failures).get(symbol)}",
                        }
                    )
                    continue

                try:
                    candidate = _latest_candidate(reg, symbol=symbol)
                    if not candidate:
                        raise RuntimeError(f"No PPO candidate found after training for {symbol}")

                    champion = reg.load_active_model(prefer_canary=False, symbol=symbol)
                    logger.info(f"Evaluate {symbol} candidate={os.path.basename(candidate)} vs champion={champion}")
                    report = evaluate_candidate_vs_champion(
                        candidate,
                        champion,
                        symbols=[symbol],
                        period=eval_period,
                        interval=eval_interval,
                        reward_weights=reward_weights,
                        gates=gates,
                    )

                    if report.get("error"):
                        raise RuntimeError(f"Evaluator error on {symbol}: {report['error']}")

                    wins = bool(report.get("wins"))
                    passes = bool(report.get("passes_thresholds"))
                    logger.info(f"{symbol} | wins={wins} passes_thresholds={passes} pass_rate={report.get('pass_rate', 0):.2f}")

                    if wins and passes:
                        reg.set_canary(candidate, symbol=symbol)
                        logger.success(f"Canary set for {symbol}: {candidate}")
                    else:
                        logger.warning(f"Candidate blocked for {symbol}; champion unchanged")

                    cycle_report["symbols"].append(
                        {
                            "symbol": symbol,
                            "candidate": candidate,
                            "champion": champion,
                            "wins": wins,
                            "passes_thresholds": passes,
                            "pass_rate": report.get("pass_rate"),
                            "per_symbol_gates": report.get("per_symbol_gates", []),
                            "forward_windows": report.get("forward_windows", []),
                        }
                    )
                except Exception as exc:
                    # Log and continue; one symbol must not abort the remaining cycle.
                    logger.exception(f"Cycle evaluation failed | symbol={symbol} | error={exc}")
                    cycle_report["symbols"].append(
                        {
                            "symbol": symbol,
                            "candidate": None,
                            "champion": None,
                            "wins": False,
                            "passes_thresholds": False,
                            "pass_rate": 0.0,
                            "per_symbol_gates": [],
                            "forward_windows": [],
                            "error": str(exc),
                        }
                    )
        else:
            logger.info("Cycle step: train PPO candidate")
            _run_train_drl(symbol=None)

            candidate = _latest_candidate(reg)
            if not candidate:
                raise RuntimeError("No PPO candidate found after training")

            champion = reg.load_active_model(prefer_canary=False)
            logger.info(f"Cycle step: evaluate candidate={os.path.basename(candidate)} vs champion={champion}")
            report = evaluate_candidate_vs_champion(
                candidate,
                champion,
                symbols=symbols,
                period=eval_period,
                interval=eval_interval,
                reward_weights=reward_weights,
                gates=gates,
            )

            if report.get("error"):
                raise RuntimeError(f"Evaluator error: {report['error']}")

            wins = bool(report.get("wins"))
            passes = bool(report.get("passes_thresholds"))
            logger.info(f"Evaluation result | wins={wins} passes_thresholds={passes} pass_rate={report.get('pass_rate', 0):.2f}")

            if wins and passes:
                reg.set_canary(candidate)
                logger.success(f"Canary set to {candidate}")
            else:
                logger.warning("Candidate blocked by gates; champion unchanged")

            cycle_report["symbols"].append(
                {
                    "symbol": "GLOBAL",
                    "candidate": candidate,
                    "champion": champion,
                    "wins": wins,
                    "passes_thresholds": passes,
                    "pass_rate": report.get("pass_rate"),
                    "per_symbol_gates": report.get("per_symbol_gates", []),
                    "forward_windows": report.get("forward_windows", []),
                }
            )
    except Exception as exc:
        cycle_report["error"] = str(exc)
        _write_cycle_report(cycle_report)
        alerter.training_cycle("failed", symbols, cycle_report, detail=str(exc))
        raise

    _write_cycle_report(cycle_report)
    promoted = [row.get("symbol") for row in cycle_report["symbols"] if row.get("wins") and row.get("passes_thresholds")]

    # Check for existing canaries ready for champion promotion
    # Note: requires run_cycle.py stage_demo_canary to have run first and updated canary_state via update_canary_metrics
    champion_promoted = []
    for sym in symbols:
        try:
            can_promote, reason = reg.can_promote_canary(symbol=sym)
            if can_promote:
                reg.promote_canary_to_champion(symbol=sym)
                champion_promoted.append(sym)
                logger.success(f"Canary promoted to champion for {sym}")
        except Exception as e:
            logger.warning(f"Champion promotion check for {sym}: {e}")

    if champion_promoted:
        promoted.extend(champion_promoted)
    detail = f"promoted={','.join(str(x) for x in promoted) if promoted else 'none'}"
    alerter.training_cycle("complete", symbols, cycle_report, detail=detail)


if __name__ == "__main__":
    main()
