# 02_Core_Python

Self-contained trading-research codebase. Holds DRL agents, autonomous
meta-learning loops, backtests, live/paper trading entrypoints, and the tests
for them.

Heavy domain deps (`torch`, `metatrader5`, `xgboost`, ...) are gated behind
`pytest.importorskip`, so the test surface only needs the minimal pinned set in
`requirements.txt` (`pytest`, `loguru`, `bottle`, `requests`, `numpy`).

## Layout

| Subdir | What's inside |
| --- | --- |
| `artifacts/` | Replay bundles and validation-harness outputs (`replay_builder/`, `validation_harness/`). |
| `config/` | Per-symbol configs (e.g. `XAUUSDm.yaml`). |
| `data/` | Replay data snapshots used by tests and backtests. |
| `drl/` | Reinforcement-learning core — agents (`ppo_agent.py`, `dreamer_agent.py`), feature extractors (`adaptive_feature_extractor.py`, `chronos_extractor.py`, ...), shared policies and training utilities. |
| `live/` | Live and paper trading entrypoints (`live_trade.py`, `paper_trade.py`). |
| `logs/` | Run logs (`ppo_training.log`, `dreamer_training.log`, `server.log`) plus trader-review notes under `trade_reviews/`. |
| `models/` | Saved checkpoints. Empty in the bare checkout; populated at runtime. Usually `.gitignore`d. |
| `reports/` | Generated reports; `validation/` holds canary / harness outputs. |
| `runtime/` | Live state + inter-process queues: `agent_status/`, `mql5_commands/`, `execution_reports/`, `retraining_jobs/`, `meta_optimizer/`, `self_evolution/`, `validation_results/`. |
| `training/` | End-to-end training scripts — `enhanced_train_drl.py`, `eval_harness.py`, plus the `run_*` / `plot_*` family. |
| `tests/` | pytest suite — see below. |

### `Python/`

Domain modules grouped by concern. Top-level subdirs:
`alerts/`, `analysis/`, `autonomous/`, `backtest/`, `canary/`, `compat/`,
`data/`, `datasets/`, `ensemble/`, `execution/`, `features/`, `feedback/`.

- `analysis/` — diagnostics such as the LSTM gradient-flow analyzer.
- `autonomous/` — meta-learning loop pieces: continual-learner, experience
  memory, meta-optimizer, regime controller, self-monitor, validation
  harness.
- `backtest/`, `canary/`, `ensemble/`, `execution/`, `features/` — execution
  pathway on the agent side.
- `compat/`, `data/`, `datasets/`, `feedback/`, `alerts/` — supporting glue.

## `tests/`

Pytest discovers here. Eight test modules cover:

- `test_api_server_health.py` *(integration)* — api-server + Server_AGI traffic contract.
- `test_feature_registry.py` — feature-import leak guards.
- `test_no_leakage.py` — train/test split leakage checks.
- `test_server_agi_writer.py` — agi writer round-trips.
- `test_sync_rainforest_model.py` — RandomForest sync path.
- `test_trading_env.py` — gym-style trading env behaviour.
- `test_training_analyzer_cooldown.py` — analyzer cooldown bounds.
- `test_training_analyzer_logging.py` — analyzer logging shape.

`conftest.py` adds `02_Core_Python/Python` to `sys.path`, registers the
`clock_mock`, `build_ollama_response`, `assert_cooldown_bounds`, and
`live_state_factory` fixtures, and bridges `loguru` into pytest's `caplog`
via an autouse handler.

## Running the tests

From this directory:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

## See also

- `.github/workflows/python-tests.yml` — CI workflow that runs this suite
  on Python 3.12 / 3.13 for pushes to `main`, `master`, or `feature/**`.
- `requirements.txt` — minimal pinned dependencies. Heavy domain packages
  (torch, MT5, xgboost) are optional and only loaded behind
  `pytest.importorskip` in tests.
