# Full Stack Profitability Review — June 2026

**Goal:** Make the autonomous trading bot reliably profitable on XAUUSDm (and other symbols) after realistic costs, slippage, and live execution.

**Context:** Extensive local experimentation (Lane A/B Tier 1-3) on `experiment/xauusd-regime-baseline`. Multiple prior "fix" attempts in `archive/`. The main production path lives in `drl/trading_env.py`, `Python/rewards/reward_function.py`, `Python/feature_pipeline.py`, `Python/hybrid_brain.py`, `training/train_drl.py`, promotion gates, and MT5 execution.

## Key Findings

### 1. Reward Is the Primary Bottleneck (Highest Leverage)
- Rich `TradingReward` class exists with excellent terms:
  - Explicit spread + commission + slippage costs
  - Drawdown tail penalties, overtrading/churn
  - Anti-HOLD: `excessive_hold_penalty` + **fresh entry bonus** + `hold_persist_penalty`
  - DSR (Differential Sharpe), asymmetric loss amplification (Xia 2023)
  - Vol-adjusted costs, Calmar/Sortino/IR/B&H benchmark, news timing, curriculum stages, regime-adaptive hints
- **Problem:** In `drl/trading_env.py:step()`, it is only used in an 80/20 blend with legacy `shaped_reward` (growth/payoff/churn etc. from a weights dict). Comments admit: "Long-term target: delegate core reward to TradingReward (currently unused)" and "Future: ... as primary".
- Result: Noisy advantages → policy collapses to flat or perpetual HOLD (seen in Lane A blow-ups to -100%, Lane B seed 42 flat at 0%).
- Evidence from diagnostics: On real 2500-bar XAU slices, fixed Short produced +110 scaled reward while Flat was worst (-303). Signal *exists* in the environment on some windows — the learned policy just doesn't discover/exploit it reliably.

### 2. Features Have Dead Weight + Missing Priors
- `Python/feature_registry.py`: ENGINEERED_V2 (43 cols) explicitly documents dead columns (`pattern_*` detector missing, many `cross_asset_*` have no live data).
- MTF builder, `TrendMomentumBiasLayer` (trend/momentum/direction_bias/confidence/agreement/persistent — exactly "momentum and trend features should be a layer before the rest"), Chronos, Sentiment are powerful but **opt-in** via env vars (`AGI_USE_*`) and not default in XAUUSDm.yaml (`feature_version: engineered_v3`).
- The Lane B "raw" path deliberately used only 7 simple cols (OHLCV logrets + RSI + MACD) + market symmetry to isolate the problem. It still collapsed for the same reasons.

### 3. Action Space & Policy Complexity
- Rich `DecisionSpec` (18-dim) + parametrized head for full trade planning (direction, lots, entry, TP/SL, trailing, partials, breakeven, risk) is expressive for live execution.
- But history shows repeated "dead Gaussian" or near-zero policy collapse. Low `ent_coef: 0.005` in config + complex decoder makes credit assignment brutal on noisy 5m rewards.
- Regime routing + HybridBrain blending (PPO primary + optional Dreamer + LSTM context + Rainforest gate) adds power but also variance.

### 4. Costs, Execution Fidelity & Sim-Live Gap
- Training now models realistic costs (good progress from the Tier 1-3 "honest costs" commits).
- Live: `Python/mt5_executor.py` has spread guards, requote retries, slippage logging. `risk_supervisor.py` + `RiskEngine` add daily limits and halts.
- Remaining risks: XAU config has loose `max_spread_bps: 500`; any mismatch in bps/slippage numbers or latency means a marginal edge disappears after fills.
- Backtester re-uses the env (good), but promotion gates (now stricter: min PF 1.15, sharpe 0.50, oos return 2%, rich execution gates for trailing/R-multiple) are the real filter — many candidates still fail to stay profitable live.

### 5. Historical Pattern
- Repeated cycles of "more bars / more seeds / discrete actions / symmetry / inactivity penalty / reward scaling" followed by flat or blowup results.
- `archive/` contains dozens of targeted reward/fix patches. This is symptom of reward hacking + insufficient base edge rather than one missing line of code.

## Evidence from Recent Runs
- Lane A: complete -100% blowups on multiple seeds.
- Lane B Tier 3 (visible pre-completion): seed 42 100% flat, 0% return, position=0 after hundreds of steps. Active PID was still running when reviewed.
- Sanity test (tools/reward_sanity_check.py): proved directional fixed policies can beat flat on the actual reward signal on real data slices.

## Prioritized Fix Roadmap

### Tier 0 (Immediate — This Week)
- [ ] Force `TradingReward` as 100% primary in `drl/trading_env.py` (remove the 0.8/0.2 blend or make it 1.0 default). Re-run small sanity + 10k-step smoke on XAU.
- [ ] Promote & default-enable strong features for XAU: MTF + `TrendMomentumBiasLayer` + clean dead columns from the matrix. Update `configs/XAUUSDm.yaml` + training launchers.
- [ ] Run the reward sanity checker on multiple recent slices + the exact data the current champion was trained on. Require "at least one simple rule or fixed side beats flat + positive expectancy after modeled costs" before any big training.

### Tier 1 (High-ROI)
- Tune anti-HOLD / exploration terms (AGI_TRADE_EXPLORATION_BONUS, hold penalties, cost_penalty via AGI_COST_PENALTY=1.5-2.0) so the gradient prefers "try a small correct-side trade" over flat.
- Tighten XAU risk in config (lower practical max_spread_bps, require higher post-cost PF in gates).
- Add a reusable "profitability scorecard" that runs a candidate through backtest + simulated fills and emits edge-vs-costs, regime breakdown, etc.

### Tier 2 (Architecture)
- Unify reward completely; consider making the rich terms (asymmetric, DSR, regime hints) dominant early.
- Consider a simpler base policy head + separate high-quality exit/risk manager (keeps expressivity without making the PPO output 18-d of correlated params).
- Strengthen live → retrain loop (trade_coroner, outcome_labels, prioritized replay from real fills).

### Tier 3
- Source better edge (multi-horizon labels, external regime signals, focus on high-quality sessions only, or ensemble of specialists).
- Full MQL5 + Python parity for execution if latency matters.

## Recommended Next Experiments
1. Small controlled training diff: 100% TradingReward + MTF + bias layer + slightly higher entropy on a 20-30k step XAU run (3 seeds). Compare flat% / turnover / final NW vs current baseline.
2. Use `tools/reward_sanity_check.py` (and extend it) as a gate in the ablation harness and promotion flow.
3. After any promising run, immediately run paper trading via the execution layer with real MT5 spreads for 1-2 days before canary.

## Files Touched in This Review Cycle
- `training/run_lane_b_raw_lstm.py` + `training/taming_shared.py` (Tier 3 discrete + symmetry + inactivity + market inversion)
- `tools/reward_sanity_check.py` (new diagnostic proving signal exists)
- `docs/FULL_STACK_PROFITABILITY_REVIEW.md` (this document)
- `.gitignore` (hygiene for experiment artifacts so future pushes stay clean)

## References
- `Python/rewards/reward_function.py` (the rich implementation)
- `drl/trading_env.py` (the blend + DecisionSpec)
- `Python/feature_registry.py` (dead columns)
- `drl/trend_momentum_bias.py` (the missing prior layer)
- Promotion gates: `Python/registry/promotion_gates.py`
- CI triggers on `experiment/**` branches (compile + core tests).

**Status (staying on `feature/profitability-tier0-reward` branch as requested):**
- GitHub hygiene setup on separate branch, then profitability work isolated here.
- Tier 0 in progress on this branch:
  - TradingReward 100% primary (drl/trading_env.py) — committed.
  - Configs/XAUUSDm.yaml updated: ent_coef=0.02 (more exploration), feature_version=ultimate_150, explicit use_trend_momentum_bias + use_mtf, dead columns noted.
  - train_drl.py updated to pass use_trend_momentum_bias=True by default (via AGI_USE_TREND_MOMENTUM_BIAS env, default on). This wires the "momentum and trend features as a layer before the rest".
- Next on this branch: run updated sanity as gate, small controlled XAU run (20-30k steps), update review with metrics.
- All changes pushed to this branch only. No switching.

This review is living documentation. Update it as experiments produce measurable OOS improvements after costs.
