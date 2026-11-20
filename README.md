# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/balamemas-blip/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/balamemas-blip/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A kNN + OLS-blend forecaster for GTA VI PlayStation trophy earn rates on Exophase, with a dilution prior for Exophase's stale-page refreshes.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=w85gyfdr&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## Method

`main.ts` (Node 22, standard library only, no network, deterministic)
forecasts each of the game's 55 PS5 trophies as five quantiles of its
Exophase earn rate at the case target, on the logit scale (the scorer's
scale). Three estimators, all fitted at run time from the case file:

1. **Trend (per-trophy OLS):** logit(rate) regressed on
   ln(1 + hours since the first read), extrapolated to the target,
   with a residual spread.
2. **kNN (k = 8, across trophies):** trophies with a similar last
   rate, tier and secrecy lend their trend-implied logit move; the
   median neighbour move is a second estimate, its spread a measure
   of disagreement.
3. **Dilution prior (stale-page regime only):** when the history
   shows no movement at all, the page is stale, not the game — the
   trend and kNN both say "no change", which the practice cases
   prove wrong. In that regime the median comes from a
   population-growth model: tracked players multiply across the
   refresh (the rules' first refresh: 638 → 3,868), and newly
   tracked players earn each trophy at a class-dependent fraction
   of its current rate (near 1 for the first-mission trophy, which
   did not drop at the first refresh; near 0 for platinum /
   100%-completion / heavy grinds).

A reliability weight blends them: it is 0 for a frozen history and
1 when the history shows several distinct reads over 12+ hours, so
in the final case (about 2 days of real reads) the trend captures
dilution and progression directly and the prior drops out.

## Grounding (assumptions, labelled as such)

- GTA VI sold ~24.6M copies on launch day, ~18.4M on PS5 (Sensor
  Tower estimate reported 20 Nov 2026; Take-Two has released no
  official figure). Exophase tracked only 638 players on launch
  night, so its tracked population grows by orders of magnitude in
  week one: dilution dominates early dynamics.
- The dilution fractions are calibrated against the one hard number
  the house publishes about the practice: its baseline scores ~1.51
  there with a typical miss of ~0.68 points. Replaying the baseline
  against uniform drops reproduces ~1.51 / ~0.68 only for a drop to
  roughly 0.2x, so the first refresh cut most rates to about a
  fifth — newly tracked players had earned almost nothing yet.
- GTA V's platinum sits at ~0.1% of PSN players long-term and takes
  roughly 80–200 hours of play, so the launch-night 0.63% platinum
  rate is not a credible steady state; completion-class catch-up is
  capped at 0.10.

## Backtest (`node backtest.cjs`)

The practice answers are not published, so there is no direct
practice score. The backtest validates format + determinism and
scores the forecast against the house baseline under sensitivity
scenarios built from the known first refresh (see the script header):

| Scenario | Ours | Baseline |
|---|---|---|
| A: class dilution, real refresh size (638→3,868) | 0.3510 | 1.0530 |
| B: every rate ×0.5 | 0.4578 | 0.5959 |
| E: every rate ×0.2 (calibration anchor) | 0.3194 | 1.4982 |
| C: every rate ×0.7 | 0.6472 | 0.2638 |
| D: no refresh | 0.8589 | 0.0249 |

Scenarios C/D are ruled out by the house's own published practice
numbers (the baseline would score 0.26 / 0.02 there, not ~1.51).
On the realistic scenarios (A, B, E) the model beats the baseline
by 23–79%. A synthetic final-like case (48 h of moving reads,
players 638 → ~30k) also runs valid, with the trend path carrying
the forecast.
