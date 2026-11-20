// GTA VI trophy rarity forecast — gta6-trophy-rarity-forecast-01
// ---------------------------------------------------------------------------
// What this program does
// -----------------------
// For each PlayStation 5 trophy of Grand Theft Auto VI, forecast its earn
// rate on Exophase (percent of tracked players who earned it) at a case's
// target time, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95).
// Input:  /input/<id>.in  (schema "makefun/gta6-trophies-case/v1")
// Output: /output/<id>.out  {"trophies": [{"id": n, "q": [q05..q95]}, ...]}
// Node 22, standard library only, no network, fully deterministic
// (no clock, no randomness: two runs write identical bytes).
//
// Method: a blend of three estimators, all fitted at run time from the
// case file itself, on the logit scale (the scorer's scale, so a miss
// from 1% to 2% costs about what a miss from 33% to 50% does):
//
//   1. TREND (per-trophy OLS): logit(rate) regressed on ln(1 + hours
//      since the first read). Extrapolated to the target. This is the
//      house curve baseline's idea, but fitted over the whole history
//      instead of only the last 24 hours, and with a residual spread.
//   2. kNN (k = 8, across trophies): trophies whose last rate, tier and
//      secrecy are closest lend their trend-implied logit move
//      (target minus last). The median neighbour move is a second
//      estimate of this trophy's move; its spread measures how much
//      trophies like this one disagree. This shrinks noisy single-trophy
//      trends toward what similar trophies do.
//   3. DILUTION PRIOR (stale-page regime only): Exophase refreshes a
//      game's counts on its own schedule, not with every player, so a
//      history of identical reads means the page is stale, not that
//      nothing changed. When the history shows no movement at all, the
//      trend and kNN both say "no change" — which the practice cases
//      prove wrong (the house baseline scores ~1.51 there, with ~0
//      coverage, because the first refresh drops almost every rate).
//      In that regime the median comes from a population-growth model:
//      tracked players multiply across the refresh, and newly tracked
//      players earn each trophy at a class-dependent fraction of the
//      current rate (near 1 for the first-mission trophy, which the
//      rules say did NOT drop at the first refresh; small for
//      platinum / 100%-completion / heavy grinds, which need far more
//      play time than a new player has had).
//
// Blending: a reliability weight `rel` in [0, 1] measures how much real
// movement the history contains (number of distinct rate vectors and
// time span). median = rel * (trend + kNN)/2 + (1 - rel) * dilution.
// When the history moves (the final case, ~2 days of reads), the trend
// already captures dilution and progression, so the prior drops out;
// when it is frozen (the practice cases), the prior carries the median
// and a wide spread covers the refresh shock.
//
// Grounding for the priors (assumptions, labelled as such — none of
// these is presented as an official number):
//   - GTA VI sold ~24.6M copies on launch day, ~18.4M on PS5
//     (Sensor Tower estimate reported 20 Nov 2026; Take-Two has
//     released no official figure). Exophase tracked only 638 players
//     on launch night, so its tracked population must grow by orders
//     of magnitude during week one: dilution dominates the early
//     dynamics, and rates for most trophies fall even as players
//     progress. The first refresh the rules describe (638 -> 3,868
//     tracked players, every rate but the 100% one lower) is the
//     direct evidence for this.
//   - GTA V's platinum sits at ~0.1% of PSN players long-term
//     (PSN-wide; ~5% on PSNProfiles, a self-selected hunter sample)
//     and takes roughly 80-200 hours of play. A week-one platinum
//     rate anywhere near the launch-night 0.63% (4 of 638 players,
//     almost certainly early/test accounts) is therefore not credible
//     as a steady state: the completion-class catch-up fraction is
//     capped at 0.10.
//   - Story trophies in GTA V settle at roughly 15-35% long-term;
//     week-one rates are far lower because most buyers have only
//     played a few hours. The class fractions below encode that the
//     first mission is earned by essentially every new tracked player
//     while later chapters and grinds lag well behind.
//
// Usage (outside the grader):  MAKEFUN_INPUT_DIR=input \
//   MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

/** Quantile levels, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard-normal quantiles at LEVELS. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Neighbours borrowed per trophy in the kNN step. */
const KNN_K = 8;
/** Random-walk spread of the house baseline, logit units per sqrt(day). */
const BASE_SIGMA = 0.5;
/** Smallest spread we ever report (logit units): the scorer punishes
 *  over-confident bands far more than slightly wide ones. */
const SIGMA_FLOOR = 0.35;
/** Largest spread we ever report. */
const SIGMA_CAP = 2.5;

// ---------------------------------------------------------------------------
// Small numeric helpers (all pure, all deterministic)
// ---------------------------------------------------------------------------

/** Clamp a percent into the scorer's logit domain. */
function clampPct(x: number): number {
  if (!Number.isFinite(x)) return 1;
  return Math.min(99.99, Math.max(0.01, x));
}

function logit(pct: number): number {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
}

function invLogit(l: number): number {
  if (l >= 20) return 99.99;
  if (l <= -20) return 0.01;
  return 100 / (1 + Math.exp(-l));
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / (xs.length - 1));
}

/** Least-squares line y ~ a + b*x. Returns slope, intercept, residual std. */
function ols(xs: number[], ys: number[]): { slope: number; intercept: number; resid: number } {
  const n = xs.length;
  if (n < 2) return { slope: 0, intercept: n === 1 ? ys[0] : 0, resid: 0 };
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx < 1e-12) return { slope: 0, intercept: my, resid: std(ys) };
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  const resid = std(ys.map((y, i) => y - (intercept + slope * xs[i])));
  return { slope, intercept, resid };
}

// ---------------------------------------------------------------------------
// Trophy classes (for the dilution prior only)
// ---------------------------------------------------------------------------
// Classified from the trophy's own description in the case file, so the
// program needs no hard-coded trophy list and works if the list grows.

type TrophyClass = "onboard" | "story" | "grind" | "completion";

function classify(t: { name: string; description: string; tier: string }): TrophyClass {
  const text = `${t.name} ${t.description}`.toLowerCase();
  if (t.tier === "platinum") return "completion";
  if (/100% completion|attain 100%/.test(text)) return "completion";
  if (/gone fishing/.test(text)) return "onboard";
  if (/complete chapter [12]\b/.test(text)) return "story";
  if (/complete (chapter|the final mission)/.test(text)) return "story";
  if (/complete the final mission for/.test(text)) return "story";
  if (/complete \d+ mission strands|complete 8 mission/.test(text)) return "story";
  if (/250 stunts|100 locations|50 challenges|100%|earn \$250,000|purchase all/.test(text)) return "grind";
  if (/gold medal in all|win .* at every location|win 20 different/.test(text)) return "grind";
  return "grind";
}

/**
 * Newly tracked players' earn rate for a trophy of this class, as a
 * fraction of the trophy's current rate, after `horizonDays` days.
 * f0 is the launch-night value. It is calibrated against the one
 * hard number the house publishes about the practice: its baseline
 * scores a loss of ~1.51 there with a typical miss of ~0.68 points.
 * Replaying the baseline against uniform drops of the launch-night
 * rates reproduces ~1.51 / ~0.68 only for a drop to roughly 0.2x —
 * i.e. the first refresh cut most rates to about a fifth, so newly
 * tracked players had earned almost nothing yet (f0 near 0.1 for
 * story trophies, near 0 for grinds and completion). The first-mission
 * trophy is the exception the rules state: it did not drop at all,
 * so its f0 stays 1. The fraction then catches up toward its cap as
 * the horizon lengthens, because new players do play the game —
 * slowly for grinds, never for the 80-200 hour completion trophies
 * within a week.
 */
function catchUpFraction(cls: TrophyClass, horizonDays: number): number {
  const base: Record<TrophyClass, number> = {
    onboard: 1.0,
    story: 0.1,
    grind: 0.05,
    completion: 0.02,
  };
  const cap: Record<TrophyClass, number> = {
    onboard: 1.0,
    story: 0.9,
    grind: 0.3,
    completion: 0.1,
  };
  const f0 = base[cls];
  const caught = f0 + (1 - f0) * (1 - Math.exp(-horizonDays / 3));
  return Math.min(cap[cls], caught);
}

/**
 * Prior for how many times the tracked-player count multiplies by the
 * target, when the history shows no growth at all (stale page).
 * One refresh multiplies it ~6x (638 -> 3,868 at the first refresh the
 * rules describe); we take that published jump as the short-horizon
 * prior and let it keep growing, capped, over a long one.
 */
function growthPrior(horizonDays: number): number {
  if (horizonDays <= 0.5) return 6;
  return Math.min(80, 6 * Math.exp(0.8 * (horizonDays - 0.5)));
}

// ---------------------------------------------------------------------------
// Case types
// ---------------------------------------------------------------------------

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

// ---------------------------------------------------------------------------
// Forecast one case
// ---------------------------------------------------------------------------

function forecastCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  const horizonDays = Math.max(0, c.horizon_minutes / 1440);
  const trophies = c.trophies;
  const n = trophies.length;
  const history = c.history ?? [];

  // --- History summary -----------------------------------------------------
  // Distinct rate vectors: 1 means the page never moved (stale).
  const seenVectors = new Set<string>();
  for (const r of history) {
    seenVectors.add(JSON.stringify(r.rates));
  }
  const distinct = seenVectors.size;
  const firstAt = history.length > 0 ? Date.parse(history[0].at) : NaN;
  const lastAt = history.length > 0 ? Date.parse(history[history.length - 1].at) : NaN;
  const spanHours = Number.isFinite(firstAt) && Number.isFinite(lastAt)
    ? Math.max(0, (lastAt - firstAt) / 3600000)
    : 0;
  // Reliability of the trend: 0 when frozen, 1 when the history shows
  // several distinct reads spread over at least 12 hours.
  const rel = history.length === 0
    ? 0
    : Math.min(1, Math.max(0, (distinct - 1) / 4)) * Math.min(1, spanHours / 12);

  // Target position on the trend axis: ln(1 + hours since first read).
  const xTarget = Math.log(1 + spanHours + c.horizon_minutes / 60);

  // Per-trophy last rate and logit.
  const lastRate: number[] = trophies.map((_, j) => {
    for (let i = history.length - 1; i >= 0; i--) {
      const r = history[i].rates[j];
      if (r !== null && r !== undefined && Number.isFinite(r)) return r;
    }
    return 1; // no read ever listed this trophy: neutral 1% prior
  });
  const lastLogit = lastRate.map((r) => logit(r));

  // --- 1. Trend: OLS of logit(rate) on ln(1 + hours since first read) ------
  const trendLogit: number[] = new Array(n).fill(0);
  const trendResid: number[] = new Array(n).fill(0);
  const trendDelta: number[] = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const r of history) {
      const rate = r.rates[j];
      if (rate === null || rate === undefined || !Number.isFinite(rate)) continue;
      const t = (Date.parse(r.at) - firstAt) / 3600000;
      if (!Number.isFinite(t) || t < 0) continue;
      xs.push(Math.log(1 + t));
      ys.push(logit(rate));
    }
    if (xs.length >= 3) {
      const fit = ols(xs, ys);
      trendLogit[j] = fit.intercept + fit.slope * xTarget;
      trendResid[j] = fit.resid;
    } else if (xs.length >= 1) {
      trendLogit[j] = ys[ys.length - 1];
      trendResid[j] = 0;
    } else {
      trendLogit[j] = lastLogit[j];
      trendResid[j] = 0;
    }
    trendDelta[j] = trendLogit[j] - lastLogit[j];
  }

  // --- 2. kNN: borrow the trend move of similar trophies -------------------
  const knnDelta: number[] = new Array(n).fill(0);
  const knnSpread: number[] = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    const dists: Array<{ d: number; delta: number }> = [];
    for (let i = 0; i < n; i++) {
      if (i === j) continue;
      let d = Math.abs(lastLogit[i] - lastLogit[j]);
      if (trophies[i].tier !== trophies[j].tier) d += 0.6;
      if (trophies[i].secret !== trophies[j].secret) d += 0.3;
      dists.push({ d, delta: trendDelta[i] });
    }
    dists.sort((a, b) => a.d - b.d);
    const near = dists.slice(0, Math.min(KNN_K, dists.length));
    const moves = near.map((x) => x.delta);
    knnDelta[j] = median(moves);
    knnSpread[j] = std(moves);
  }

  // --- 3. Dilution prior (stale-page regime) -------------------------------
  // Extrapolate the tracked-player count. If the history shows growth,
  // fit log(players) on the trend axis and cap the multiplier; if it
  // shows none, fall back to the growth prior above.
  let playerMult = growthPrior(horizonDays);
  const lastPlayers = history.length > 0 ? history[history.length - 1].players_tracked : 0;
  // Only trust a fitted growth rate when the player count actually moved:
  // a flat series fits slope 0 and would report "no growth", which is
  // exactly the stale-page case the prior exists for.
  const playersMoved = history.some((r) => r.players_tracked !== lastPlayers);
  if (history.length >= 3 && lastPlayers > 0 && playersMoved) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const r of history) {
      if (!(r.players_tracked > 0)) continue;
      const t = (Date.parse(r.at) - firstAt) / 3600000;
      if (!Number.isFinite(t) || t < 0) continue;
      xs.push(Math.log(1 + t));
      ys.push(Math.log(r.players_tracked));
    }
    if (xs.length >= 3) {
      const fit = ols(xs, ys);
      const pred = Math.exp(fit.intercept + fit.slope * xTarget);
      if (Number.isFinite(pred) && pred > 0) {
        playerMult = Math.min(80, Math.max(1, pred / lastPlayers));
      }
    }
  }
  const dilutionLogit: number[] = trophies.map((t, j) => {
    const r = lastRate[j];
    if (r >= 99.99) return logit(99.5); // first-mission trophy: stays ~full
    const cls = classify(t);
    const f = catchUpFraction(cls, horizonDays);
    const u = f * r; // newly tracked players' earn rate for this trophy
    const mixed = (r + (playerMult - 1) * u) / playerMult;
    return logit(mixed);
  });

  // --- Blend median + spread ----------------------------------------------
  const baseSigma = BASE_SIGMA * Math.sqrt(horizonDays);
  const rows: Array<{ id: number; q: number[] }> = [];
  for (let j = 0; j < n; j++) {
    const blendedTrend = (trendLogit[j] + (lastLogit[j] + knnDelta[j])) / 2;
    let med = rel * blendedTrend + (1 - rel) * dilutionLogit[j];
    let sigma = Math.sqrt(
      baseSigma * baseSigma +
        rel * (trendResid[j] * trendResid[j] + knnSpread[j] * knnSpread[j]) +
        (1 - rel) * (1.0 * 1.0 + 0.5 * 0.5),
    );
    if (lastRate[j] >= 99.99) {
      // The one trophy the first refresh left at 100%: median just under
      // full, with a real (not token) downward spread — a flood of new
      // tracked players could still pull it down.
      med = logit(99.0);
      sigma = Math.max(sigma, 0.8);
    }
    sigma = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, sigma));
    const qs = Z.map((z) => round4(clampPct(invLogit(med + z * sigma))));
    // Enforce non-decreasing (rounding can tie, never invert).
    for (let k = 1; k < qs.length; k++) {
      if (qs[k] < qs[k - 1]) qs[k] = qs[k - 1];
    }
    rows.push({ id: trophies[j].id, q: qs });
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Driver: read every *.in in the input dir, write one *.out per case
// ---------------------------------------------------------------------------

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith(".in")).sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as CaseFile;
  if (!Array.isArray(c.levels) || c.levels.length !== LEVELS.length ||
      c.levels.some((t, k) => t !== LEVELS[k])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!Array.isArray(c.trophies) || c.trophies.length === 0) {
    throw new Error(`${file} lists no trophies`);
  }
  const rows = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
}
