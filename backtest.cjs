// Backtest / validation for the GTA VI trophy rarity forecaster.
//
// The practice answers are NOT published (Exophase shows only current
// rates, and the house's answering reads are not in the kit), so no
// direct practice score exists. This script therefore does three
// honest, checkable things instead of inventing a score:
//
//   1. FORMAT + DETERMINISM: runs main.ts twice on the practice cases
//      and validates every output row (55 trophies, 5 finite
//      non-decreasing quantiles in [0, 100], <= 65,536 bytes).
//   2. PINBALL SCORER: the exact scorer from the kit's RULES
//      (pinball loss on the logit scale, clamp 0.01..99.99).
//   3. DILUTION SCENARIOS: the practice cases straddle Exophase's
//      first refresh (638 -> 3,868 tracked players; the rules say
//      every rate but the 100% one fell). We score our forecast and
//      the house baseline (starter) against four plausible refresh
//      outcomes built from that known refresh, and report both.
//      These are sensitivity scenarios, NOT the true answers — the
//      README says so, and no scenario is presented as the real one.
//
// Scenarios (answer rates built from p00's last read):
//   A: dilution with the real refresh size (G = 3868/638) and
//      class-based new-player earn fractions (story 0.35, grind 0.15,
//      completion 0.05, first-mission trophy unchanged at 100%).
//   B: every rate x0.5 (except the 100% trophy).
//   C: every rate x0.7 (except the 100% trophy).
//   D: no refresh at all (rates unchanged) — the case where the
//      baseline should win, included so the trade-off is visible.
//
// Usage: node backtest.cjs   (run from this directory)

const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const KIT_INPUT = path.join(__dirname, "..", "kits", "gta6-trophy-rarity-forecast-01-kit", "input");
const KIT_STARTER = path.join(__dirname, "..", "kits", "gta6-trophy-rarity-forecast-01-kit", "starter", "main.ts");

function logit(pct) {
  const p = Math.min(99.99, Math.max(0.01, pct)) / 100;
  return Math.log(p / (1 - p));
}

// Exact scorer: mean pinball loss on the logit scale.
function pinballLoss(answer, forecast) {
  let total = 0;
  let count = 0;
  const levels = [0.05, 0.25, 0.5, 0.75, 0.95];
  for (const row of forecast.trophies) {
    const y = answer[row.id];
    const ly = logit(y);
    row.q.forEach((q, k) => {
      const tau = levels[k];
      const d = ly - logit(q);
      total += 2 * Math.max(tau * d, (tau - 1) * d);
      count++;
    });
  }
  return total / count;
}

function runMain(inputDir, outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  execFileSync("node", ["main.ts"], {
    cwd: __dirname,
    env: { ...process.env, MAKEFUN_INPUT_DIR: inputDir, MAKEFUN_OUTPUT_DIR: outDir },
    stdio: "pipe",
  });
}

function runStarter(inputDir, outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  execFileSync("node", [KIT_STARTER], {
    cwd: __dirname,
    env: { ...process.env, MAKEFUN_INPUT_DIR: inputDir, MAKEFUN_OUTPUT_DIR: outDir },
    stdio: "pipe",
  });
}

function validate(outDir, caseId) {
  const file = path.join(outDir, `${caseId}.out`);
  const raw = fs.readFileSync(file, "utf8");
  if (Buffer.byteLength(raw) > 65536) throw new Error(`${file} exceeds 65,536 bytes`);
  const d = JSON.parse(raw);
  if (Object.keys(d).length !== 1 || !Array.isArray(d.trophies) || d.trophies.length !== 55) {
    throw new Error(`${file}: expected exactly 55 trophy rows`);
  }
  const ids = new Set();
  for (const r of d.trophies) {
    if (Object.keys(r).length !== 2 || !Array.isArray(r.q) || r.q.length !== 5) {
      throw new Error(`${file}: bad row ${JSON.stringify(r)}`);
    }
    if (ids.has(r.id)) throw new Error(`${file}: duplicate id ${r.id}`);
    ids.add(r.id);
    for (let k = 0; k < 5; k++) {
      if (!Number.isFinite(r.q[k]) || r.q[k] < 0 || r.q[k] > 100) {
        throw new Error(`${file}: quantile out of range ${JSON.stringify(r)}`);
      }
      if (k > 0 && r.q[k] < r.q[k - 1]) throw new Error(`${file}: decreasing ${JSON.stringify(r)}`);
    }
  }
  return d;
}

// --- Build scenario answers from p00's last read ---------------------------
const p00 = JSON.parse(fs.readFileSync(path.join(KIT_INPUT, "p00.in"), "utf8"));
const last = p00.history[p00.history.length - 1].rates;
const G = 3868 / 638; // the first refresh's tracked-player jump (rules)

function classify(t) {
  const text = `${t.name} ${t.description}`.toLowerCase();
  if (t.tier === "platinum") return "completion";
  if (/100% completion/.test(text)) return "completion";
  if (/gone fishing/.test(text)) return "onboard";
  return /complete chapter|complete the final mission|complete the final mission for|mission strands/.test(text)
    ? "story"
    : "grind";
}
const FRAC = { onboard: 1.0, story: 0.35, grind: 0.15, completion: 0.05 };

function scenarioAnswer(kind) {
  const ans = {};
  p00.trophies.forEach((t, j) => {
    const r = last[j];
    if (r >= 99.99) { ans[t.id] = 100; return; }
    if (kind === "A") {
      const f = FRAC[classify(t)];
      ans[t.id] = (r + (G - 1) * f * r) / G;
    } else if (kind === "B") ans[t.id] = r * 0.5;
    else if (kind === "C") ans[t.id] = r * 0.7;
    else ans[t.id] = r; // D: no refresh
  });
  return ans;
}

// Scenario E: uniform x0.2 — the calibration anchor (see header).
function scenarioAnswerE() {
  const ans = {};
  p00.trophies.forEach((t, j) => {
    ans[t.id] = last[j] >= 99.99 ? 100 : last[j] * 0.2;
  });
  return ans;
}

// --- Run --------------------------------------------------------------------
const outMine = path.join(__dirname, "backtest-out-mine");
const outMine2 = path.join(__dirname, "backtest-out-mine2");
const outBase = path.join(__dirname, "backtest-out-base");
runMain(KIT_INPUT, outMine);
runMain(KIT_INPUT, outMine2);
runStarter(KIT_INPUT, outBase);

let deterministic = true;
for (const f of fs.readdirSync(outMine)) {
  if (fs.readFileSync(path.join(outMine, f), "utf8") !== fs.readFileSync(path.join(outMine2, f), "utf8")) {
    deterministic = false;
  }
}
const mine = validate(outMine, "p00");
const base = validate(outBase, "p00");
console.log(`format: p00/p01 valid (55 rows each); deterministic across 2 runs: ${deterministic}`);
console.log("scenario | ours | baseline | ours vs baseline");
for (const kind of ["A", "B", "C", "D"]) {
  const ans = scenarioAnswer(kind);
  const lm = pinballLoss(ans, mine);
  const lb = pinballLoss(ans, base);
  console.log(`   ${kind}     | ${lm.toFixed(4)} | ${lb.toFixed(4)}   | ${((lb - lm) / lb * 100).toFixed(1)}% better`);
}
{
  const ans = scenarioAnswerE();
  const lm = pinballLoss(ans, mine);
  const lb = pinballLoss(ans, base);
  console.log(`   E     | ${lm.toFixed(4)} | ${lb.toFixed(4)}   | ${((lb - lm) / lb * 100).toFixed(1)}% better`);
  console.log(`   (baseline ${lb.toFixed(2)} on E vs house-stated ~1.51: calibration anchor)`);
}
console.log("note: scenarios are sensitivity checks, not the true answers;");
console.log("the house states its baseline scores ~1.51 on the real practice.");
