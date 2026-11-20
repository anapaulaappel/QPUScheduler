"""
X4 — Cost and quota under the E3 placement policies (offline, no QPU; illustrative list prices).

Per (circuit, target) cost from the usage model on the transpiled circuit (same as E3; cached in
results/x4_costs.csv). For each policy and requirement tau (102 jobs, 8 simulated candidates):
  - total estimated cost (USD) and QPU-seconds of the jobs that ran;
  - jobs that met the requirement per 100 USD (fidelity-per-dollar, illustrative);
  - quota: jobs processed in a fixed random order against a budget equal to 25% / 50% of the
    raw-fidelity policy's total cost at the same tau; a job whose cost exceeds the remaining budget is
    blocked (counted as blocked by quota).
Prices: IBM pay-as-you-go per second (1.60 USD/s) and per-gate-shot prices for the trapped-ion models
(common.py). They are illustrative list prices, not quotes.

Run:  python x4_cost_quota.py
Out:  results/x4_cost_quota.json, results/x4_cost_quota.csv
"""
import json

import numpy as np
import pandas as pd

from common import RESULTS
from e3_routing_policies import measure_costs_and_latency

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)
POLICIES = ("raw_best", "lowest_cost", "estimate_filter", "transpile_all")
SEED = 20261120


def choose(g, pol, tau):
    if pol == "raw_best":
        return g.loc[g.raw_fidelity.idxmax()]
    if pol == "lowest_cost":
        return g.loc[g.effective_cost.idxmin()]
    col = "esp_estimated" if pol == "estimate_filter" else "esp_actual"
    ok = g[g[col] >= tau]
    return ok.loc[ok[col].idxmax()] if len(ok) else None


def main():
    cache = RESULTS / "x4_costs.csv"
    e2 = pd.read_csv(RESULTS / "e2_dataset_large.csv")
    if not cache.exists():
        measure_costs_and_latency(sizes=sorted(e2.n.unique()), instances=3).to_csv(cache, index=False)
    df = e2.merge(pd.read_csv(cache), on=["circuit", "target"])
    order = np.random.default_rng(SEED).permutation(df.circuit.unique())
    rows = []
    for tau in TAUS:
        picks = {}
        for pol in POLICIES:
            picks[pol] = {c: choose(g, pol, tau) for c, g in df.groupby("circuit")}
        ref_cost = sum(p.effective_cost for p in picks["raw_best"].values())
        for pol in POLICIES:
            ran = [p for p in picks[pol].values() if p is not None]
            cost = sum(p.effective_cost for p in ran)
            met = sum(p.success_probability >= tau for p in ran)
            r = dict(tau=tau, policy=pol, ran=len(ran), met_requirement=int(met),
                     violations=int(len(ran) - met), total_cost_usd=round(cost, 2),
                     qpu_seconds=round(sum(p.qpu_seconds for p in ran), 1),
                     met_per_100usd=round(100 * met / cost, 3) if cost else None)
            for frac in (0.25, 0.5):
                budget, blocked, done_ok = frac * ref_cost, 0, 0
                for c in order:
                    p = picks[pol][c]
                    if p is None:
                        continue
                    if p.effective_cost <= budget:
                        budget -= p.effective_cost
                        done_ok += int(p.success_probability >= tau)
                    else:
                        blocked += 1
                r[f"blocked_at_{int(frac * 100)}pct"] = blocked
                r[f"met_within_{int(frac * 100)}pct"] = done_ok
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "x4_cost_quota.csv", index=False)
    (RESULTS / "x4_cost_quota.json").write_text(json.dumps(
        {"prices": "illustrative list prices (common.py)", "seed": SEED, "rows": rows}, indent=1))
    pd.set_option("display.width", 180)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
