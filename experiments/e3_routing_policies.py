"""
E3 — Routing policies: SLA violations, cost and scheduling latency.

Each benchmark (mirror) circuit is a job with a minimum-fidelity requirement
τ, to be placed on one of the candidate QPUs of E2. Ground truth for a
placement = the simulated success probability from E2.

Policies:
  raw_best         highest raw device fidelity (ignores the circuit)
  lowest_cost      cheapest QPU (no transpilation awareness)
  estimate_filter  OURS: keep candidates with ESP(topology estimate) − margin ≥ τ,
                   pick the highest predicted value; unschedulable if none
  perqubit_filter  OURS, per-qubit ESP (noise-aware placement + routing replay)
  calibrated_filter OURS, ESP(topology estimate) mapped to success probability by an
                   isotonic calibration fitted on the OTHER circuit families,
                   with a margin = 90th percentile of the training
                   over-prediction (leave-one-family-out)
  transpile_all    transpile for EVERY candidate, filter on ESP(actual)
  oracle           true best placement (upper bound; not implementable)

Metrics vs τ: SLA violations (ran but success < τ), unschedulable jobs,
false unschedulable (a feasible QPU existed), effective cost, and per-job
scheduling latency (estimates vs transpilations over all candidates).

Run:  python e3_routing_policies.py      (requires results/e2_dataset.csv)
Out:  results/e3_policies.csv, results/e3_summary.json, results/e3_policies.png
"""

import json
import warnings

import numpy as np
import pandas as pd

from common import RESULTS, benchmark_suite, count_ops, load_targets, timed, transpile_best
from e2_fidelity_prediction import mirror

warnings.filterwarnings("ignore")
SHOTS = 1024
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)


def measure_costs_and_latency(sizes=(4, 6, 8, 10), instances=1):
    """Per (circuit, target): estimate time, transpile time, effective cost."""
    from circuit_analysis import circuit_stats, estimate_divergence
    from usage import estimate_usage
    targets = load_targets()
    rows = []
    for i, b in enumerate(benchmark_suite(sizes=sizes, instances=instances)):
        circ, _ = mirror(b.circuit, seed=100 + i)
        st, t_stats = timed(circuit_stats, circ)
        qc = circ.copy()
        qc.measure_all()
        for t in targets:
            tel = t.telemetry
            _, t_est = timed(estimate_divergence, st, tel.coupling_map, tel.qubit_count,
                             tel.native_gates)
            out, t_tr = transpile_best(qc, t.backend)
            n1, n2, depth = count_ops(out)
            u = estimate_usage(tel.pricing, tel.durations, depth, n1, n2, SHOTS)
            rows.append(dict(circuit=b.name, target=t.name, stats_ms=t_stats * 1e3,
                             estimate_ms=t_est * 1e3, transpile_ms=t_tr * 1e3,
                             effective_cost=u.effective_cost_usd, qpu_seconds=u.qpu_seconds))
    return pd.DataFrame(rows)


def choose(g, policy, tau, margin=0.0):
    """Return the chosen row (or None = unschedulable)."""
    if policy == "raw_best":
        return g.loc[g.raw_fidelity.idxmax()]
    if policy == "lowest_cost":
        return g.loc[g.effective_cost.idxmin()]
    if policy == "oracle":
        ok = g[g.success_probability >= tau]
        return ok.loc[ok.success_probability.idxmax()] if len(ok) else None
    col = {"estimate_filter": "esp_estimated", "transpile_all": "esp_actual",
           "learned_filter": "learned", "perqubit_filter": "esp_per_qubit",
           "calibrated_filter": "calibrated"}[policy]
    if policy == "calibrated_filter":
        ok = g[g.calibrated - g.calibration_margin >= tau]
        return ok.loc[ok[col].idxmax()] if len(ok) else None
    ok = g[g[col] - margin >= tau]
    return ok.loc[ok[col].idxmax()] if len(ok) else None


def evaluate(df):
    policies = [("raw_best", 0), ("lowest_cost", 0), ("estimate_filter", 0.0),
                ("perqubit_filter", 0.0), ("calibrated_filter", 0.0),
                ("learned_filter", 0.0), ("transpile_all", 0.0), ("oracle", 0)]
    rows = []
    for tau in TAUS:
        for pol, margin in policies:
            ran = viol = unsched = false_unsched = 0
            succ, cost, lat = [], [], []
            for _, g in df.groupby("circuit"):
                feasible = (g.success_probability >= tau).any()
                pick = choose(g, pol, tau, margin)
                # Scheduling latency for this job
                if pol == "transpile_all":
                    lat.append(g.stats_ms.iloc[0] + g.transpile_ms.sum())
                elif pol in ("estimate_filter", "learned_filter", "perqubit_filter",
                             "calibrated_filter"):
                    lat.append(g.stats_ms.iloc[0] + g.estimate_ms.sum())
                else:
                    lat.append(0.0)
                if pick is None:
                    unsched += 1
                    false_unsched += int(feasible)
                    continue
                ran += 1
                succ.append(pick.success_probability)
                cost.append(pick.effective_cost)
                viol += int(pick.success_probability < tau)
            n = df.circuit.nunique()
            rows.append(dict(tau=tau, policy=pol if not margin else f"{pol}(margin={margin})",
                             jobs=n, ran=ran, sla_violations=viol, unschedulable=unsched,
                             false_unschedulable=false_unsched,
                             violation_rate=viol / n, mean_success=float(np.mean(succ)) if succ else np.nan,
                             mean_effective_cost=float(np.mean(cost)) if cost else np.nan,
                             median_latency_ms=float(np.median(lat))))
    return pd.DataFrame(rows)


LABELS = {"raw_best": "Highest raw fidelity", "lowest_cost": "Lowest cost",
          "estimate_filter": "Estimate filter (ours)", "perqubit_filter": "Per-qubit estimate filter",
          "calibrated_filter": "Calibrated estimate filter", "learned_filter": "Learned predictor filter",
          "transpile_all": "Transpile every candidate", "oracle": "Oracle (true success)"}


def plot(res, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    order = [p for p in LABELS if p in set(res.policy)]
    colors = dict(zip(order, plt.cm.tab10.colors))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), layout="constrained")
    for pol in order:
        d = res[res.policy == pol].sort_values("tau")
        kw = dict(color=colors[pol], marker="o", ms=4, lw=2.4 if pol == "estimate_filter" else 1.4,
                  label=LABELS[pol])
        axes[0].plot(d.tau, d.sla_violations, **kw)
        axes[1].plot(d.tau, d.unschedulable, **kw)
    axes[0].set_title("SLA violations")
    axes[1].set_title("Jobs declared unschedulable")
    for ax in axes[:2]:
        ax.set_xlabel("Required success probability τ")
        ax.set_ylabel(f"Jobs (of {int(res.jobs.iloc[0])})")
    lat = res[res.tau == res.tau.min()].set_index("policy").median_latency_ms
    lat = lat[[p for p in order if p in lat.index and lat[p] > 0]]
    axes[2].bar(range(len(lat)), lat.values, color=[colors[p] for p in lat.index])
    axes[2].set_xticks(range(len(lat)))
    axes[2].set_xticklabels([LABELS[p].replace(" (ours)", "*") for p in lat.index], rotation=35, ha="right", fontsize=10)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Median latency per job (ms)")
    axes[2].set_title("Decision latency")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=11, frameon=False)
    fig.savefig(RESULTS / f"e3_policies{tag}.png", dpi=200)
    fig.savefig(RESULTS / f"e3_policies{tag}.pdf")


def calibrate_lofo(df, col="esp_estimated", q=0.9):
    """Isotonic map predictor → success, fitted on the other families."""
    from sklearn.isotonic import IsotonicRegression
    cal = pd.Series(np.nan, index=df.index)
    margin = pd.Series(np.nan, index=df.index)
    for fam in df.family.unique():
        tr, te = df.family != fam, df.family == fam
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(df.loc[tr, col], df.loc[tr, "success_probability"])
        over = iso.predict(df.loc[tr, col]) - df.loc[tr, "success_probability"]
        cal[te] = iso.predict(df.loc[te, col])
        margin[te] = max(0.0, float(np.quantile(over, q)))
    return cal, margin


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--instances", type=int, default=1)
    args = ap.parse_args()
    e2 = pd.read_csv(RESULTS / f"e2_dataset{args.tag}.csv")
    from e2_fidelity_prediction import learned_lofo
    e2["learned"] = learned_lofo(e2)
    e2["calibrated"], e2["calibration_margin"] = calibrate_lofo(e2)
    extra = measure_costs_and_latency(sizes=sorted(e2.n.unique()), instances=args.instances)
    df = e2.merge(extra, on=["circuit", "target"])
    res = evaluate(df)
    res.to_csv(RESULTS / f"e3_policies{args.tag}.csv", index=False)
    summary = {
        "jobs": int(df.circuit.nunique()), "candidates_per_job": int(df.target.nunique()),
        "by_tau": {str(t): res[res.tau == t].set_index("policy")[
            ["sla_violations", "unschedulable", "false_unschedulable", "mean_success",
             "mean_effective_cost", "median_latency_ms"]].round(4).to_dict(orient="index")
            for t in TAUS},
    }
    (RESULTS / f"e3_summary{args.tag}.json").write_text(json.dumps(summary, indent=2))
    plot(res, args.tag)
    print(res.round(3).to_string(index=False))
