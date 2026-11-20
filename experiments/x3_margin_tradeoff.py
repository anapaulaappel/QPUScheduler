"""
X3 — Operating curve of the safety margin (offline, no QPU).

The hard filter keeps a candidate when predicted - margin >= tau. Sweeping the margin trades SLA
violations (job placed where its true success < tau) for false rejections (job rejected although some
candidate met tau). Totals are summed over tau = 0.3, ..., 0.8 and over all jobs.

  - additive margin m in [-0.10, 0.30] on ESP(topology estimate), for IBM measured outcomes (E6/E9)
    and for simulated outcomes (E2/E3 labels);
  - split-conformal margin at levels delta in {0.05, 0.1, 0.2, 0.3, 0.5}, calibrated on held-out circuit
    folds (E9 --split circuit), IBM measured outcomes.

Run:  python x3_margin_tradeoff.py
Out:  results/x3_margin_tradeoff.json, results/fig_margin_tradeoff.{png,pdf}
"""
import json

import numpy as np
import pandas as pd

from common import RESULTS
from e9_hardware_policies import conformal_margin

TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MARGINS = np.round(np.arange(-0.10, 0.301, 0.02), 2)
DELTAS = (0.05, 0.1, 0.2, 0.3, 0.5)


def totals(d, label, pred, margin):
    """margin: scalar or a per-row Series. Returns (violations, false rejections) summed over taus."""
    d = d.assign(_adj=d[pred] - margin)
    v = f = 0
    for tau in TAUS:
        for _, g in d.groupby("circuit"):
            ok = g[g._adj >= tau]
            if len(ok):
                v += int(ok.loc[ok[pred].idxmax(), label] < tau)
            else:
                f += int((g[label] >= tau).any())
    return v, f


def main():
    ibm = pd.read_csv(RESULTS / "e6_hardware_large.csv")
    sim = pd.read_csv(RESULTS / "e2_dataset_large.csv")
    out = {"taus": TAUS, "additive": {}, "conformal_ibm": []}
    for name, d, label in (("ibm_measured", ibm, "measured"), ("simulated", sim, "success_probability")):
        out["additive"][name] = [dict(margin=float(m), **dict(zip(("violations", "false_rejections"),
                                 totals(d, label, "esp_estimated", m)))) for m in MARGINS]
    for delta in DELTAS:
        m = conformal_margin(ibm.reset_index(drop=True), "esp_estimated", delta, "circuit")
        v, f = totals(ibm.reset_index(drop=True), "measured", "esp_estimated", m)
        out["conformal_ibm"].append(dict(delta=delta, median_margin=float(m.median()), violations=v,
                                         false_rejections=f))
    raw_v = sum(int(g.loc[g.raw_fidelity.idxmax(), "measured"] < t) for t in TAUS for _, g in ibm.groupby("circuit"))
    out["raw_fidelity_ibm_violations"] = raw_v
    (RESULTS / "x3_margin_tradeoff.json").write_text(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, name, title in ((axes[0], "ibm_measured", "IBM hardware (measured, 102 jobs)"),
                            (axes[1], "simulated", "Simulation (102 jobs, 8 devices)")):
        a = pd.DataFrame(out["additive"][name])
        ax.plot(a.margin, a.violations, "o-", ms=4, color="#d62728", label="SLA violations")
        ax.plot(a.margin, a.false_rejections, "s-", ms=4, color="#1f77b4", label="false rejections")
        ax.axvline(0, color="#888", lw=0.8, ls=":")
        ax.set_xlabel("Safety margin subtracted from the estimate")
        ax.set_ylabel(f"Jobs, summed over τ = {TAUS[0]}–{TAUS[-1]}")
        ax.set_title(title, fontsize=11)
    c = pd.DataFrame(out["conformal_ibm"])
    axes[0].scatter(c.median_margin, c.violations, marker="*", s=90, color="#d62728", zorder=5,
                    label="conformal margin (δ = 0.05–0.5)")
    axes[0].scatter(c.median_margin, c.false_rejections, marker="*", s=90, color="#1f77b4", zorder=5)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_margin_tradeoff.png", dpi=200)
    fig.savefig(RESULTS / "fig_margin_tradeoff.pdf")
    print(json.dumps({k: out[k] for k in ("conformal_ibm", "raw_fidelity_ibm_violations")}, indent=1))
    for name in ("ibm_measured", "simulated"):
        a = pd.DataFrame(out["additive"][name]); print(name); print(a[a.margin.isin([-0.1, 0.0, 0.1, 0.2, 0.3])].to_string(index=False))


if __name__ == "__main__":
    main()
