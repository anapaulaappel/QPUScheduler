"""
X2 — 95% bootstrap confidence intervals for the main claims (offline, no QPU).

Paired cluster bootstrap over circuits: each resample draws circuits with replacement and keeps all
rows (devices) of each drawn circuit, because observations of one circuit on several devices are not
independent. B = 2000 resamples, fixed seed. Point estimates are computed on the full data and must
match the paper.

  E1  Spearman rho of predicted vs actual 2Q gates            (results/e1_divergence.csv)
  E2  rho, mean regret, top-1 vs simulated success             (results/e2_dataset_large.csv)
  E6  rho, mean regret, top-1 vs measured success (IBM x3)     (results/e6_hardware_large.csv)
  E7  selection over 5 and 6 devices (measured)                (results/e7/e7_hardware_with_ibm.csv)
  E3  SLA violations / false rejections per tau, simulated     (e2 labels; E3 policy definitions)
  E9  same on measured outcomes (IBM x3)                       (e6 labels)

Run:  python x2_bootstrap.py
Out:  results/x2_bootstrap.json, results/x2_bootstrap.csv
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS

B, SEED = 2000, 20261120
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def rho(d, x, y):
    return float(spearmanr(d[x], d[y]).correlation)


def selection(d, pred, label):
    reg, hit = [], []
    for _, g in d.groupby("circuit"):
        if g.target.nunique() < 2 or g[pred].isna().all():
            continue
        g = g.dropna(subset=[pred])
        pick = g.loc[g[pred].idxmax(), label]
        best = g[label].max()
        reg.append(best - pick)
        hit.append(pick >= best - 1e-12)          # ties count as hits
    return float(np.mean(reg)), float(np.mean(hit))


def policy_counts(d, label, col, tau):
    """E3/E9 rule: keep candidates with predicted >= tau, pick the highest predicted; raw_best picks the
    highest raw fidelity regardless of tau. Returns (violations, false_rejections) summed over circuits."""
    viol = fr = 0
    for _, g in d.groupby("circuit"):
        feasible = bool((g[label] >= tau).any())
        if col == "raw_fidelity":
            pick = g.loc[g.raw_fidelity.idxmax()] if g.raw_fidelity.notna().any() else None
        else:
            ok = g[g[col] >= tau]
            pick = ok.loc[ok[col].idxmax()] if len(ok) else None
        if pick is None:
            fr += int(feasible)
        else:
            viol += int(pick[label] < tau)
    return viol, fr


def boot(d, fn, rng):
    circ = d.circuit.unique()
    groups = {c: g for c, g in d.groupby("circuit")}
    out = []
    for _ in range(B):
        draw = rng.choice(circ, size=len(circ), replace=True)
        # relabel duplicates so groupby keeps them as separate circuits
        parts = [groups[c].assign(circuit=f"{c}#{k}") for k, c in enumerate(draw)]
        out.append(fn(pd.concat(parts, ignore_index=True)))
    return np.array(out)


def ci(arr):
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    def add(exp, metric, pred, point, samples, n):
        rows.append(dict(experiment=exp, metric=metric, predictor=pred, point=round(point, 4),
                         lo=round(ci(samples)[0], 4), hi=round(ci(samples)[1], 4), n_circuits=n))

    e1 = pd.read_csv(RESULTS / "e1_divergence.csv")
    for p in ("pred_none", "pred_table", "pred_static", "pred_static_calibrated", "pred_calibrated"):
        add("E1", "spearman", p, rho(e1, p, "actual_n2"),
            boot(e1, lambda d, p=p: rho(d, p, "actual_n2"), rng), e1.circuit.nunique())

    e2 = pd.read_csv(RESULTS / "e2_dataset_large.csv")
    for p in ("raw_fidelity", "esp_logical", "esp_estimated", "esp_per_qubit", "esp_actual", "learned"):
        add("E2", "spearman", p, rho(e2, p, "success_probability"),
            boot(e2, lambda d, p=p: rho(d, p, "success_probability"), rng), e2.circuit.nunique())
        s = boot(e2, lambda d, p=p: selection(d, p, "success_probability"), rng)
        r, t = selection(e2, p, "success_probability")
        add("E2", "regret", p, r, s[:, 0], e2.circuit.nunique())
        add("E2", "top1", p, t, s[:, 1], e2.circuit.nunique())

    e6 = pd.read_csv(RESULTS / "e6_hardware_large.csv")
    for p in ("simulated", "raw_fidelity", "esp_estimated", "esp_actual"):
        add("E6", "spearman", p, rho(e6, p, "measured"),
            boot(e6, lambda d, p=p: rho(d, p, "measured"), rng), e6.circuit.nunique())
        if p != "simulated":
            s = boot(e6, lambda d, p=p: selection(d, p, "measured"), rng)
            r, t = selection(e6, p, "measured")
            add("E6", "regret", p, r, s[:, 0], e6.circuit.nunique())
            add("E6", "top1", p, t, s[:, 1], e6.circuit.nunique())

    e7 = pd.read_csv(RESULTS / "e7" / "e7_hardware_with_ibm.csv")
    for label, d in (("E7-5dev", e7[e7.target != "aqt_ibex_q1"]), ("E7-6dev", e7)):
        for p in ("esp_estimated", "esp_actual"):
            s = boot(d, lambda x, p=p: selection(x, p, "measured"), rng)
            r, t = selection(d, p, "measured")
            add(label, "regret", p, r, s[:, 0], d.circuit.nunique())
            add(label, "top1", p, t, s[:, 1], d.circuit.nunique())

    for exp, d, label in (("E3-sim", e2, "success_probability"), ("E9-ibm", e6, "measured")):
        for col in ("raw_fidelity", "esp_estimated", "esp_actual"):
            for tau in TAUS:
                v, f = policy_counts(d, label, col, tau)
                s = boot(d, lambda x, col=col, tau=tau: policy_counts(x, label, col, tau), rng)
                add(exp, f"violations@{tau}", col, v, s[:, 0], d.circuit.nunique())
                add(exp, f"false_rejections@{tau}", col, f, s[:, 1], d.circuit.nunique())

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "x2_bootstrap.csv", index=False)
    (RESULTS / "x2_bootstrap.json").write_text(json.dumps(
        {"B": B, "seed": SEED, "method": "paired cluster bootstrap over circuits, percentile 95% CI",
         "rows": rows}, indent=1))
    pd.set_option("display.width", 160)
    print(df[~df.metric.str.contains("@")].to_string(index=False))


if __name__ == "__main__":
    main()
