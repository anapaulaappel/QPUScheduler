"""E7 follow-up: cross-vendor selection with error rates learned from the scheduler's own measurements.

No QPU time: uses results/e7/e7_hardware_with_ibm.csv (written by e7_analyze.py).

Model: one factor gamma_d per device, predicted success = ESP ** gamma_d. Since
log ESP ~ -sum(n_k * e_k), gamma_d scales all of the device's error rates by the same factor
(gamma > 1: the device is worse than its stated error rates). gamma_d is fitted by least squares on
the linear success scale, without the circuit being predicted:

- leave-one-circuit-out (LOCO): gamma_d from all other circuits run on device d;
- k probes: gamma_d from k random other circuits of device d (repeated), i.e. what the scheduler
  would learn from k probe jobs per device.

Selection metrics are the same as in e7_analyze.selection (mean regret, best device picked).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

E7 = Path(__file__).parent / "results" / "e7"
COLS = ("esp_estimated", "esp_actual")
GAMMA_BOUNDS = (0.05, 20.0)


def fit_gamma(esp, measured):
    esp = np.clip(np.asarray(esp, float), 1e-9, 1.0)
    measured = np.asarray(measured, float)
    res = minimize_scalar(lambda g: np.sum((measured - esp ** g) ** 2), bounds=GAMMA_BOUNDS, method="bounded")
    return float(res.x)


def selection(df, col):
    reg, hit = [], []
    for _, g in df.groupby("circuit"):
        if g.target.nunique() < 2:
            continue
        pick, best = g.loc[g[col].idxmax()], g.loc[g.measured.idxmax()]
        reg.append(best.measured - pick.measured)
        hit.append(pick.target == best.target)
    return {"mean_regret": float(np.mean(reg)), "top1": float(np.mean(hit)), "circuits": len(reg)}


def per_device(df, col):
    out = {}
    for t, g in df.groupby("target"):
        out[t] = {"spearman": float(spearmanr(g[col], g.measured).correlation),
                  "mae": float((g[col] - g.measured).abs().mean())}
    return out


def loco(df, col):
    pred = pd.Series(index=df.index, dtype=float)
    for t, g in df.groupby("target"):
        for i, row in g.iterrows():
            train = g.drop(index=i)
            pred[i] = row[col] ** fit_gamma(train[col], train.measured)
    return pred


def k_probes(df, col, k, reps, rng):
    regs, hits = [], []
    for _ in range(reps):
        d = df.copy()
        d["pred"] = np.nan
        for t, g in df.groupby("target"):
            for i, row in g.iterrows():
                others = g.drop(index=i)
                train = others.loc[rng.choice(others.index, size=min(k, len(others)), replace=False)]
                d.loc[i, "pred"] = row[col] ** fit_gamma(train[col], train.measured)
        s = selection(d, "pred")
        regs.append(s["mean_regret"])
        hits.append(s["top1"])
    return {"mean_regret": float(np.mean(regs)), "regret_p10_p90": [float(np.percentile(regs, 10)),
            float(np.percentile(regs, 90))], "top1": float(np.mean(hits))}


def main():
    df = pd.read_csv(E7 / "e7_hardware_with_ibm.csv")
    rng = np.random.default_rng(0)
    out = {"model": "predicted = ESP ** gamma_device, gamma by least squares on linear success",
           "devices": sorted(df.target.unique().tolist()),
           "always_device_regret": {}, "gamma_all_data": {}, "published": {}, "loco": {}, "k_probes": {}}
    for t in df.target.unique():
        out["always_device_regret"][t] = float(np.mean([g.measured.max() - g[g.target == t].measured.iloc[0]
                                                        for _, g in df.groupby("circuit")
                                                        if (g.target == t).any() and g.target.nunique() > 1]))
    for col in COLS:
        out["gamma_all_data"][col] = {t: fit_gamma(g[col], g.measured) for t, g in df.groupby("target")}
        out["published"][col] = {"all6": selection(df, col),
                                 "no_aqt": selection(df[df.target != "aqt_ibex_q1"], col)}
        d = df.assign(pred=loco(df, col))
        out["loco"][col] = {"all6": selection(d, "pred"),
                            "no_aqt": selection(d[d.target != "aqt_ibex_q1"], "pred"),
                            "per_device": per_device(d, "pred")}
        out["k_probes"][col] = {k: k_probes(df, col, k, reps=200, rng=rng) for k in (1, 2, 3, 5)}
    (E7 / "e7_learned_rates.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
