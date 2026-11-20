"""
E9 — Placement policies evaluated on MEASURED hardware outcomes (offline counterfactual),
     with conformal margins for the hard fidelity constraint.

In E6 every circuit ran on every IBM device, and in E7 the 28 circuits ran on IBM, IQM and
(subsets) Rigetti and AQT. So any placement policy can be replayed after the fact: we know the
measured success the job WOULD have had on each candidate. No new QPU time.

Policies (same definitions as E3, ground truth = measured success):
  random           uniform over candidates (mean over all choices)
  raw_best         highest raw device fidelity
  estimate_filter  OURS: ESP(estimate) - margin >= tau, pick highest predicted; else unschedulable
  transpile_all    same on ESP(transpiled circuit)
  *_conformal      margin = split-conformal quantile of (predicted - measured), fitted on held-out
                   groups: other circuit families (--split family, default) or circuit folds
                   (--split circuit), level 1 - alpha
  oracle           best feasible placement (upper bound)

Conformal guarantee (marginal, under exchangeability across families): for a new (circuit,
device) pair, P(measured >= predicted - margin) >= 1 - alpha. Coverage is reported per run.

Run:  python e9_hardware_policies.py [--dataset ibm|crossvendor|stale] [--alpha 0.1]
Out:  results/e9_policies_<dataset>.csv, results/e9_summary_<dataset>.json, results/e9_<dataset>.png
"""

import argparse
import json

import numpy as np
import pandas as pd

from common import RESULTS

TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def load(dataset: str) -> pd.DataFrame:
    if dataset == "ibm":
        df = pd.read_csv(RESULTS / "e6_hardware_large.csv")
    elif dataset == "stale":
        df = pd.read_csv(RESULTS / "e6_hardware_stale.csv")
    elif dataset == "crossvendor":
        df = pd.read_csv(RESULTS / "e7" / "e7_hardware_with_ibm.csv")
    else:
        raise ValueError(dataset)
    df = df.dropna(subset=["measured", "esp_estimated", "esp_actual"]).reset_index(drop=True)
    return df


def conformal_margin(df: pd.DataFrame, col: str, alpha: float, split: str,
                     k: int = 5, seed: int = 0) -> pd.Series:
    """Split-conformal margin per row, calibrated on held-out groups.
    split='family': calibrate on the other circuit families (new family; exchangeability
    is not guaranteed, so coverage may fall below 1 - alpha).
    split='circuit': k folds of circuits (exchangeable; the formal guarantee applies)."""
    if split == "family":
        groups = df.family
    else:
        circ = df.circuit.unique()
        rng = np.random.default_rng(seed)
        fold = dict(zip(rng.permutation(circ), np.arange(len(circ)) % k))
        groups = df.circuit.map(fold)
    m = pd.Series(np.nan, index=df.index)
    for g in groups.unique():
        cal = df[groups != g]
        r = (cal[col] - cal.measured).to_numpy()          # over-prediction
        n = len(r)
        q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        m[groups == g] = float(np.quantile(r, q, method="higher"))
    return m


def choose(g, policy, tau, rng=None):
    if policy == "raw_best":
        if g.raw_fidelity.isna().all():
            return None
        return g.loc[g.raw_fidelity.idxmax()]
    if policy == "oracle":
        ok = g[g.measured >= tau]
        return ok.loc[ok.measured.idxmax()] if len(ok) else None
    col = "esp_estimated" if policy.startswith("estimate") else "esp_actual"
    margin = g[f"margin_{col}"] if policy.endswith("_conformal") else 0.0
    ok = g[g[col] - margin >= tau]
    return ok.loc[ok[col].idxmax()] if len(ok) else None


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    policies = ["random", "raw_best", "estimate_filter", "estimate_filter_conformal",
                "transpile_all", "transpile_all_conformal", "oracle"]
    if df.raw_fidelity.isna().all():
        policies.remove("raw_best")
    rows = []
    for tau in TAUS:
        for pol in policies:
            ran = viol = unsched = false_unsched = 0
            succ = []
            for _, g in df.groupby("circuit"):
                feasible = bool((g.measured >= tau).any())
                if pol == "random":                 # expectation over a uniform choice
                    ran += 1
                    viol += float((g.measured < tau).mean())
                    succ.append(float(g.measured.mean()))
                    continue
                pick = choose(g, pol, tau)
                if pick is None:
                    unsched += 1
                    false_unsched += int(feasible)
                    continue
                ran += 1
                succ.append(float(pick.measured))
                viol += int(pick.measured < tau)
            n = df.circuit.nunique()
            rows.append(dict(tau=tau, policy=pol, jobs=n, ran=ran, sla_violations=round(viol, 2),
                             unschedulable=unsched, false_unschedulable=false_unsched,
                             feasible_jobs=int(sum((g.measured >= tau).any()
                                                   for _, g in df.groupby("circuit"))),
                             mean_success=float(np.mean(succ)) if succ else np.nan))
    return pd.DataFrame(rows)


def plot(res, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for pol, d in res.groupby("policy"):
        if pol == "oracle":
            continue
        axes[0].plot(d.tau, d.sla_violations, "o-", ms=3, label=pol)
        axes[1].plot(d.tau, d.false_unschedulable, "o-", ms=3, label=pol)
    axes[0].set_title("SLA violations (measured)", fontsize=9)
    axes[1].set_title("False unschedulable (a feasible device existed)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("τ (min. success)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(RESULTS / f"e9_{dataset}.png", dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ibm", choices=["ibm", "crossvendor", "stale"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--split", default="family", choices=["family", "circuit"],
                    help="conformal calibration groups (see conformal_margin)")
    args = ap.parse_args()
    df = load(args.dataset)
    cov = {}
    for col in ("esp_estimated", "esp_actual"):
        df[f"margin_{col}"] = conformal_margin(df, col, args.alpha, args.split)
        cov[col] = dict(
            coverage=float((df.measured >= df[col] - df[f"margin_{col}"]).mean()),
            margin_median=float(df[f"margin_{col}"].median()),
            margin_range=[float(df[f"margin_{col}"].min()), float(df[f"margin_{col}"].max())])
    res = evaluate(df)
    tag = f"{args.dataset}" + ("" if args.split == "family" else "_circuitsplit")
    res.to_csv(RESULTS / f"e9_policies_{tag}.csv", index=False)
    summary = {"dataset": args.dataset, "conformal_split": args.split, "rows": int(len(df)), "circuits": int(df.circuit.nunique()),
               "devices": sorted(df.target.unique().tolist()), "alpha": args.alpha,
               "conformal": cov,
               "by_tau": {str(t): res[res.tau == t].set_index("policy")[
                   ["ran", "sla_violations", "unschedulable", "false_unschedulable",
                    "mean_success"]].round(3).to_dict(orient="index") for t in TAUS}}
    (RESULTS / f"e9_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    plot(res, tag)
    print(json.dumps({k: summary[k] for k in ("dataset", "rows", "circuits", "devices", "conformal")}, indent=1))
    print(res.pivot_table(index="policy", columns="tau",
                          values=["sla_violations", "false_unschedulable"]).round(1).to_string())
