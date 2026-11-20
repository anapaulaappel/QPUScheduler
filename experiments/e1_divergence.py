"""
E1 — Does the topology-divergence estimate predict real transpilation overhead?

For every benchmark circuit × target QPU:
  - graph estimate, static: placement held fixed (scheduler default);
  - graph estimate, dynamic: greedy routing replay;
  - table estimate: family-level lookup table (source = logical/all-to-all);
  - actual: Qiskit transpilation for the target (opt. level 1, best of 3 seeds).

Reports Spearman ρ and MAE of the predicted vs actual two-qubit gate count,
a leave-one-family-out calibrated estimate, and estimate vs transpile time.

Run:  python e1_divergence.py
Out:  results/e1_divergence.csv, results/e1_summary.json, results/e1_divergence.png
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS, benchmark_suite, count_ops, load_targets, timed, transpile_best

warnings.filterwarnings("ignore")


def run():
    from circuit_analysis import circuit_stats, estimate_divergence
    from cost_fidelity_model import SWAP_COST_2Q, topology_divergence

    targets = load_targets()
    benches = benchmark_suite()
    rows = []
    for b in benches:
        st = circuit_stats(b.circuit)
        qc = b.circuit.copy()
        qc.measure_all()
        for t in targets:
            if b.num_qubits > t.telemetry.qubit_count:
                continue
            tel = t.telemetry
            est, t_est = timed(estimate_divergence, st, tel.coupling_map, tel.qubit_count,
                               tel.native_gates, routing="dynamic")
            table_added = SWAP_COST_2Q * topology_divergence("all-to-all", t.topology) \
                * st.num_two_qubit_gates
            out, t_tr = transpile_best(qc, t.backend)
            _, n2_out, depth_out = count_ops(out)
            rows.append(dict(
                circuit=b.name, family=b.family, n=b.num_qubits, target=t.name,
                topology=t.topology, n2_logical=st.num_two_qubit_gates,
                est_swaps=est.swaps, est_added_2q=est.added_two_qubit_gates,
                static_added_2q=(0 if t.topology == "all-to-all" else 3 * est.static_swaps),
                exact_embedding=est.exact_embedding, table_added_2q=table_added,
                actual_n2=n2_out, actual_added_2q=n2_out - st.num_two_qubit_gates,
                actual_depth=depth_out,
                est_ms=t_est * 1e3, transpile_ms_per_seed=t_tr * 1e3 / 3,
            ))
        print(f"  {b.name:<24} done")
    return pd.DataFrame(rows)


def loo_calibration(df, col="est_added_2q"):
    """Leave-one-family-out fit of actual_added ≈ α·est_added (sparse targets)."""
    pred = pd.Series(np.nan, index=df.index)
    for fam in df.family.unique():
        train = df[(df.family != fam) & (df[col] > 0)]
        alpha = (train[col] * train.actual_added_2q).sum() / max((train[col] ** 2).sum(), 1e-9)
        m = df.family == fam
        pred[m] = df.n2_logical[m] + alpha * df[col][m]
    return pred


def summarize(df):
    df = df.copy()
    df["pred_static"] = df.n2_logical + df.static_added_2q
    df["pred_graph"] = df.n2_logical + df.est_added_2q
    df["pred_table"] = df.n2_logical + df.table_added_2q
    sparse = df.topology != "all-to-all"
    for name, col, base in (("pred_calibrated", "est_added_2q", "pred_graph"),
                            ("pred_static_calibrated", "static_added_2q", "pred_static")):
        df[name] = np.nan
        df.loc[sparse, name] = loo_calibration(df[sparse], col)
        df.loc[~sparse, name] = df.loc[~sparse, base]
    df["pred_none"] = df.n2_logical          # "raw": ignore transpilation overhead

    out = {}
    for scope, d in [("all", df)] + [(t, df[df.topology == t]) for t in df.topology.unique()]:
        res = {"n": int(len(d))}
        for col in ("pred_none", "pred_table", "pred_static", "pred_static_calibrated",
                    "pred_graph", "pred_calibrated"):
            rho = spearmanr(d[col], d.actual_n2).correlation if d[col].nunique() > 1 else float("nan")
            res[col] = {"spearman": float(rho),
                        "mae_2q": float((d[col] - d.actual_n2).abs().mean()),
                        "median_rel_err": float(((d[col] - d.actual_n2).abs()
                                                 / d.actual_n2.clip(lower=1)).median())}
        res["est_ms_median"] = float(d.est_ms.median())
        res["transpile_ms_median"] = float(d.transpile_ms_per_seed.median())
        out[scope] = res
    return df, out


def plot(df):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.ravel()
    colors = {"heavy-hex": "#4a9eed", "square-lattice": "#ef4444", "all-to-all": "#22c55e"}
    for ax, col, title in [(axes[0], "pred_table", "Family lookup table"),
                           (axes[1], "pred_static", "Graph estimate, static placement"),
                           (axes[2], "pred_graph", "Graph estimate, routing replay"),
                           (axes[3], "pred_calibrated", "Routing replay, calibrated (LOFO)")]:
        for topo, d in df.groupby("topology"):
            ax.scatter(d.actual_n2, d[col], s=14, alpha=0.7, color=colors.get(topo), label=topo)
        lim = max(df.actual_n2.max(), df[col].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xscale("symlog"); ax.set_yscale("symlog")
        ax.set_xlabel("Actual two-qubit gates after transpilation")
        ax.set_ylabel("Predicted two-qubit gates")
        rho = spearmanr(df[col], df.actual_n2).correlation
        ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS / "e1_divergence.png", dpi=200)
    plt.savefig(RESULTS / "e1_divergence.pdf")


if __name__ == "__main__":
    df = run()
    df, summary = summarize(df)
    df.to_csv(RESULTS / "e1_divergence.csv", index=False)
    (RESULTS / "e1_summary.json").write_text(json.dumps(summary, indent=2))
    plot(df)
    print(json.dumps(summary["all"], indent=2))
