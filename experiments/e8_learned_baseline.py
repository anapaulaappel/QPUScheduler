"""
E8 — Does a LEARNED predictor beat the training-free estimate, and does it
generalize to a device it was not trained on?

Learned baseline in the spirit of Fulginiti et al. (arXiv:2609.09980; GNN on
pre-compilation circuit features -> post-compilation fidelity per QPU) and
MQT Predictor (arXiv:2310.06889), implemented with scikit-learn on tabular
features (no GNN, no external code):

  circuit features (pre-transpilation, mirror circuit): n, 1Q/2Q counts, depth,
      interaction-graph stats (edges, density, degree, diameter, max weight)
  device features: mean 1Q/2Q/readout error, T2, qubit count, coupling-graph
      stats (mean/max degree, diameter, mean distance), all-to-all flag, native 2Q

Predictors compared on exactly the same test rows:
  esp_estimated   ours (training-free, no transpilation)
  esp_actual      ESP of the actually transpiled circuit (needs transpilation)
  ours_cal        ours with a 2-parameter log-linear recalibration fitted on train
  hgb             HistGradientBoosting on the success probability
  hgb_log         HistGradientBoosting on log(success)
  mlp_log         MLP (64,64) on log(success)
  hgb_log+ours    hgb_log with log(esp_estimated) as an extra feature
  ours*corr       ours times a learned correction: HGB on log(y / esp_estimated)

Evaluations (out-of-fold predictions; metrics as in e6_summary_large.json):
  sim_random, sim_lodo, sim_lodo_strict, sim_lotopo     simulated labels (E2 large)
  hw_random, hw_lodo, hw_lodo_strict                    IBM hardware (E6 large)
  sim_to_hw                                             train sim, test IBM hardware
  e7_from_{sim,hw,sim+hw}                               test IQM / Rigetti / AQT hardware

"strict" = the held-out device AND the held-out circuit fold are both unseen.

Run:  python e8_learned_baseline.py [--boot 500] [--seed 0]
Out:  results/e8_predictions.csv, results/e8_results.csv,
      results/e8_summary.json, results/e8_learned_baseline.png
"""

import argparse
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS, benchmark_suite
from e2_fidelity_prediction import mirror

warnings.filterwarnings("ignore")
E7 = RESULTS / "e7"
FLOOR = 1e-3            # success floor for log targets (shots 2000 -> 1/shots = 5e-4)
SEED = 0

CIRC_FEATS = ["n", "n1", "n2", "depth", "ig_edges", "ig_density", "ig_max_deg",
              "ig_mean_deg", "ig_diameter", "ig_max_w", "n2_per_qubit", "frac_2q"]
DEV_FEATS = ["e1", "e2", "ero", "t2_us", "d_qubits", "d_mean_deg", "d_max_deg",
             "d_diameter", "d_mean_dist", "d_all2all", "nat_cz", "nat_ecr", "nat_ms"]
FEATS = CIRC_FEATS + DEV_FEATS
LEARNED = ["ours_cal", "hgb", "hgb_log", "mlp_log", "hgb_log+ours", "ours*corr"]
BASE = ["esp_estimated", "esp_actual"]


# ─────────────────────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────────────────────

def _graph_stats(n_nodes, edges):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edges)
    deg = np.array([d for _, d in g.degree()]) if n_nodes else np.array([0])
    if nx.is_connected(g):
        diam = nx.diameter(g)
        mdist = nx.average_shortest_path_length(g)
    else:   # use largest component
        h = g.subgraph(max(nx.connected_components(g), key=len))
        diam = nx.diameter(h) if h.number_of_nodes() > 1 else 0
        mdist = nx.average_shortest_path_length(h) if h.number_of_nodes() > 1 else 0
    return deg, diam, mdist, g.number_of_edges()


def circuit_features(sizes, instances):
    """Pre-transpilation features of the mirror circuits, rebuilt exactly as in E2/E6/E7."""
    from circuit_analysis import circuit_stats
    rows = []
    for i, b in enumerate(benchmark_suite(sizes=sizes, instances=instances)):
        circ, _ = mirror(b.circuit, seed=100 + i)
        st = circuit_stats(circ)
        nq = st.num_qubits
        deg, diam, _, ne = _graph_stats(nq, list(st.interaction))
        rows.append(dict(
            circuit=b.name, n=b.num_qubits, n1=st.num_single_qubit_gates,
            n2=st.num_two_qubit_gates, depth=st.depth, ig_edges=ne,
            ig_density=ne / max(nq * (nq - 1) / 2, 1), ig_max_deg=int(deg.max()),
            ig_mean_deg=float(deg.mean()), ig_diameter=diam,
            ig_max_w=max(st.interaction.values()) if st.interaction else 0,
            n2_per_qubit=st.num_two_qubit_gates / nq,
            frac_2q=st.num_two_qubit_gates / max(st.num_two_qubit_gates + st.num_single_qubit_gates, 1)))
    return pd.DataFrame(rows).set_index("circuit")


def topo_features(n_qubits, edges, native):
    edges = [tuple(sorted(e)) for e in edges]
    all2all = not edges or len(set(edges)) == n_qubits * (n_qubits - 1) // 2
    if not edges:
        edges = [(a, b) for a in range(n_qubits) for b in range(a + 1, n_qubits)]
    deg, diam, mdist, _ = _graph_stats(n_qubits, set(edges))
    native = {g.lower() for g in native}
    return dict(d_qubits=n_qubits, d_mean_deg=float(deg.mean()), d_max_deg=int(deg.max()),
                d_diameter=diam, d_mean_dist=mdist, d_all2all=int(all2all),
                nat_cz=int("cz" in native), nat_ecr=int("ecr" in native),
                nat_ms=int(bool(native & {"rxx", "ms", "xx"})))


def _backend_topo(backend):
    t = backend.target
    edges = t.build_coupling_map().get_edges()
    return topo_features(backend.num_qubits, edges, t.operation_names)


def sim_dataset():
    """E2 large dataset (8 targets, noisy-simulation labels)."""
    from common import load_targets
    df = pd.read_csv(RESULTS / "e2_dataset_large.csv").rename(columns={"success_probability": "y"})
    cf = circuit_features((4, 6, 8, 10, 12, 14), 3)
    assert (cf.loc[df.circuit, "n2"].values == df.n2.values).all(), "circuit features mismatch"
    topo = {t.name: _backend_topo(t.backend) for t in load_targets()}
    df = df.drop(columns=["n", "n1", "n2", "depth"]).join(cf, on="circuit")
    df = df.join(pd.DataFrame(topo).T.astype(float), on="target")
    df["dataset"] = "sim"
    return df


def _ibm_calibration(name, when):
    """Device means from the locally collected history: latest calibration <= `when`."""
    h = pd.read_csv(RESULTS.parent.parent / "data" / "real_history" / f"{name}.csv")
    h["cal"] = pd.to_datetime(h.calibration_timestamp, utc=True, format="ISO8601")
    h["ts"] = pd.to_datetime(h.timestamp, utc=True, format="ISO8601")
    h = h[(h.cal <= when) & (h.ts >= when - pd.Timedelta("3h"))]
    r = h.sort_values(["cal", "ts"]).iloc[-1]
    return dict(e2=r.mean_tq_error, ero=r.mean_ro_error, t2_us=r.mean_t2_us,
                hist_qubit_fidelity=r.mean_qubit_fidelity, hist_cal=str(r.calibration_timestamp))


def hw_dataset():
    """E6 large IBM hardware (3 Heron r2 devices, measured success)."""
    from qiskit_ibm_runtime import fake_provider as fp
    df = pd.read_csv(RESULTS / "e6_hardware_large.csv").rename(columns={"measured": "y"})
    jobs = json.loads((RESULTS / "e6_jobs_large.json").read_text())
    cf = circuit_features((4, 6, 8, 10, 12, 14), 3)
    assert (cf.loc[df.circuit, "n2"].values == df.n2_logical.values).all()
    fake = {"ibm_fez": fp.FakeFez, "ibm_marrakesh": fp.FakeMarrakesh, "ibm_kingston": fp.FakeKingston}
    dev, cal_info = {}, {}
    for name in df.target.unique():
        when = pd.Timestamp(jobs[name]["submitted"])
        cal = _ibm_calibration(name, when)
        e1 = 1 - df.loc[df.target == name, "raw_fidelity"].iloc[0]      # exact value used by E6
        cal_info[name] = {**cal, "e1_from_e6": e1, "submitted": str(when)}
        dev[name] = {"e1": e1, "e2": cal["e2"], "ero": cal["ero"], "t2_us": cal["t2_us"],
                     **_backend_topo(fake[name]())}
    df = df.drop(columns=["n"]).join(cf, on="circuit").join(pd.DataFrame(dev).T.astype(float), on="target")
    df["dataset"] = "hw_ibm"
    return df, cal_info


def e7_dataset():
    """E7 non-IBM hardware rows (IQM Garnet 28, Rigetti Cepheus 12, AQT IBEX Q1 6)."""
    from e7_analyze import device_errors
    df = pd.read_csv(E7 / "e7_hardware_with_ibm.csv").rename(columns={"measured": "y"})
    df = df[df.vendor != "ibm"].copy()
    cf = circuit_features((4, 6, 8, 10), 1)
    dev = {}
    for name in df.target.unique():
        d = json.loads((E7 / f"device_{name}.json").read_text())
        err = device_errors(name, d)
        dev[name] = {"e1": err["e1"], "e2": err["e2"], "ero": err["ero"],
                     "t2_us": err["t2_us"] or np.nan,     # 0 = not published
                     **topo_features(d["n_qubits"], d.get("edges") or [], d["native_tag"])}
    df = df.drop(columns=["n"]).join(cf, on="circuit").join(pd.DataFrame(dev).T.astype(float), on="target")
    df["dataset"] = "hw_e7"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def _hgb():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=10, random_state=SEED)


def _mlp():
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=3000,
                                      early_stopping=True, random_state=SEED))


def _logy(y):
    return np.log(np.clip(y, FLOOR, 1.0))


def _logesp(d):
    return np.log(np.clip(d.esp_estimated.values, 1e-6, 1.0))


def fit_predict(tr, te):
    """All learned predictors, trained on `tr`, predicted on `te` (probabilities)."""
    X_tr, X_te = tr[FEATS].astype(float), te[FEATS].astype(float)
    ly = _logy(tr.y.values)
    out = {}
    a = np.polyfit(_logesp(tr), ly, 1)                                   # ours_cal
    out["ours_cal"] = np.exp(np.polyval(a, _logesp(te)))
    out["hgb"] = _hgb().fit(X_tr, tr.y.values).predict(X_te)
    out["hgb_log"] = np.exp(_hgb().fit(X_tr, ly).predict(X_te))
    out["mlp_log"] = np.exp(_mlp().fit(X_tr, ly).predict(X_te))
    Xo_tr, Xo_te = X_tr.assign(log_esp=_logesp(tr)), X_te.assign(log_esp=_logesp(te))
    out["hgb_log+ours"] = np.exp(_hgb().fit(Xo_tr, ly).predict(Xo_te))
    corr = _hgb().fit(Xo_tr, ly - _logesp(tr)).predict(Xo_te)
    out["ours*corr"] = te.esp_estimated.values * np.exp(corr)
    return {k: np.clip(v, 0.0, 1.0) for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Splits and metrics
# ─────────────────────────────────────────────────────────────────────────────

def circuit_folds(df, k=5):
    from sklearn.model_selection import GroupKFold
    fold = pd.Series(-1, index=df.index)
    for i, (_, te) in enumerate(GroupKFold(k, shuffle=True, random_state=SEED)
                                .split(df, groups=df.circuit)):
        fold.iloc[te] = i
    return fold


def splits_random(df):
    f = circuit_folds(df)
    return [(f != i, f == i, f"fold{i}") for i in sorted(f.unique())]


def splits_by(col):
    def s(df):
        return [(df[col] != v, df[col] == v, v) for v in sorted(df[col].unique())]
    return s


def splits_lodo_strict(df):
    f = circuit_folds(df)
    return [((df.target != d) & (f != i), (df.target == d) & (f == i), d, f == i)
            for d in sorted(df.target.unique()) for i in sorted(f.unique())]


def run_cv(df, split_fn, deploy=None):
    """Out-of-fold predictions. If a split carries a 4th mask (all devices of the
    held-out circuit fold), the same model also scores those rows and the device
    choice is evaluated there ("deployment": device d is new, the circuits are new
    on every device, the other devices were seen in training); results are appended
    to `deploy` as (heldout device, frame)."""
    parts = []
    for sp in split_fn(df):
        tr, te, grp = sp[:3]
        sel = sp[3] if len(sp) > 3 else te
        tr, te_all = df[tr.values], df[sel.values]
        if df[te.values].empty:
            continue
        p = te_all[["circuit", "family", "n", "target", "y"] + BASE].copy()
        p["heldout"] = grp
        for k, v in fit_predict(tr, te_all).items():
            p[k] = v
        if deploy is not None and len(sp) > 3:
            deploy.append((grp, p))
        parts.append(p[te[sel].values])
    return pd.concat(parts, ignore_index=True)


def deploy_selection(deploy, cols):
    """Mean regret / top-1 of the device choice, averaged over (new device, circuit fold)."""
    out = {}
    for c in cols:
        reg, hit = [], []
        for _, p in deploy:
            for _, g in p.groupby("circuit"):
                pick, best = g.loc[g[c].idxmax()], g.loc[g.y.idxmax()]
                reg.append(best.y - pick.y)
                hit.append(pick.target == best.target)
        out[c] = {"mean_regret": float(np.mean(reg)), "top1_accuracy": float(np.mean(hit)),
                  "n_decisions": len(reg)}
    return out


def run_transfer(train, test):
    p = test[["circuit", "family", "n", "target", "y"] + BASE].copy()
    p["heldout"] = "transfer"
    for k, v in fit_predict(train, test).items():
        p[k] = v
    return p


def _selection(p, col):
    reg, hit = [], []
    for _, g in p.groupby("circuit"):
        if len(g) < 2:
            continue
        pick, best = g.loc[g[col].idxmax()], g.loc[g.y.idxmax()]
        reg.append(best.y - pick.y)
        hit.append(pick.target == best.target)
    return (float(np.mean(reg)), float(np.mean(hit)), len(reg)) if reg else (None, None, 0)


def metrics(p, cols, boot=0, rng=None):
    res = {}
    circuits = p.circuit.unique()
    idx = {c: np.where(p.circuit.values == c)[0] for c in circuits}
    samples = [np.concatenate([idx[c] for c in rng.choice(circuits, len(circuits))])
               for _ in range(boot)] if boot else []
    for c in cols:
        reg, top1, nsel = _selection(p, c)
        r = {"spearman": float(spearmanr(p[c], p.y).correlation),
             "mae": float((p[c] - p.y).abs().mean()),
             "mean_regret": reg, "top1_accuracy": top1, "n_selection_circuits": nsel}
        if samples:
            x, y = p[c].values, p.y.values
            rho = [spearmanr(x[s], y[s]).correlation for s in samples]
            mae = [np.abs(x[s] - y[s]).mean() for s in samples]
            r["spearman_ci95"] = [float(np.nanpercentile(rho, 2.5)), float(np.nanpercentile(rho, 97.5))]
            r["mae_ci95"] = [float(np.percentile(mae, 2.5)), float(np.percentile(mae, 97.5))]
        res[c] = r
    return res


def paired_delta(p, a, b, boot, rng):
    """Bootstrap (by circuit) of MAE(a) - MAE(b) and rho(a) - rho(b)."""
    circuits = p.circuit.unique()
    idx = {c: np.where(p.circuit.values == c)[0] for c in circuits}
    y, xa, xb = p.y.values, p[a].values, p[b].values
    dm, dr = [], []
    for _ in range(boot):
        s = np.concatenate([idx[c] for c in rng.choice(circuits, len(circuits))])
        dm.append(np.abs(xa[s] - y[s]).mean() - np.abs(xb[s] - y[s]).mean())
        dr.append(spearmanr(xa[s], y[s]).correlation - spearmanr(xb[s], y[s]).correlation)
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    return {"mae_diff_ci95": q(dm), "spearman_diff_ci95": q(dr)}


# ─────────────────────────────────────────────────────────────────────────────

def plot(results, path):
    import matplotlib.pyplot as plt
    evals = [e for e in results if not e.startswith("e7_")] + [e for e in results if e.startswith("e7_")]
    show = [("esp_estimated", "Ours (training-free)", "#2a78d6"),
            ("esp_actual", "ESP of transpiled circuit", "#1baf7a"),
            ("hgb_log", "Learned (HGB, log target)", "#eb6834"),
            ("ours*corr", "Ours × learned correction", "#eda100")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    y = np.arange(len(evals))
    h = 0.8 / len(show)
    for ax, metric, title in [(axes[0], "spearman", "Spearman ρ vs label (higher is better)"),
                              (axes[1], "mae", "MAE vs label (lower is better)")]:
        for j, (col, lab, colr) in enumerate(show):
            v = [results[e]["overall"][col][metric] for e in evals]
            ax.barh(y + (j - 1.5) * h, v, height=h * 0.9, color=colr, label=lab)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_xlim(0, 1)
    axes[0].set_yticks(y, [f"{e} (n={results[e]['n_test_rows']})" for e in evals], fontsize=8)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False,
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("E8: learned predictors vs training-free estimate (same test rows)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.91))
    plt.savefig(path, dpi=200)


def main():
    global SEED
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples (by circuit)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    SEED = args.seed
    rng = np.random.default_rng(SEED)

    sim = sim_dataset()
    hw, cal_info = hw_dataset()
    e7 = e7_dataset()
    print(f"sim {len(sim)} rows / {sim.target.nunique()} devices; IBM hw {len(hw)}; "
          f"E7 {len(e7)} ({e7.groupby('target').size().to_dict()})", flush=True)

    deploys = {}
    plans = {
        "sim_random": lambda: run_cv(sim, splits_random),
        "sim_lodo": lambda: run_cv(sim, splits_by("target")),
        "sim_lodo_strict": lambda: run_cv(sim, splits_lodo_strict, deploys.setdefault("sim_lodo_strict", [])),
        "sim_lotopo": lambda: run_cv(sim, splits_by("topology")),
        "hw_random": lambda: run_cv(hw, splits_random),
        "hw_lodo": lambda: run_cv(hw, splits_by("target")),
        "hw_lodo_strict": lambda: run_cv(hw, splits_lodo_strict, deploys.setdefault("hw_lodo_strict", [])),
        "sim_to_hw": lambda: run_transfer(sim, hw),
        "e7_from_sim": lambda: run_transfer(sim, e7),
        "e7_from_hw": lambda: run_transfer(hw, e7),
        "e7_from_sim+hw": lambda: run_transfer(pd.concat([sim, hw], ignore_index=True), e7),
    }
    cols = BASE + LEARNED
    results, preds, rows = {}, [], []
    for name, fn in plans.items():
        p = fn()
        p.insert(0, "evaluation", name)
        preds.append(p)
        r = {"n_test_rows": int(len(p)), "n_circuits": int(p.circuit.nunique()),
             "devices": sorted(p.target.unique()),
             "overall": metrics(p, cols, args.boot, rng)}
        # chosen on the test rows: optimistic for the learned side, informational only
        r["best_learned_by_mae_on_test"] = min(LEARNED[1:], key=lambda c: r["overall"][c]["mae"])
        r["paired_minus_ours"] = {c: paired_delta(p, c, "esp_estimated", args.boot, rng)
                                  for c in ["esp_actual"] + LEARNED}
        if name in deploys:
            r["deployment_selection"] = deploy_selection(deploys[name], cols)
        if name.startswith(("sim_lodo", "sim_lotopo", "hw_lodo", "e7_")):
            r["per_device"] = {d: {**metrics(g, cols), "n": int(len(g))}
                               for d, g in p.groupby("target")}
        results[name] = r
        for c in cols:
            m = r["overall"][c]
            rows.append({"evaluation": name, "predictor": c, "n": len(p), **{
                k: m[k] for k in ("spearman", "mae", "mean_regret", "top1_accuracy",
                                  "n_selection_circuits")},
                "spearman_lo": m["spearman_ci95"][0] if "spearman_ci95" in m else None,
                "spearman_hi": m["spearman_ci95"][1] if "spearman_ci95" in m else None,
                "mae_lo": m["mae_ci95"][0] if "mae_ci95" in m else None,
                "mae_hi": m["mae_ci95"][1] if "mae_ci95" in m else None,
                "deploy_regret": r.get("deployment_selection", {}).get(c, {}).get("mean_regret"),
                "deploy_top1": r.get("deployment_selection", {}).get(c, {}).get("top1_accuracy")})
        o = r["overall"]
        print(f"{name:<16} n={len(p):4d}  ours rho={o['esp_estimated']['spearman']:.3f} "
              f"mae={o['esp_estimated']['mae']:.3f} | hgb_log rho={o['hgb_log']['spearman']:.3f} "
              f"mae={o['hgb_log']['mae']:.3f} | ours*corr rho={o['ours*corr']['spearman']:.3f} "
              f"mae={o['ours*corr']['mae']:.3f}", flush=True)

    pd.concat(preds, ignore_index=True).to_csv(RESULTS / "e8_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "e8_results.csv", index=False)
    summary = {"created": datetime.now(timezone.utc).isoformat(), "seed": SEED,
               "bootstrap": args.boot, "log_floor": FLOOR, "features": FEATS,
               "ibm_calibration_used": cal_info, "evaluations": results}
    (RESULTS / "e8_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot(results, RESULTS / "e8_learned_baseline.png")


if __name__ == "__main__":
    main()
