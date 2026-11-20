"""
E7 — multi-vendor hardware validation: analysis (main venv, no QPU).

Combines results/e7/run_<device>.json with the manifest, computes ESP from the topology
estimate and from the actually transpiled circuit, and compares both with the
measured success probability of the mirror circuits. Because the circuits are the
same as in the 28-circuit E6 run (same mirror seeds), the IBM Heron measurements
from results/e6_hardware.csv are added for a cross-vendor target-selection test.

Error rates: IQM from IQM Resonance calibration metrics (device JSON); other devices
from results/e7/errors_<device>.json if present (e.g. calibration downloaded with the
job), else vendor-published figures listed below (marked as such in the summary).

Out: results/e7/e7_hardware.csv, results/e7/e7_summary.json, results/e7/e7_hardware.png
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS, esp

E7 = RESULTS / "e7"
# Devices whose results are kept on disk but excluded from the analysis.
# Rigetti (via Open Quantum): the provider compiler optimizes across the barrier of the
# mirror circuits (e.g. real_amplitudes_8 with 256 CZ measured 0.74 success), so the
# executed circuit is not the submitted one. Needs a verbatim-box submission.
EXCLUDE = {"rigetti_cepheus": "provider compiler simplified the mirror circuits (C·C† cancelled across the barrier)"}
PUBLISHED = {   # fallback error rates (source noted in summary)
    "aqt_ibex_q1": dict(e1=3e-4, e2=1.15e-2, ero=5e-3, t2_us=0, source="AQT published IBEX Q1 figures "
                        "(1Q 99.97%, 2Q 98.85%); readout assumed 0.5%"),
    "rigetti_cepheus": dict(e1=1e-3, e2=1.5e-2, ero=3e-2, t2_us=0, source="assumed typical values; "
                            "replace with job calibration"),
}


def device_errors(name, dev):
    p = E7 / f"errors_{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    if "e1" in dev:
        return {k: dev[k] for k in ("e1", "e2", "ero", "t2_us")} | {"source": dev.get("source", "")}
    return PUBLISHED[name]


def load_runs():
    man = json.loads((E7 / "manifest.json").read_text())
    rows, sources = [], {}
    for name, m in man["devices"].items():
        if name in EXCLUDE:
            continue
        run_p = E7 / f"run_{name}.json"
        if not run_p.exists():
            continue
        run = json.loads(run_p.read_text())
        run = run.get("rows", run)
        dev = json.loads((E7 / f"device_{name}.json").read_text())
        err = device_errors(name, dev)
        sources[name] = err.get("source", "")
        for r in m["rows"]:
            x = run.get(r["circuit"])
            if not x or not x.get("counts"):
                continue
            shots = sum(x["counts"].values())
            meas = x["counts"].get(r["key"], 0) / shots
            n = r["n"]
            rows.append(dict(
                circuit=r["circuit"], family=r["family"], n=n, target=name, vendor=dev["provider"],
                measured=meas, shots=shots,
                esp_estimated=esp(r["est_n1"], r["est_n2"], n, err["e1"], err["e2"], err["ero"]),
                esp_actual=esp(x["actual_n1"], x["actual_n2"], n, err["e1"], err["e2"], err["ero"]),
                est_swaps=r["est_swaps"], actual_n2=x["actual_n2"], est_n2=r["est_n2"]))

    # Verbatim submissions (raw provider bitstrings: choose the bit order from the data,
    # using the circuits whose ideal output is not a palindrome).
    for name, fname in (("rigetti_cepheus", "run_rigetti_verbatim.json"),
                        ("aqt_ibex_q1", "run_aqt_verbatim.json")):
        vp = E7 / fname
        if not vp.exists():
            continue
        run = {k: v for k, v in json.loads(vp.read_text()).items() if v.get("raw_counts")}
        dev = json.loads((E7 / f"device_{name}.json").read_text())
        err = device_errors(name, dev)
        sources[name] = err.get("source", "") + " | verbatim submission"
        mrows = {r["circuit"]: r for r in man["devices"][name]["rows"]}

        def succ(x, key):
            return x["raw_counts"].get(key, 0) / sum(x["raw_counts"].values())
        nonpal = [c for c in run if mrows[c]["key"] != mrows[c]["key"][::-1]]
        as_is = sum(succ(run[c], mrows[c]["key"]) for c in nonpal)
        rev = sum(succ(run[c], mrows[c]["key"][::-1]) for c in nonpal)
        order = "reversed" if rev > as_is else "as_is"
        sources[name] += f" | bit order {order} (sum over {len(nonpal)} circuits: as_is {as_is:.2f}, reversed {rev:.2f})"
        for c, x in run.items():
            r = mrows[c]
            key = r["key"][::-1] if order == "reversed" else r["key"]
            n = r["n"]
            rows.append(dict(
                circuit=c, family=r["family"], n=n, target=name, vendor=dev["provider"],
                measured=succ(x, key), shots=sum(x["raw_counts"].values()),
                esp_estimated=esp(r["est_n1"], r["est_n2"], n, err["e1"], err["e2"], err["ero"]),
                esp_actual=esp(x["actual_n1"], x["actual_n2"], n, err["e1"], err["e2"], err["ero"]),
                est_swaps=r["est_swaps"], actual_n2=x["actual_n2"], est_n2=r["est_n2"]))
    return pd.DataFrame(rows), sources


def stats(df, cols=("esp_estimated", "esp_actual")):
    out = {"n": int(len(df))}
    for c in cols:
        out[c] = {"spearman": float(spearmanr(df[c], df.measured).correlation) if len(df) > 2 else None,
                  "mae": float((df[c] - df.measured).abs().mean())}
    return out


def selection(df, col):
    reg, hit = [], []
    for _, g in df.groupby("circuit"):
        if g.target.nunique() < 2:
            continue
        pick, best = g.loc[g[col].idxmax()], g.loc[g.measured.idxmax()]
        reg.append(best.measured - pick.measured)
        hit.append(pick.target == best.target)
    return {"circuits": len(reg), "mean_regret": float(np.mean(reg)) if reg else None,
            "top1": float(np.mean(hit)) if hit else None}


def main():
    df, sources = load_runs()
    summary = {"error_sources": sources, "excluded": EXCLUDE, "per_device": {}, "pooled_non_ibm": stats(df)}
    for t, g in df.groupby("target"):
        summary["per_device"][t] = stats(g) | {
            "mean_measured": float(g.measured.mean()),
            "mean_est_swaps": float(g.est_swaps.mean()),
            "est_vs_actual_n2_spearman": float(spearmanr(g.est_n2, g.actual_n2).correlation) if len(g) > 2 else None}

    # Cross-vendor selection with the IBM Heron measurements of the same circuits (E6).
    ibm_p = RESULTS / "e6_hardware.csv"
    if ibm_p.exists():
        ibm = pd.read_csv(ibm_p)
        ibm = ibm[ibm.circuit.isin(df.circuit)][["circuit", "family", "n", "target", "measured",
                                                 "esp_estimated", "esp_actual", "raw_fidelity"]]
        ibm["vendor"] = "ibm"
        both = pd.concat([df.assign(raw_fidelity=np.nan), ibm], ignore_index=True)
        summary["cross_vendor"] = {
            "devices": sorted(both.target.unique().tolist()),
            "pooled": stats(both),
            "selection_esp_estimated": selection(both, "esp_estimated"),
            "selection_esp_actual": selection(both, "esp_actual"),
            "always_device_regret": {t: float(np.mean([g.measured.max() - g[g.target == t].measured.iloc[0]
                                                       for _, g in both.groupby("circuit")
                                                       if (g.target == t).any() and g.target.nunique() > 1]))
                                     for t in both.target.unique()},
        }
        both.to_csv(E7 / "e7_hardware_with_ibm.csv", index=False)
    df.to_csv(E7 / "e7_hardware.csv", index=False)
    (E7 / "e7_summary.json").write_text(json.dumps(summary, indent=2))
    plot(df)
    print(json.dumps(summary, indent=2))


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in [(axes[0], "esp_estimated", "ESP, topology estimate (no transpilation)"),
                           (axes[1], "esp_actual", "ESP, actual transpilation")]:
        for t, d in df.groupby("target"):
            ax.scatter(d[col], d.measured, s=18, alpha=0.8, label=t)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        rho = spearmanr(df[col], df.measured).correlation if len(df) > 2 else float("nan")
        ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}", fontsize=10)
        ax.set_xlabel("Predicted success")
        ax.set_ylabel("Measured success on hardware")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(E7 / "e7_hardware.png", dpi=200)


if __name__ == "__main__":
    main()
