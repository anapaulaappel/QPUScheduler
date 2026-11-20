"""
E10 — analysis of application circuits on hardware (IBM ×3, Rigetti, AQT; IQM if run).

Quality = normalized Hellinger fidelity to the ideal distribution (Lubinski et al.):
    F = (sum_x sqrt(p_ideal(x) p_meas(x)))^2,   F_norm = max(0, (F - F_unif) / (1 - F_unif)).
Predictors: ESP from the estimate without transpiling (ours) and ESP of the transpiled circuit,
with the same device error rates as E7 for non-IBM devices and the published IBM calibration.
ESP is the probability of an error-free run; under a depolarizing picture F_norm ≈ ESP, so we
report Spearman ρ, MAE and device-selection regret/top-1, as for the mirror circuits.

Raw provider bitstrings (Open Quantum) may be in either bit order; the order is chosen from the
data (sum of F over circuits with asymmetric ideal distributions), as in E7, and reported.

Out: results/e10/e10_hardware.csv, results/e10/e10_summary.json, results/e10/e10_hardware.png
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS, esp
from e7_analyze import device_errors, selection

E10 = RESULTS / "e10"
E7 = RESULTS / "e7"


def hellinger(ideal: dict, counts: dict) -> float:
    tot = sum(counts.values())
    return float(sum(np.sqrt(p * counts.get(k, 0) / tot) for k, p in ideal.items()) ** 2)


def f_norm(ideal: dict, counts: dict, n: int) -> float:
    f = hellinger(ideal, counts)
    f_u = float(sum(np.sqrt(p / 2 ** n) for p in ideal.values()) ** 2)
    return max(0.0, (f - f_u) / (1 - f_u))


def normalize_counts(counts: dict, n: int, reverse: bool) -> dict:
    out = {}
    for k, v in counts.items():
        k = k.replace(" ", "")[-n:] if len(k.replace(" ", "")) >= n else k.zfill(n)
        k = k[::-1] if reverse else k
        out[k] = out.get(k, 0) + v
    return out


def load():
    man = json.loads((E10 / "manifest.json").read_text())
    ideal = {b["name"]: json.loads((E10 / "ideal" / f"{b['name']}.json").read_text()) for b in man["all"]}
    meta = {b["name"]: b for b in man["all"]}
    rows, notes = [], {}
    # IBM (Qiskit bit order)
    for p in sorted(E10.glob("run_ibm_*.json")):
        run = json.loads(p.read_text())["rows"]
        for c, r in run.items():
            n = r["n"]
            cnt = normalize_counts(r["raw_counts"], n, False)
            rows.append(dict(circuit=c, family=r["family"], n=n, target=r["target"], vendor="ibm",
                             shots=r["shots"], measured=f_norm(ideal[c], cnt, n),
                             hellinger=hellinger(ideal[c], cnt), raw_fidelity=r["raw_fidelity"],
                             esp_estimated=r["esp_estimated"], esp_actual=r["esp_actual"],
                             actual_n2=r["actual_n2"]))
        notes[p.stem[4:]] = f"published calibration {next(iter(run.values()))['published_calibration']}"
    # Open Quantum devices
    for dev_name in ("rigetti_cepheus", "aqt_ibex_q1"):
        p = E10 / f"run_{dev_name}.json"
        if not p.exists():
            continue
        run = {c: x for c, x in json.loads(p.read_text()).items() if x.get("raw_counts")}
        dev = json.loads((E7 / f"device_{dev_name}.json").read_text())
        err = device_errors(dev_name, dev)
        mrows = {r["circuit"]: r for r in man["devices"][dev_name]["rows"]}
        asym = [c for c in run if ideal[c] != {k[::-1]: v for k, v in ideal[c].items()}]
        score = {rev: sum(hellinger(ideal[c], normalize_counts(run[c]["raw_counts"], meta[c]["n"], rev))
                          for c in asym) for rev in (False, True)}
        rev = score[True] > score[False]
        notes[dev_name] = (f"{err.get('source', '')} | bit order {'reversed' if rev else 'as_is'} "
                           f"(sum F over {len(asym)} asymmetric circuits: as_is {score[False]:.2f}, "
                           f"reversed {score[True]:.2f})")
        for c, x in run.items():
            r, n = mrows[c], meta[c]["n"]
            cnt = normalize_counts(x["raw_counts"], n, rev)
            rows.append(dict(circuit=c, family=r["family"], n=n, target=dev_name, vendor=dev["provider"],
                             shots=x["shots"], measured=f_norm(ideal[c], cnt, n),
                             hellinger=hellinger(ideal[c], cnt), raw_fidelity=np.nan,
                             esp_estimated=esp(r["est_n1"], r["est_n2"], n, err["e1"], err["e2"], err["ero"]),
                             esp_actual=esp(x["actual_n1"], x["actual_n2"], n, err["e1"], err["e2"], err["ero"]),
                             actual_n2=x["actual_n2"]))
    return pd.DataFrame(rows), notes


def stats(df):
    out = {"n": int(len(df))}
    for c in ("esp_estimated", "esp_actual", "raw_fidelity"):
        d = df.dropna(subset=[c])
        if len(d) > 2 and d[c].nunique() > 1:
            out[c] = {"spearman": round(float(spearmanr(d[c], d.measured).correlation), 3),
                      "mae": round(float((d[c] - d.measured).abs().mean()), 3) if c != "raw_fidelity" else None}
    return out


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    for ax, col, title in [(axes[0], "esp_estimated", "Estimate (no transpilation)"),
                           (axes[1], "esp_actual", "Transpiled circuit")]:
        for t, d in df.groupby("target"):
            ax.scatter(d[col], d.measured, s=14, label=t, alpha=0.8)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel(f"Predicted ESP: {title}", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Measured normalized Hellinger fidelity", fontsize=8)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(E10 / "e10_hardware.png", dpi=200)


def main():
    df, notes = load()
    df.to_csv(E10 / "e10_hardware.csv", index=False)
    summary = {"notes": notes, "all": stats(df),
               "per_device": {t: stats(d) for t, d in df.groupby("target")},
               "per_family": {f: stats(d) for f, d in df.groupby("family")},
               "selection": {c: selection(df, c) for c in ("esp_estimated", "esp_actual")}}
    ibm = df[df.vendor == "ibm"]
    if ibm.target.nunique() > 1:
        summary["selection_ibm_only"] = {c: selection(ibm, c) for c in ("esp_estimated", "esp_actual",
                                                                        "raw_fidelity")}
    (E10 / "e10_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    plot(df)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
