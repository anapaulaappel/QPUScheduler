"""
X7 — controlled cross-vendor analysis: the same 12 mirror circuits (4 and 8 qubits, E7 seeds), 1000 shots
on every device, all runs on the same day (20 Nov 2026): IBM fez/marrakesh/kingston (calibration read at
preparation), IQM Garnet (IQM Resonance; error rates as in E7), Rigetti Cepheus (verbatim via Open
Quantum; published medians, bit order chosen from the data as in E7). AQT was not included (12 circuits
x 1000 shots quoted at 180 credits).

Run:  python x7_analyze.py      Out: results/x7/x7_summary.json, results/x7/x7_hardware.csv
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS, esp
from e7_analyze import device_errors
from x2_bootstrap import boot, ci, selection

X7 = RESULTS / "x7"


def load():
    man = json.loads((X7 / "manifest.json").read_text())["devices"]
    names = [r["circuit"] for r in man["rigetti_cepheus"]["rows"]]
    rows, notes = [], {}
    ibm = pd.read_csv(RESULTS / "e6_hardware_x7.csv")
    # IBM ran all 28 E7 circuits (same seeds); subsets are taken in main()
    for _, r in ibm.iterrows():
        rows.append(dict(circuit=r.circuit, family=r.family, n=r.n, target=r.target, vendor="ibm",
                         measured=r.measured, raw_fidelity=r.raw_fidelity,
                         esp_estimated=r.esp_estimated, esp_actual=r.esp_actual))
    dev = json.loads((X7 / "device_iqm_garnet.json").read_text()); err = device_errors("iqm_garnet", dev)
    notes["iqm_garnet"] = err.get("source", "")
    iqm, iqm_rows = {}, list(man["iqm_garnet"]["rows"])
    for dname in ("x7", "x7b"):                      # x7b: the 16 remaining E7 circuits, same day, 1000 shots
        f = RESULTS / dname / "run_iqm_garnet.json"
        if f.exists():
            j = json.loads(f.read_text()); iqm.update(j.get("rows", j))
    if (RESULTS / "x7b" / "manifest.json").exists():
        iqm_rows += json.loads((RESULTS / "x7b" / "manifest.json").read_text())["devices"]["iqm_garnet"]["rows"]
    for r in iqm_rows:
        x = iqm.get(r["circuit"])
        if not x or not x.get("counts"):
            continue
        n = r["n"]
        rows.append(dict(circuit=r["circuit"], family=r["family"], n=n, target="iqm_garnet", vendor="iqm",
                         measured=x["counts"].get(r["key"], 0) / sum(x["counts"].values()), raw_fidelity=np.nan,
                         esp_estimated=esp(r["est_n1"], r["est_n2"], n, err["e1"], err["e2"], err["ero"]),
                         esp_actual=esp(x["actual_n1"], x["actual_n2"], n, err["e1"], err["e2"], err["ero"])))
    rf = X7 / "run_rigetti_cepheus.json"
    rg = {k: v for k, v in json.loads(rf.read_text()).items() if v.get("raw_counts")} if rf.exists() else {}
    dev = json.loads((X7 / "device_rigetti_cepheus.json").read_text()); err = device_errors("rigetti_cepheus", dev)
    mrows = {r["circuit"]: r for r in man["rigetti_cepheus"]["rows"]}
    def succ(x, key): return x["raw_counts"].get(key, 0) / sum(x["raw_counts"].values())
    nonpal = [c for c in rg if mrows[c]["key"] != mrows[c]["key"][::-1]] or [None]
    nonpal = [c for c in nonpal if c]
    as_is = sum(succ(rg[c], mrows[c]["key"]) for c in nonpal); rev = sum(succ(rg[c], mrows[c]["key"][::-1]) for c in nonpal)
    order = "reversed" if rev > as_is else "as_is"
    notes["rigetti_cepheus"] = err.get("source", "") + f" | bit order {order} (as_is {as_is:.2f}, reversed {rev:.2f})"
    for c, x in rg.items():
        r = mrows[c]; n = r["n"]; key = r["key"][::-1] if order == "reversed" else r["key"]
        rows.append(dict(circuit=c, family=r["family"], n=n, target="rigetti_cepheus", vendor="rigetti",
                         measured=succ(x, key), raw_fidelity=np.nan,
                         esp_estimated=esp(r["est_n1"], r["est_n2"], n, err["e1"], err["e2"], err["ero"]),
                         esp_actual=esp(x["actual_n1"], x["actual_n2"], n, err["e1"], err["e2"], err["ero"])))
    return pd.DataFrame(rows), notes, names


def main():
    d, notes, names12 = load()
    d.to_csv(X7 / "x7_hardware.csv", index=False)
    rng = np.random.default_rng(20261120)
    out = {"notes": notes, "n_rows": int(len(d)), "per_device": {}, "selection": {}}
    for t, g in d.groupby("target"):
        out["per_device"][t] = {c: {"rho": round(float(spearmanr(g[c], g.measured).correlation), 3),
                                    "mae": round(float((g[c] - g.measured).abs().mean()), 3)}
                                for c in ("esp_estimated", "esp_actual")} | {"n": int(len(g))}
    ibm_iqm = d[d.target != "rigetti_cepheus"]
    common_iqm = set(d[d.target == "iqm_garnet"].circuit)
    subsets = [("ibm3+iqm (all circuits run on IQM)", ibm_iqm[ibm_iqm.circuit.isin(common_iqm)])]
    if (d.target == "rigetti_cepheus").any():
        subsets.append(("ibm3+iqm+rigetti (12 circuits)", d[d.circuit.isin(names12)]))
    for label, sub in subsets:
        out["selection"][f"{label}/n_circuits"] = int(sub.circuit.nunique())
        for c in ("esp_estimated", "esp_actual"):
            r, t = selection(sub, c, "measured")
            s = boot(sub, lambda x, c=c: selection(x, c, "measured"), rng)
            out["selection"][f"{label}/{c}"] = {"regret": round(r, 4), "regret_ci": [round(v, 4) for v in ci(s[:, 0])],
                                                "top1": round(t, 3), "top1_ci": [round(v, 3) for v in ci(s[:, 1])]}
        always = {tt: float(sub[sub.target == tt].measured.mean()) for tt in sub.target.unique()}
        out["selection"][f"{label}/best_single_device"] = max(always, key=always.get)
    (X7 / "x7_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
