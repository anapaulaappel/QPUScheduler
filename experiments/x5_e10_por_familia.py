"""
X5 (path A) — E10 by output type (offline). Spearman rho of ESP vs normalized Hellinger fidelity, split
into families with a concentrated ideal output (BV, GHZ, adder, Grover: one or two dominant bitstrings)
and families with a broad output (QAOA, VQE), on IBM devices and on all devices.
Run: python x5_e10_por_familia.py   Out: results/e10/x5_por_familia.json
"""
import json
import pandas as pd
from scipy.stats import spearmanr
from common import RESULTS

d = pd.read_csv(RESULTS / "e10" / "e10_hardware.csv")
CONC = {"bernstein_vazirani", "ghz", "adder", "grover"}
d["saida"] = d.family.map(lambda f: "concentrada" if f in CONC else "espalhada")
out = {}
for escopo, x in (("ibm", d[d.vendor == "ibm"]), ("todos", d)):
    for grupo, g in x.groupby("saida"):
        out[f"{escopo}/{grupo}"] = {"n": int(len(g)),
            "rho_estimate": round(float(spearmanr(g.esp_estimated, g.measured).correlation), 3),
            "rho_transpiled": round(float(spearmanr(g.esp_actual, g.measured).correlation), 3),
            "mae_estimate": round(float((g.esp_estimated - g.measured).abs().mean()), 3)}
    for fam, g in x.groupby("family"):
        if len(g) > 3:
            out[f"{escopo}/familia/{fam}"] = {"n": int(len(g)),
                "rho_estimate": round(float(spearmanr(g.esp_estimated, g.measured).correlation), 3)}
(RESULTS / "e10" / "x5_por_familia.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    if "/familia/" not in k: print(k, v)
