"""
X8 — Layout-consistent ESP (offline; read-only historical calibration, no QPU).

E6 showed that device-mean estimates cannot separate ibm_fez from ibm_marrakesh (48% correct). Here we
take the circuits as actually transpiled for E6 (results/e6_prepared_large.pkl, physical qubits) and the
calibration each device published on 3 Nov 2026 just before the run (IBM properties(datetime=...)), and
compute:
  esp_layout : product over the transpiled gates of (1 - error of that gate on those physical qubits)
               times (1 - readout error) of each measured physical qubit;
  esp_mean   : the same formula with the device-mean 1Q, 2Q and readout errors (isolates the layout effect).
Both ignore decoherence, so they are compared with each other; the published E6 ESP columns are reported
for reference. Selection between fez and marrakesh (and among all three) uses the measured success.

Run:  python x8_layout_esp.py
Out:  results/x8_layout_esp.csv, results/x8_layout_esp.json
"""
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS

RUN_TIME = datetime(2026, 11, 3, 14, 35, tzinfo=timezone.utc)
TWO_Q = ("cz", "ecr", "cx")
VIRTUAL = ("rz", "barrier", "delay", "reset")


def calibration(name):
    from qiskit_ibm_runtime import QiskitRuntimeService
    p = QiskitRuntimeService().backend(name).properties(datetime=RUN_TIME)
    gate_err = {}
    for g in p.gates:
        e = next((x.value for x in g.parameters if x.name == "gate_error"), None)
        if e is not None:
            gate_err[(g.gate, tuple(g.qubits))] = e
    ro = {q: p.readout_error(q) for q in range(len(p.qubits))}
    one = [v for (g, q), v in gate_err.items() if len(q) == 1 and g in ("sx", "x")]
    two = [v for (g, q), v in gate_err.items() if len(q) == 2]
    return gate_err, ro, float(np.mean(one)), float(np.mean(two)), float(np.mean(list(ro.values()))), str(p.last_update_date)


def esps(circ, gate_err, ro, m1, m2, mro):
    lay = mean = 1.0
    for inst in circ.data:
        name = inst.operation.name
        qs = tuple(circ.find_bit(q).index for q in inst.qubits)
        if name in VIRTUAL:
            continue
        if name == "measure":
            lay *= 1 - ro[qs[0]]; mean *= 1 - mro
        elif len(qs) == 2:
            e = gate_err.get((name, qs), gate_err.get((name, qs[::-1]), m2))
            lay *= 1 - e; mean *= 1 - m2
        else:
            lay *= 1 - gate_err.get((name, qs), m1); mean *= 1 - m1
    return lay, mean


def selection(d, col):
    reg, hit = [], []
    for _, g in d.groupby("circuit"):
        pick, best = g.loc[g[col].idxmax(), "measured"], g.measured.max()
        reg.append(best - pick); hit.append(pick >= best - 1e-12)
    return round(float(np.mean(reg)), 4), round(float(np.mean(hit)), 3)


def main():
    prep = pickle.load(open(RESULTS / "e6_prepared_large.pkl", "rb"))
    hw = pd.read_csv(RESULTS / "e6_hardware_large.csv")
    rows, cal_info = [], {}
    for name, d in prep["backends"].items():
        gate_err, ro, m1, m2, mro, upd = calibration(name)
        cal_info[name] = {"published": upd, "mean_1q": m1, "mean_2q": m2, "mean_ro": mro}
        for r, circ in zip(d["rows"], d["circuits"]):
            lay, mean = esps(circ, gate_err, ro, m1, m2, mro)
            rows.append(dict(circuit=r["circuit"], target=name, esp_layout=lay, esp_mean=mean))
    df = hw.merge(pd.DataFrame(rows), on=["circuit", "target"])
    df.to_csv(RESULTS / "x8_layout_esp.csv", index=False)
    out = {"calibration": cal_info, "n_rows": int(len(df))}
    for col in ("esp_layout", "esp_mean", "esp_estimated", "esp_actual"):
        out[col] = {"spearman_all": round(float(spearmanr(df[col], df.measured).correlation), 3),
                    "mae_all": round(float((df[col] - df.measured).abs().mean()), 3)}
    pair = df[df.target.isin(["ibm_fez", "ibm_marrakesh"])]
    for col in ("esp_layout", "esp_mean", "esp_estimated", "esp_actual", "raw_fidelity"):
        out[col] = out.get(col, {})
        out[col]["fez_vs_marrakesh_regret_top1"] = selection(pair, col)
        out[col]["three_devices_regret_top1"] = selection(df, col)
    (RESULTS / "x8_layout_esp.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
