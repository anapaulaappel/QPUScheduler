"""
E1b — Latency: graph estimate vs real transpilation as circuits grow.

For QFT and random circuits of increasing width on a heavy-hex device
(FakeFez, 156 qubits), measures the time of the topology estimate and of a real
Qiskit transpilation (optimization level 1 and 3, one seed). The scheduler's
Filter phase must evaluate every candidate; this quantifies why the estimate,
not a transpilation, runs there.

Run:  python e1b_latency.py
Out:  results/e1b_latency.csv, results/e1b_latency.png
"""

import warnings

import pandas as pd

from common import RESULTS, FAMILIES, load_targets, timed

warnings.filterwarnings("ignore")


def run(sizes=(10, 20, 30, 40, 60, 80), families=("qft", "random")):
    from qiskit import transpile
    from circuit_analysis import circuit_stats, estimate_divergence

    fez = next(t for t in load_targets(include=("heavy-hex",)) if t.name == "FakeFez")
    tel = fez.telemetry
    rows = []
    for fam in families:
        for n in sizes:
            qc = FAMILIES[fam](n, 7 + n)
            st, t_stats = timed(circuit_stats, qc)
            _, t_est = timed(estimate_divergence, st, tel.coupling_map, tel.qubit_count,
                             tel.native_gates)
            m = qc.copy()
            m.measure_all()
            _, t1 = timed(transpile, m, backend=fez.backend, optimization_level=1, seed_transpiler=1)
            _, t3 = timed(transpile, m, backend=fez.backend, optimization_level=3, seed_transpiler=1)
            rows.append(dict(family=fam, n=n, n2=st.num_two_qubit_gates,
                             stats_ms=t_stats * 1e3, estimate_ms=t_est * 1e3,
                             transpile_l1_ms=t1 * 1e3, transpile_l3_ms=t3 * 1e3))
            print(rows[-1])
    return pd.DataFrame(rows)


def plot(df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fam, d in df.groupby("family"):
        ax.plot(d.n, d.estimate_ms, "o-", label=f"{fam}: topology estimate")
        ax.plot(d.n, d.transpile_l1_ms, "s--", label=f"{fam}: transpile (level 1)")
        ax.plot(d.n, d.transpile_l3_ms, "^:", label=f"{fam}: transpile (level 3)")
    ax.set_yscale("log")
    ax.set_xlabel("Circuit width (qubits)")
    ax.set_ylabel("Time per candidate QPU (ms)")
    ax.set_title("Per-candidate cost: estimate vs transpilation (FakeFez, 156 qubits)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS / "e1b_latency.png", dpi=200)


if __name__ == "__main__":
    df = run()
    df.to_csv(RESULTS / "e1b_latency.csv", index=False)
    plot(df)
    print(df.to_string(index=False))
