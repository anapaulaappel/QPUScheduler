"""
E7 — multi-vendor hardware validation: preparation (main venv, no QPU).

Builds the E6 mirror circuits for each non-IBM device and computes the topology
estimate (no transpilation) from the device description exported by the
provider venvs (results/e7/device_<name>.json). The provider scripts then
transpile for the real backend, run, and record counts; e7_analyze.py
computes ESP and compares with the measured success.

Out: results/e7/circuits/<circuit>.qasm (logical mirror circuit, measured),
     results/e7/manifest.json
"""
import json

from qiskit import qasm2

from common import RESULTS, benchmark_suite
from e2_fidelity_prediction import e1_alpha, mirror

E7 = RESULTS / "e7"

# Circuit sets per device (subsets of the 28-circuit E6 set, same mirror seeds).
SELECT = {
    "iqm_garnet": lambda b: True,                                         # all 28
    "rigetti_cepheus": lambda b: b.num_qubits in (4, 8) and b.family != "qaoa_complete",  # 12
    "aqt_ibex_q1": lambda b: b.num_qubits == 6 and b.family != "qaoa_complete",           # 6
}
SHOTS = {"iqm_garnet": 500, "rigetti_cepheus": 1000, "aqt_ibex_q1": 100}


def main():
    from circuit_analysis import circuit_stats, estimate_divergence
    alpha = e1_alpha("static_added_2q")
    suite = benchmark_suite(sizes=(4, 6, 8, 10))
    (E7 / "circuits").mkdir(parents=True, exist_ok=True)

    circuits = {}
    for i, b in enumerate(suite):
        circ, key = mirror(b.circuit, seed=100 + i)          # same as E6
        qc = circ.copy()
        qc.measure_all()
        (E7 / "circuits" / f"{b.name}.qasm").write_text(qasm2.dumps(qc))
        circuits[b.name] = dict(bench=b, circ=circ, key=key)

    manifest = {"alpha": alpha, "devices": {}}
    for dev_name, pick in SELECT.items():
        dev = json.loads((E7 / f"device_{dev_name}.json").read_text())
        rows = []
        for name, c in circuits.items():
            b = c["bench"]
            if not pick(b) or b.num_qubits > dev["n_qubits"]:
                continue
            st = circuit_stats(c["circ"])
            est = estimate_divergence(st, dev["edges"], dev["n_qubits"], dev["native_tag"])
            rows.append(dict(
                circuit=name, family=b.family, n=b.num_qubits, key=c["key"],
                n1_logical=st.num_single_qubit_gates, n2_logical=st.num_two_qubit_gates,
                depth_logical=st.depth,
                est_n1=st.num_single_qubit_gates * est.gate_overhead,
                est_n2=st.num_two_qubit_gates + alpha * est.added_two_qubit_gates,
                est_depth=st.depth * (1 + est.swap_ratio * alpha),
                est_swaps=est.swaps))
        manifest["devices"][dev_name] = {"shots": SHOTS[dev_name], "rows": rows}
        print(f"{dev_name}: {len(rows)} circuits, {SHOTS[dev_name]} shots")
    (E7 / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
