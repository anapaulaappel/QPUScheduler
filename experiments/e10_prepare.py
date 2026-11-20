"""
E10 — application circuits on hardware: preparation (main venv, no QPU).

The earlier hardware runs (E6, E7) used mirror circuits, whose ideal output is a single known
bitstring. E10 uses application circuits as they are (no mirroring): Bernstein–Vazirani, GHZ,
Grover, a ripple-carry adder, QAOA MaxCut and a hardware-efficient VQE ansatz. Quality is the
normalized Hellinger fidelity to the ideal output distribution (Lubinski et al., 2110.03137):
    F = (sum_x sqrt(p_ideal(x) p_meas(x)))^2,   F_norm = (F - F_unif) / (1 - F_unif),
where F_unif is the fidelity of the uniform distribution. For deterministic circuits (BV, adder)
we also record the success probability of the ideal bitstring.

Out: results/e10/circuits/<name>.qasm (logical circuit with measurements),
     results/e10/ideal/<name>.json (ideal distribution, bitstrings in Qiskit order c[n-1]..c[0]),
     results/e10/manifest.json (per device: circuits, shots, topology estimate without transpiling).
"""
import json

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector

from common import RESULTS, _bv, _ghz, _qaoa
from e2_fidelity_prediction import e1_alpha

E10 = RESULTS / "e10"
SEED = 2026


def grover(n, marked):
    """Grover search for one marked n-bit state, optimal number of iterations."""
    qc = QuantumCircuit(n)
    qc.h(range(n))
    iters = int(np.floor(np.pi / 4 * np.sqrt(2 ** n)))
    bits = [(marked >> i) & 1 for i in range(n)]
    for _ in range(iters):
        for i, b in enumerate(bits):           # oracle: phase flip on |marked>
            if not b:
                qc.x(i)
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        for i, b in enumerate(bits):
            if not b:
                qc.x(i)
        qc.h(range(n)); qc.x(range(n))         # diffusion
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        qc.x(range(n)); qc.h(range(n))
    return qc


def adder(k, a, b):
    """Cuccaro ripple-carry adder on k-bit inputs a, b (2k + 2 qubits), basis-state input."""
    from qiskit.circuit.library import CDKMRippleCarryAdder
    add = CDKMRippleCarryAdder(k, kind="full")
    qc = QuantumCircuit(add.num_qubits)
    # register order of the library circuit: cin, a[0..k-1], b[0..k-1], cout
    for i in range(k):
        if (a >> i) & 1:
            qc.x(1 + i)
        if (b >> i) & 1:
            qc.x(1 + k + i)
    qc.compose(add.decompose(), inplace=True)
    return qc


def vqe_hea(n, seed):
    """Hardware-efficient ansatz: RY layer, linear CZ chain, RY layer (2 reps)."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(2):
        for q in range(n):
            qc.ry(rng.uniform(0, 2 * np.pi), q)
        for q in range(n - 1):
            qc.cz(q, q + 1)
    for q in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), q)
    return qc


def random_3_regular(n, seed):
    import networkx as nx
    return list(nx.random_regular_graph(3, n, seed=seed).edges())


def suite():
    s = []
    for n in (4, 6, 8, 10):
        s.append(("bernstein_vazirani", n, _bv(n, SEED + n)))
        s.append(("ghz", n, _ghz(n)))
    s.append(("grover", 3, grover(3, 0b101)))
    s.append(("grover", 4, grover(4, 0b1011)))
    s.append(("adder", 6, adder(2, 0b10, 0b11)))
    s.append(("adder", 8, adder(3, 0b101, 0b011)))
    for n in (4, 6):
        s.append(("qaoa_ring", n, _qaoa(n, [(i, (i + 1) % n) for i in range(n)], 1, SEED + n)))
    for n in (6, 8):
        s.append(("qaoa_3reg", n, _qaoa(n, random_3_regular(n, SEED + n), 1, SEED + 100 + n)))
    for n in (4, 6, 8):
        s.append(("vqe_hea", n, vqe_hea(n, SEED + 200 + n)))
    return s


def ideal_distribution(qc, tol=1e-10):
    p = Statevector(qc).probabilities_dict()
    return {k: float(v) for k, v in p.items() if v > tol}


# Devices: IBM (estimate from live calibration is done in e10_run_ibm.py, like E6); the
# non-IBM device descriptions are the ones exported for E7.
DEVICES = {"rigetti_cepheus": 1000, "aqt_ibex_q1": 200, "iqm_garnet": 500}
MAX_QUBITS = {"aqt_ibex_q1": 8}          # keep Hellinger estimates meaningful at 200 shots


def main():
    from circuit_analysis import circuit_stats, estimate_divergence
    alpha = e1_alpha("static_added_2q")
    (E10 / "circuits").mkdir(parents=True, exist_ok=True)
    (E10 / "ideal").mkdir(parents=True, exist_ok=True)
    benches = []
    for fam, n, circ in suite():
        name = f"{fam}_{n}"
        ideal = ideal_distribution(circ)
        qc = circ.copy()
        qc.measure_all()
        (E10 / "circuits" / f"{name}.qasm").write_text(qasm2.dumps(qc))
        (E10 / "ideal" / f"{name}.json").write_text(json.dumps(ideal))
        top = max(ideal, key=ideal.get)
        benches.append(dict(name=name, family=fam, n=n, circ=circ,
                            deterministic=ideal[top] > 0.999, top=top, top_p=ideal[top],
                            support=len(ideal)))
        print(f"{name:22s} support={len(ideal):4d} top_p={ideal[top]:.3f}")
    manifest = {"alpha": alpha, "seed": SEED, "devices": {}}
    for dev_name, shots in DEVICES.items():
        dev = json.loads((RESULTS / "e7" / f"device_{dev_name}.json").read_text())
        cap = min(dev["n_qubits"], MAX_QUBITS.get(dev_name, 99))
        rows = []
        for b in benches:
            if b["n"] > cap:
                continue
            st = circuit_stats(b["circ"])
            est = estimate_divergence(st, dev["edges"], dev["n_qubits"], dev["native_tag"])
            rows.append(dict(
                circuit=b["name"], family=b["family"], n=b["n"],
                deterministic=b["deterministic"], top=b["top"], support=b["support"],
                n1_logical=st.num_single_qubit_gates, n2_logical=st.num_two_qubit_gates,
                depth_logical=st.depth,
                est_n1=st.num_single_qubit_gates * est.gate_overhead,
                est_n2=st.num_two_qubit_gates + alpha * est.added_two_qubit_gates,
                est_depth=st.depth * (1 + est.swap_ratio * alpha),
                est_swaps=est.swaps))
        manifest["devices"][dev_name] = {"shots": shots, "rows": rows}
        print(f"{dev_name}: {len(rows)} circuits, {shots} shots")
    manifest["all"] = [{k: v for k, v in b.items() if k != "circ"} for b in benches]
    (E10 / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
