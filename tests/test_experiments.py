"""Smoke tests for the experiment setup (fast subset)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))


def test_mirror_circuits_return_ideal_bitstring():
    from qiskit_aer import AerSimulator
    from common import benchmark_suite, transpile_best
    from e2_fidelity_prediction import mirror
    from qiskit_ibm_runtime.fake_provider import FakeFez
    sim, fez = AerSimulator(), FakeFez()
    for i, b in enumerate(benchmark_suite(sizes=(4,), families=["qft", "random", "ghz"])):
        m, key = mirror(b.circuit, 100 + i)
        m.measure_all()
        out, _ = transpile_best(m, fez, seeds=(11,))
        counts = sim.run(out, shots=100).result().get_counts()
        assert counts.get(key, 0) == 100


def test_targets_have_realistic_telemetry():
    from common import load_targets
    for t in load_targets():
        tel = t.telemetry
        assert 0 < tel.mean_tq_error < 0.05, t.name
        assert tel.coupling_map and tel.durations.get("gate_2q_ns")
