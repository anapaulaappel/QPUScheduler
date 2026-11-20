"""
Shared setup for the paper experiments (E1–E5).

All experiments run offline:
  - heavy-hex and square-lattice targets use IBM fake backends, which carry
    real calibration snapshots (error rates, T1/T2, durations, coupling maps);
  - all-to-all (trapped-ion-like) targets use GenericBackendV2 with a full
    coupling map and an MS-like (RXX) native gate.
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
for sub in ("probe", "scheduler", "forecaster"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Illustrative list prices (verify with providers)
IBM_PRICING = {"scheme": "time", "price_per_second": 1.60}
ION_PRICING = {"scheme": "per_gate_shot", "price_1q": 0.000220, "price_2q": 0.000975}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark circuits (logical; no topology)
# ─────────────────────────────────────────────────────────────────────────────

def _ghz(n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def _qft(n):
    from qiskit.synthesis import synth_qft_full
    return synth_qft_full(n)


def _random(n, seed):
    from qiskit.circuit.random import random_circuit
    return random_circuit(n, depth=max(4, n), max_operands=2, seed=seed)


def _qaoa(n, edges, p, seed):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(p):
        g, b = rng.uniform(0, np.pi, 2)
        for a, c in edges:
            qc.rzz(g, a, c)
        for q in range(n):
            qc.rx(2 * b, q)
    return qc


def _real_amplitudes(n, seed):
    from qiskit.circuit.library import real_amplitudes
    qc = real_amplitudes(n, reps=2, entanglement="full")
    rng = np.random.default_rng(seed)
    return qc.assign_parameters(rng.uniform(0, 2 * np.pi, qc.num_parameters))


def _bv(n, seed):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    secret = rng.integers(0, 2, n - 1)
    qc = QuantumCircuit(n)
    qc.x(n - 1)
    qc.h(range(n))
    for i, bit in enumerate(secret):
        if bit:
            qc.cx(i, n - 1)
    qc.h(range(n - 1))
    return qc


FAMILIES = {
    "ghz":             lambda n, s: _ghz(n),
    "qft":             lambda n, s: _qft(n),
    "random":          lambda n, s: _random(n, s),
    "qaoa_ring":       lambda n, s: _qaoa(n, [(i, (i + 1) % n) for i in range(n)], 2, s),
    "qaoa_complete":   lambda n, s: _qaoa(n, [(i, j) for i in range(n) for j in range(i + 1, n)], 1, s),
    "real_amplitudes": lambda n, s: _real_amplitudes(n, s),
    "bernstein_vazirani": lambda n, s: _bv(n, s),
}


@dataclass
class Bench:
    name: str
    family: str
    num_qubits: int
    circuit: object          # logical circuit WITHOUT measurements


def benchmark_suite(sizes=(4, 6, 8, 10, 12), families=None, seed=7,
                    instances: int = 1) -> List[Bench]:
    """
    Benchmark circuits. `instances` > 1 adds instances with different random
    seeds (names get a suffix _s<k>); deterministic families (GHZ, QFT) are not
    duplicated because their circuits do not depend on the seed.
    """
    out = []
    deterministic = {"ghz", "qft"}
    for fam in (families or FAMILIES):
        for n in sizes:
            for k in range(1 if fam in deterministic else instances):
                qc = FAMILIES[fam](n, seed + n + 1000 * k)
                name = f"{fam}_{n}" if k == 0 else f"{fam}_{n}_s{k}"
                out.append(Bench(name, fam, n, qc))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Targets
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Target:
    name: str
    topology: str
    backend: object          # Qiskit BackendV2 (fake / generic)
    telemetry: object        # QPUTelemetry


def _ion_backend(n, seed, e1=2e-4, e2=5e-3, ero=4e-3):
    """
    Trapped-ion-like backend: all-to-all RXX (MS-like) + RX/RZ, with error
    rates and durations of the order reported for current trapped-ion devices
    (randomized per qubit/pair around the given means).
    """
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import Measure, RXGate, RXXGate, RZGate
    from qiskit.providers import BackendV2, Options
    from qiskit.providers.backend import QubitProperties
    from qiskit.transpiler import InstructionProperties, Target

    rng = np.random.default_rng(seed)
    jitter = lambda m: float(m * rng.uniform(0.7, 1.3))   # noqa: E731
    qprops = [QubitProperties(t1=jitter(10.0), t2=jitter(1.0), frequency=None)
              for _ in range(n)]
    target = Target(num_qubits=n, qubit_properties=qprops)
    theta = Parameter("θ")
    target.add_instruction(RZGate(theta), {(q,): InstructionProperties(0.0, 0.0) for q in range(n)})
    target.add_instruction(RXGate(theta), {(q,): InstructionProperties(63e-6, jitter(e1))
                                           for q in range(n)})
    target.add_instruction(RXXGate(theta), {(a, b): InstructionProperties(650e-6, jitter(e2))
                                            for a in range(n) for b in range(n) if a != b})
    target.add_instruction(Measure(), {(q,): InstructionProperties(250e-6, jitter(ero))
                                       for q in range(n)})

    class IonBackend(BackendV2):
        def __init__(self):
            super().__init__(name=f"ion_{seed}")

        @property
        def target(self):
            return target

        @property
        def max_circuits(self):
            return None

        @classmethod
        def _default_options(cls):
            return Options(shots=1024)

        def run(self, run_input, **options):   # pragma: no cover - simulated via Aer
            raise NotImplementedError

    return IonBackend()


def load_targets(include=("heavy-hex", "square-lattice", "all-to-all")) -> List[Target]:
    from qiskit_ibm_runtime import fake_provider as fp
    from backends.target_telemetry import telemetry_from_backend

    specs = []
    if "heavy-hex" in include:
        for cls in ("FakeFez", "FakeTorino", "FakeBrisbane", "FakeSherbrooke", "FakeKyiv"):
            specs.append((cls, "heavy-hex", getattr(fp, cls)(), IBM_PRICING))
    if "square-lattice" in include:
        specs.append(("FakeNighthawk", "square-lattice", fp.FakeNighthawk(), IBM_PRICING))
    if "all-to-all" in include:
        for seed in (1, 2):
            specs.append((f"ion_generic_{seed}", "all-to-all", _ion_backend(16, seed), ION_PRICING))
    out = []
    for name, topo, backend, pricing in specs:
        tel = telemetry_from_backend(backend, name=name, topology=topo,
                                     provider=topo, pricing=pricing)
        out.append(Target(name, topo, backend, tel))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_ops(circuit):
    n1 = sum(1 for i in circuit.data if i.operation.num_qubits == 1
             and i.operation.name not in ("measure", "barrier", "reset", "delay"))
    n2 = sum(1 for i in circuit.data if i.operation.num_qubits == 2)
    depth = circuit.depth(lambda i: i.operation.name not in ("barrier",))
    return n1, n2, depth


def transpile_best(circuit, backend, levels=(1,), seeds=(11, 23, 37)):
    """Transpile for a backend; keep the variant with the fewest 2Q gates."""
    from qiskit import transpile
    best, best_n2, t0 = None, None, time.perf_counter()
    for lvl in levels:
        for s in seeds:
            out = transpile(circuit, backend=backend, optimization_level=lvl, seed_transpiler=s)
            n2 = count_ops(out)[1]
            if best is None or n2 < best_n2:
                best, best_n2 = out, n2
    return best, time.perf_counter() - t0


def esp(n1, n2, n_meas, e1, e2, ero, duration_s=0.0, t2_us=0.0):
    """Estimated success probability of a circuit (product of gate fidelities)."""
    val = (1 - e1) ** n1 * (1 - e2) ** n2 * (1 - ero) ** n_meas
    if duration_s and t2_us:
        val *= np.exp(-duration_s / (t2_us * 1e-6))
    return float(val)


def timed(fn: Callable, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0
