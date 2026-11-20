"""
Circuit Retranspilation Broker 
------------------------------------------------------------
Recompiles a circuit for the RESERVED candidate QPU only (slow path), and
reports the ACTUAL post-transpilation metrics used to re-validate the
scheduling decision.

Toolchains: the design maps vendors to toolchains (IBM → Qiskit,
Quantinuum/IonQ → tket, Rigetti → PyQuil, Google → Cirq). This prototype uses
tket for trapped-ion targets (MS native gate) when pytket is installed, and
Qiskit otherwise, mapping each native gate set to toolchain-equivalent gates.
Other toolchains can be registered per provider with `register_toolchain`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

# Native gate set (as reported in telemetry) → Qiskit basis used for transpilation
QISKIT_BASIS_BY_NATIVE = [
    ({"ms"},  ["rxx", "rz", "rx", "ry"]),          # trapped ion (MS + GPi/GPi2)
    ({"ecr"}, ["ecr", "rz", "sx", "x"]),
    ({"cz"},  None),                               # use reported 1Q gates + cz
    ({"cx"},  ["cx", "rz", "sx", "x"]),
]
QISKIT_STANDARD_1Q = {"rz", "sx", "x", "rx", "ry", "id"}


def qiskit_basis(native_gates: Sequence[str]) -> Optional[list]:
    native = {g.lower() for g in native_gates}
    if "any" in native or not native:
        return None
    for key, basis in QISKIT_BASIS_BY_NATIVE:
        if key & native:
            if basis is not None:
                return basis
            one_q = sorted(native & QISKIT_STANDARD_1Q - {"id"}) or ["rz", "sx", "x"]
            if one_q == ["rz"]:
                one_q = ["rz", "sx", "x"]
            return ["cz"] + one_q
    return ["cx", "rz", "sx", "x"]


@dataclass
class TranspileResult:
    qasm: Optional[str]
    depth: int
    num_single_qubit_gates: int
    num_two_qubit_gates: int
    added_two_qubit_gates: int      # vs. the circuit in the neutral {u, cx} basis
    swap_ratio: float               # actual SWAP-equivalents per original 2Q gate
    seconds: float
    toolchain: str
    variant: dict = field(default_factory=dict)


def tket_available() -> bool:
    try:
        import pytket  # noqa: F401
        import pytket.extensions.qiskit  # noqa: F401
        return True
    except ImportError:
        return False


def tket_toolchain(circuit, backend):
    """
    Compile with tket: peephole optimisation, placement + routing on the
    device architecture, rebase to the native gate set. Returns a Qiskit circuit.
    """
    from pytket.architecture import Architecture
    from pytket.circuit import OpType
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    from pytket.passes import (AutoRebase, DecomposeBoxes, DefaultMappingPass,
                               FullPeepholeOptimise, RemoveRedundancies, SequencePass)
    from topology import is_all_to_all

    native = {g.lower() for g in backend.native_gates}
    if "ms" in native or "rxx" in native:
        gateset = {OpType.XXPhase, OpType.Rz, OpType.Rx}
    elif "ecr" in native:
        gateset = {OpType.ECR, OpType.Rz, OpType.SX, OpType.X}
    elif "cz" in native:
        gateset = {OpType.CZ, OpType.Rz, OpType.Rx, OpType.SX, OpType.X}
    else:
        gateset = {OpType.CX, OpType.Rz, OpType.SX, OpType.X}
    gateset |= {OpType.Measure, OpType.Barrier}

    edges = [tuple(e) for e in (backend.coupling_map or [])]
    passes = [DecomposeBoxes(), FullPeepholeOptimise()]
    if edges and not is_all_to_all(edges, backend.qubit_count):
        passes.append(DefaultMappingPass(Architecture(edges)))   # placement + routing
    passes += [AutoRebase(gateset), RemoveRedundancies()]
    tk = qiskit_to_tk(circuit)
    SequencePass(passes).apply(tk)
    return tk_to_qiskit(tk)


class RetranspilationBroker:
    def __init__(self, optimization_levels=(1, 2), seeds=(11, 23, 37),
                 swap_cost_2q: int = 3, use_tket_for_ions: bool = True):
        self.optimization_levels = tuple(optimization_levels)
        self.seeds = tuple(seeds)
        self.swap_cost_2q = swap_cost_2q
        self.use_tket_for_ions = use_tket_for_ions and tket_available()
        self._toolchains: Dict[str, Callable] = {}

    def register_toolchain(self, provider: str, fn: Callable):
        """fn(circuit, backend_telemetry) -> QuantumCircuit transpiled for the QPU."""
        self._toolchains[provider] = fn

    def transpile(self, circuit_profile, backend) -> TranspileResult:
        """Transpile for one QPU, trying several variants; keep the fewest 2Q gates."""
        from qiskit import qasm2, transpile
        from qiskit.transpiler import CouplingMap
        from circuit_analysis import circuit_stats, load_circuit

        if not circuit_profile.qasm:
            raise ValueError("circuit has no QASM; cannot retranspile")
        t0 = time.perf_counter()
        circuit = load_circuit(circuit_profile.qasm)
        base = circuit_stats(circuit)

        native = {g.lower() for g in backend.native_gates}
        if backend.provider in self._toolchains:
            best = self._toolchains[backend.provider](circuit, backend)
            toolchain, variant = f"custom:{backend.provider}", {}
        elif self.use_tket_for_ions and native & {"ms", "rxx"}:
            best = tket_toolchain(circuit, backend)
            toolchain, variant = "tket", {}
        else:
            basis = qiskit_basis(backend.native_gates)
            cmap = None
            if backend.coupling_map and basis is not None:
                edges = [tuple(e) for e in backend.coupling_map]
                cmap = CouplingMap(edges + [(b, a) for a, b in edges])
            best, variant = None, {}
            for level in self.optimization_levels:
                for seed in self.seeds:
                    out = transpile(circuit, coupling_map=cmap, basis_gates=basis,
                                    optimization_level=level, seed_transpiler=seed)
                    n2 = sum(1 for i in out.data if i.operation.num_qubits == 2)
                    if best is None or n2 < variant["n2q"]:
                        best, variant = out, {"level": level, "seed": seed, "n2q": n2}
            toolchain = "qiskit"

        n1 = sum(1 for i in best.data if i.operation.num_qubits == 1
                 and i.operation.name not in ("measure", "barrier", "reset", "delay"))
        n2 = sum(1 for i in best.data if i.operation.num_qubits == 2)
        depth = best.depth(lambda i: i.operation.name != "barrier")
        added = max(0, n2 - base.num_two_qubit_gates)
        swap_ratio = added / self.swap_cost_2q / max(base.num_two_qubit_gates, 1)
        try:
            qasm_out = qasm2.dumps(best)
        except Exception:     # non-QASM2 gates (e.g. rxx is fine; custom may not be)
            qasm_out = None
        return TranspileResult(
            qasm=qasm_out, depth=depth, num_single_qubit_gates=n1,
            num_two_qubit_gates=n2, added_two_qubit_gates=added,
            swap_ratio=swap_ratio, seconds=time.perf_counter() - t0,
            toolchain=toolchain, variant=variant,
        )
