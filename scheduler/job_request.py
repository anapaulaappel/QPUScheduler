"""
JobRequest
----------
Represents a quantum workload's scheduling requirements.
In a real OpenShift deployment these come from Pod annotations.
In the demo/test, you instantiate this directly.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CircuitProfile:
    """
    Describes the quantum circuit to be scheduled.
    Used by the Cost-Fidelity Prediction Module to estimate
    post-transpilation fidelity when routing to a different topology.
    """
    # Circuit structure
    num_qubits: int
    depth: int                        # total circuit depth (layers)
    num_two_qubit_gates: int          # number of 2Q gates (CX, ECR, CZ...)
    num_single_qubit_gates: int       # number of 1Q gates

    # Topology the circuit was originally compiled for
    # e.g. "heavy-hex", "all-to-all", "square-lattice"
    compiled_for_topology: str = "heavy-hex"

    # OpenQASM string (optional). When present, the scheduler computes the
    # graph-based topology divergence and the retranspilation broker can
    # recompile the circuit for the selected QPU.
    qasm: Optional[str] = None

    # Interaction graph {(u, v): two-qubit gate count}, two-qubit gate order and
    # per-qubit 1Q gate counts, derived from qasm
    interaction: Optional[dict] = field(default=None, repr=False)
    two_qubit_sequence: Optional[list] = field(default=None, repr=False)
    one_qubit_counts: Optional[dict] = field(default=None, repr=False)

    @classmethod
    def from_qasm(cls, qasm: str, compiled_for_topology: str = "heavy-hex") -> "CircuitProfile":
        """Build a profile (sizes + interaction graph) from an OpenQASM string."""
        from circuit_analysis import circuit_stats, load_circuit
        st = circuit_stats(load_circuit(qasm))
        return cls(num_qubits=st.num_qubits, depth=st.depth,
                   num_two_qubit_gates=st.num_two_qubit_gates,
                   num_single_qubit_gates=st.num_single_qubit_gates,
                   compiled_for_topology=compiled_for_topology,
                   qasm=qasm, interaction=st.interaction,
                   two_qubit_sequence=st.two_qubit_sequence,
                   one_qubit_counts=st.one_qubit_counts)

    def stats(self):
        """CircuitStats view used by circuit_analysis.estimate_divergence()."""
        from circuit_analysis import CircuitStats
        return CircuitStats(num_qubits=self.num_qubits, depth=self.depth,
                            num_two_qubit_gates=self.num_two_qubit_gates,
                            num_single_qubit_gates=self.num_single_qubit_gates,
                            interaction=dict(self.interaction or {}),
                            two_qubit_sequence=list(self.two_qubit_sequence or []),
                            one_qubit_counts=dict(self.one_qubit_counts or {}))


@dataclass
class JobRequest:
    """
    Scheduling requirements declared by a quantum workload.
    Mirrors the Pod annotation schema used by the scheduler.
    """
    # ── Identity ──────────────────────────────────────────────────────────
    job_id: str
    circuit: CircuitProfile

    # ── Hard constraints (Filter phase) ──────────────────────────────────
    min_fidelity: float = 0.99        # minimum mean qubit fidelity
    max_gate_error_rate: float = 0.02 # maximum mean 2Q gate error rate
    max_readout_error: float = 0.05   # maximum mean readout error rate
    max_queue_depth: int = 50         # maximum jobs ahead in queue
    max_calibration_age_minutes: float = 720.0  # max time since calibration
    min_qubit_count: int = 1          # minimum qubit count

    # ── Cost / deadline constraints ───────────────────────────────────────
    max_cost_per_shot: float = 1.0    # USD per shot (0.0 = no limit)
    max_cost_per_job: float = 0.0     # USD, effective cost of the job (0.0 = no limit)
    deadline_minutes: float = 0.0     # 0 = execute immediately, >0 = deferrable

    # ── Execution / tenancy ───────────────────────────────────────────────
    shots: int = 1024
    namespace: str = "default"

    # ── Preferences ───────────────────────────────────────────────────────
    preferred_providers: List[str] = field(default_factory=list)
    allow_simulator_fallback: bool = True

    # ── Score weight overrides (optional) ────────────────────────────────
    # Any subset of DEFAULT_SCORE_WEIGHTS may be overridden; missing keys
    # keep their default value.
    score_weights: Optional[dict] = None

    def __post_init__(self):
        weights = dict(DEFAULT_SCORE_WEIGHTS)
        if self.score_weights:
            unknown = set(self.score_weights) - set(DEFAULT_SCORE_WEIGHTS)
            if unknown:
                raise ValueError(f"Unknown score weights: {sorted(unknown)}")
            weights.update(self.score_weights)
        self.score_weights = weights


# Default score weights. 'fidelity' refers to the predicted post-transpilation
# fidelity; 'cost' to the per-shot price; 'provider_pref' is the additive
# preference bonus β. The composite is divided by the sum of all weights.
DEFAULT_SCORE_WEIGHTS = {
    "fidelity":        0.40,
    "error_rate":      0.25,
    "coherence":       0.15,
    "queue_depth":     0.15,
    "cal_freshness":   0.05,
    "cost":            0.05,
    "provider_pref":   0.10,
}
