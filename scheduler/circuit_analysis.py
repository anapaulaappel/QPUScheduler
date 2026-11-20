"""
Circuit analysis and graph-based topology divergence
-----------------------------------------------------------------------------
Estimates how much extra work a circuit needs to run on a candidate QPU
WITHOUT transpiling it, so that the estimate can run inside the scheduler's
fast Filter phase.

Steps:
  1. Circuit interaction graph G_c: vertices = logical qubits, edge weight
     w(u, v) = number of two-qubit gates acting on (u, v).
  2. Target coupling graph G_q and all-pairs shortest-path distances d_q.
  3. Initial placement φ minimizing Σ w(u,v)·(d_q(φ(u), φ(v)) − 1):
     exact embedding by subgraph isomorphism (VF2) when one exists — chosen
     among several embeddings by lowest error cost when per-qubit errors are
     known — otherwise a greedy heuristic.
  4. Routing overhead, two variants:
       static : S = Σ w(u,v)·max(0, d_q(φ(u), φ(v)) − 1)  (placement fixed;
                default — best target selection in experiment E2)
       dynamic: a lightweight greedy router replays the two-qubit gates in
                order, moving a qubit along a shortest path when its partner
                is not adjacent and updating the placement (best calibrated
                gate-count accuracy in experiment E1)
     ΔN_2q = k_swap(q)·S (k_swap = 3 on sparse devices, 0 if all-to-all).
  With per-qubit/per-pair error rates, the same replay accumulates an
  estimated success probability (ESP) on the qubits actually used.
  5. Gate-set translation overhead r_gate(q) from per-native-gate rules.
  6. D(c, q) = ΔN_2q / max(N_2q, 1) + λ·(r_gate − 1).

The cost-fidelity model consumes `swap_ratio` = S / N_2q, the same unit as the
family-level lookup table (SWAP insertions per original two-qubit gate).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Edge = Tuple[int, int]

SWAP_COST_2Q = 3            # two-qubit gates per SWAP on sparse devices
GATE_TERM_WEIGHT = 0.1      # λ: relative error weight of extra 1Q gates
VF2_CALL_LIMIT = 20_000     # bound the exact-embedding search

# Gate-translation rules from the neutral {u, cx} basis to a native gate set:
#   one_q_factor: native 1Q gates per generic 1Q rotation (u)
#   extra_1q_per_cx: extra native 1Q gates per CX when the native 2Q gate is not CX
NATIVE_RULES = {
    "cx":  {"one_q_factor": 2.5, "extra_1q_per_cx": 0.0},
    "ecr": {"one_q_factor": 2.5, "extra_1q_per_cx": 4.0},
    "cz":  {"one_q_factor": 2.5, "extra_1q_per_cx": 4.0},
    "ms":  {"one_q_factor": 2.0, "extra_1q_per_cx": 4.0},
    "rzz": {"one_q_factor": 2.5, "extra_1q_per_cx": 4.0},
}
DEFAULT_RULE = {"one_q_factor": 2.5, "extra_1q_per_cx": 4.0}


# ─────────────────────────────────────────────────────────────────────────────
# Circuit profile
# ─────────────────────────────────────────────────────────────────────────────

def load_circuit(qasm: str):
    """Parse an OpenQASM 2 (or 3, if qiskit_qasm3_import is installed) string."""
    from qiskit import qasm2
    text = qasm.lstrip()
    if text.startswith("OPENQASM 3"):
        from qiskit import qasm3
        return qasm3.loads(qasm)
    return qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def to_neutral_basis(circuit):
    """Unroll to {u, cx} without routing (cheap, topology-independent)."""
    from qiskit import transpile
    return transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)


@dataclass
class CircuitStats:
    num_qubits: int                 # active (used) qubits
    depth: int
    num_two_qubit_gates: int
    num_single_qubit_gates: int
    interaction: Dict[Edge, int] = field(default_factory=dict)  # logical edges
    two_qubit_sequence: List[Edge] = field(default_factory=list)   # gate order
    one_qubit_counts: Dict[int, int] = field(default_factory=dict)  # per logical qubit


def circuit_stats(circuit) -> CircuitStats:
    """Statistics and interaction graph of a circuit in the neutral basis."""
    c = to_neutral_basis(circuit)
    index = {q: i for i, q in enumerate(c.qubits)}
    used, interaction = set(), {}
    sequence, one_q = [], {}
    n1 = n2 = 0
    for inst in c.data:
        name = inst.operation.name
        if name in ("barrier", "measure", "reset", "delay"):
            for q in inst.qubits:
                if name != "barrier":
                    used.add(index[q])
            continue
        qs = [index[q] for q in inst.qubits]
        used.update(qs)
        if len(qs) == 1:
            n1 += 1
            one_q[qs[0]] = one_q.get(qs[0], 0) + 1
        elif len(qs) == 2:
            n2 += 1
            e = tuple(sorted(qs))
            interaction[e] = interaction.get(e, 0) + 1
            sequence.append(e)
    # Re-index active qubits densely
    remap = {q: i for i, q in enumerate(sorted(used))}
    interaction = {(remap[a], remap[b]): w for (a, b), w in interaction.items()}
    sequence = [(remap[a], remap[b]) for a, b in sequence]
    one_q = {remap[q]: k for q, k in one_q.items()}
    depth = c.depth(lambda inst: inst.operation.name not in ("barrier",))
    return CircuitStats(num_qubits=max(len(used), 1), depth=depth,
                        num_two_qubit_gates=n2, num_single_qubit_gates=n1,
                        interaction=interaction, two_qubit_sequence=sequence,
                        one_qubit_counts=one_q)


# ─────────────────────────────────────────────────────────────────────────────
# Divergence estimate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DivergenceEstimate:
    swaps: float                    # S: estimated SWAP insertions
    added_two_qubit_gates: float    # ΔN_2q = k_swap · S
    gate_overhead: float            # r_gate (≥ 1)
    score: float                    # D(c, q) as defined above
    swap_ratio: float               # S / N_2q (cost-fidelity model input)
    exact_embedding: bool           # VF2 found a SWAP-free placement
    placement: Dict[int, int] = field(default_factory=dict)
    static_swaps: Optional[float] = None    # S with the placement held fixed
    esp: Optional[float] = None             # per-qubit ESP (gates + readout), if errors known
    swap_neglog: float = 0.0                # −log fidelity contributed by inserted SWAPs
    esp_neglog: Optional[float] = None      # −log(esp), kept to avoid underflow

    def esp_calibrated(self, alpha: float) -> Optional[float]:
        """ESP with the SWAP contribution scaled by a calibration factor α (log space)."""
        if self.esp_neglog is None:
            return None
        return float(np.exp(-(self.esp_neglog - (1.0 - alpha) * self.swap_neglog)))


def _distance_matrix(num_physical: int, edges: Sequence[Edge]) -> np.ndarray:
    import rustworkx as rx
    g = rx.PyGraph()
    g.add_nodes_from(range(num_physical))
    g.add_edges_from_no_data([tuple(e) for e in edges])
    d = rx.distance_matrix(g, null_value=np.inf)
    np.fill_diagonal(d, 0)
    return d


def _vf2_embedding(interaction: Dict[Edge, int], n_logical: int,
                   num_physical: int, edges: Sequence[Edge],
                   cost_fn=None, max_mappings: int = 64) -> Optional[Dict[int, int]]:
    """
    SWAP-free placement via subgraph isomorphism, if one is found quickly.
    With cost_fn, up to max_mappings embeddings are scored and the cheapest
    (lowest error cost) is returned.
    """
    import rustworkx as rx
    if not interaction:
        return {i: i for i in range(n_logical)}
    gq = rx.PyGraph()
    gq.add_nodes_from(range(num_physical))
    gq.add_edges_from_no_data([tuple(e) for e in edges])
    gc = rx.PyGraph()
    gc.add_nodes_from(range(n_logical))
    gc.add_edges_from_no_data(list(interaction.keys()))
    best, best_cost = None, None
    try:
        it = rx.vf2_mapping(gq, gc, subgraph=True, induced=False, call_limit=VF2_CALL_LIMIT)
        for k, mapping in enumerate(it):
            # rustworkx returns {physical: logical}
            cand = {log: phys for phys, log in mapping.items()}
            if cost_fn is None:
                return cand
            c = cost_fn(cand)
            if best is None or c < best_cost:
                best, best_cost = cand, c
            if k + 1 >= max_mappings:
                break
    except Exception:  # pragma: no cover - defensive
        pass
    return best


def _greedy_placement(interaction: Dict[Edge, int], n_logical: int,
                      dist: np.ndarray) -> Dict[int, int]:
    """Place heavily-interacting logical qubits on nearby physical qubits."""
    n_phys = dist.shape[0]
    weight_deg = np.zeros(n_logical)
    nbrs: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(n_logical)}
    for (a, b), w in interaction.items():
        weight_deg[a] += w
        weight_deg[b] += w
        nbrs[a].append((b, w))
        nbrs[b].append((a, w))

    finite = np.where(np.isinf(dist), n_phys, dist)
    # Start on the most central physical qubit
    start = int(np.argmin(finite.sum(axis=1)))
    placement: Dict[int, int] = {}
    free = set(range(n_phys))

    order = list(np.argsort(-weight_deg))
    for log in order:
        log = int(log)
        placed_nbrs = [(placement[v], w) for v, w in nbrs[log] if v in placement]
        if not placement:
            phys = start
        elif not placed_nbrs:
            # Unconnected so far: closest free qubit to the placed set
            placed = list(placement.values())
            phys = min(free, key=lambda p: finite[p, placed].min())
        else:
            phys = min(free, key=lambda p: sum(w * finite[p, q] for q, w in placed_nbrs))
        placement[log] = phys
        free.discard(phys)
    return placement


class ErrorModel:
    """Per-qubit / per-pair error rates with device-mean fallbacks."""

    def __init__(self, error_map: Optional[dict], mean_1q: float, mean_2q: float,
                 mean_ro: float):
        em = error_map or {}
        self.e1 = {int(k): float(v) for k, v in em.get("sq", {}).items()}
        self.e2 = {}
        for k, v in em.get("tq", {}).items():
            a, b = (map(int, k.split("-")) if isinstance(k, str) else k)
            self.e2[tuple(sorted((a, b)))] = float(v)
        self.ero = {int(k): float(v) for k, v in em.get("ro", {}).items()}
        self.m1, self.m2, self.mro = mean_1q, mean_2q, mean_ro

    @staticmethod
    def _nl(e):          # negative log fidelity, robust to e >= 1
        return -np.log(max(1.0 - min(e, 0.999999), 1e-12))

    def c1(self, q):
        return self._nl(self.e1.get(q, self.m1))

    def c2(self, a, b):
        return self._nl(self.e2.get((min(a, b), max(a, b)), self.m2))

    def cro(self, q):
        return self._nl(self.ero.get(q, self.mro))


def _placement_cost(placement, stats: CircuitStats, errors: ErrorModel) -> float:
    cost = sum(w * errors.c2(placement[a], placement[b])
               for (a, b), w in stats.interaction.items())
    cost += sum(k * errors.c1(placement[q]) for q, k in stats.one_qubit_counts.items())
    cost += sum(errors.cro(placement[q]) for q in range(stats.num_qubits))
    return cost


def _route(stats: CircuitStats, placement: Dict[int, int], dist: np.ndarray,
           adjacency: Dict[int, List[int]], errors: Optional[ErrorModel],
           one_q_factor: float) -> Tuple[float, Optional[float], float]:
    """
    Greedy routing replay. Returns (swaps, esp or None). The qubit with the
    lower remaining distance contribution is moved one step at a time along a
    shortest path (preferring low-error couplers when errors are known).
    """
    pos = dict(placement)
    occ = {p: q for q, p in pos.items()}
    swaps = 0
    nl = 0.0
    nl_swap = 0.0
    for a, b in stats.two_qubit_sequence:
        while dist[pos[a], pos[b]] > 1:
            u, target = pos[a], pos[b]
            d = dist[u, target]
            steps = [w for w in adjacency[u] if dist[w, target] == d - 1]
            if errors is not None:
                w = min(steps, key=lambda w: errors.c2(u, w))
                nl += 3 * errors.c2(u, w)
                nl_swap += 3 * errors.c2(u, w)
            else:
                w = steps[0]
            other = occ.get(w)
            occ[w], pos[a] = a, w
            if other is not None:
                occ[u], pos[other] = other, u
            else:
                occ.pop(u, None)
            swaps += 1
        if errors is not None:
            nl += errors.c2(pos[a], pos[b])
    if errors is None:
        return float(swaps), None, 0.0
    # 1Q gates on their initial qubits; readout on final positions
    nl += sum(k * one_q_factor * errors.c1(placement[q])
              for q, k in stats.one_qubit_counts.items())
    nl += sum(errors.cro(pos[q]) for q in range(stats.num_qubits))
    return float(swaps), float(nl), float(nl_swap)


def estimate_divergence(
    stats: CircuitStats,
    coupling_map: Optional[Iterable[Iterable[int]]],
    num_physical: int,
    native_gates: Optional[Sequence[str]] = None,
    gate_term_weight: float = GATE_TERM_WEIGHT,
    routing: str = "static",
    errors: Optional[ErrorModel] = None,
) -> DivergenceEstimate:
    """
    Graph-based topology divergence of a circuit on a candidate QPU.

    Args:
        routing: "static" (default) or "dynamic" (greedy routing replay).
        errors:  optional per-qubit/per-pair error model; enables noise-aware
                 embedding selection and the per-qubit ESP estimate.
    """
    from topology import is_all_to_all, normalize_edges

    edges = normalize_edges(coupling_map or [])
    n_log = stats.num_qubits
    n2 = stats.num_two_qubit_gates

    # 5. Gate-set translation overhead
    native = {g.lower() for g in (native_gates or [])}
    two_q_native = next((g for g in ("cx", "ecr", "cz", "ms", "rzz") if g in native), None)
    rule = NATIVE_RULES.get(two_q_native, DEFAULT_RULE)
    n1 = stats.num_single_qubit_gates
    total = max(n1 + n2, 1)
    r_gate = (n1 * rule["one_q_factor"] + n2 * (1 + rule["extra_1q_per_cx"])) / total
    r_gate = max(1.0, r_gate)

    all_to_all = (not edges and native_gates is not None and "any" in native) or \
        is_all_to_all(edges, num_physical)
    if n_log > num_physical:
        raise ValueError(f"circuit needs {n_log} qubits, QPU has {num_physical}")
    esp_val = None
    swap_nl = 0.0
    if all_to_all or n2 == 0:
        if errors is not None and edges:
            cost_fn = lambda pl: _placement_cost(pl, stats, errors)   # noqa: E731
            # choose the lowest-error qubits for an all-to-all device
            order = sorted(range(num_physical), key=lambda q: errors.c1(q) + errors.cro(q))
            placement = {i: order[i] for i in range(n_log)}
        else:
            placement = {i: i for i in range(n_log)}
        swaps = static_swaps = 0.0
        exact = True
        if errors is not None:
            esp_val = float(_placement_cost(placement, stats, errors)
                            + sum((rule["one_q_factor"] - 1) * k * errors.c1(placement[q])
                                  for q, k in stats.one_qubit_counts.items()))   # neglog
    else:
        dist = _distance_matrix(num_physical, edges)
        cost_fn = (lambda pl: _placement_cost(pl, stats, errors)) if errors is not None else None
        placement = _vf2_embedding(stats.interaction, n_log, num_physical, edges, cost_fn)
        exact = placement is not None
        if not exact:
            placement = _greedy_placement(stats.interaction, n_log, dist)
        static_swaps = float(sum(
            w * max(0.0, dist[placement[a], placement[b]] - 1)
            for (a, b), w in stats.interaction.items()))
        if np.isinf(static_swaps):
            raise ValueError("coupling map is disconnected for this placement")
        dyn_swaps = static_swaps
        if (routing == "dynamic" or errors is not None) and stats.two_qubit_sequence:
            adjacency: Dict[int, List[int]] = {}
            for a, b in edges:
                adjacency.setdefault(a, []).append(b)
                adjacency.setdefault(b, []).append(a)
            dyn_swaps, esp_val, swap_nl = _route(stats, placement, dist, adjacency, errors,
                                                 rule["one_q_factor"])
        # Dynamic routing needs the gate order; without it fall back to static
        use_dynamic = routing == "dynamic" and bool(stats.two_qubit_sequence)
        swaps = dyn_swaps if use_dynamic else static_swaps

    k_swap = 0 if all_to_all else SWAP_COST_2Q
    added_2q = k_swap * swaps
    score = added_2q / max(n2, 1) + gate_term_weight * (r_gate - 1.0)
    return DivergenceEstimate(
        swaps=swaps, added_two_qubit_gates=added_2q, gate_overhead=r_gate,
        score=score, swap_ratio=swaps / max(n2, 1), exact_embedding=exact,
        placement=placement, static_swaps=static_swaps,
        esp=float(np.exp(-esp_val)) if esp_val is not None else None,
        swap_neglog=swap_nl, esp_neglog=esp_val,
    )
