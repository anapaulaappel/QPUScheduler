"""
Topology helpers
----------------
Coupling maps for QPU topology families, as undirected edge lists.

Real backends report their own coupling map (IBM: configuration.coupling_map).
Simulated backends use a representative graph for their topology family, so
that graph-based scheduling logic (topology divergence, retranspilation) can be
exercised without hardware.
"""

import math
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

Edge = Tuple[int, int]

ALL_TO_ALL_FAMILIES = {"all-to-all", "trapped-ion"}


def normalize_edges(edges: Iterable[Iterable[int]]) -> List[Edge]:
    """Undirected, de-duplicated, sorted edge list."""
    out = {tuple(sorted((int(a), int(b)))) for a, b in edges if int(a) != int(b)}
    return sorted(out)


def _connected_truncation(edges, num_qubits: int) -> List[Edge]:
    """
    Keep a CONNECTED region of `num_qubits` nodes (BFS order from node 0) and
    relabel it 0..num_qubits-1. Truncating by node index instead can split a
    lattice into disconnected pieces.
    """
    from collections import deque
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    if not adj:
        return []
    start = min(adj)
    order, seen, queue = [], {start}, deque([start])
    while queue and len(order) < num_qubits:
        v = queue.popleft()
        order.append(v)
        for w in sorted(adj[v]):
            if w not in seen:
                seen.add(w)
                queue.append(w)
    keep = {v: i for i, v in enumerate(order)}
    return [(keep[a], keep[b]) for a, b in edges if a in keep and b in keep]


@lru_cache(maxsize=64)
def _family_edges(family: str, num_qubits: int) -> Tuple[Edge, ...]:
    from qiskit.transpiler import CouplingMap

    family = family.lower()
    if family in ALL_TO_ALL_FAMILIES:
        edges = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
    elif family in ("heavy-hex", "falcon"):
        # Smallest odd code distance whose heavy-hex lattice has >= num_qubits
        d = 3
        while (5 * d * d - 2 * d - 1) // 2 < num_qubits:
            d += 2
        cmap = CouplingMap.from_heavy_hex(d, bidirectional=False)
        edges = _connected_truncation(cmap.get_edges(), num_qubits)
    elif family == "square-lattice":
        rows = max(1, int(math.floor(math.sqrt(num_qubits))))
        cols = math.ceil(num_qubits / rows)
        cmap = CouplingMap.from_grid(rows, cols, bidirectional=False)
        edges = _connected_truncation(cmap.get_edges(), num_qubits)
    else:
        # Unknown family: a line is the most conservative assumption
        edges = [(i, i + 1) for i in range(num_qubits - 1)]
    return tuple(normalize_edges(edges))


def family_coupling_map(family: str, num_qubits: int) -> List[Edge]:
    """Representative undirected coupling map for a topology family."""
    return list(_family_edges(family, num_qubits))


def is_all_to_all(edges: Optional[List[Edge]], num_qubits: int) -> bool:
    if not edges:
        return False
    return len(normalize_edges(edges)) >= num_qubits * (num_qubits - 1) // 2
