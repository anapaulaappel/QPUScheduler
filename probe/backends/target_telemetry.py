"""
Telemetry from a Qiskit BackendV2 Target
---------------------------------------
Builds a QPUTelemetry snapshot from any Qiskit BackendV2 (IBM fake backends
with real calibration snapshots, GenericBackendV2, or real backends), using
only the Target API. Used by the experiments to obtain realistic telemetry
for many devices without network access.
"""

import statistics
from datetime import datetime, timezone
from typing import Optional

from backends.simulator_backend import QPUTelemetry
from topology import normalize_edges

ONE_Q = ("sx", "x", "rx", "ry", "gpi2", "gpi")
TWO_Q = ("cz", "ecr", "cx", "rxx", "ms", "rzz")


def _errors(target, names, arity):
    errs, durs = [], []
    for name in names:
        if name not in target.operation_names:
            continue
        for qargs, props in target[name].items():
            if props is None or qargs is None or len(qargs) != arity:
                continue
            if props.error is not None and 0 <= props.error < 1:
                errs.append(props.error)
            if props.duration:
                durs.append(props.duration)
    return errs, durs


def error_map_from_target(target) -> dict:
    """Per-qubit / per-pair error rates from a Target (keys as in QPUTelemetry.error_map)."""
    sq, tq, ro = {}, {}, {}
    for name in target.operation_names:
        for qargs, props in target[name].items():
            if props is None or qargs is None or props.error is None:
                continue
            if name in ONE_Q and len(qargs) == 1:
                q = qargs[0]
                sq[q] = max(sq.get(q, 0.0), props.error)
            elif name in TWO_Q and len(qargs) == 2:
                a, b = sorted(qargs)
                key = f"{a}-{b}"
                tq[key] = min(tq.get(key, 1.0), props.error)    # best direction
            elif name == "measure" and len(qargs) == 1:
                ro[qargs[0]] = props.error
    return {"sq": sq, "tq": tq, "ro": ro}


def telemetry_from_backend(backend, name: Optional[str] = None, topology: str = "heavy-hex",
                           provider: str = "experiment", pricing: Optional[dict] = None,
                           now: Optional[datetime] = None) -> QPUTelemetry:
    now = now or datetime.now(timezone.utc)
    target = backend.target
    e1, d1 = _errors(target, ONE_Q, 1)
    e2, d2 = _errors(target, TWO_Q, 2)
    ero, dm = _errors(target, ("measure",), 1)
    t1s, t2s = [], []
    for q in range(backend.num_qubits):
        qp = target.qubit_properties[q] if target.qubit_properties else None
        if qp is not None:
            if qp.t1:
                t1s.append(qp.t1 * 1e6)
            if qp.t2:
                t2s.append(qp.t2 * 1e6)
    edges = normalize_edges(target.build_coupling_map().get_edges()) \
        if target.build_coupling_map() is not None else []
    fid = [1 - e for e in e1] or [1.0]
    mean = lambda xs, d=0.0: statistics.mean(xs) if xs else d   # noqa: E731
    durations = {}
    if d1: durations["gate_1q_ns"] = mean(d1) * 1e9
    if d2: durations["gate_2q_ns"] = mean(d2) * 1e9
    if dm: durations["measure_ns"] = mean(dm) * 1e9
    durations["reset_us"] = 250.0
    native = sorted(n for n in target.operation_names
                    if n not in ("measure", "reset", "delay", "if_else", "switch_case",
                                 "for_loop", "while_loop"))
    return QPUTelemetry(
        backend_name=name or backend.name, provider=provider,
        qubit_count=backend.num_qubits, topology=topology,
        is_simulator=True, is_simulator_fallback=False,
        mean_qubit_fidelity=mean(fid, 1.0), min_qubit_fidelity=min(fid),
        max_qubit_fidelity=max(fid),
        mean_t1_us=mean(t1s), min_t1_us=min(t1s) if t1s else 0.0,
        mean_t2_us=mean(t2s), min_t2_us=min(t2s) if t2s else 0.0,
        mean_sq_error=mean(e1), max_sq_error=max(e1) if e1 else 0.0,
        mean_tq_error=mean(e2), max_tq_error=max(e2) if e2 else 0.0,
        mean_ro_error=mean(ero),
        queue_depth=0, estimated_queue_wait_minutes=0.0,
        cost_per_shot=float((pricing or {}).get("price_per_shot", 0.0)),
        native_gates=native,
        last_updated=now.isoformat(), calibration_timestamp=now.isoformat(),
        calibration_age_minutes=0.0, available=True,
        coupling_map=[list(e) for e in edges], pricing=dict(pricing or {}),
        durations=durations, error_map=error_map_from_target(target),
    )
