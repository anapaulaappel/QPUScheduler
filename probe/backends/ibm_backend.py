"""
IBM Quantum Backend
-------------------
Fetches real QPU telemetry from IBM Quantum via qiskit-ibm-runtime.

IMPORTANT: Only used when PROBE_MODE=real.
Each call to fetch() does NOT submit any circuits — it only reads
calibration data from the IBM API (free, does not consume QPU minutes).
QPU minutes are only consumed when you actually RUN a circuit.
"""

import logging
import statistics
from datetime import datetime, timezone
from typing import Optional

from backends.simulator_backend import QPUTelemetry
from topology import normalize_edges

logger = logging.getLogger(__name__)

# Lazy import — only loaded when PROBE_MODE=real
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    logger.warning("[IBMBackend] qiskit-ibm-runtime not installed. "
                   "Run: pip install qiskit-ibm-runtime")


# IBM processor families and their qubit connectivity
IBM_FAMILY_TOPOLOGY = {
    "falcon": "heavy-hex", "hummingbird": "heavy-hex", "eagle": "heavy-hex",
    "osprey": "heavy-hex", "heron": "heavy-hex",
    "nighthawk": "square-lattice",
}


def ibm_topology_family(processor_type) -> str:
    """
    Map IBM's processor_type metadata to a scheduler topology-family label.
    Unknown or missing metadata defaults to 'heavy-hex', IBM's standard
    connectivity for most current devices.
    """
    family = None
    if isinstance(processor_type, dict):
        family = processor_type.get("family")
    elif processor_type is not None:
        family = getattr(processor_type, "family", None) or str(processor_type)
    key = str(family).strip().lower() if family else ""
    if key in IBM_FAMILY_TOPOLOGY:
        return IBM_FAMILY_TOPOLOGY[key]
    if family:
        logger.warning(f"[IBMBackend] Unknown processor family '{family}', "
                       f"assuming heavy-hex topology")
    return "heavy-hex"


class IBMBackend:
    """
    Fetches real telemetry from IBM Quantum.

    Quota note:
    - Reading backend properties (calibration data) is FREE and does not
      consume any QPU time.
    - QPU time is only consumed when you call backend.run() with a circuit.
    - This probe NEVER calls backend.run() — safe to use anytime.
    """

    def __init__(self, token: Optional[str] = None, instance: Optional[str] = None,
                 preferred_backends: Optional[list] = None,
                 channel: str = "ibm_quantum_platform",
                 pricing: Optional[dict] = None):
        """
        Args:
            token: API key. If empty, the saved account is used
                   (QiskitRuntimeService.save_account()).
            instance: optional instance CRN / name.
            preferred_backends: backend names to monitor; empty = all operational.
            channel: Qiskit Runtime channel (the legacy "ibm_quantum" channel
                     has been retired).
            pricing: pricing model attached to the telemetry (see config.IBM_PRICING).
        """
        if not IBM_AVAILABLE:
            raise RuntimeError("qiskit-ibm-runtime not installed. "
                               "Run: pip install qiskit-ibm-runtime")
        self.token = token or None
        self.instance = instance or None
        self.channel = channel
        self.preferred_backends = list(preferred_backends or [])
        self.pricing = dict(pricing or {})
        self._service = None
        self._backend_cache = {}
        logger.info("[IBMBackend] Initialized (real hardware mode)")
        logger.warning("[IBMBackend] Reading calibration data only — no QPU minutes consumed.")

    def _get_service(self) -> "QiskitRuntimeService":
        if self._service is None:
            logger.info("[IBMBackend] Connecting to IBM Quantum...")
            if self.token:
                self._service = QiskitRuntimeService(
                    channel=self.channel, token=self.token, instance=self.instance)
            else:
                # Saved account (~/.qiskit/qiskit-ibm.json)
                self._service = QiskitRuntimeService(instance=self.instance) \
                    if self.instance else QiskitRuntimeService()
            logger.info("[IBMBackend] Connected.")
        return self._service

    def list_available_backends(self) -> list:
        """Returns list of operational IBM backends you have access to."""
        svc = self._get_service()
        backends = svc.backends(operational=True, simulator=False)
        return [b.name for b in backends]

    def fetch(self, backend_name: str) -> Optional[QPUTelemetry]:
        """
        Fetch real calibration data for a specific IBM backend.
        This is READ-ONLY — no QPU minutes are consumed.
        """
        try:
            svc = self._get_service()
            backend = svc.backend(backend_name)
            props = backend.properties()
            config = backend.configuration()
            status = backend.status()

            if props is None:
                logger.warning(f"[IBMBackend] No calibration data for {backend_name}")
                return self._unavailable(backend_name)

            now = datetime.now(timezone.utc)

            # ── Extract per-qubit T1 / T2 ─────────────────────────────────
            t1_values, t2_values = [], []
            for qubit_idx in range(config.n_qubits):
                try:
                    t1 = props.t1(qubit_idx) * 1e6  # seconds -> microseconds
                    t2 = props.t2(qubit_idx) * 1e6
                    if t1 > 0:
                        t1_values.append(t1)
                    if t2 > 0:
                        t2_values.append(t2)
                except Exception:
                    pass

            mean_t1 = statistics.mean(t1_values) if t1_values else 0.0
            min_t1  = min(t1_values) if t1_values else 0.0
            mean_t2 = statistics.mean(t2_values) if t2_values else 0.0
            min_t2  = min(t2_values) if t2_values else 0.0

            # ── Extract single-qubit gate errors ──────────────────────────
            sq_errors = []
            for qubit_idx in range(config.n_qubits):
                for gate in ["sx", "x", "rz"]:
                    try:
                        err = props.gate_error(gate, qubit_idx)
                        if 0 < err < 1:
                            sq_errors.append(err)
                    except Exception:
                        pass

            mean_sq = statistics.mean(sq_errors) if sq_errors else 0.0
            max_sq  = max(sq_errors) if sq_errors else 0.0

            # ── Extract two-qubit gate errors ─────────────────────────────
            tq_errors = []
            cx_pairs = config.coupling_map if hasattr(config, 'coupling_map') else []
            for pair in cx_pairs:
                for gate in ["ecr", "cx", "cz"]:
                    try:
                        err = props.gate_error(gate, pair)
                        if 0 < err < 1:
                            tq_errors.append(err)
                    except Exception:
                        pass

            mean_tq = statistics.mean(tq_errors) if tq_errors else 0.0
            max_tq  = max(tq_errors) if tq_errors else 0.0

            # ── Readout errors ────────────────────────────────────────────
            ro_errors = []
            for qubit_idx in range(config.n_qubits):
                try:
                    err = props.readout_error(qubit_idx)
                    if 0 < err < 1:
                        ro_errors.append(err)
                except Exception:
                    pass
            mean_ro = statistics.mean(ro_errors) if ro_errors else 0.0

            # ── Estimate fidelity from error rates ────────────────────────
            # Mean single-qubit gate fidelity: F_hw = mean(1 - e_i) over the
            # reported single-qubit gate errors (sx, x, rz) of all qubits.
            fidelity_per_qubit = [1.0 - e for e in sq_errors if e > 0]
            if fidelity_per_qubit:
                mean_fidelity = statistics.mean(fidelity_per_qubit)
                min_fidelity  = min(fidelity_per_qubit)
                max_fidelity  = max(fidelity_per_qubit)
            else:
                mean_fidelity = min_fidelity = max_fidelity = 0.0

            # ── Calibration age ───────────────────────────────────────────
            cal_time = props.last_update_date
            if cal_time.tzinfo is None:
                cal_time = cal_time.replace(tzinfo=timezone.utc)
            cal_age_minutes = (now - cal_time).total_seconds() / 60.0

            # ── Queue depth ───────────────────────────────────────────────
            queue_depth = status.pending_jobs if hasattr(status, 'pending_jobs') else 0

            # ── Native gates ──────────────────────────────────────────────
            native_gates = list(config.basis_gates) if hasattr(config, 'basis_gates') else []

            # ── Operation durations (for QPU-time estimates) ──────────────
            durations = self._durations(backend, config, cx_pairs)

            # ── Topology ──────────────────────────────────────────────────
            # Normalize to the topology-family labels used by the scheduler
            # (processor_type is e.g. {"family": "Heron", "revision": "2"}).
            processor_type = getattr(config, 'processor_type', None)
            topology = ibm_topology_family(processor_type)

            telemetry = QPUTelemetry(
                backend_name=backend_name,
                provider="ibm-quantum",
                qubit_count=config.n_qubits,
                topology=topology,
                is_simulator=False,
                is_simulator_fallback=False,
                mean_qubit_fidelity=round(mean_fidelity, 5),
                min_qubit_fidelity=round(min_fidelity, 5),
                max_qubit_fidelity=round(max_fidelity, 5),
                mean_t1_us=round(mean_t1, 2),
                min_t1_us=round(min_t1, 2),
                mean_t2_us=round(mean_t2, 2),
                min_t2_us=round(min_t2, 2),
                mean_sq_error=round(mean_sq, 6),
                max_sq_error=round(max_sq, 6),
                mean_tq_error=round(mean_tq, 6),
                max_tq_error=round(max_tq, 6),
                mean_ro_error=round(mean_ro, 6),
                queue_depth=queue_depth,
                estimated_queue_wait_minutes=round(queue_depth * 1.5, 1),
                cost_per_shot=0.0,  # IBM open plan: free (quota-limited)
                native_gates=native_gates,
                last_updated=now.isoformat(),
                calibration_timestamp=cal_time.isoformat(),
                calibration_age_minutes=round(cal_age_minutes, 1),
                available=status.operational,
                raw={
                    "mode": "real",
                    "n_qubits": config.n_qubits,
                    "processor_type": processor_type,
                    "status_msg": status.status_msg if hasattr(status, 'status_msg') else "",
                    "pending_jobs": queue_depth,
                },
                coupling_map=[list(e) for e in normalize_edges(cx_pairs)],
                pricing=dict(self.pricing),
                durations=durations,
                error_map=self._error_map(backend),
            )

            logger.info(f"[IBMBackend] {backend_name}: fidelity={mean_fidelity:.4f} "
                        f"tq_error={mean_tq:.4f} queue={queue_depth} "
                        f"cal_age={cal_age_minutes:.0f}min")
            return telemetry

        except Exception as e:
            logger.error(f"[IBMBackend] Failed to fetch {backend_name}: {e}")
            return self._unavailable(backend_name)

    @staticmethod
    def _durations(backend, config, pairs) -> dict:
        """Mean 1Q/2Q gate, measurement and reset (rep delay) durations."""
        def mean_duration(gate_names, qargs_list):
            vals = []
            target = getattr(backend, "target", None)
            if target is None:
                return None
            for g in gate_names:
                if g not in target:
                    continue
                for qargs in qargs_list:
                    props = target[g].get(tuple(qargs))
                    if props is not None and props.duration:
                        vals.append(props.duration)
            return statistics.mean(vals) * 1e9 if vals else None   # s -> ns

        qubits = [(q,) for q in range(config.n_qubits)]
        out = {}
        g1 = mean_duration(["sx", "x"], qubits)
        g2 = mean_duration(["cz", "ecr", "cx"], pairs)
        ms = mean_duration(["measure"], qubits)
        if g1: out["gate_1q_ns"] = round(g1, 1)
        if g2: out["gate_2q_ns"] = round(g2, 1)
        if ms: out["measure_ns"] = round(ms, 1)
        rep_delay = getattr(config, "default_rep_delay", None)
        if rep_delay:
            out["reset_us"] = round(rep_delay * 1e6, 1)       # s -> us
        return out

    @staticmethod
    def _error_map(backend) -> dict:
        try:
            from backends.target_telemetry import error_map_from_target
            return error_map_from_target(backend.target)
        except Exception as e:      # pragma: no cover - defensive
            logger.debug(f"[IBMBackend] no per-qubit errors: {e}")
            return {}

    def backend_names(self) -> list:
        """Configured backends, or all operational backends of the account."""
        return self.preferred_backends or self.list_available_backends()

    def fetch_all(self) -> list:
        """Fetch telemetry for every monitored backend (read-only)."""
        return [t for t in (self.fetch(n) for n in self.backend_names()) if t]

    def fetch_best_available(self) -> Optional[QPUTelemetry]:
        """
        Try preferred backends in order and return the first available one.
        Falls back gracefully if a backend is offline or inaccessible.
        """
        for name in self.backend_names():
            logger.info(f"[IBMBackend] Trying backend: {name}")
            telemetry = self.fetch(name)
            if telemetry and telemetry.available:
                return telemetry
            logger.warning(f"[IBMBackend] {name} unavailable, trying next...")
        logger.error("[IBMBackend] No preferred backends available.")
        return None

    def _unavailable(self, backend_name: str) -> QPUTelemetry:
        """Return an unavailable telemetry record when fetch fails."""
        now = datetime.now(timezone.utc).isoformat()
        return QPUTelemetry(
            backend_name=backend_name,
            provider="ibm-quantum",
            qubit_count=0, topology="unknown",
            is_simulator=False, is_simulator_fallback=False,
            mean_qubit_fidelity=0.0, min_qubit_fidelity=0.0, max_qubit_fidelity=0.0,
            mean_t1_us=0.0, min_t1_us=0.0,
            mean_t2_us=0.0, min_t2_us=0.0,
            mean_sq_error=0.0, max_sq_error=0.0,
            mean_tq_error=0.0, max_tq_error=0.0,
            mean_ro_error=0.0,
            queue_depth=999, estimated_queue_wait_minutes=999.0,
            cost_per_shot=0.0, native_gates=[],
            last_updated=now, calibration_timestamp=now,
            calibration_age_minutes=9999.0,
            available=False,
            raw={"mode": "real", "error": "fetch_failed"}
        )
