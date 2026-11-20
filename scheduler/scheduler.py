"""
Quantum-Aware Scheduler
------------------------
Implements the Filter → Score → Select pipeline.

In production this runs as a Kubernetes Scheduling Framework plugin (Go).
Here it runs as a Python service called by the demo notebook.

Input:
  - job: JobRequest  (scheduling requirements)
  - backends: List[QPUTelemetry]  (current state of all QPU backends from CRDs)

Output:
  - SchedulingDecision  (selected backend + full audit trail)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Collection, List, Optional

from job_request import JobRequest
from cost_fidelity_model import CostFidelityModel, estimate_overhead, topology_divergence
from usage import UsageEstimate, estimate_usage

# Import QPUTelemetry from probe (shared dataclass)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'probe'))
from backends.simulator_backend import QPUTelemetry, telemetry_age_minutes
from config import STALENESS_THRESHOLD_MINUTES

logger = logging.getLogger(__name__)

# Fixed reference ranges used to normalize each score term to [0, 1].
# Grading against fixed ranges (rather than relative to the other candidates)
# keeps scores meaningful even when only one candidate passes the Filter.
SCORE_RANGES = {
    "fidelity":      (0.95, 1.0),
    "tq_error":      (0.0, 0.05),
    "t2_us":         (20.0, 500.0),
    "queue_depth":   (0, 100),
    "cal_age_min":   (0, 720),
    "cost_per_shot": (0.0, 0.01),   # effective USD per shot (C_eff / shots)
}

# Names of the individual Filter checks (can be skipped selectively, e.g. when
# evaluating a projected future state for deferral).
FILTER_CHECKS = (
    "available", "staleness", "qubits", "fidelity", "gate_error",
    "readout_error", "queue", "calibration_age", "cost", "post_transpilation",
)


@dataclass
class FilterResult:
    """Result of the Filter phase for one backend."""
    backend: QPUTelemetry
    passed: bool
    rejection_reason: Optional[str] = None
    predicted_post_transpilation_fidelity: Optional[float] = None
    cost_fidelity_explanation: Optional[str] = None
    requires_retranspilation: bool = False
    divergence_estimate: Optional[object] = None   # circuit_analysis.DivergenceEstimate
    usage: Optional[UsageEstimate] = None          # estimated usage/cost on this QPU


@dataclass
class ScoreResult:
    """Result of the Score phase for one backend."""
    backend: QPUTelemetry
    composite_score: float          # final score in [0, 100]
    predicted_fidelity: float
    score_breakdown: dict = field(default_factory=dict)


@dataclass
class SchedulingDecision:
    """Full output of the scheduling pipeline."""
    job_id: str
    selected_backend: Optional[QPUTelemetry]
    selected_score: Optional[ScoreResult]
    is_simulator_fallback: bool
    is_unschedulable: bool
    unschedulable_reason: Optional[str]

    # Full audit trail
    filter_results: List[FilterResult] = field(default_factory=list)
    score_results: List[ScoreResult] = field(default_factory=list)


class QuantumScheduler:
    """
    Implements PreFilter → Filter → Score → Select.

    Mirrors the Kubernetes Scheduling Framework extension points.
    """

    def __init__(self, model_mode: str = "physics",
                 staleness_threshold_minutes: float = STALENESS_THRESHOLD_MINUTES):
        self.cf_model = CostFidelityModel(mode=model_mode)
        self.staleness_threshold_minutes = staleness_threshold_minutes
        self._divergence_cache: dict = {}
        logger.info(f"[Scheduler] Initialized with Cost-Fidelity model mode: {model_mode}")

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def schedule(
        self,
        job: JobRequest,
        backends: List[QPUTelemetry],
        now: Optional[datetime] = None,
    ) -> SchedulingDecision:
        """
        Main scheduling pipeline. Returns a SchedulingDecision with full audit.

        Args:
            now: evaluation time (defaults to the real clock); used for the
                 telemetry staleness check.
        """
        now = now or datetime.now(timezone.utc)
        logger.info(f"\n{'='*60}")
        logger.info(f"  SCHEDULING JOB: {job.job_id}")
        logger.info(f"  min_fidelity={job.min_fidelity}  max_cost={job.max_cost_per_shot}"
                    f"  deadline={job.deadline_minutes}min  fallback={job.allow_simulator_fallback}")
        logger.info(f"{'='*60}")

        # ── Phase 1: PreFilter — separate real QPUs from simulators ───────
        real_backends    = [b for b in backends if not b.is_simulator_fallback]
        fallback_backends = [b for b in backends if b.is_simulator_fallback]

        # ── Phase 2: Filter real QPUs ─────────────────────────────────────
        filter_results = [self.filter_backend(job, b, now=now) for b in real_backends]
        passed = [r for r in filter_results if r.passed]

        self._log_filter_summary(filter_results)

        # ── Phase 3: Score passing QPUs ───────────────────────────────────
        if passed:
            score_results = [self._score(job, r) for r in passed]
            score_results.sort(key=lambda s: s.composite_score, reverse=True)
            self._log_score_summary(score_results)

            best = score_results[0]
            logger.info(f"\n[Scheduler] SELECTED: {best.backend.backend_name} "
                        f"(score={best.composite_score:.1f}, "
                        f"predicted_fidelity={best.predicted_fidelity:.4f})")

            return SchedulingDecision(
                job_id=job.job_id,
                selected_backend=best.backend,
                selected_score=best,
                is_simulator_fallback=False,
                is_unschedulable=False,
                unschedulable_reason=None,
                filter_results=filter_results,
                score_results=score_results,
            )

        # ── Phase 4: Simulator fallback ───────────────────────────────────
        if job.allow_simulator_fallback and fallback_backends:
            fallback = fallback_backends[0]
            logger.warning(f"\n[Scheduler] No real QPU meets constraints. "
                           f"FALLBACK to simulator: {fallback.backend_name}")
            fake_score = ScoreResult(
                backend=fallback,
                composite_score=0.0,
                predicted_fidelity=1.0,
                score_breakdown={"note": "simulator_fallback"},
            )
            return SchedulingDecision(
                job_id=job.job_id,
                selected_backend=fallback,
                selected_score=fake_score,
                is_simulator_fallback=True,
                is_unschedulable=False,
                unschedulable_reason=None,
                filter_results=filter_results,
                score_results=[],
            )

        # ── Phase 5: Unschedulable ────────────────────────────────────────
        reasons = [r.rejection_reason for r in filter_results if r.rejection_reason]
        reason_str = " | ".join(set(reasons)) if reasons else "No QPU available"
        logger.error(f"\n[Scheduler] UNSCHEDULABLE: {reason_str}")

        return SchedulingDecision(
            job_id=job.job_id,
            selected_backend=None,
            selected_score=None,
            is_simulator_fallback=False,
            is_unschedulable=True,
            unschedulable_reason=reason_str,
            filter_results=filter_results,
            score_results=[],
        )

    # ──────────────────────────────────────────────────────────────────────
    # FILTER PHASE
    # ──────────────────────────────────────────────────────────────────────

    def filter_backend(
        self,
        job: JobRequest,
        backend: QPUTelemetry,
        now: Optional[datetime] = None,
        skip_checks: Collection[str] = (),
    ) -> FilterResult:
        """
        Evaluate one backend against the job's hard constraints.
        Returns FilterResult with passed=True or rejection_reason.

        Args:
            now:         evaluation time for the staleness check.
            skip_checks: names from FILTER_CHECKS to skip. Used when evaluating
                         a projected future state (deferral), where some
                         properties (e.g. queue depth) cannot be forecast and
                         are re-checked live at execution time.
        """
        b = backend
        now = now or datetime.now(timezone.utc)
        skip = set(skip_checks)
        unknown = skip - set(FILTER_CHECKS)
        if unknown:
            raise ValueError(f"Unknown filter checks: {sorted(unknown)}")

        # 1. Availability
        if "available" not in skip and not b.available:
            return FilterResult(b, False, "backend_unavailable")

        # 2. Telemetry freshness ("temporal safety"): unknown current quality
        if "staleness" not in skip:
            age = telemetry_age_minutes(b, now)
            if age > self.staleness_threshold_minutes:
                return FilterResult(b, False,
                    f"telemetry_stale({age:.0f}min>{self.staleness_threshold_minutes:.0f}min)")

        # 3. Qubit count
        if "qubits" not in skip and b.qubit_count < job.circuit.num_qubits:
            return FilterResult(b, False,
                f"insufficient_qubits({b.qubit_count}<{job.circuit.num_qubits})")

        # 4. Raw fidelity threshold
        if "fidelity" not in skip and b.mean_qubit_fidelity < job.min_fidelity:
            return FilterResult(b, False,
                f"fidelity_too_low({b.mean_qubit_fidelity:.4f}<{job.min_fidelity})")

        # 5. Gate error rate
        if "gate_error" not in skip and b.mean_tq_error > job.max_gate_error_rate:
            return FilterResult(b, False,
                f"gate_error_too_high({b.mean_tq_error:.4f}>{job.max_gate_error_rate})")

        # 6. Readout error
        if "readout_error" not in skip and b.mean_ro_error > job.max_readout_error:
            return FilterResult(b, False,
                f"readout_error_too_high({b.mean_ro_error:.4f}>{job.max_readout_error})")

        # 7. Queue depth
        if "queue" not in skip and b.queue_depth > job.max_queue_depth:
            return FilterResult(b, False,
                f"queue_too_deep({b.queue_depth}>{job.max_queue_depth})")

        # 8. Calibration age
        if "calibration_age" not in skip and \
                b.calibration_age_minutes > job.max_calibration_age_minutes:
            return FilterResult(b, False,
                f"calibration_stale({b.calibration_age_minutes:.0f}min"
                f">{job.max_calibration_age_minutes}min)")

        # 9. Cost
        if "cost" not in skip and job.max_cost_per_shot > 0 and \
                b.cost_per_shot > job.max_cost_per_shot:
            return FilterResult(b, False,
                f"cost_too_high(${b.cost_per_shot:.4f}>${job.max_cost_per_shot:.4f})")

        # 10. ── Post-transpilation fidelity prediction ─────────────
        #    If the circuit's compiled topology differs from this QPU's topology,
        #    predict whether retranspilation will still meet the fidelity threshold.
        predicted_fidelity = b.mean_qubit_fidelity  # default: no change needed
        cf_explanation = "same_topology_no_transpilation_overhead"

        needs_retranspile = job.circuit.compiled_for_topology.lower() != b.topology.lower()
        estimate = None
        if "post_transpilation" not in skip and needs_retranspile:
            # Graph-based divergence when the circuit and the QPU's coupling
            # map are known; otherwise the family-level lookup table.
            estimate = self._graph_divergence(job, b)
            passes, predicted_fidelity, cf_explanation = self.cf_model.will_meet_threshold(
                min_fidelity=job.min_fidelity,
                src_topology=job.circuit.compiled_for_topology,
                dst_topology=b.topology,
                circuit_depth=job.circuit.depth,
                num_two_qubit_gates=job.circuit.num_two_qubit_gates,
                target_mean_tq_error=b.mean_tq_error,
                target_mean_t2_us=b.mean_t2_us,
                target_mean_fidelity=b.mean_qubit_fidelity,
                divergence=estimate.swap_ratio if estimate is not None else None,
            )
            cf_explanation = ("graph-divergence: " if estimate is not None
                              else "table-divergence: ") + cf_explanation
            if not passes:
                return FilterResult(
                    backend=b,
                    passed=False,
                    rejection_reason=(
                        f"post_transpilation_fidelity_too_low"
                        f"(predicted={predicted_fidelity:.4f}<{job.min_fidelity})"
                    ),
                    predicted_post_transpilation_fidelity=predicted_fidelity,
                    cost_fidelity_explanation=cf_explanation,
                    requires_retranspilation=True,
                    divergence_estimate=estimate,
                )

        # 11. Estimated usage / effective cost of THIS job on this QPU
        usage = self.estimate_job_usage(job, b, needs_retranspile, estimate)
        if "cost" not in skip and job.max_cost_per_job > 0 and \
                usage.effective_cost_usd > job.max_cost_per_job:
            return FilterResult(b, False,
                f"job_cost_too_high(${usage.effective_cost_usd:.2f}>${job.max_cost_per_job:.2f})",
                predicted_post_transpilation_fidelity=predicted_fidelity,
                requires_retranspilation=needs_retranspile,
                divergence_estimate=estimate, usage=usage)

        return FilterResult(
            backend=b,
            passed=True,
            predicted_post_transpilation_fidelity=predicted_fidelity,
            cost_fidelity_explanation=cf_explanation,
            requires_retranspilation=needs_retranspile,
            divergence_estimate=estimate,
            usage=usage,
        )

    @staticmethod
    def estimate_job_usage(job: JobRequest, b: QPUTelemetry, needs_retranspile: bool,
                           estimate=None) -> UsageEstimate:
        """Usage estimate from the circuit size expected AFTER transpilation."""
        c = job.circuit
        n1, n2, depth = c.num_single_qubit_gates, c.num_two_qubit_gates, c.depth
        if needs_retranspile:
            if estimate is not None:
                added_2q = estimate.added_two_qubit_gates
                n1 = int(round(n1 * estimate.gate_overhead))
                added_layers = estimate.swap_ratio * depth
            else:
                div = topology_divergence(c.compiled_for_topology, b.topology)
                added_2q, added_layers = estimate_overhead(div, depth, n2)
            n2 = int(round(n2 + added_2q))
            depth = int(round(depth + added_layers))
        return estimate_usage(b.pricing, b.durations, depth, n1, n2, job.shots,
                              cost_per_shot=b.cost_per_shot)

    def _graph_divergence(self, job: JobRequest, b: QPUTelemetry):
        """Cached graph-based divergence estimate, or None if not computable."""
        if not job.circuit.interaction and not job.circuit.qasm:
            return None
        if not b.coupling_map and "any" not in [g.lower() for g in b.native_gates]:
            return None
        if job.circuit.interaction is None or (job.circuit.qasm and
                                               job.circuit.two_qubit_sequence is None):
            from circuit_analysis import circuit_stats, load_circuit
            st = circuit_stats(load_circuit(job.circuit.qasm))
            job.circuit.interaction = st.interaction
            job.circuit.two_qubit_sequence = st.two_qubit_sequence
            job.circuit.one_qubit_counts = st.one_qubit_counts
        key = (id(job.circuit), b.backend_name, len(b.coupling_map or []), b.qubit_count)
        if key not in self._divergence_cache:
            from circuit_analysis import estimate_divergence
            try:
                self._divergence_cache[key] = estimate_divergence(
                    job.circuit.stats(), b.coupling_map, b.qubit_count, b.native_gates)
            except ValueError as e:
                logger.warning(f"[Scheduler] graph divergence failed for "
                               f"{b.backend_name}: {e}; using lookup table")
                self._divergence_cache[key] = None
        return self._divergence_cache[key]

    # ──────────────────────────────────────────────────────────────────────
    # SCORE PHASE
    # ──────────────────────────────────────────────────────────────────────

    def _score(self, job: JobRequest, filter_result: FilterResult) -> ScoreResult:
        """
        Compute composite quality score for a QPU that passed filtering.
        Score is in [0, 100].
        """
        b = filter_result.backend
        w = job.score_weights
        r = SCORE_RANGES

        # The fidelity term uses the PREDICTED post-transpilation fidelity
        # (equal to the raw fidelity when no retranspilation is needed).
        predicted_fidelity = filter_result.predicted_post_transpilation_fidelity
        if predicted_fidelity is None:
            predicted_fidelity = b.mean_qubit_fidelity

        # Each term normalized to [0, 1] against fixed reference ranges;
        # "lower is better" quantities are inverted.
        terms = {
            "fidelity":      self._norm(predicted_fidelity, *r["fidelity"]),
            "error_rate":    self._norm_inv(b.mean_tq_error, *r["tq_error"]),
            "coherence":     self._norm(b.mean_t2_us, *r["t2_us"]),
            "queue_depth":   self._norm_inv(b.queue_depth, *r["queue_depth"]),
            "cal_freshness": self._norm_inv(b.calibration_age_minutes, *r["cal_age_min"]),
            "cost":          self._norm_inv(
                filter_result.usage.effective_cost_per_shot if filter_result.usage
                else b.cost_per_shot, *r["cost_per_shot"]),
        }

        # Provider preference indicator (1 if the provider is preferred)
        pref = 1.0 if (job.preferred_providers and
                       b.provider in job.preferred_providers) else 0.0
        beta = w.get("provider_pref", 0.0)

        # Composite = (Σ w_i·s_i + β·pref) / (Σ w_i + β), mapped to [0, 100].
        # Dividing by the total weight keeps the maximum at 100; as a result the
        # preference bonus is worth β/(Σw+β)·100 points (≈8.7 with defaults).
        weighted = sum(w.get(k, 0.0) * s for k, s in terms.items())
        total_weight = sum(w.get(k, 0.0) for k in terms) + beta
        composite = (weighted + beta * pref) * 100.0 / total_weight if total_weight > 0 else 0.0
        composite = max(0.0, min(100.0, composite))
        pref_points = beta * pref * 100.0 / total_weight if total_weight > 0 else 0.0

        breakdown = {f"{k}_score": round(s, 4) for k, s in terms.items()}
        breakdown.update({
            "provider_pref":        pref,
            "provider_pref_points": round(pref_points, 2),
            "composite_score":      round(composite, 2),
        })

        return ScoreResult(
            backend=b,
            composite_score=round(composite, 2),
            predicted_fidelity=predicted_fidelity,
            score_breakdown=breakdown,
        )

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _norm(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to [0, 1]. Higher value = higher score."""
        if max_val == min_val:
            return 1.0
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    @staticmethod
    def _norm_inv(value: float, min_val: float, max_val: float) -> float:
        """Normalize and invert. Lower value = higher score."""
        if max_val == min_val:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (value - min_val) / (max_val - min_val)))

    def _log_filter_summary(self, results: List[FilterResult]):
        logger.info(f"\n{'─'*70}")
        logger.info(f"  FILTER PHASE RESULTS")
        logger.info(f"{'─'*70}")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            reason = r.rejection_reason or ""
            pred   = f"pred_fidelity={r.predicted_post_transpilation_fidelity:.4f}" \
                     if r.predicted_post_transpilation_fidelity is not None else ""
            logger.info(f"  [{status}] {r.backend.backend_name:<35} {reason} {pred}")

    def _log_score_summary(self, results: List[ScoreResult]):
        logger.info(f"\n{'─'*70}")
        logger.info(f"  SCORE PHASE RESULTS (sorted by composite score)")
        logger.info(f"{'─'*70}")
        for r in results:
            logger.info(f"  {r.backend.backend_name:<35} score={r.composite_score:>6.1f}  "
                        f"pred_fidelity={r.predicted_fidelity:.4f}")
