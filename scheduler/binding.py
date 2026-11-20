"""
Two-stage scheduling with retranspilation, re-validation and rollback
--------------------------------------------------------------------
Two-stage binding (estimate, reserve, transpile, re-validate, bind or roll back), mapped onto the Kubernetes
Scheduling Framework:

  1. Filter (fast path): every candidate is checked with the ESTIMATED
     divergence and predicted post-transpilation fidelity.
  2. Score / Reserve: the best candidate is reserved.
  3. Permit / PreBind (slow path): the Retranspilation Broker transpiles the
     circuit for the reserved QPU only.
  4. Re-validation: the ACTUAL transpiled metrics replace the estimates;
     fidelity and cost are recomputed and the constraints re-checked.
  5. Quota (180): usage is reserved atomically in its native units.
  6. Commit (Bind) or rollback (Unreserve → exclude candidate → re-queue),
     bounded by max_retries; when exhausted, the fallback (160) applies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from job_request import JobRequest
from quota import QuotaController, QuotaExceeded, Reservation
from retranspilation_broker import RetranspilationBroker, TranspileResult
from scheduler import QuantumScheduler, SchedulingDecision
from usage import UsageEstimate, estimate_usage

logger = logging.getLogger(__name__)


@dataclass
class Attempt:
    backend_name: str
    outcome: str                       # bound | revalidation_failed | quota_exceeded
    estimated_fidelity: Optional[float] = None
    revalidated_fidelity: Optional[float] = None
    transpile_seconds: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class BindingDecision:
    job_id: str
    bound: bool
    backend_name: Optional[str] = None
    is_simulator_fallback: bool = False
    unschedulable_reason: Optional[str] = None
    predicted_fidelity: Optional[float] = None
    transpiled: Optional[TranspileResult] = None
    usage: Optional[UsageEstimate] = None
    reservation: Optional[Reservation] = None
    attempts: List[Attempt] = field(default_factory=list)
    last_decision: Optional[SchedulingDecision] = None

    def audit_annotations(self, now: Optional[datetime] = None) -> dict:
        """Audit record written to the workload at binding time (component 210)."""
        now = now or datetime.now(timezone.utc)
        a = {
            "quantum/scheduled-at": now.isoformat(),
            "quantum/backend": self.backend_name or "",
            "quantum/simulator-fallback": str(self.is_simulator_fallback).lower(),
            "quantum/attempts": str(len(self.attempts)),
        }
        if self.predicted_fidelity is not None:
            a["quantum/predicted-fidelity"] = f"{self.predicted_fidelity:.5f}"
        if self.usage is not None:
            a["quantum/estimated-qpu-seconds"] = f"{self.usage.qpu_seconds:.3f}"
            a["quantum/estimated-cost-usd"] = f"{self.usage.cost_usd:.4f}"
            a["quantum/effective-cost-usd"] = f"{self.usage.effective_cost_usd:.4f}"
        if self.transpiled is not None:
            a["quantum/retranspiled"] = "true"
            a["quantum/transpiled-2q-gates"] = str(self.transpiled.num_two_qubit_gates)
        if self.unschedulable_reason:
            a["quantum/unschedulable-reason"] = self.unschedulable_reason
        return a


class SchedulingPipeline:
    def __init__(self, scheduler: Optional[QuantumScheduler] = None,
                 broker: Optional[RetranspilationBroker] = None,
                 quota: Optional[QuotaController] = None,
                 max_retries: int = 3):
        self.scheduler = scheduler or QuantumScheduler()
        self.broker = broker
        self.quota = quota
        self.max_retries = max_retries

    def schedule_and_bind(self, job: JobRequest, backends, now: Optional[datetime] = None
                          ) -> BindingDecision:
        now = now or datetime.now(timezone.utc)
        excluded: set = set()       # exclusion annotation on the workload
        attempts: List[Attempt] = []
        decision: Optional[SchedulingDecision] = None

        for _ in range(self.max_retries + 1):
            candidates = [b for b in backends if b.backend_name not in excluded]
            decision = self.scheduler.schedule(job, candidates, now=now)
            if decision.is_unschedulable or decision.is_simulator_fallback:
                break

            # ── Reserve the best candidate ────────────────────────────────
            best = decision.selected_backend
            fr = next(r for r in decision.filter_results
                      if r.backend.backend_name == best.backend_name)
            predicted = decision.selected_score.predicted_fidelity
            usage = fr.usage
            transpiled = None

            # ── Slow path: retranspile + re-validate (Permit / PreBind) ───
            if fr.requires_retranspilation and self.broker and job.circuit.qasm:
                transpiled = self.broker.transpile(job.circuit, best)
                passes, revalidated, expl = self.scheduler.cf_model.will_meet_threshold(
                    min_fidelity=job.min_fidelity,
                    src_topology=job.circuit.compiled_for_topology,
                    dst_topology=best.topology,
                    circuit_depth=job.circuit.depth,
                    num_two_qubit_gates=job.circuit.num_two_qubit_gates,
                    target_mean_tq_error=best.mean_tq_error,
                    target_mean_t2_us=best.mean_t2_us,
                    target_mean_fidelity=best.mean_qubit_fidelity,
                    divergence=transpiled.swap_ratio,       # ACTUAL overhead
                )
                usage = estimate_usage(best.pricing, best.durations, transpiled.depth,
                                       transpiled.num_single_qubit_gates,
                                       transpiled.num_two_qubit_gates, job.shots,
                                       cost_per_shot=best.cost_per_shot)
                cost_ok = not (job.max_cost_per_job > 0 and
                               usage.effective_cost_usd > job.max_cost_per_job)
                if not passes or not cost_ok:
                    reason = (f"revalidated_fidelity={revalidated:.4f}<{job.min_fidelity}"
                              if not passes else
                              f"revalidated_cost=${usage.effective_cost_usd:.2f}"
                              f">${job.max_cost_per_job:.2f}")
                    logger.info(f"[Pipeline] {best.backend_name}: re-validation failed "
                                f"({reason}); Unreserve and re-queue")
                    attempts.append(Attempt(best.backend_name, "revalidation_failed",
                                            predicted, revalidated, transpiled.seconds, reason))
                    excluded.add(best.backend_name)
                    continue
                predicted = revalidated

            # ── Quota (180): atomic reservation in native units ──────────
            reservation = None
            if self.quota is not None:
                try:
                    reservation = self.quota.reserve(job.namespace, best.provider,
                                                     usage.units(), now=now)
                except QuotaExceeded as e:
                    # Budget failures do NOT trigger the simulator fallback
                    attempts.append(Attempt(best.backend_name, "quota_exceeded",
                                            predicted, reason=str(e)))
                    return BindingDecision(job.job_id, bound=False,
                                           unschedulable_reason=str(e),
                                           attempts=attempts, last_decision=decision)

            attempts.append(Attempt(best.backend_name, "bound", fr.predicted_post_transpilation_fidelity,
                                    predicted if transpiled else None,
                                    transpiled.seconds if transpiled else None))
            return BindingDecision(job.job_id, bound=True, backend_name=best.backend_name,
                                   predicted_fidelity=predicted, transpiled=transpiled,
                                   usage=usage, reservation=reservation,
                                   attempts=attempts, last_decision=decision)

        # ── Retries exhausted or no candidate: fallback (160) ─────────────
        if decision is not None and decision.is_simulator_fallback:
            return BindingDecision(job.job_id, bound=True,
                                   backend_name=decision.selected_backend.backend_name,
                                   is_simulator_fallback=True, attempts=attempts,
                                   last_decision=decision)
        sims = [b for b in backends if b.is_simulator_fallback]
        if job.allow_simulator_fallback and sims and attempts:
            return BindingDecision(job.job_id, bound=True, backend_name=sims[0].backend_name,
                                   is_simulator_fallback=True, attempts=attempts,
                                   last_decision=decision)
        reason = (decision.unschedulable_reason if decision and decision.is_unschedulable
                  else None) or "retranspilation_revalidation_failed"
        if attempts and decision and not decision.is_unschedulable:
            reason = "retranspilation_revalidation_failed"
        elif attempts:
            reason = f"{reason} (after {len(attempts)} failed re-validation(s))"
        return BindingDecision(job.job_id, bound=False, unschedulable_reason=reason,
                               attempts=attempts, last_decision=decision)
