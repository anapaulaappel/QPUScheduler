"""
Deferral Scheduler
------------------
Extends the base QuantumScheduler with deferral logic:
if no QPU currently meets the job's constraints but a quality window
is predicted within the job's deadline, the scheduler returns a
DeferralDecision instead of Unschedulable.

Example: schedule job X in 2 hours, when the forecast indicates a
calibration peak on vendor Y.
"""

import logging
import sys
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'probe'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scheduler'))

from backends.simulator_backend import QPUTelemetry
from job_request import JobRequest
from scheduler import QuantumScheduler, SchedulingDecision
from forecaster import (QualityForecaster, QualityWindow, ForecastResult,
                        CALIBRATION_CYCLE_HOURS)
from history_store import HistoryStore

logger = logging.getLogger(__name__)

# Properties that the fidelity-only forecaster cannot predict. They are not
# checked against the projected window state; they are re-checked LIVE when
# the full scheduling pipeline re-runs at the start of the window.
NOT_FORECAST_CHECKS = ("staleness", "gate_error", "readout_error", "queue")


@dataclass
class DeferralDecision:
    """
    Returned when no QPU currently meets constraints but a future
    quality window exists within the job's deadline.
    """
    job_id: str
    deferred: bool
    immediate_decision: SchedulingDecision  # result of immediate scheduling attempt

    # Deferral details (populated if deferred=True)
    recommended_backend: Optional[QPUTelemetry] = None
    recommended_window:  Optional[QualityWindow] = None
    defer_minutes:       float = 0.0
    expected_fidelity:   float = 0.0
    reasoning:           str = ""

    # All forecasts computed (for audit/demo)
    forecasts: List[ForecastResult] = field(default_factory=list)


class DeferralScheduler:
    """
    Wraps QuantumScheduler with forecast-aware deferral logic.

    Decision tree:
      1. Try immediate scheduling (base scheduler)
      2. If successful → return immediately
      3. If failed + deadline_minutes > 0 → check quality forecasts
      4. If a window exists within deadline → return DeferralDecision
      5. If no window → Unschedulable
    """

    def __init__(
        self,
        history: HistoryStore,
        forecast_model: str = "physics",
        scheduler_model: str = "ml",
    ):
        self.forecaster = QualityForecaster(history, model=forecast_model)
        self.scheduler  = QuantumScheduler(model_mode=scheduler_model)

    def schedule(
        self,
        job: JobRequest,
        backends: List[QPUTelemetry],
        now: Optional[datetime] = None,
    ) -> DeferralDecision:
        """
        Main entry point. Returns a DeferralDecision with full audit trail.
        """
        now = now or datetime.now(timezone.utc)

        # ── Step 1: Try immediate scheduling ─────────────────────────────
        immediate = self.scheduler.schedule(job, backends, now=now)

        if not immediate.is_unschedulable and not immediate.is_simulator_fallback:
            logger.info(f"[DeferralScheduler] Job {job.job_id}: immediate scheduling succeeded.")
            return DeferralDecision(
                job_id=job.job_id,
                deferred=False,
                immediate_decision=immediate,
                reasoning="Immediate scheduling succeeded — no deferral needed.",
            )

        # ── Step 2: Check if deferral is possible ────────────────────────
        if job.deadline_minutes <= 0:
            logger.info(f"[DeferralScheduler] Job {job.job_id}: "
                        f"no deadline set, cannot defer.")
            return DeferralDecision(
                job_id=job.job_id,
                deferred=False,
                immediate_decision=immediate,
                reasoning="No deadline set — deferral not applicable.",
            )

        logger.info(f"\n[DeferralScheduler] Job {job.job_id}: "
                    f"immediate scheduling failed. "
                    f"Checking quality forecasts (deadline={job.deadline_minutes}min)...")

        # ── Step 3: Forecast quality windows for each real backend ────────
        real_backends = [b for b in backends if not b.is_simulator_fallback]
        forecasts = []
        best_window: Optional[QualityWindow] = None
        best_backend: Optional[QPUTelemetry] = None

        for backend in real_backends:
            forecast = self.forecaster.forecast(
                backend_name=backend.backend_name,
                min_fidelity=job.min_fidelity,
                horizon_hours=job.deadline_minutes / 60.0,
                now=now,
                # Anchor the calibration-cycle phase on LIVE telemetry
                current_calibration_age_minutes=backend.calibration_age_minutes,
            )
            forecasts.append(forecast)

            # Find the earliest window that fits within the deadline
            for window in forecast.quality_windows:
                if window.minutes_from_now <= job.deadline_minutes:
                    # Check if this backend would also pass the other hard
                    # constraints in the projected state at window start
                    if self._would_pass_other_constraints(job, backend, window, now):
                        if best_window is None or \
                           window.minutes_from_now < best_window.minutes_from_now:
                            best_window  = window
                            best_backend = backend

        # ── Step 4: Return deferral or unschedulable ──────────────────────
        if best_window and best_backend:
            reasoning = (
                f"No QPU currently meets constraints. "
                f"Quality window found on '{best_backend.backend_name}' "
                f"starting in {best_window.minutes_from_now:.0f} min "
                f"(predicted fidelity={best_window.predicted_fidelity:.4f}, "
                f"confidence={best_window.confidence:.2f}). "
                f"Defer job until {best_window.start_time.strftime('%H:%M UTC')}."
            )
            logger.info(f"\n[DeferralScheduler] DEFERRED: {reasoning}")

            return DeferralDecision(
                job_id=job.job_id,
                deferred=True,
                immediate_decision=immediate,
                recommended_backend=best_backend,
                recommended_window=best_window,
                defer_minutes=best_window.minutes_from_now,
                expected_fidelity=best_window.predicted_fidelity,
                reasoning=reasoning,
                forecasts=forecasts,
            )

        else:
            reasoning = (
                f"No QPU currently meets constraints and no quality window "
                f"predicted within {job.deadline_minutes} min deadline."
            )
            logger.warning(f"\n[DeferralScheduler] UNSCHEDULABLE: {reasoning}")
            return DeferralDecision(
                job_id=job.job_id,
                deferred=False,
                immediate_decision=immediate,
                reasoning=reasoning,
                forecasts=forecasts,
            )

    def _would_pass_other_constraints(
        self,
        job: JobRequest,
        backend: QPUTelemetry,
        window: QualityWindow,
        now: datetime,
    ) -> bool:
        """
        Run the full scheduler Filter against the backend's PROJECTED state at
        the start of the window:
          - fidelity  = the window's most pessimistic lower bound;
          - calibration age advanced to the window start;
          - everything else (topology, T2, 2Q error, qubits, cost) as observed now.
        This includes the post-transpilation fidelity check, which uses the
        current (degraded) 2Q error and T2 and is therefore conservative.
        Properties that cannot be forecast (NOT_FORECAST_CHECKS) are skipped
        here and re-checked live when the job is re-scheduled at window start.
        """
        if backend.is_simulator_fallback:
            return False
        projected_cal_age = (backend.calibration_age_minutes + window.minutes_from_now) \
            % (CALIBRATION_CYCLE_HOURS * 60)
        projected = replace(
            backend,
            mean_qubit_fidelity=window.min_lower_bound,
            calibration_age_minutes=projected_cal_age,
        )
        result = self.scheduler.filter_backend(
            job, projected, now=now, skip_checks=NOT_FORECAST_CHECKS)
        if not result.passed:
            logger.info(f"[DeferralScheduler] Window on {backend.backend_name} at "
                        f"+{window.minutes_from_now:.0f}min rejected: "
                        f"{result.rejection_reason}")
        return result.passed
