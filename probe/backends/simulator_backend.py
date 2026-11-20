"""
Simulator Backend
-----------------
Generates realistic QPU telemetry from configurable noise ranges in
config.py. Simulates intra-day drift with a cosine degradation curve over
each calibration cycle, mimicking real calibration-cycle degradation
without using any IBM quota.
"""

import random
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from dataclasses import dataclass, field

from topology import family_coupling_map

logger = logging.getLogger(__name__)


@dataclass
class QPUTelemetry:
    """Normalized telemetry data for one QPU backend."""
    backend_name: str
    provider: str
    qubit_count: int
    topology: str
    is_simulator: bool
    is_simulator_fallback: bool

    # Physical properties
    mean_qubit_fidelity: float
    min_qubit_fidelity: float
    max_qubit_fidelity: float
    mean_t1_us: float
    min_t1_us: float
    mean_t2_us: float
    min_t2_us: float
    mean_sq_error: float
    max_sq_error: float
    mean_tq_error: float
    max_tq_error: float
    mean_ro_error: float
    queue_depth: int
    estimated_queue_wait_minutes: float
    cost_per_shot: float
    native_gates: list

    # Metadata
    last_updated: str
    calibration_timestamp: str
    calibration_age_minutes: float
    available: bool = True

    # Raw source data (for debugging)
    raw: dict = field(default_factory=dict)

    # Undirected coupling map (list of [q0, q1]); None if unknown
    coupling_map: Optional[list] = None

    # Pricing model (see scheduler/usage.py), e.g.
    #   {"scheme": "per_shot", "task_fee": 0.30, "price_per_shot": 0.0009}
    #   {"scheme": "time", "price_per_second": 1.60}
    #   {"scheme": "prepaid_time", "plan_price": 1000.0, "plan_seconds": 3600}
    pricing: dict = field(default_factory=dict)

    # Mean operation durations used to estimate QPU time, e.g.
    #   {"gate_1q_ns": 50, "gate_2q_ns": 300, "measure_ns": 1500, "reset_us": 250}
    durations: dict = field(default_factory=dict)

    # Per-qubit / per-pair error rates (optional):
    #   {"sq": {q: e}, "tq": {"a-b": e}, "ro": {q: e}}
    error_map: dict = field(default_factory=dict, repr=False)


def telemetry_age_minutes(telemetry: "QPUTelemetry",
                          now: Optional[datetime] = None) -> float:
    """
    Minutes elapsed since the telemetry snapshot was taken (last_updated).
    Returns +inf if the timestamp is missing or cannot be parsed, so that
    unknown-age telemetry is always treated as stale.
    """
    now = now or datetime.now(timezone.utc)
    try:
        ts = datetime.fromisoformat(telemetry.last_updated)
    except (TypeError, ValueError):
        return float("inf")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (now - ts).total_seconds() / 60.0)


class SimulatorBackend:
    """
    Generates realistic multi-vendor QPU telemetry without using real hardware.

    For each fake backend, simulates:
    - Intra-day fidelity drift using a sine wave (mimics calibration cycle)
    - Random noise on all metrics within realistic ranges
    - Queue depth fluctuations
    - Periodic calibration events (fidelity resets to peak)
    """

    CALIBRATION_CYCLE_HOURS = 12  # how often fake QPU "recalibrates"

    def __init__(self, backend_config: dict, now: Optional[datetime] = None):
        self.cfg = backend_config
        self.name = backend_config["name"]
        # _start_time is the reference time of a calibration event.
        # 'calibration_phase_hours' shifts each backend's cycle so that
        # different QPUs are not all calibrated at the same instant.
        phase_h = float(backend_config.get("calibration_phase_hours", 0.0))
        self._start_time = (now or datetime.now(timezone.utc)) - timedelta(hours=phase_h)
        logger.info(f"[SimulatorBackend] Initialized fake backend: {self.name}")

    def fetch(self, now: Optional[datetime] = None) -> Optional[QPUTelemetry]:
        """
        Generate a telemetry snapshot with realistic drift simulation.

        Args:
            now: simulated current time. Defaults to the real clock. Passing an
                 explicit time is required when generating historical data,
                 otherwise every snapshot reflects the same instant.
        """
        try:
            now = now or datetime.now(timezone.utc)
            elapsed_hours = (now - self._start_time).total_seconds() / 3600.0

            # ── Simulate calibration cycle ────────────────────────────────
            # Hours since last "calibration" (resets every CALIBRATION_CYCLE_HOURS)
            hours_since_cal = elapsed_hours % self.CALIBRATION_CYCLE_HOURS
            cal_timestamp = now - timedelta(hours=hours_since_cal)
            cal_age_minutes = hours_since_cal * 60.0

            # Degradation factor: 1.0 right after calibration, 0.0 just before
            # Uses a cosine so it starts high and degrades smoothly
            degradation = math.cos(math.pi * hours_since_cal / self.CALIBRATION_CYCLE_HOURS)
            degradation = max(0.0, degradation)  # clamp to [0, 1]

            # ── Fidelity with drift ───────────────────────────────────────
            f_min, f_max = self.cfg["fidelity_range"]
            f_range = f_max - f_min
            # Peak fidelity right after calibration, degrades toward f_min
            base_fidelity = f_min + f_range * degradation
            # Add small random noise
            noise = random.gauss(0, f_range * 0.05)
            mean_fidelity = max(f_min, min(f_max, base_fidelity + noise))
            min_fidelity  = max(f_min - 0.005, mean_fidelity - random.uniform(0.002, 0.010))
            max_fidelity  = min(1.0,            mean_fidelity + random.uniform(0.001, 0.005))

            # ── T1 / T2 with drift ────────────────────────────────────────
            t1_min, t1_max = self.cfg["t1_range_us"]
            t2_min, t2_max = self.cfg["t2_range_us"]
            mean_t1 = t1_min + (t1_max - t1_min) * degradation + random.gauss(0, (t1_max-t1_min)*0.05)
            mean_t1 = max(t1_min * 0.8, min(t1_max * 1.1, mean_t1))
            min_t1  = mean_t1 * random.uniform(0.6, 0.85)
            mean_t2 = t2_min + (t2_max - t2_min) * degradation + random.gauss(0, (t2_max-t2_min)*0.05)
            mean_t2 = max(t2_min * 0.8, min(t2_max * 1.1, mean_t2))
            min_t2  = mean_t2 * random.uniform(0.5, 0.80)

            # ── Gate errors (increase as calibration ages) ────────────────
            sq_min, sq_max = self.cfg["sq_error_range"]
            tq_min, tq_max = self.cfg["tq_error_range"]
            ro_min, ro_max = self.cfg["ro_error_range"]

            error_inflation = 1.0 + (1.0 - degradation) * 2.0  # up to 3x at end of cycle
            mean_sq = min(sq_max * 1.5, (sq_min + (sq_max - sq_min) * (1 - degradation)) * error_inflation)
            max_sq  = mean_sq * random.uniform(1.5, 3.0)
            mean_tq = min(tq_max * 1.5, (tq_min + (tq_max - tq_min) * (1 - degradation)) * error_inflation)
            max_tq  = mean_tq * random.uniform(1.5, 2.5)
            mean_ro = min(ro_max * 1.5, (ro_min + (ro_max - ro_min) * (1 - degradation)) * error_inflation)

            # ── Queue depth (random, time-of-day-like pattern) ────────────
            q_min, q_max = self.cfg["queue_range"]
            # Simulate busier queue during "business hours" (hours 8-18 UTC)
            hour_utc = now.hour
            peak_factor = 1.0 + 0.5 * math.sin(math.pi * max(0, hour_utc - 8) / 10) if 8 <= hour_utc <= 18 else 0.3
            queue = int(q_min + (q_max - q_min) * peak_factor * random.uniform(0.5, 1.5))
            queue = max(q_min, min(q_max * 3, queue))
            queue_wait = queue * random.uniform(0.5, 2.0)  # minutes per job in queue

            is_fallback = self.cfg.get("is_simulator_fallback", False)

            telemetry = QPUTelemetry(
                backend_name=self.name,
                provider=self.cfg["provider"],
                qubit_count=self.cfg["qubit_count"],
                topology=self.cfg["topology"],
                is_simulator=True,
                is_simulator_fallback=is_fallback,
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
                queue_depth=queue,
                estimated_queue_wait_minutes=round(queue_wait, 1),
                cost_per_shot=self.cfg.get("cost_per_shot", 0.0),
                native_gates=self.cfg.get("native_gates", []),
                last_updated=now.isoformat(),
                calibration_timestamp=cal_timestamp.isoformat(),
                calibration_age_minutes=round(cal_age_minutes, 1),
                available=True,
                raw={
                    "degradation_factor": round(degradation, 3),
                    "hours_since_calibration": round(hours_since_cal, 2),
                    "mode": "simulator",
                },
                coupling_map=[list(e) for e in family_coupling_map(
                    self.cfg["topology"], self.cfg["qubit_count"])],
                pricing=dict(self.cfg.get("pricing", {})),
                durations=dict(self.cfg.get("durations", {})),
            )
            logger.debug(f"[SimulatorBackend] {self.name}: fidelity={mean_fidelity:.4f} "
                         f"tq_error={mean_tq:.4f} queue={queue} cal_age={cal_age_minutes:.0f}min")
            return telemetry

        except Exception as e:
            logger.error(f"[SimulatorBackend] {self.name} fetch failed: {e}")
            return None
