"""
Predictive Quality Forecaster
------------------------------
Trains a per-backend time-series model on historical telemetry and
forecasts future QPU quality windows — intervals during which fidelity
is predicted to exceed a threshold.

Goal: predict WHEN the fidelity will be high, not just WHAT the fidelity
is now, enabling deferral scheduling to calibration peaks.

Two model options:
  - 'physics': Uses the known calibration cycle sine-wave model.
                Accurate when the cycle period is stable.
                Always available with no training data needed.
  - 'prophet': Facebook Prophet for robust time-series forecasting.
                Handles missing data, outliers, daily seasonality.
                Requires >= MIN_RECORDS_FOR_TRAINING records.

The forecaster writes QualityWindow predictions back into the
QuantumProcessor CRD (qualityForecast field) for the scheduler to use.
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

from history_store import HistoryStore, TelemetryRecord

logger = logging.getLogger(__name__)

MIN_RECORDS_FOR_TRAINING = 50     # need at least this many points for Prophet
FORECAST_HORIZON_HOURS   = 24     # how far ahead to forecast
FORECAST_STEP_MINUTES    = 15     # resolution of forecast time steps
CALIBRATION_CYCLE_HOURS  = 12     # assumed period between calibrations


@dataclass
class QualityWindow:
    """A predicted future interval of high QPU fidelity."""
    start_time: datetime
    end_time:   datetime
    predicted_fidelity: float
    confidence: float             # 0.0 (low) to 1.0 (high)
    minutes_from_now: float       # convenience: how far in the future
    min_lower_bound: float = 0.0  # most pessimistic fidelity inside the window

    def to_dict(self) -> dict:
        return {
            "startTime":          self.start_time.isoformat(),
            "endTime":            self.end_time.isoformat(),
            "predictedFidelity":  round(self.predicted_fidelity, 5),
            "confidence":         round(self.confidence, 3),
            "minutesFromNow":     round(self.minutes_from_now, 1),
            "minLowerBound":      round(self.min_lower_bound, 5),
        }


@dataclass
class ForecastResult:
    """Full forecast for one backend."""
    backend_name: str
    generated_at: datetime
    model_used: str
    horizon_hours: float
    time_steps: List[datetime]          = field(default_factory=list)
    predicted_fidelity: List[float]     = field(default_factory=list)
    confidence_lower: List[float]       = field(default_factory=list)
    confidence_upper: List[float]       = field(default_factory=list)
    quality_windows: List[QualityWindow]= field(default_factory=list)


class QualityForecaster:
    """
    Per-backend quality forecaster.

    Usage:
        forecaster = QualityForecaster(history_store)
        result = forecaster.forecast("fake_ibm_brisbane", min_fidelity=0.99)
        for w in result.quality_windows:
            print(w)
    """

    def __init__(self, history: HistoryStore, model: str = "physics"):
        self.history = history
        self.model   = model
        logger.info(f"[Forecaster] Initialized with model='{model}'")

    def forecast(
        self,
        backend_name: str,
        min_fidelity: float = 0.99,
        horizon_hours: float = FORECAST_HORIZON_HOURS,
        now: Optional[datetime] = None,
        current_calibration_age_minutes: Optional[float] = None,
    ) -> ForecastResult:
        """
        Forecast fidelity over the next horizon_hours and identify
        quality windows where the lower confidence bound >= min_fidelity.

        Args:
            now: forecast origin (defaults to the real clock).
            current_calibration_age_minutes: calibration age from LIVE
                telemetry, if available. It anchors the calibration-cycle
                phase more accurately than the last stored history record.
        """
        records = self.history.get(backend_name)
        now = now or datetime.now(timezone.utc)

        if self.model == "prophet" and len(records) >= MIN_RECORDS_FOR_TRAINING:
            return self._forecast_prophet(backend_name, records, min_fidelity,
                                          horizon_hours, now)
        else:
            if self.model == "prophet" and len(records) < MIN_RECORDS_FOR_TRAINING:
                logger.warning(
                    f"[Forecaster] {backend_name}: only {len(records)} records "
                    f"(need {MIN_RECORDS_FOR_TRAINING} for Prophet). "
                    f"Using physics model."
                )
            return self._forecast_physics(backend_name, records, min_fidelity,
                                          horizon_hours, now,
                                          current_calibration_age_minutes)

    # ──────────────────────────────────────────────────────────────────────
    # PHYSICS MODEL
    # ──────────────────────────────────────────────────────────────────────

    def _forecast_physics(
        self,
        backend_name: str,
        records: List[TelemetryRecord],
        min_fidelity: float,
        horizon_hours: float,
        now: datetime,
        current_calibration_age_minutes: Optional[float] = None,
    ) -> ForecastResult:
        """
        Physics-based forecast using the known calibration cycle model.

        Assumes fidelity follows:
          F(t) = F_low + (F_high - F_low) * max(0, cos(pi * t_since_cal / T_cal))

        where t_since_cal resets to 0 at each calibration event, F_high and
        F_low are the highest and lowest fidelities observed in recent history.
        """
        # Estimate the amplitude from the most recent full calibration cycle.
        # A time-based window is required: a fixed record count (e.g. the last
        # 200 records) covers less than one cycle at short polling intervals,
        # which collapses F_high - F_low and hides the degradation.
        if records:
            cutoff = records[-1].timestamp - timedelta(hours=CALIBRATION_CYCLE_HOURS)
            fidelities = [r.mean_qubit_fidelity for r in records if r.timestamp >= cutoff]
            f_high = max(fidelities)
            f_low  = min(fidelities)
        else:
            f_high, f_low = 0.999, 0.985

        # Estimate the current phase (hours since last calibration)
        if current_calibration_age_minutes is not None:
            cal_age_now = current_calibration_age_minutes / 60.0
        elif records:
            # Last stored calibration age, advanced by the time elapsed since
            # that record was taken
            elapsed_h = max(0.0, (now - records[-1].timestamp).total_seconds() / 3600.0)
            cal_age_now = records[-1].calibration_age_minutes / 60.0 + elapsed_h
        else:
            cal_age_now = 0.0

        f_range = max(f_high - f_low, 0.001)
        cycle   = CALIBRATION_CYCLE_HOURS

        # Generate time steps
        steps = int(horizon_hours * 60 / FORECAST_STEP_MINUTES)
        times, preds, lowers, uppers = [], [], [], []

        for i in range(steps + 1):
            t = now + timedelta(minutes=i * FORECAST_STEP_MINUTES)
            # Hours since last calibration at this future time
            hours_from_now = i * FORECAST_STEP_MINUTES / 60.0
            t_since_cal = (cal_age_now + hours_from_now) % cycle

            # Cosine degradation model
            degradation = math.cos(math.pi * t_since_cal / cycle)
            degradation = max(0.0, degradation)
            pred = f_low + f_range * degradation

            # Uncertainty grows with distance from last known point
            uncertainty = 0.002 + 0.001 * hours_from_now

            times.append(t)
            preds.append(round(pred, 5))
            lowers.append(round(max(0.0, pred - uncertainty), 5))
            uppers.append(round(min(1.0, pred + uncertainty), 5))

        result = ForecastResult(
            backend_name=backend_name,
            generated_at=now,
            model_used="physics",
            horizon_hours=horizon_hours,
            time_steps=times,
            predicted_fidelity=preds,
            confidence_lower=lowers,
            confidence_upper=uppers,
        )
        result.quality_windows = self._extract_windows(
            times, preds, lowers, min_fidelity, now, horizon_hours
        )
        logger.info(f"[Forecaster] {backend_name}: physics model, "
                    f"{len(result.quality_windows)} quality windows found "
                    f"(threshold={min_fidelity})")
        return result

    # ──────────────────────────────────────────────────────────────────────
    # PROPHET MODEL
    # ──────────────────────────────────────────────────────────────────────

    def _forecast_prophet(
        self,
        backend_name: str,
        records: List[TelemetryRecord],
        min_fidelity: float,
        horizon_hours: float,
        now: datetime,
    ) -> ForecastResult:
        """
        Prophet-based forecast for robust time-series prediction.
        Handles missing data, calibration events as changepoints,
        and daily seasonality patterns.
        """
        try:
            import pandas as pd
            from prophet import Prophet
        except ImportError:
            logger.warning("[Forecaster] Prophet not installed. "
                           "pip install prophet. Falling back to physics.")
            return self._forecast_physics(
                backend_name, records, min_fidelity, horizon_hours, now
            )

        import pandas as pd

        # Build training DataFrame
        df = pd.DataFrame([{
            "ds": r.timestamp.astimezone(timezone.utc).replace(tzinfo=None),  # naive UTC
            "y":  r.mean_qubit_fidelity,
        } for r in records])
        df = df.sort_values("ds").reset_index(drop=True)

        # Detect calibration events as potential changepoints
        # (sharp increases in fidelity after a degradation period)
        fidelity_diff = df["y"].diff()
        changepoint_idx = fidelity_diff[fidelity_diff > 0.005].index.tolist()
        changepoints = df.loc[changepoint_idx, "ds"].tolist() if changepoint_idx else None

        model = Prophet(
            changepoints=changepoints,
            changepoint_prior_scale=0.05,
            seasonality_mode="additive",
            daily_seasonality=True,
            weekly_seasonality=False,
            uncertainty_samples=200,
            interval_width=0.80,
        )
        # Add custom calibration cycle seasonality (12h period)
        model.add_seasonality(
            name="calibration_cycle",
            period=CALIBRATION_CYCLE_HOURS / 24,  # Prophet uses days
            fourier_order=5,
        )

        logger.info(f"[Forecaster] {backend_name}: fitting Prophet on "
                    f"{len(df)} records...")
        model.fit(df)

        # Forecast grid anchored at `now` (not at the last history record,
        # which may be older), matching the physics model's grid
        periods = int(horizon_hours * 60 / FORECAST_STEP_MINUTES)
        grid = [now + timedelta(minutes=i * FORECAST_STEP_MINUTES) for i in range(periods + 1)]
        future = pd.DataFrame({"ds": [t.astimezone(timezone.utc).replace(tzinfo=None)
                                      for t in grid]})
        forecast_df = model.predict(future)

        # Convert to lists (clip to valid fidelity range)
        times  = [datetime.fromisoformat(str(ds)).replace(tzinfo=timezone.utc)
                  for ds in forecast_df["ds"]]
        preds  = [float(np.clip(v, 0, 1)) for v in forecast_df["yhat"]]
        lowers = [float(np.clip(v, 0, 1)) for v in forecast_df["yhat_lower"]]
        uppers = [float(np.clip(v, 0, 1)) for v in forecast_df["yhat_upper"]]

        result = ForecastResult(
            backend_name=backend_name,
            generated_at=now,
            model_used="prophet",
            horizon_hours=horizon_hours,
            time_steps=times,
            predicted_fidelity=preds,
            confidence_lower=lowers,
            confidence_upper=uppers,
        )
        result.quality_windows = self._extract_windows(
            times, preds, lowers, min_fidelity, now, horizon_hours
        )
        logger.info(f"[Forecaster] {backend_name}: Prophet model, "
                    f"{len(result.quality_windows)} quality windows found "
                    f"(threshold={min_fidelity})")
        return result

    # ──────────────────────────────────────────────────────────────────────
    # WINDOW EXTRACTION
    # ──────────────────────────────────────────────────────────────────────

    def _extract_windows(
        self,
        times: List[datetime],
        preds: List[float],
        lowers: List[float],
        min_fidelity: float,
        now: datetime,
        horizon_hours: float = FORECAST_HORIZON_HOURS,
        min_window_minutes: int = FORECAST_STEP_MINUTES,
    ) -> List[QualityWindow]:
        """
        Identify maximal contiguous runs of forecast steps in which the LOWER
        confidence bound is >= min_fidelity (conservative window detection).

        Each grid point represents one forecast step, so a window spanning k
        points lasts k steps; a window must last at least min_window_minutes.
        Confidence decreases linearly with the window's distance from now,
        relative to the actual forecast horizon.
        """
        windows: List[QualityWindow] = []
        if not times:
            return windows
        step = (times[1] - times[0]) if len(times) > 1 else timedelta(minutes=FORECAST_STEP_MINUTES)
        horizon_s = max(horizon_hours * 3600.0, 1.0)

        def close(start_idx: int, end_idx: int):
            win_start, win_end = times[start_idx], times[end_idx]
            duration_min = ((win_end - win_start) + step).total_seconds() / 60.0
            if duration_min < min_window_minutes:
                return
            lead_s = (win_start - now).total_seconds()
            confidence = 1.0 - lead_s / horizon_s
            windows.append(QualityWindow(
                start_time=win_start,
                end_time=win_end,
                predicted_fidelity=round(float(np.mean(preds[start_idx:end_idx + 1])), 5),
                confidence=round(max(0.0, min(1.0, confidence)), 3),
                minutes_from_now=round(lead_s / 60.0, 1),
                min_lower_bound=round(float(min(lowers[start_idx:end_idx + 1])), 5),
            ))

        start_idx = None
        for i, lower in enumerate(lowers):
            qualifies = lower >= min_fidelity
            if qualifies and start_idx is None:
                start_idx = i
            elif not qualifies and start_idx is not None:
                close(start_idx, i - 1)
                start_idx = None
        if start_idx is not None:
            close(start_idx, len(times) - 1)

        return windows
