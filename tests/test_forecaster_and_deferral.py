from datetime import timedelta

import pytest

from conftest import make_backend, make_job
from deferral_scheduler import DeferralScheduler
from forecaster import CALIBRATION_CYCLE_HOURS, QualityForecaster
from history_store import HistoryStore, TelemetryRecord


def make_history(now, backend_name="qpu_a", hours=24, step_min=15,
                 f_high=0.999, f_low=0.985, phase_h=0.0):
    """Synthetic history following the calibration-cycle curve."""
    import math
    store = HistoryStore()
    steps = int(hours * 60 / step_min)
    start = now - timedelta(hours=hours)
    for i in range(steps):
        t = start + timedelta(minutes=i * step_min)
        age_h = ((t - start).total_seconds() / 3600 + phase_h) % CALIBRATION_CYCLE_HOURS
        f = f_low + (f_high - f_low) * max(0.0, math.cos(math.pi * age_h / CALIBRATION_CYCLE_HOURS))
        store.add(TelemetryRecord(t, backend_name, f, 0.005, 150.0, age_h * 60, 2))
    return store


def test_phase_advances_with_time_since_last_record(now):
    # Bug: phase came from the last record's calibration age only
    hist = make_history(now - timedelta(hours=3))   # history ends 3h ago
    fc = QualityForecaster(hist)
    last = hist.get("qpu_a")[-1]
    a = fc.forecast("qpu_a", min_fidelity=0.99, horizon_hours=1, now=now)
    elapsed_min = (now - last.timestamp).total_seconds() / 60
    b = fc.forecast("qpu_a", min_fidelity=0.99, horizon_hours=1, now=now,
                    current_calibration_age_minutes=last.calibration_age_minutes + elapsed_min)
    assert a.predicted_fidelity[0] == pytest.approx(b.predicted_fidelity[0])


def test_window_confidence_uses_actual_horizon(now):
    # Bug 11: confidence was computed against a fixed 24h horizon
    hist = make_history(now)
    fc = QualityForecaster(hist)
    res = fc.forecast("qpu_a", min_fidelity=0.99, horizon_hours=4, now=now,
                      current_calibration_age_minutes=10.5 * 60)
    assert res.quality_windows, "a post-calibration window is expected"
    w = res.quality_windows[0]
    expected = 1.0 - w.minutes_from_now / (4 * 60)
    assert w.confidence == pytest.approx(expected, abs=0.002)


def test_window_uses_lower_bound_and_reports_it(now):
    hist = make_history(now)
    res = QualityForecaster(hist).forecast(
        "qpu_a", min_fidelity=0.99, horizon_hours=4, now=now,
        current_calibration_age_minutes=10.5 * 60)
    w = res.quality_windows[0]
    assert w.min_lower_bound >= 0.99
    assert w.predicted_fidelity >= w.min_lower_bound


def test_single_step_window_is_kept(now):
    fc = QualityForecaster(HistoryStore())
    times = [now + timedelta(minutes=15 * i) for i in range(4)]
    lowers = [0.98, 0.995, 0.98, 0.98]
    wins = fc._extract_windows(times, lowers, lowers, 0.99, now, horizon_hours=1)
    assert len(wins) == 1 and wins[0].start_time == times[1]


def test_deferral_targets_next_calibration(now):
    # Degraded backend 10h into a 12h cycle; next calibration in ~2h
    hist = make_history(now, phase_h=0.0)
    degraded = make_backend(mean_qubit_fidelity=0.986, calibration_age_minutes=600)
    job = make_job(min_fidelity=0.993, deadline_minutes=180)
    d = DeferralScheduler(hist, scheduler_model="physics").schedule(job, [degraded], now=now)
    assert d.deferred
    assert 105 <= d.defer_minutes <= 135


def test_deferral_rechecks_post_transpilation(now):
    # Bug 8: deferral ignored the topology / post-transpilation check
    hist = make_history(now)
    lattice = make_backend(topology="square-lattice", mean_qubit_fidelity=0.986,
                           mean_tq_error=0.02, mean_t2_us=40.0,
                           calibration_age_minutes=600)
    job = make_job(min_fidelity=0.993, deadline_minutes=180)
    d = DeferralScheduler(hist, scheduler_model="physics").schedule(job, [lattice], now=now)
    assert not d.deferred


def test_amplitude_uses_a_full_cycle_even_with_dense_polling(now):
    # Bug: amplitude came from the last 200 records (~1.7h at 30s polling)
    hist = make_history(now, hours=24, step_min=0.5)
    degraded = make_backend(mean_qubit_fidelity=0.986, calibration_age_minutes=600)
    job = make_job(min_fidelity=0.993, deadline_minutes=180)
    d = DeferralScheduler(hist, scheduler_model="physics").schedule(job, [degraded], now=now)
    assert d.deferred and d.defer_minutes > 60


def test_no_deferral_without_deadline(now):
    hist = make_history(now)
    degraded = make_backend(mean_qubit_fidelity=0.986, calibration_age_minutes=600)
    d = DeferralScheduler(hist, scheduler_model="physics").schedule(
        make_job(min_fidelity=0.993, deadline_minutes=0), [degraded], now=now)
    assert not d.deferred
