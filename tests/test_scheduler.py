from dataclasses import replace
from datetime import timedelta

import pytest

from conftest import make_backend, make_job
from job_request import DEFAULT_SCORE_WEIGHTS, JobRequest
from scheduler import QuantumScheduler


@pytest.fixture
def scheduler():
    return QuantumScheduler(model_mode="physics", staleness_threshold_minutes=15)


def test_healthy_backend_is_selected(scheduler, now):
    d = scheduler.schedule(make_job(), [make_backend()], now=now)
    assert not d.is_unschedulable
    assert d.selected_backend.backend_name == "qpu_a"


def test_stale_telemetry_is_rejected(scheduler, now):
    # Bug 5: staleness threshold was defined but never enforced
    stale = make_backend(last_updated=(now - timedelta(minutes=30)).isoformat())
    d = scheduler.schedule(make_job(), [stale], now=now)
    assert d.is_unschedulable
    assert "telemetry_stale" in d.filter_results[0].rejection_reason


def test_unparseable_timestamp_is_treated_as_stale(scheduler, now):
    d = scheduler.schedule(make_job(), [make_backend(last_updated="n/a")], now=now)
    assert d.is_unschedulable


def test_post_transpilation_check_rejects_high_raw_fidelity_backend(scheduler, now):
    lattice = make_backend(backend_name="qpu_lattice", topology="square-lattice",
                           mean_qubit_fidelity=0.999, mean_tq_error=0.01, mean_t2_us=50.0)
    d = scheduler.schedule(make_job(), [lattice], now=now)
    assert d.is_unschedulable
    assert "post_transpilation_fidelity_too_low" in d.filter_results[0].rejection_reason


def test_score_uses_predicted_fidelity(scheduler, now):
    # Bug 6: the score used raw fidelity even after retranspilation
    job = make_job(min_fidelity=0.90)
    native = make_backend(backend_name="native", mean_qubit_fidelity=0.996)
    foreign = make_backend(backend_name="foreign", topology="all-to-all",
                           mean_qubit_fidelity=0.996)
    d = scheduler.schedule(job, [native, foreign], now=now)
    by_name = {s.backend.backend_name: s for s in d.score_results}
    assert by_name["foreign"].predicted_fidelity < 0.996
    assert by_name["foreign"].score_breakdown["fidelity_score"] < \
        by_name["native"].score_breakdown["fidelity_score"]
    assert d.selected_backend.backend_name == "native"


def test_single_candidate_score_is_well_defined(scheduler, now):
    d = scheduler.schedule(make_job(), [make_backend()], now=now)
    assert 0 < d.selected_score.composite_score <= 100


def test_cost_term_prefers_cheaper_backend(scheduler, now):
    cheap = make_backend(backend_name="cheap", cost_per_shot=0.0002)
    pricey = make_backend(backend_name="pricey", cost_per_shot=0.0008)
    d = scheduler.schedule(make_job(), [pricey, cheap], now=now)
    assert d.selected_backend.backend_name == "cheap"


def test_provider_preference_bonus_points(scheduler, now):
    job = make_job(preferred_providers=["vendor-a"])
    d = scheduler.schedule(job, [make_backend()], now=now)
    w = DEFAULT_SCORE_WEIGHTS
    total = sum(w.values())
    expected = w["provider_pref"] * 100 / total
    assert d.selected_score.score_breakdown["provider_pref_points"] == pytest.approx(expected, abs=0.01)


def test_partial_weight_override_keeps_defaults():
    job = make_job(score_weights={"cost": 0.5})
    assert job.score_weights["cost"] == 0.5
    assert job.score_weights["fidelity"] == DEFAULT_SCORE_WEIGHTS["fidelity"]


def test_unknown_weight_raises():
    with pytest.raises(ValueError):
        make_job(score_weights={"typo": 1.0})


def test_skip_checks_validation(scheduler, now):
    with pytest.raises(ValueError):
        scheduler.filter_backend(make_job(), make_backend(), now=now, skip_checks=["nope"])


def test_simulator_fallback(scheduler, now):
    bad = make_backend(mean_qubit_fidelity=0.90)
    sim = make_backend(backend_name="sim", is_simulator_fallback=True)
    d = scheduler.schedule(make_job(allow_simulator_fallback=True), [bad, sim], now=now)
    assert d.is_simulator_fallback and d.selected_backend.backend_name == "sim"
