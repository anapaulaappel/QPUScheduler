from datetime import timedelta

import pytest

from backends.ibm_backend import ibm_topology_family
from backends.simulator_backend import SimulatorBackend, telemetry_age_minutes
from config import SIMULATOR_BACKENDS
from conftest import make_backend
from crd_writer import telemetry_to_crd
from history_store import HistoryStore
import simulate_history


def _cfg(name):
    return next(c for c in SIMULATOR_BACKENDS if c["name"] == name)


def test_fetch_uses_simulated_time(now):
    # Bug 1: fetch() ignored the simulated time, freezing the calibration phase
    b = SimulatorBackend(_cfg("fake_ibm_brisbane"), now=now)
    phase = _cfg("fake_ibm_brisbane")["calibration_phase_hours"]
    t1 = b.fetch(now=now)
    t2 = b.fetch(now=now + timedelta(hours=3))
    assert t1.calibration_age_minutes == pytest.approx(phase * 60, abs=0.1)
    assert t2.calibration_age_minutes == pytest.approx((phase + 3) * 60, abs=0.1)


def test_simulate_history_varies_and_overwrites(tmp_path):
    simulate_history.simulate(hours=13, output_dir=tmp_path, step_seconds=600)
    store = HistoryStore.from_csv_dir(tmp_path)
    ages = [r.calibration_age_minutes for r in store.get("fake_ibm_brisbane")]
    assert len(set(ages)) > 10 and max(ages) > 600 and min(ages) < 60
    n_first = store.size("fake_ibm_brisbane")
    # Re-running must not duplicate the history (this created a 211 MB CSV)
    simulate_history.simulate(hours=13, output_dir=tmp_path, step_seconds=600)
    assert HistoryStore.from_csv_dir(tmp_path).size("fake_ibm_brisbane") == n_first


def test_telemetry_age(now):
    b = make_backend(last_updated=(now - timedelta(minutes=7)).isoformat())
    assert telemetry_age_minutes(b, now) == pytest.approx(7)


def test_crd_marks_stale_telemetry():
    stale = make_backend(last_updated="2000-01-01T00:00:00+00:00")
    cond = {c["type"]: c for c in telemetry_to_crd(stale)["status"]["conditions"]}
    assert cond["TelemetryFresh"]["status"] == "False"
    assert cond["TelemetryFresh"]["reason"] == "TelemetryStale"


@pytest.mark.parametrize("ptype", [{"family": "Eagle", "revision": 3},
                                   {"family": "Heron"}, None, "unknown"])
def test_ibm_topology_family(ptype):
    # Bug 10: str(processor_type) did not match the divergence table keys
    assert ibm_topology_family(ptype) == "heavy-hex"
