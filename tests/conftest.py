"""Shared fixtures for QPUScheduler tests."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from backends.simulator_backend import QPUTelemetry
from job_request import CircuitProfile, JobRequest

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_backend(**overrides) -> QPUTelemetry:
    """A healthy heavy-hex QPU snapshot taken at NOW; override any field."""
    base = QPUTelemetry(
        backend_name="qpu_a", provider="vendor-a", qubit_count=127,
        topology="heavy-hex", is_simulator=True, is_simulator_fallback=False,
        mean_qubit_fidelity=0.997, min_qubit_fidelity=0.99, max_qubit_fidelity=0.999,
        mean_t1_us=200.0, min_t1_us=150.0, mean_t2_us=150.0, min_t2_us=100.0,
        mean_sq_error=0.001, max_sq_error=0.002,
        mean_tq_error=0.005, max_tq_error=0.01, mean_ro_error=0.01,
        queue_depth=2, estimated_queue_wait_minutes=3.0,
        cost_per_shot=0.0, native_gates=["ecr", "rz", "sx", "x"],
        last_updated=NOW.isoformat(), calibration_timestamp=NOW.isoformat(),
        calibration_age_minutes=30.0, available=True,
    )
    return replace(base, **overrides)


def make_job(**overrides) -> JobRequest:
    circuit = overrides.pop("circuit", CircuitProfile(
        num_qubits=20, depth=85, num_two_qubit_gates=42,
        num_single_qubit_gates=110, compiled_for_topology="heavy-hex",
    ))
    defaults = dict(job_id="job-1", circuit=circuit, min_fidelity=0.99,
                    allow_simulator_fallback=False)
    defaults.update(overrides)
    return JobRequest(**defaults)


@pytest.fixture
def now():
    return NOW
