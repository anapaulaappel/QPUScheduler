"""
QPU Telemetry Probe
-------------------
Main entry point. Polls QPU backends and writes QuantumProcessor CRDs.

Usage:
    # Simulator mode (safe, no IBM quota used):
    export PROBE_MODE=simulator
    python probe.py

    # Real IBM Quantum mode (uses calibration API only — no QPU minutes):
    export PROBE_MODE=real
    # Either a saved account (QiskitRuntimeService.save_account) or:
    export IBM_QUANTUM_TOKEN=your_api_key
    python probe.py

    # Apply to Kubernetes (requires kubectl configured):
    export K8S_APPLY=true
    python probe.py
"""

import logging
import time
import sys
import os

# Add probe and forecaster directories to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "forecaster"))

from config import (
    PROBE_MODE, IBM_TOKEN, IBM_INSTANCE, IBM_CHANNEL, IBM_PREFERRED_BACKENDS,
    IBM_PRICING, SIMULATOR_BACKENDS, POLLING_INTERVAL_SECONDS, LOG_LEVEL
)
from backends.simulator_backend import SimulatorBackend
from crd_writer import apply_crd

# Quality forecast written to status.qualityForecast (Predictive Quality Forecaster)
FORECAST_ENABLED = os.environ.get("FORECAST_ENABLED", "true").lower() == "true"
FORECAST_MIN_FIDELITY = float(os.environ.get("FORECAST_MIN_FIDELITY", "0.99"))
FORECAST_HORIZON_HOURS = float(os.environ.get("FORECAST_HORIZON_HOURS", "12"))
FORECAST_MODEL = os.environ.get("FORECAST_MODEL", "physics")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def build_backends():
    """Initialize backend objects based on PROBE_MODE."""
    if PROBE_MODE == "simulator":
        logger.info("="*60)
        logger.info("  PROBE MODE: SIMULATOR")
        logger.info("  No IBM Quantum quota will be used.")
        logger.info("  Simulating realistic QPU drift for all backends.")
        logger.info("="*60)
        return [SimulatorBackend(cfg) for cfg in SIMULATOR_BACKENDS]

    elif PROBE_MODE == "real":
        logger.info("="*60)
        logger.info("  PROBE MODE: REAL IBM QUANTUM HARDWARE")
        logger.info("  Reading calibration data only — NO QPU minutes consumed.")
        logger.info("  QPU minutes are only used when you RUN circuits.")
        logger.info("="*60)

        from backends.ibm_backend import IBMBackend
        # Uses IBM_QUANTUM_TOKEN if set, otherwise the saved Qiskit account
        ibm = IBMBackend(IBM_TOKEN, IBM_INSTANCE, IBM_PREFERRED_BACKENDS,
                         channel=IBM_CHANNEL, pricing=IBM_PRICING)

        logger.info("Discovering IBM backends...")
        try:
            names = ibm.backend_names()
            logger.info(f"Monitoring backends: {names}")
        except Exception as e:
            logger.error(f"Could not connect to IBM Quantum: {e}")
            sys.exit(1)

        # One fetchable wrapper per backend
        class _SingleIBM:
            def __init__(self, b, n):
                self._b, self._n, self.name = b, n, n
            def fetch(self):
                return self._b.fetch(self._n)

        return [_SingleIBM(ibm, name) for name in names]

    else:
        logger.error(f"Unknown PROBE_MODE: {PROBE_MODE}. Use 'simulator' or 'real'.")
        sys.exit(1)


def print_summary(results: list):
    """Print a formatted summary table of all backend telemetry."""
    print("\n" + "─"*100)
    print(f"{'BACKEND':<30} {'FIDELITY':>8} {'2Q-ERR':>8} {'T2(us)':>8} "
          f"{'QUEUE':>6} {'CAL-AGE':>8} {'AVAIL':>6}")
    print("─"*100)
    for t in results:
        if t is None:
            continue
        flag = "SIM" if t.is_simulator else "REAL"
        fb   = " [FALLBACK]" if t.is_simulator_fallback else ""
        print(f"{t.backend_name:<30} {t.mean_qubit_fidelity:>8.4f} "
              f"{t.mean_tq_error:>8.4f} {t.mean_t2_us:>8.1f} "
              f"{t.queue_depth:>6} {t.calibration_age_minutes:>7.0f}m "
              f"  [{flag}]{fb}")
    print("─"*100)


def _forecast_windows(forecaster, history, telemetry):
    """Record the snapshot in the history and forecast quality windows."""
    from datetime import datetime
    from history_store import TelemetryRecord
    ts = datetime.fromisoformat(telemetry.last_updated)
    history.add(TelemetryRecord(ts, telemetry.backend_name, telemetry.mean_qubit_fidelity,
                                telemetry.mean_tq_error, telemetry.mean_t2_us,
                                telemetry.calibration_age_minutes, telemetry.queue_depth))
    if telemetry.is_simulator_fallback:
        return []
    result = forecaster.forecast(telemetry.backend_name, FORECAST_MIN_FIDELITY,
                                 FORECAST_HORIZON_HOURS,
                                 current_calibration_age_minutes=telemetry.calibration_age_minutes)
    return result.quality_windows


def run():
    backends = build_backends()
    poll = 0
    history = forecaster = None
    if FORECAST_ENABLED:
        from history_store import HistoryStore
        from forecaster import QualityForecaster
        history = HistoryStore()
        forecaster = QualityForecaster(history, model=FORECAST_MODEL)

    while True:
        poll += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"  Poll #{poll} — fetching telemetry from {len(backends)} backends")
        logger.info(f"{'='*60}")

        results = []
        for backend in backends:
            telemetry = backend.fetch()
            if telemetry:
                windows = _forecast_windows(forecaster, history, telemetry) \
                    if forecaster is not None and telemetry.available else []
                apply_crd(telemetry, windows)
                results.append(telemetry)
            else:
                logger.warning(f"No telemetry returned from {backend.name}")

        print_summary(results)

        logger.info(f"\nNext poll in {POLLING_INTERVAL_SECONDS}s "
                    f"(Ctrl+C to stop)...\n")
        time.sleep(POLLING_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("\nProbe stopped by user.")
