"""
Real calibration collector (for experiments E4 / E5)
---------------------------------------------------
Polls IBM Quantum calibration data at a fixed interval and appends one row
per backend to data/real_history/<backend>.csv. READ-ONLY: it never submits
circuits, so no QPU time is consumed.

Credentials: IBM_QUANTUM_TOKEN if set, otherwise the saved Qiskit account
(~/.qiskit/qiskit-ibm.json).

Usage:
    python collect_calibration.py --once              # single poll (test)
    python collect_calibration.py --interval 15       # poll every 15 min, forever

Each row also records the provider's calibration timestamp, so calibration
events can be detected exactly (needed to evaluate quality-window forecasts).
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from config import IBM_CHANNEL, IBM_INSTANCE, IBM_PREFERRED_BACKENDS, IBM_PRICING, IBM_TOKEN

logger = logging.getLogger("collector")

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "real_history"
FIELDS = ["timestamp", "backend_name", "mean_qubit_fidelity", "mean_tq_error",
          "mean_t2_us", "calibration_age_minutes", "queue_depth",
          "calibration_timestamp", "max_tq_error", "mean_ro_error", "topology",
          "available"]


def append_row(output_dir: Path, t) -> Path:
    path = output_dir / f"{t.backend_name}.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": t.last_updated, "backend_name": t.backend_name,
            "mean_qubit_fidelity": t.mean_qubit_fidelity,
            "mean_tq_error": t.mean_tq_error, "mean_t2_us": t.mean_t2_us,
            "calibration_age_minutes": t.calibration_age_minutes,
            "queue_depth": t.queue_depth,
            "calibration_timestamp": t.calibration_timestamp,
            "max_tq_error": t.max_tq_error, "mean_ro_error": t.mean_ro_error,
            "topology": t.topology, "available": t.available,
        })
    return path


def poll_once(ibm, output_dir: Path) -> int:
    n = 0
    for t in ibm.fetch_all():
        if t.available:
            append_row(output_dir, t)
            n += 1
            logger.info(f"{t.backend_name}: F={t.mean_qubit_fidelity:.5f} "
                        f"e2q={t.mean_tq_error:.5f} cal_age={t.calibration_age_minutes:.0f}min "
                        f"queue={t.queue_depth}")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=float, default=15.0, help="minutes between polls")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from backends.ibm_backend import IBMBackend

    args.output.mkdir(parents=True, exist_ok=True)
    ibm = IBMBackend(IBM_TOKEN, IBM_INSTANCE, IBM_PREFERRED_BACKENDS,
                     channel=IBM_CHANNEL, pricing=IBM_PRICING)
    logger.info(f"Collecting calibration for {ibm.backend_names()} -> {args.output}")
    while True:
        started = time.time()
        try:
            n = poll_once(ibm, args.output)
            logger.info(f"poll done: {n} backends at {datetime.now(timezone.utc).isoformat()}")
        except Exception as e:      # keep collecting through transient API errors
            logger.error(f"poll failed: {e}")
        if args.once:
            break
        time.sleep(max(0.0, args.interval * 60 - (time.time() - started)))


if __name__ == "__main__":
    main()
