"""
Simulate History
----------------
Generates N hours of synthetic telemetry history for all backends
so the forecaster has data to train on — without waiting hours of real polling.

Run:
    python simulate_history.py --hours 24 --output ../data/history

Output:
    ../data/history/<backend_name>.csv  (one file per backend)
"""

import argparse
import os
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'probe'))

from config import SIMULATOR_BACKENDS, POLLING_INTERVAL_SECONDS
from backends.simulator_backend import SimulatorBackend
from history_store import HistoryStore, TelemetryRecord


def simulate(hours: float, output_dir: Path, step_seconds: int = 30,
             append: bool = False, seed: Optional[int] = 42):
    print(f"Simulating {hours}h of history at {step_seconds}s intervals...")
    print(f"Output: {output_dir}\n")

    if seed is not None:
        random.seed(seed)

    # Overwrite by default: HistoryStore appends to CSVs, so re-running the
    # script without this would duplicate the history on every run.
    if not append:
        for cfg in SIMULATOR_BACKENDS:
            (output_dir / f"{cfg['name']}.csv").unlink(missing_ok=True)

    store = HistoryStore(csv_dir=output_dir)

    # History covers [now - hours, now). Backends are created at the start of
    # that interval so that their calibration phase is anchored correctly.
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    backends = [SimulatorBackend(cfg, now=start) for cfg in SIMULATOR_BACKENDS]
    steps = int(hours * 3600 / step_seconds)

    for i in range(steps):
        t = start + timedelta(seconds=i * step_seconds)
        for b in backends:
            # Pass the simulated time so the calibration cycle advances
            tel = b.fetch(now=t)
            if tel:
                # Override the timestamp to the simulated past time
                record = TelemetryRecord(
                    timestamp=t,
                    backend_name=tel.backend_name,
                    mean_qubit_fidelity=tel.mean_qubit_fidelity,
                    mean_tq_error=tel.mean_tq_error,
                    mean_t2_us=tel.mean_t2_us,
                    calibration_age_minutes=tel.calibration_age_minutes,
                    queue_depth=tel.queue_depth,
                )
                store.add(record)

        if i % 100 == 0:
            pct = i / steps * 100
            print(f"  Progress: {pct:.0f}%  ({i}/{steps} steps, "
                  f"t={t.strftime('%H:%M')})", end="\r")

    print(f"\n\nDone! Records per backend:")
    for b in backends:
        n = store.size(b.name)
        print(f"  {b.name:<35} {n} records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours",  type=float, default=24.0,
                        help="Hours of history to simulate (default: 24)")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "history"),
                        help="Output directory for CSV files")
    parser.add_argument("--step",   type=int, default=30,
                        help="Polling step in seconds (default: 30)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSVs instead of overwriting them")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    simulate(args.hours, Path(args.output), args.step,
             append=args.append, seed=args.seed)
