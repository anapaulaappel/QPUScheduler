"""
History Store
-------------
Accumulates time-indexed telemetry readings per QPU backend.
In production this would be backed by a time-series database (e.g. Prometheus TSDB).
Here it uses an in-memory deque with optional CSV persistence.
"""

import csv
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

MAX_HISTORY_PER_BACKEND = 2000   # keep last ~16h at 30s polling


@dataclass
class TelemetryRecord:
    """One time-indexed telemetry snapshot for a backend."""
    timestamp: datetime
    backend_name: str
    mean_qubit_fidelity: float
    mean_tq_error: float
    mean_t2_us: float
    calibration_age_minutes: float
    queue_depth: int


class HistoryStore:
    """
    Per-backend ring buffer of telemetry records.
    Supports CSV export for offline analysis and notebook plotting.
    """

    def __init__(self, csv_dir: Optional[Path] = None):
        self._store: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY_PER_BACKEND)
        )
        self._csv_dir = csv_dir
        if csv_dir:
            csv_dir.mkdir(parents=True, exist_ok=True)

    def add(self, record: TelemetryRecord):
        self._store[record.backend_name].append(record)
        if self._csv_dir:
            self._append_csv(record)

    def get(self, backend_name: str) -> List[TelemetryRecord]:
        return list(self._store[backend_name])

    def all_backends(self) -> List[str]:
        return list(self._store.keys())

    def size(self, backend_name: str) -> int:
        return len(self._store[backend_name])

    def _append_csv(self, r: TelemetryRecord):
        path = self._csv_dir / f"{r.backend_name}.csv"
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "backend_name", "mean_qubit_fidelity",
                "mean_tq_error", "mean_t2_us",
                "calibration_age_minutes", "queue_depth"
            ])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": r.timestamp.isoformat(),
                "backend_name": r.backend_name,
                "mean_qubit_fidelity": r.mean_qubit_fidelity,
                "mean_tq_error": r.mean_tq_error,
                "mean_t2_us": r.mean_t2_us,
                "calibration_age_minutes": r.calibration_age_minutes,
                "queue_depth": r.queue_depth,
            })

    @classmethod
    def from_csv_dir(cls, csv_dir: Path) -> "HistoryStore":
        """Load history from persisted CSVs using pandas for speed."""
        try:
            import pandas as pd
        except ImportError:
            pd = None

        store = cls(csv_dir=csv_dir)
        for csv_file in csv_dir.glob("*.csv"):
            if pd is not None:
                df = pd.read_csv(csv_file, parse_dates=["timestamp"])
                for row in df.itertuples(index=False):
                    ts = row.timestamp
                    if hasattr(ts, 'to_pydatetime'):
                        ts = ts.to_pydatetime()
                    if ts.tzinfo is None:
                        from datetime import timezone as tz
                        ts = ts.replace(tzinfo=tz.utc)
                    store._store[row.backend_name].append(TelemetryRecord(
                        timestamp=ts,
                        backend_name=row.backend_name,
                        mean_qubit_fidelity=float(row.mean_qubit_fidelity),
                        mean_tq_error=float(row.mean_tq_error),
                        mean_t2_us=float(row.mean_t2_us),
                        calibration_age_minutes=float(row.calibration_age_minutes),
                        queue_depth=int(row.queue_depth),
                    ))
            else:
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        store.add(TelemetryRecord(
                            timestamp=datetime.fromisoformat(row["timestamp"]),
                            backend_name=row["backend_name"],
                            mean_qubit_fidelity=float(row["mean_qubit_fidelity"]),
                            mean_tq_error=float(row["mean_tq_error"]),
                            mean_t2_us=float(row["mean_t2_us"]),
                            calibration_age_minutes=float(row["calibration_age_minutes"]),
                            queue_depth=int(row["queue_depth"]),
                        ))
            logger.info(f"[HistoryStore] Loaded {store.size(csv_file.stem)} "
                        f"records for {csv_file.stem}")
        return store
