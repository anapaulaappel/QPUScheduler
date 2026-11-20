"""
Quantum usage quota controller
---------------------------------------------------------------
Namespace-scoped quotas in one or more units (USD, QPU seconds, shots), with
optional per-vendor sub-limits in the unit each vendor sells (e.g. QPU
seconds for a prepaid plan, USD for pay-per-use).

Concurrency: mirrors Kubernetes optimistic concurrency. Each namespace's
usage record carries a version; a reservation reads (usage, version),
checks the limits and commits only if the version is unchanged, retrying on
conflict. Concurrent reservations therefore cannot jointly exceed a limit.

In a cluster this state lives in the QuantumQuota custom resource status and
the version is its resourceVersion; here it is kept in memory.
"""

import itertools
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

UNITS = ("usd", "qpu_seconds", "shots")


class QuotaExceeded(Exception):
    def __init__(self, namespace: str, unit: str, scope: str, used: float,
                 requested: float, limit: float):
        self.namespace, self.unit, self.scope = namespace, unit, scope
        self.used, self.requested, self.limit = used, requested, limit
        super().__init__(f"quota_exceeded({scope}:{unit} used={used:.4g} "
                         f"+ requested={requested:.4g} > limit={limit:.4g})")


@dataclass
class QuotaSpec:
    """Limits for one namespace (None = unlimited)."""
    limits: Dict[str, float] = field(default_factory=dict)             # e.g. {"usd": 50}
    vendor_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)
    period: str = "daily"           # "daily" or "billing_cycle"
    period_start: Optional[datetime] = None      # for billing_cycle
    period_length: timedelta = timedelta(days=30)  # for billing_cycle


@dataclass
class Reservation:
    id: int
    namespace: str
    vendor: str
    amounts: Dict[str, float]
    state: str = "reserved"          # reserved | reconciled | released


@dataclass
class _Usage:
    total: Dict[str, float] = field(default_factory=lambda: {u: 0.0 for u in UNITS})
    vendor: Dict[str, Dict[str, float]] = field(default_factory=dict)
    version: int = 0
    period_key: Optional[str] = None


class QuotaController:
    MAX_CAS_RETRIES = 50

    def __init__(self):
        self._specs: Dict[str, QuotaSpec] = {}
        self._usage: Dict[str, _Usage] = {}
        self._reservations: Dict[int, Reservation] = {}
        self._lock = threading.Lock()       # protects the compare-and-swap only
        self._ids = itertools.count(1)

    # ── configuration ─────────────────────────────────────────────────────
    def set_quota(self, namespace: str, spec: QuotaSpec):
        self._specs[namespace] = spec
        self._usage.setdefault(namespace, _Usage())

    def usage(self, namespace: str) -> dict:
        u = self._usage.get(namespace, _Usage())
        return {"total": dict(u.total),
                "vendor": {v: dict(x) for v, x in u.vendor.items()},
                "version": u.version}

    def remaining(self, namespace: str, vendor: Optional[str] = None) -> Dict[str, float]:
        spec = self._specs.get(namespace)
        if spec is None:
            return {}
        u = self._usage.get(namespace, _Usage())
        rem = {unit: lim - u.total.get(unit, 0.0) for unit, lim in spec.limits.items()}
        if vendor and vendor in spec.vendor_limits:
            vu = u.vendor.get(vendor, {})
            for unit, lim in spec.vendor_limits[vendor].items():
                rem[f"{vendor}:{unit}"] = lim - vu.get(unit, 0.0)
        return rem

    # ── period handling ───────────────────────────────────────────────────
    @staticmethod
    def _period_key(spec: QuotaSpec, now: datetime) -> str:
        if spec.period == "billing_cycle" and spec.period_start is not None:
            n = int((now - spec.period_start) / spec.period_length)
            return f"cycle-{n}"
        return now.date().isoformat()

    def _maybe_reset(self, ns: str, now: datetime) -> _Usage:
        spec = self._specs.get(ns)
        u = self._usage.setdefault(ns, _Usage())
        if spec is None:
            return u
        key = self._period_key(spec, now)
        if u.period_key != key:
            with self._lock:
                if u.period_key != key:
                    u.total = {unit: 0.0 for unit in UNITS}
                    u.vendor = {}
                    u.period_key = key
                    u.version += 1
        return u

    # ── reservation lifecycle ─────────────────────────────────────────────
    def _check(self, ns: str, vendor: str, u_total: dict, u_vendor: dict,
               amounts: Dict[str, float]):
        spec = self._specs.get(ns)
        if spec is None:
            return
        for unit, lim in spec.limits.items():
            req = amounts.get(unit, 0.0)
            if u_total.get(unit, 0.0) + req > lim + 1e-12:
                raise QuotaExceeded(ns, unit, "namespace", u_total.get(unit, 0.0), req, lim)
        for unit, lim in spec.vendor_limits.get(vendor, {}).items():
            req = amounts.get(unit, 0.0)
            if u_vendor.get(unit, 0.0) + req > lim + 1e-12:
                raise QuotaExceeded(ns, unit, f"vendor:{vendor}",
                                    u_vendor.get(unit, 0.0), req, lim)

    def reserve(self, namespace: str, vendor: str, amounts: Dict[str, float],
                now: Optional[datetime] = None) -> Reservation:
        """Atomically reserve usage; raises QuotaExceeded naming the exhausted unit."""
        now = now or datetime.now(timezone.utc)
        u = self._maybe_reset(namespace, now)
        amounts = {k: float(v) for k, v in amounts.items() if k in UNITS}
        for _ in range(self.MAX_CAS_RETRIES):
            # Read (optimistic): snapshot + version
            version = u.version
            total = dict(u.total)
            vend = dict(u.vendor.get(vendor, {}))
            self._check(namespace, vendor, total, vend, amounts)
            # Compare-and-swap
            with self._lock:
                if u.version != version:
                    continue        # conflict: someone else committed; retry
                for k, v in amounts.items():
                    u.total[k] = u.total.get(k, 0.0) + v
                    u.vendor.setdefault(vendor, {})
                    u.vendor[vendor][k] = u.vendor[vendor].get(k, 0.0) + v
                u.version += 1
                r = Reservation(next(self._ids), namespace, vendor, amounts)
                self._reservations[r.id] = r
                return r
        raise RuntimeError("quota reservation: too many concurrent conflicts")

    def _adjust(self, r: Reservation, delta: Dict[str, float]):
        u = self._usage[r.namespace]
        with self._lock:
            for k, v in delta.items():
                u.total[k] = u.total.get(k, 0.0) + v
                u.vendor.setdefault(r.vendor, {})
                u.vendor[r.vendor][k] = u.vendor[r.vendor].get(k, 0.0) + v
            u.version += 1

    def reconcile(self, reservation: Reservation, actual: Dict[str, float]):
        """Replace the estimate with the actual usage reported by the provider."""
        if reservation.state != "reserved":
            raise ValueError(f"reservation {reservation.id} is {reservation.state}")
        delta = {k: float(actual.get(k, 0.0)) - reservation.amounts.get(k, 0.0)
                 for k in set(actual) | set(reservation.amounts) if k in UNITS}
        self._adjust(reservation, delta)
        reservation.amounts = {k: float(actual.get(k, 0.0)) for k in reservation.amounts}
        reservation.state = "reconciled"

    def release(self, reservation: Reservation):
        """Return a reservation (job unbound or failed before execution)."""
        if reservation.state != "reserved":
            return
        self._adjust(reservation, {k: -v for k, v in reservation.amounts.items()})
        reservation.state = "released"
