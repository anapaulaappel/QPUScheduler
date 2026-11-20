"""
Unified usage and cost model
-----------------------------------------------------------
Providers charge in different units: per task, per shot, per gate-shot, per
QPU second, or through prepaid QPU-time plans. For each candidate QPU this
module estimates the job's usage in every relevant unit plus an EFFECTIVE
monetary cost used to compare providers on a single scale.

  QPU time      t_est = shots · (circuit_duration + reset) + per_job_overhead
  Pay-per-use   C     = task_fee + shots · (price_per_shot
                                            + price_1q · N_1q + price_2q · N_2q)
                        + t_est · price_per_second
  Prepaid time  C = 0 (consumes t_est from the plan);
                C_eff = t_est · plan_price / plan_seconds

Quota enforcement uses each usage value in its native unit; scoring and the
routing agent use C_eff.
"""

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_JOB_OVERHEAD_S = 1.0


@dataclass
class UsageEstimate:
    qpu_seconds: float
    cost_usd: float                 # pay-per-use charge (0 for prepaid plans)
    effective_cost_usd: float       # comparable cost across pricing schemes
    shots: int
    scheme: str = "free"
    breakdown: dict = field(default_factory=dict)

    def units(self) -> dict:
        """Usage per quota unit."""
        return {"usd": self.cost_usd, "qpu_seconds": self.qpu_seconds,
                "shots": float(self.shots)}

    @property
    def effective_cost_per_shot(self) -> float:
        return self.effective_cost_usd / max(self.shots, 1)


def estimate_qpu_seconds(depth: int, n1q: int, n2q: int, shots: int,
                         durations: dict) -> float:
    """
    Estimated QPU time of a job.

    Circuit duration ≈ depth · (mean layer time) + measurement, where the mean
    layer time weights 1Q and 2Q gate durations by their share of the gates.
    """
    t1 = float(durations.get("gate_1q_ns", 0.0))
    t2 = float(durations.get("gate_2q_ns", 0.0))
    tm = float(durations.get("measure_ns", 0.0))
    reset_s = float(durations.get("reset_us", 0.0)) * 1e-6
    overhead_s = float(durations.get("job_overhead_s", DEFAULT_JOB_OVERHEAD_S)) \
        if (t1 or t2 or tm or reset_s) else 0.0
    total = max(n1q + n2q, 1)
    frac_2q = n2q / total
    layer_ns = frac_2q * t2 + (1 - frac_2q) * t1
    circuit_s = (depth * layer_ns + tm) * 1e-9
    return shots * (circuit_s + reset_s) + overhead_s


def estimate_usage(pricing: Optional[dict], durations: Optional[dict],
                   depth: int, n1q: int, n2q: int, shots: int,
                   cost_per_shot: float = 0.0) -> UsageEstimate:
    """Usage and cost of running a (transpiled) circuit on one QPU."""
    pricing = dict(pricing or {})
    durations = dict(durations or {})
    scheme = pricing.get("scheme", "per_shot" if cost_per_shot else "free")
    t_est = estimate_qpu_seconds(depth, n1q, n2q, shots, durations)

    task_fee = float(pricing.get("task_fee", 0.0))
    per_shot = float(pricing.get("price_per_shot", cost_per_shot or 0.0))
    p1 = float(pricing.get("price_1q", 0.0))
    p2 = float(pricing.get("price_2q", 0.0))
    per_second = float(pricing.get("price_per_second", 0.0))

    if scheme == "free":
        cost = eff = 0.0
    elif scheme == "prepaid_time":
        plan_price = float(pricing.get("plan_price", 0.0))
        plan_seconds = float(pricing.get("plan_seconds", 0.0))
        cost = 0.0
        eff = t_est * plan_price / plan_seconds if plan_seconds > 0 else 0.0
    else:
        # per_shot, per_gate_shot, time, or any mixture of the terms
        cost = (task_fee + shots * (per_shot + p1 * n1q + p2 * n2q)
                + t_est * per_second)
        cost = max(cost, float(pricing.get("min_charge", 0.0)))
        eff = cost

    return UsageEstimate(
        qpu_seconds=t_est, cost_usd=cost, effective_cost_usd=eff, shots=shots,
        scheme=scheme,
        breakdown={"depth": depth, "n1q": n1q, "n2q": n2q},
    )
