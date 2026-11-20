"""
X1 — Multi-job contention (discrete-event simulation, offline, no QPU).

Jobs arrive (Poisson) and compete for 8 simulated devices (the E2 targets), each with a FIFO queue.
Each job is one of the 102 E2 mirror circuits (drawn with replacement) with a requirement tau. Service
time on a device = QPU-seconds from the usage model on the transpiled circuit (results/x4_costs.csv).
Ground truth success = simulated success probability (E2 labels). Load = arrival rate x mean (over circuits)
of the service time on the job's fastest device / number of devices.

Policies (decision at arrival, no preemption):
  raw_best         device with the best average fidelity (what a user picking "the best QPU" does)
  lowest_cost      cheapest device
  shortest_queue   earliest expected start, ignoring fidelity (a quality-blind load balancer)
  estimate_greedy  ours, hard filter ESP_estimate >= tau, then highest predicted ESP (queue-blind)
  estimate_queue   ours, hard filter ESP_estimate >= tau, then earliest expected completion
  transpile_queue  same as estimate_queue on ESP of the transpiled circuit (exact, expensive oracle)
A job with no feasible candidate is rejected (false rejection if some device met tau).

Metrics per (load, policy): SLA violation rate, false-rejection rate, mean and p95 waiting time,
makespan, QPU-seconds, median decision latency (estimate vs transpile-all, from the E3 timings).

Run:  python x1_contention.py
Out:  results/x1_contention.csv, results/x1_contention.json, results/fig_x1_contention.{png,pdf}
"""
import json

import numpy as np
import pandas as pd

from common import RESULTS

SEED, N_JOBS, TAU = 20261120, 400, 0.7
LOADS = (0.25, 0.5, 0.75, 1.0)
POLICIES = ("raw_best", "lowest_cost", "shortest_queue", "estimate_greedy", "estimate_queue", "transpile_queue")


def simulate(df, load, policy, rng):
    circuits = df.circuit.unique()
    targets = sorted(df.target.unique())
    by_c = {c: g.set_index("target") for c, g in df.groupby("circuit")}
    # capacity reference: each job on its fastest device (load 1.0 = all devices busy if jobs spread
    # perfectly over the fastest options)
    mean_service = df.groupby("circuit").qpu_seconds.min().mean()
    rate = load * len(targets) / mean_service
    t, free_at = 0.0, {d: 0.0 for d in targets}
    rec = []
    for _ in range(N_JOBS):
        t += rng.exponential(1 / rate)
        c = circuits[rng.integers(len(circuits))]
        g = by_c[c]
        start = {d: max(t, free_at[d]) for d in targets}
        done = {d: start[d] + g.loc[d, "qpu_seconds"] for d in targets}
        feasible_true = bool((g.success_probability >= TAU).any())
        if policy == "raw_best":
            pick = g.raw_fidelity.idxmax()
        elif policy == "lowest_cost":
            pick = g.effective_cost.idxmin()
        elif policy == "shortest_queue":
            pick = min(targets, key=lambda d: start[d])
        else:
            col = "esp_actual" if policy == "transpile_queue" else "esp_estimated"
            ok = [d for d in targets if g.loc[d, col] >= TAU]
            if not ok:
                rec.append(dict(rejected=1, false_rej=int(feasible_true)))
                continue
            pick = (max(ok, key=lambda d: g.loc[d, col]) if policy == "estimate_greedy"
                    else min(ok, key=lambda d: done[d]))
        free_at[pick] = done[pick]
        rec.append(dict(rejected=0, false_rej=0, wait=start[pick] - t, end=done[pick],
                        violation=int(g.loc[pick, "success_probability"] < TAU),
                        qpu_s=g.loc[pick, "qpu_seconds"],
                        decision_ms=(0.0 if policy in ("raw_best", "lowest_cost", "shortest_queue") else
                                     g.stats_ms.iloc[0] + (g.transpile_ms.sum() if policy == "transpile_queue"
                                                           else g.estimate_ms.sum()))))
    r = pd.DataFrame(rec)
    ran = r[r.rejected == 0]
    return dict(load=load, policy=policy, jobs=N_JOBS, ran=int(len(ran)),
                violation_rate=round(float(ran.violation.mean()), 4) if len(ran) else None,
                false_rejection_rate=round(float(r.false_rej.mean()), 4),
                mean_wait_s=round(float(ran.wait.mean()), 2), p95_wait_s=round(float(ran.wait.quantile(0.95)), 2),
                makespan_s=round(float(ran.end.max()), 1), qpu_seconds=round(float(ran.qpu_s.sum()), 1),
                median_decision_ms=round(float(ran.decision_ms.median()), 2) if "decision_ms" in ran else 0.0)


def main():
    e2 = pd.read_csv(RESULTS / "e2_dataset_large.csv")
    costs = pd.read_csv(RESULTS / "x4_costs.csv")
    df = e2.merge(costs, on=["circuit", "target"])
    rows = []
    for load in LOADS:
        for pol in POLICIES:
            rows.append(simulate(df, load, pol, np.random.default_rng(SEED)))   # same arrivals per policy
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "x1_contention.csv", index=False)
    (RESULTS / "x1_contention.json").write_text(json.dumps(
        {"seed": SEED, "jobs": N_JOBS, "tau": TAU, "loads": LOADS, "rows": rows}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.0), layout="constrained")
    label = {"raw_best": "Highest raw fidelity", "lowest_cost": "Lowest cost",
             "shortest_queue": "Shortest queue (quality-blind)", "estimate_greedy": "Estimate filter, best ESP",
             "estimate_queue": "Estimate filter, earliest completion (ours)",
             "transpile_queue": "Transpiled ESP, earliest completion"}
    for pol, d in out.groupby("policy", sort=False):
        axes[0].plot(d.load, d.violation_rate, "o-", ms=4, label=label[pol])
        axes[1].plot(d.load, d.mean_wait_s, "o-", ms=4, label=label[pol])
        axes[2].plot(d.load, d.false_rejection_rate, "o-", ms=4, label=label[pol])
    for ax, t in zip(axes, ("SLA violations", "Waiting time", "False rejections")):
        ax.set_title(t); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Share of jobs that ran")
    axes[1].set_ylabel("Mean wait (s)"); axes[1].set_yscale("log")
    axes[2].set_ylabel("Share of all jobs")
    for ax in axes:
        ax.set_xlabel("Offered load")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=11, frameon=False)
    fig.savefig(RESULTS / "fig_x1_contention.png", dpi=200)
    fig.savefig(RESULTS / "fig_x1_contention.pdf")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
