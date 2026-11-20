"""
E6 — Hardware validation on real IBM QPUs.

Runs the E2 mirror circuits on real IBM backends and compares the MEASURED
success probability with (i) the noisy-simulation label built from the same
backend's current calibration and (ii) the predictors (raw fidelity, ESP from
the topology estimate, ESP of the actual transpiled circuit).

THIS CONSUMES QPU TIME. Phases:
  python e6_hardware_validation.py prepare   # transpile + cost estimate (no QPU)
  python e6_hardware_validation.py submit    # submit one job per backend; saves job ids
  python e6_hardware_validation.py collect   # fetch results, simulate, analyze

Options: --backends ibm_fez,ibm_marrakesh  --shots 4000  --sizes 4,6,8,10
Out: results/e6_jobs.json, results/e6_hardware.csv, results/e6_summary.json, results/e6_hardware.png
"""

import argparse
import json
import pickle
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS, benchmark_suite, count_ops, esp, transpile_best
from e2_fidelity_prediction import e1_alpha, mirror

warnings.filterwarnings("ignore")
TAG = ""   # set from --tag; keeps runs with different circuit sets apart
JOBS = RESULTS / "e6_jobs.json"
PREP = RESULTS / "e6_prepared.pkl"


def out(name):
    """results/e6_<name><tag>.<ext>"""
    stem, ext = name.rsplit(".", 1)
    return RESULTS / f"e6_{stem}{TAG}.{ext}"


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService()


def prepare(args):
    from backends.target_telemetry import telemetry_from_backend
    from circuit_analysis import circuit_stats, estimate_divergence
    from usage import estimate_qpu_seconds, estimate_usage

    svc = service()
    alpha = e1_alpha("static_added_2q")
    sizes = tuple(int(x) for x in args.sizes.split(","))
    benches = benchmark_suite(sizes=sizes, instances=args.instances)
    prepared = {"created": datetime.now(timezone.utc).isoformat(), "shots": args.shots,
                "alpha": alpha, "backends": {}}
    total_s = 0.0
    for name in args.backends.split(","):
        backend = svc.backend(name)
        tel = telemetry_from_backend(backend, name=name, topology="heavy-hex",
                                     provider="ibm-quantum",
                                     pricing={"scheme": "time", "price_per_second": 1.60})
        rows, circuits = [], []
        for i, b in enumerate(benches):
            circ, key = mirror(b.circuit, seed=100 + i)
            st = circuit_stats(circ)
            qc = circ.copy()
            qc.measure_all()
            out, _ = transpile_best(qc, backend)
            n1a, n2a, deptha = count_ops(out)
            est = estimate_divergence(st, tel.coupling_map, tel.qubit_count, tel.native_gates)
            n1e = st.num_single_qubit_gates * est.gate_overhead
            n2e = st.num_two_qubit_gates + alpha * est.added_two_qubit_gates
            de = st.depth * (1 + est.swap_ratio * alpha)
            dur = {**tel.durations, "reset_us": 0, "job_overhead_s": 0}
            t_e = estimate_qpu_seconds(de, n1e, n2e, 1, dur)
            t_a = estimate_qpu_seconds(deptha, n1a, n2a, 1, dur)
            e1, e2, ero, T2 = tel.mean_sq_error, tel.mean_tq_error, tel.mean_ro_error, tel.mean_t2_us
            rows.append(dict(
                circuit=b.name, family=b.family, n=b.num_qubits, target=name, key=key,
                n2_logical=st.num_two_qubit_gates, actual_n2=n2a, actual_depth=deptha,
                raw_fidelity=tel.mean_qubit_fidelity,
                esp_estimated=esp(n1e, n2e, b.num_qubits, e1, e2, ero, t_e, T2),
                esp_actual=esp(n1a, n2a, b.num_qubits, e1, e2, ero, t_a, T2)))
            circuits.append(out)
        u = sum(estimate_usage(tel.pricing, {**tel.durations, "job_overhead_s": 0},
                               r["actual_depth"], 0, r["actual_n2"], args.shots).qpu_seconds
                for r in rows) + 2.0
        total_s += u
        prepared["backends"][name] = {"rows": rows, "circuits": circuits,
                                      "estimated_qpu_seconds": u,
                                      "calibration": tel.calibration_timestamp}
        print(f"{name}: {len(circuits)} circuits, estimated QPU time ≈ {u:.1f} s")
    print(f"TOTAL estimated QPU time ≈ {total_s:.1f} s (shots={args.shots})")
    with open(PREP, "wb") as f:
        pickle.dump(prepared, f)


def submit(args):
    from qiskit_ibm_runtime import SamplerV2
    svc = service()
    with open(PREP, "rb") as f:
        prepared = pickle.load(f)
    jobs = json.loads(JOBS.read_text()) if JOBS.exists() else {}
    for name, d in prepared["backends"].items():
        if name in jobs:
            print(f"{name}: already submitted ({jobs[name]['job_id']})")
            continue
        sampler = SamplerV2(mode=svc.backend(name))
        job = sampler.run(d["circuits"], shots=prepared["shots"])
        jobs[name] = {"job_id": job.job_id(),
                      "submitted": datetime.now(timezone.utc).isoformat()}
        JOBS.write_text(json.dumps(jobs, indent=2))     # save immediately
        print(f"{name}: submitted job {job.job_id()}")


def collect(args):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    svc = service()
    with open(PREP, "rb") as f:
        prepared = pickle.load(f)
    jobs = json.loads(JOBS.read_text())
    shots = prepared["shots"]
    all_rows, usage = [], {}
    for name, info in jobs.items():
        job = svc.job(info["job_id"])
        status = job.status()
        status = getattr(status, "name", status)
        if str(status) != "DONE":
            print(f"{name}: job {info['job_id']} status {status}; skipping")
            continue
        try:
            usage[name] = job.usage()
        except Exception:
            usage[name] = None
        result = job.result()
        d = prepared["backends"][name]
        sim = AerSimulator(noise_model=NoiseModel.from_backend(svc.backend(name)), seed_simulator=3)
        for r, circ, pub in zip(d["rows"], d["circuits"], result):
            counts = pub.data.meas.get_counts()
            sim_counts = sim.run(circ, shots=shots).result().get_counts()
            all_rows.append({**r, "measured": counts.get(r["key"], 0) / shots,
                             "simulated": sim_counts.get(r["key"], 0) / shots})
        print(f"{name}: collected {len(d['rows'])} circuits; QPU usage {usage[name]} s")
    df = pd.DataFrame(all_rows)
    df.to_csv(out("hardware.csv"), index=False)

    summary = {"shots": shots, "usage_seconds": usage, "n": int(len(df)),
               "backends": sorted(df.target.unique().tolist())}
    for col in ("simulated", "raw_fidelity", "esp_estimated", "esp_actual"):
        summary[col] = {
            "spearman_vs_measured": float(spearmanr(df[col], df.measured).correlation),
            "mae_vs_measured": float((df[col] - df.measured).abs().mean())
            if col != "raw_fidelity" else None,
        }
    # Target selection between the real backends, per circuit
    if df.target.nunique() > 1:
        for col in ("simulated", "raw_fidelity", "esp_estimated", "esp_actual"):
            reg, hit = [], []
            for _, g in df.groupby("circuit"):
                pick, best = g.loc[g[col].idxmax()], g.loc[g.measured.idxmax()]
                reg.append(best.measured - pick.measured)
                hit.append(pick.target == best.target)
            summary[col]["mean_regret"] = float(np.mean(reg))
            summary[col]["top1_accuracy"] = float(np.mean(hit))
    out("summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot(df)
    print(json.dumps(summary, indent=2, default=str))


def plot(df):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title in [(axes[0], "simulated", "Noisy simulation (same calibration)"),
                           (axes[1], "esp_estimated", "ESP, topology estimate (no transpilation)"),
                           (axes[2], "esp_actual", "ESP, actual transpilation")]:
        for t, d in df.groupby("target"):
            ax.scatter(d[col], d.measured, s=16, alpha=0.75, label=t)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        rho = spearmanr(df[col], df.measured).correlation
        ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}", fontsize=10)
        ax.set_xlabel("Predicted / simulated success")
        ax.set_ylabel("Measured success on hardware")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out("hardware.png"), dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "submit", "collect"])
    ap.add_argument("--backends", default="ibm_fez,ibm_marrakesh")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--sizes", default="4,6,8,10")
    ap.add_argument("--instances", type=int, default=1,
                    help="random instances per (family, size), as in E2")
    ap.add_argument("--tag", default="", help="suffix for all E6 files, e.g. _large")
    args = ap.parse_args()
    TAG = args.tag
    JOBS = out("jobs.json")
    PREP = out("prepared.pkl")
    {"prepare": prepare, "submit": submit, "collect": collect}[args.phase](args)
