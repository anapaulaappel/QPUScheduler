"""
E10 — application circuits on IBM devices. CONSUMES QPU TIME in `submit`.

Same flow as E6: the estimate (no transpilation) and the ESP of the transpiled circuit use the
backend's published calibration at preparation time.
  python e10_run_ibm.py prepare   # transpile + estimates + QPU-time estimate (no QPU)
  python e10_run_ibm.py submit    # one SamplerV2 job per backend; saves job ids
  python e10_run_ibm.py collect   # fetch counts -> results/e10/run_<backend>.json
Options: --backends ibm_fez,ibm_marrakesh,ibm_kingston  --shots 2000
"""
import argparse
import json
import pickle
from datetime import datetime, timezone

from qiskit import qasm2

from common import RESULTS, count_ops, esp, transpile_best
from e2_fidelity_prediction import e1_alpha

E10 = RESULTS / "e10"
PREP = E10 / "ibm_prepared.pkl"
JOBS = E10 / "ibm_jobs.json"


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService()


def prepare(args):
    from backends.target_telemetry import telemetry_from_backend
    from circuit_analysis import circuit_stats, estimate_divergence
    from usage import estimate_qpu_seconds, estimate_usage
    svc = service()
    alpha = e1_alpha("static_added_2q")
    man = json.loads((E10 / "manifest.json").read_text())["all"]
    prepared = {"created": datetime.now(timezone.utc).isoformat(), "shots": args.shots, "backends": {}}
    total = 0.0
    for name in args.backends.split(","):
        backend = svc.backend(name)
        cal = backend.properties(refresh=True).last_update_date
        tel = telemetry_from_backend(backend, name=name, topology="heavy-hex", provider="ibm-quantum",
                                     pricing={"scheme": "time", "price_per_second": 1.60})
        rows, circuits = [], []
        for b in man:
            qc = qasm2.loads((E10 / "circuits" / f"{b['name']}.qasm").read_text(),
                             custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            logical = qc.remove_final_measurements(inplace=False)
            st = circuit_stats(logical)
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
            rows.append(dict(circuit=b["name"], family=b["family"], n=b["n"], target=name,
                             n1_logical=st.num_single_qubit_gates, n2_logical=st.num_two_qubit_gates,
                             actual_n1=n1a, actual_n2=n2a, actual_depth=deptha,
                             raw_fidelity=tel.mean_qubit_fidelity,
                             esp_estimated=esp(n1e, n2e, b["n"], e1, e2, ero, t_e, T2),
                             esp_actual=esp(n1a, n2a, b["n"], e1, e2, ero, t_a, T2)))
            circuits.append(out)
        u = sum(estimate_usage(tel.pricing, {**tel.durations, "job_overhead_s": 0},
                               r["actual_depth"], 0, r["actual_n2"], args.shots).qpu_seconds
                for r in rows) + 2.0
        total += u
        prepared["backends"][name] = {"rows": rows, "circuits": circuits, "estimated_qpu_seconds": u,
                                      "published_calibration": str(cal)}
        print(f"{name}: {len(rows)} circuits, ≈ {u:.1f} s; published calibration {cal}")
    print(f"TOTAL ≈ {total:.1f} s")
    PREP.write_bytes(pickle.dumps(prepared))


def submit(args):
    from qiskit_ibm_runtime import SamplerV2
    svc = service()
    prepared = pickle.loads(PREP.read_bytes())
    jobs = json.loads(JOBS.read_text()) if JOBS.exists() else {}
    for name, d in prepared["backends"].items():
        if name in jobs:
            print(f"{name}: already submitted"); continue
        job = SamplerV2(mode=svc.backend(name)).run(d["circuits"], shots=prepared["shots"])
        jobs[name] = {"job_id": job.job_id(), "submitted": datetime.now(timezone.utc).isoformat()}
        JOBS.write_text(json.dumps(jobs, indent=2))
        print(f"{name}: submitted {job.job_id()}")


def collect(args):
    svc = service()
    prepared = pickle.loads(PREP.read_bytes())
    for name, j in json.loads(JOBS.read_text()).items():
        job = svc.job(j["job_id"])
        if str(job.status()) != "DONE":
            print(f"{name}: {job.status()}"); continue
        res = job.result()
        d = prepared["backends"][name]
        out = {}
        for r, pub in zip(d["rows"], res):
            creg = next(iter(pub.data.__dict__.values()))       # measure_all -> 'meas'
            out[r["circuit"]] = {**r, "shots": prepared["shots"], "raw_counts": creg.get_counts(),
                                 "published_calibration": d["published_calibration"],
                                 "job_id": j["job_id"]}
        try:
            usage = job.usage()
        except Exception:
            usage = None
        (E10 / f"run_{name}.json").write_text(json.dumps({"usage_seconds": usage, "rows": out}, indent=1))
        print(f"{name}: collected {len(out)} circuits; usage {usage} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "submit", "collect"])
    ap.add_argument("--backends", default="ibm_fez,ibm_marrakesh,ibm_kingston")
    ap.add_argument("--shots", type=int, default=2000)
    args = ap.parse_args()
    {"prepare": prepare, "submit": submit, "collect": collect}[args.phase](args)
