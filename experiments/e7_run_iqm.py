"""
E7 — run on IQM Garnet through IQM Resonance (Starter credits). USES CREDITS.

Run with the IQM venv:  ../.venv-iqm/bin/python e7_run_iqm.py [submit|collect]

submit: transpile every manifest circuit for the real Garnet backend (best of three
        seeds; these are the "actual" gate counts), save them, and submit ONE batch job.
collect: wait for the job and save counts. Token: macOS Keychain, service qsched-iqm.
Out: results/e7/run_iqm_garnet.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from qiskit import qasm2, transpile

import os
E7 = Path(__file__).resolve().parent / "results" / os.environ.get("QSCHED_RUN_DIR", "e7")   # X7: QSCHED_RUN_DIR=x7
OUT = E7 / "run_iqm_garnet.json"


def backend():
    from iqm.qiskit_iqm import IQMProvider
    tok = subprocess.run(["security", "find-generic-password", "-s", "qsched-iqm", "-w"],
                         capture_output=True, text=True).stdout.strip()
    return IQMProvider("https://resonance.iqm.tech", quantum_computer="garnet", token=tok).get_backend()


def counts_of(qc):
    ops = qc.count_ops()
    n2 = ops.get("cz", 0)
    n1 = sum(v for k, v in ops.items() if k not in ("cz", "measure", "barrier", "delay"))
    return n1, n2, qc.depth()


def submit():
    man = json.loads((E7 / "manifest.json").read_text())["devices"]["iqm_garnet"]
    if OUT.exists() and json.loads(OUT.read_text()).get("job_id"):
        sys.exit("already submitted; use collect")
    b = backend()
    names, circs, rows = [], [], {}
    for r in man["rows"]:
        qc = qasm2.loads((E7 / "circuits" / f"{r['circuit']}.qasm").read_text(),
                         custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        best = None
        for s in (11, 23, 37):
            out = transpile(qc, backend=b, optimization_level=1, seed_transpiler=s)
            if best is None or counts_of(out)[1] < counts_of(best)[1]:
                best = out
        n1, n2, d = counts_of(best)
        names.append(r["circuit"]); circs.append(best)
        rows[r["circuit"]] = dict(actual_n1=n1, actual_n2=n2, actual_depth=d)
    job = b.run(circs, shots=man["shots"])
    OUT.write_text(json.dumps({"job_id": str(job.job_id()), "names": names, "rows": rows,
                               "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    print("submitted IQM job", job.job_id(), "with", len(circs), "circuits")


def collect():
    data = json.loads(OUT.read_text())
    b = backend()
    job = b.retrieve_job(data["job_id"])
    res = job.result()
    for i, name in enumerate(data["names"]):
        data["rows"][name]["counts"] = res.get_counts(i)
    OUT.write_text(json.dumps(data, indent=1))
    print("collected", len(data["names"]), "circuits")


if __name__ == "__main__":
    {"submit": submit, "collect": collect}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()
