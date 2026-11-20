"""
E7 — verbatim submission to AQT IBEX Q1 (trapped ion) through Open Quantum.
USES CREDITS unless --quote-only.

Run with the Open Quantum venv:
  ../.venv-oq/bin/python e7_verbatim_aqt.py --quote-only          # free: validates the files
  ../.venv-oq/bin/python e7_verbatim_aqt.py --circuits ghz_6,...  # runs (2 credits/circuit at 100 shots)

Native AQT operations (Open Quantum device document): prx(theta, phi), rz(phi), xx(theta).
Each circuit is transpiled to Qiskit's {r, rz, rxx} (r(θ,φ) ≡ prx(θ,φ); rxx(θ) ≡ xx(θ) =
exp(-iθ/2 X⊗X)), written as OpenQASM 3 with physical qubits inside a `#pragma braket verbatim`
box, angles with 16 decimals, and submitted with disable_qubit_rewiring=True. IBEX Q1 is
fully connected, so no routing is needed.

Before trusting results, run one shallow circuit first and check (i) that it succeeds with
high probability and (ii) the bit order, as done for Rigetti; the xx(θ) convention is
assumed to match Qiskit's rxx(θ). Negative xx angles are kept as they are; if the
device rejects them, rewrite xx(-θ) = (Z⊗I) xx(θ) (Z⊗I).
Out: results/e7/verbatim_aqt/<circuit>.qasm, results/e7/run_aqt_verbatim.json
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

from qiskit import qasm2, transpile

E7 = Path(__file__).resolve().parent / "results" / "e7"
BACKEND_CLASS = "aqt:ibex-q1"
os.environ.setdefault("OPENQUANTUM_SDK_KEY", os.path.expanduser("~/.config/qsched/openquantum-sdk-key.json"))


def to_native(qc, n_qubits):
    full = [[a, b] for a in range(n_qubits) for b in range(n_qubits) if a != b]
    best = None
    for s in (11, 23, 37):
        pre = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
        out = transpile(pre, basis_gates=["r", "rz", "rxx"], coupling_map=full,
                        optimization_level=1, seed_transpiler=s)
        if best is None or out.count_ops().get("rxx", 0) < best.count_ops().get("rxx", 0):
            best = out
    return best


def to_verbatim_qasm3(circ):
    body, meas = [], []
    for inst in circ.data:
        op, qs = inst.operation, [circ.find_bit(q).index for q in inst.qubits]
        p = [float(x) for x in op.params]
        if op.name in ("barrier", "delay"):
            continue
        if op.name == "measure":
            meas.append(f"b[{circ.find_bit(inst.clbits[0]).index}] = measure ${qs[0]};")
        elif op.name == "r":
            theta, phi = p
            if theta < 0:                    # prx(-θ, φ) = prx(θ, φ + π): keep θ in [0, π]
                theta, phi = -theta, phi + math.pi
            phi = phi % (2 * math.pi)        # non-negative: a job with phi < 0 failed on the device
            if phi > 2 * math.pi - 1e-9:     # rounding: 2π is the same rotation as 0
                phi = 0.0
            body.append(f"  prx({theta:.16f}, {phi:.16f}) ${qs[0]};")
        elif op.name == "rz":
            body.append(f"  rz({p[0]:.16f}) ${qs[0]};")
        elif op.name == "rxx":
            body.append(f"  xx({p[0]:.16f}) ${qs[0]}, ${qs[1]};")
        else:
            raise ValueError(f"unexpected gate {op.name}")
    return "\n".join(["OPENQASM 3.0;", f"bit[{circ.num_clbits}] b;", "#pragma braket verbatim",
                      "box {", *body, "}", *meas]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="")
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--quote-only", action="store_true")
    args = ap.parse_args()

    dev = json.loads((E7 / "device_aqt_ibex_q1.json").read_text())
    man = json.loads((E7 / "manifest.json").read_text())["devices"]["aqt_ibex_q1"]["rows"]
    names = args.circuits.split(",") if args.circuits else [r["circuit"] for r in man]
    (E7 / "verbatim_aqt").mkdir(exist_ok=True)
    out_p = E7 / "run_aqt_verbatim.json"
    done = json.loads(out_p.read_text()) if out_p.exists() else {}

    from openquantum_sdk.clients import JobSubmissionConfig, select_plan_and_priority
    from openquantum_sdk.models import JobPreparationCreate
    from openquantum_sdk.session import OpenQuantumClient
    client = OpenQuantumClient.from_env()
    s = client.scheduler
    org = s._resolve_organization_id(None)
    cfg = {"shots": args.shots, "disable_qubit_rewiring": True}

    for name in names:
        qc = qasm2.loads((E7 / "circuits" / f"{name}.qasm").read_text(),
                         custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        nat = to_native(qc, dev["n_qubits"])
        ops = nat.count_ops()
        n2 = ops.get("rxx", 0)
        n1 = ops.get("r", 0) + ops.get("rz", 0)
        path = E7 / "verbatim_aqt" / f"{name}.qasm"
        path.write_text(to_verbatim_qasm3(nat))

        if args.quote_only:
            up = s.upload_job_input(file_path=str(path))
            r = s.prepare_job(JobPreparationCreate(
                organization_id=org, backend_class_id=BACKEND_CLASS, name=f"qsched-e7a-{name}",
                upload_endpoint_id=up, job_subcategory_id="phys:oth", shots=args.shots,
                configuration_data=cfg, submitted_with="sdk", input_format="qasm"))
            res = s.wait_for_preparation(r.id if hasattr(r, "id") else r.preparation_id)
            sel = select_plan_and_priority(res.quote, "auto", "auto")
            cost = s._compute_total_cost(res.quote, sel.execution_plan_id, sel.queue_priority_id)
            print(f"{name}: accepted by quote; xx={n2}, 1q={n1}; {cost} credits", flush=True)
            continue

        if name in done and done[name].get("raw_counts"):
            print(f"{name}: already done"); continue
        t0 = time.time()
        try:
            job = s.submit_job(JobSubmissionConfig(
                backend_class_id=BACKEND_CLASS, name=f"qsched-e7a-{name}", job_subcategory_id="phys:oth",
                shots=args.shots, configuration_data=cfg, auto_approve_quote=True, verbose=True), file_path=str(path))
            counts, err, jid = s.download_job_output(job), None, str(job.id)
        except Exception as e:
            counts, err, jid = None, f"{type(e).__name__}: {e}"[:400], None
        done[name] = dict(job_id=jid, actual_n1=n1, actual_n2=n2, actual_depth=nat.depth(),
                          raw_counts=counts, error=err, seconds=round(time.time() - t0, 1))
        out_p.write_text(json.dumps(done, indent=1))
        print(f"{name}: {'ok' if counts else 'ERROR ' + str(err)}", flush=True)
        if err:                              # do not keep spending credits after a failure
            print("STOP: check this error (the job may still be running) before rerunning", flush=True)
            break


if __name__ == "__main__":
    main()
