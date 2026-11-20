"""
E7 — verbatim submission to Rigetti through Open Quantum (prevents the provider
compiler from simplifying the mirror circuits). USES CREDITS unless --quote-only.

Run with the Open Quantum venv:
  ../.venv-oq/bin/python e7_verbatim_oq.py --circuits bernstein_vazirani_4,real_amplitudes_8 --quote-only
  ../.venv-oq/bin/python e7_verbatim_oq.py --circuits bernstein_vazirani_4,real_amplitudes_8

Each circuit is transpiled to Rigetti's native set {rz, rx(±pi/2), rx(pi), cz} on the
device coupling map, written as OpenQASM 3 with physical qubits inside a
`#pragma braket verbatim` box (measurements outside the box), and submitted with
disable_qubit_rewiring=True. Raw counts are saved without reordering, so the bit
order can be checked against a shallow circuit.
Out: results/e7/run_rigetti_verbatim.json, results/e7/verbatim/<circuit>.qasm
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

from qiskit import QuantumCircuit, qasm2, transpile

E7 = Path(__file__).resolve().parent / "results" / "e7"
DEVICE = "rigetti_cepheus"
BACKEND_CLASS = "rigetti:cepheus-1-108q"
os.environ.setdefault("OPENQUANTUM_SDK_KEY", os.path.expanduser("~/.config/qsched/openquantum-sdk-key.json"))


def transpile_best(qc, both):
    best = None
    for s in (11, 23, 37):
        pre = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
        out = transpile(pre, basis_gates=["rz", "sx", "x", "cz"], coupling_map=both,
                        optimization_level=1, seed_transpiler=s)
        if best is None or out.count_ops().get("cz", 0) < best.count_ops().get("cz", 0):
            best = out
    return best


def to_native(qc, edges):
    both = edges + [[b, a] for a, b in edges]
    best = transpile_best(qc, both)
    # A qubit with no two-qubit gates may land on a qubit with no couplers (Cepheus $8), which
    # the device rejects with "Invalid qubit(s)" only after the job has waited in the queue.
    connected = sorted({q for e in edges for q in e})
    used = {best.find_bit(q).index for inst in best.data if inst.operation.name != "barrier"
            for q in inst.qubits}
    if used <= set(connected):
        return best
    # Transpile on the connected qubits only, then write the result on the physical indices.
    idx = {q: i for i, q in enumerate(connected)}
    compact = transpile_best(qc, [[idx[a], idx[b]] for a, b in both])
    out = QuantumCircuit(max(connected) + 1, compact.num_clbits)
    for inst in compact.data:
        out.append(inst.operation, [connected[compact.find_bit(q).index] for q in inst.qubits],
                   [compact.find_bit(c).index for c in inst.clbits])
    return out


def to_verbatim_qasm3(circ):
    """OpenQASM 3 with physical qubits; gates in a verbatim box, measures after it."""
    nclb = circ.num_clbits
    body, meas = [], []
    for inst in circ.data:
        op, qs = inst.operation, [circ.find_bit(q).index for q in inst.qubits]
        if op.name in ("barrier", "delay"):
            continue
        if op.name == "measure":
            meas.append(f"b[{circ.find_bit(inst.clbits[0]).index}] = measure ${qs[0]};")
        elif op.name == "rz":
            body.append(f"  rz({float(op.params[0]):.16f}) ${qs[0]};")
        elif op.name == "sx":
            body.append(f"  rx({math.pi / 2:.16f}) ${qs[0]};")
        elif op.name == "x":
            body.append(f"  rx({math.pi:.16f}) ${qs[0]};")
        elif op.name == "cz":
            body.append(f"  cz ${qs[0]}, ${qs[1]};")
        else:
            raise ValueError(f"unexpected gate {op.name}")
    return "\n".join(["OPENQASM 3.0;", f"bit[{nclb}] b;", "#pragma braket verbatim", "box {",
                      *body, "}", *meas]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--quote-only", action="store_true")
    args = ap.parse_args()

    dev = json.loads((E7 / f"device_{DEVICE}.json").read_text())
    (E7 / "verbatim").mkdir(exist_ok=True)
    out_p = E7 / "run_rigetti_verbatim.json"
    done = json.loads(out_p.read_text()) if out_p.exists() else {}

    from openquantum_sdk.clients import JobSubmissionConfig, select_plan_and_priority
    from openquantum_sdk.models import JobPreparationCreate
    from openquantum_sdk.session import OpenQuantumClient
    client = OpenQuantumClient.from_env()
    s = client.scheduler
    org = s._resolve_organization_id(None)

    for name in args.circuits.split(","):
        if name in done and done[name].get("counts") and not args.quote_only:
            print(f"{name}: already done"); continue
        qc = qasm2.loads((E7 / "circuits" / f"{name}.qasm").read_text(),
                         custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        nat = to_native(qc, dev["edges"])
        ops = nat.count_ops()
        n2 = ops.get("cz", 0)
        n1 = sum(v for k, v in ops.items() if k in ("rz", "sx", "x"))
        text = to_verbatim_qasm3(nat)
        path = E7 / "verbatim" / f"{name}.qasm"
        path.write_text(text)
        cfg = {"shots": args.shots, "disable_qubit_rewiring": True}

        if args.quote_only:
            up = s.upload_job_input(file_path=str(path))
            r = s.prepare_job(JobPreparationCreate(
                organization_id=org, backend_class_id=BACKEND_CLASS, name=f"qsched-e7v-{name}",
                upload_endpoint_id=up, job_subcategory_id="phys:oth", shots=args.shots,
                configuration_data=cfg, submitted_with="sdk", input_format="qasm"))
            res = s.wait_for_preparation(r.id if hasattr(r, "id") else r.preparation_id)
            sel = select_plan_and_priority(res.quote, "auto", "auto")
            print(f"{name}: valid; cz={n2}; quote {s._compute_total_cost(res.quote, sel.execution_plan_id, sel.queue_priority_id)} credits")
            continue

        bal = client.management.get_credit_balance(org)
        print(f"{name}: submitting verbatim (cz={n2}; {bal.spark_credits} credits left)", flush=True)
        t0 = time.time()
        try:
            job = s.submit_job(JobSubmissionConfig(
                backend_class_id=BACKEND_CLASS, name=f"qsched-e7v-{name}", job_subcategory_id="phys:oth",
                shots=args.shots, configuration_data=cfg, auto_approve_quote=True, verbose=True), file_path=str(path))
            counts, err, jid = s.download_job_output(job), None, str(job.id)
        except Exception as e:
            counts, err, jid = None, f"{type(e).__name__}: {e}"[:400], None
        done[name] = dict(job_id=jid, actual_n1=n1, actual_n2=n2, actual_depth=nat.depth(),
                          raw_counts=counts, error=err, seconds=round(time.time() - t0, 1))
        out_p.write_text(json.dumps(done, indent=1))
        print(f"{name}: {'ok' if counts else 'ERROR ' + str(err)} in {done[name]['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
