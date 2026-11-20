"""
E10 — application circuits on Rigetti Cepheus and AQT IBEX Q1 through Open Quantum, verbatim.
USES CREDITS unless --quote-only.

Reuses the E7 verbatim conversion (native gates on physical qubits inside a
`#pragma braket verbatim` box, so the provider compiler cannot simplify the circuit).

Run with the Open Quantum venv:
  ../.venv-oq/bin/python e10_run_oq.py --device rigetti_cepheus --quote-only   # free
  ../.venv-oq/bin/python e10_run_oq.py --device aqt_ibex_q1 --quote-only
  ../.venv-oq/bin/python e10_run_oq.py --device rigetti_cepheus                # spends credits
Options: --circuits a,b,c   --shots N (default: manifest)   --max-credits N (stop above it)

Out: results/e10/verbatim_<device>/<circuit>.qasm, results/e10/run_<device>.json,
     results/e10/quote_<device>.json
"""
import argparse
import json
import os
import time
from pathlib import Path

from qiskit import qasm2

import e7_verbatim_aqt as aqt
import e7_verbatim_oq as rig

E10 = Path(__file__).resolve().parent / "results" / os.environ.get("QSCHED_RUN_DIR", "e10")   # X7: QSCHED_RUN_DIR=x7
E7 = Path(__file__).resolve().parent / "results" / "e7"
os.environ.setdefault("OPENQUANTUM_SDK_KEY", os.path.expanduser("~/.config/qsched/openquantum-sdk-key.json"))

DEV = {
    "rigetti_cepheus": dict(backend=rig.BACKEND_CLASS, two_q="cz", one_q=("rz", "sx", "x"),
                            native=lambda qc, d: rig.to_native(qc, d["edges"]),
                            qasm3=rig.to_verbatim_qasm3),
    "aqt_ibex_q1": dict(backend=aqt.BACKEND_CLASS, two_q="rxx", one_q=("r", "rz"),
                        native=lambda qc, d: aqt.to_native(qc, d["n_qubits"]),
                        qasm3=aqt.to_verbatim_qasm3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=list(DEV))
    ap.add_argument("--circuits", default="")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--quote-only", action="store_true")
    ap.add_argument("--max-credits", type=float, default=0,
                    help="stop before a submission that would take total spend above this")
    args = ap.parse_args()

    spec = DEV[args.device]
    dev = json.loads((E7 / f"device_{args.device}.json").read_text())
    man = json.loads((E10 / "manifest.json").read_text())["devices"][args.device]
    shots = args.shots or man["shots"]
    names = args.circuits.split(",") if args.circuits else [r["circuit"] for r in man["rows"]]
    vdir = E10 / f"verbatim_{args.device}"
    vdir.mkdir(exist_ok=True)
    out_p = E10 / f"run_{args.device}.json"
    done = json.loads(out_p.read_text()) if out_p.exists() else {}
    quote_p = E10 / f"quote_{args.device}.json"
    quotes = json.loads(quote_p.read_text()) if quote_p.exists() else {}

    from openquantum_sdk.clients import JobSubmissionConfig, select_plan_and_priority
    from openquantum_sdk.models import JobPreparationCreate
    from openquantum_sdk.session import OpenQuantumClient
    client = OpenQuantumClient.from_env()
    s = client.scheduler
    org = s._resolve_organization_id(None)
    cfg = {"shots": shots, "disable_qubit_rewiring": True}
    spent = 0.0

    for name in names:
        qc = qasm2.loads((E10 / "circuits" / f"{name}.qasm").read_text(),
                         custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        nat = spec["native"](qc, dev)
        ops = nat.count_ops()
        n2 = ops.get(spec["two_q"], 0)
        n1 = sum(v for k, v in ops.items() if k in spec["one_q"])
        path = vdir / f"{name}.qasm"
        path.write_text(spec["qasm3"](nat))

        if args.quote_only:
            try:
                up = s.upload_job_input(file_path=str(path))
                r = s.prepare_job(JobPreparationCreate(
                    organization_id=org, backend_class_id=spec["backend"], name=f"qsched-e10-{name}",
                    upload_endpoint_id=up, job_subcategory_id="phys:oth", shots=shots,
                    configuration_data=cfg, submitted_with="sdk", input_format="qasm"))
                res = s.wait_for_preparation(r.id if hasattr(r, "id") else r.preparation_id)
                sel = select_plan_and_priority(res.quote, "auto", "auto")
                cost = float(s._compute_total_cost(res.quote, sel.execution_plan_id, sel.queue_priority_id))
                quotes[name] = dict(credits=cost, shots=shots, n2=n2, n1=n1, depth=nat.depth())
                print(f"{name}: valid; 2q={n2} 1q={n1}; quote {cost} credits", flush=True)
            except Exception as e:
                quotes[name] = dict(error=f"{type(e).__name__}: {e}"[:300])
                print(f"{name}: QUOTE FAILED {quotes[name]['error']}", flush=True)
            quote_p.write_text(json.dumps(quotes, indent=1))
            continue

        if name in done and done[name].get("raw_counts"):
            print(f"{name}: already done"); continue
        cost = quotes.get(name, {}).get("credits")
        if args.max_credits and (cost is None or spent + cost > args.max_credits):
            print(f"STOP before {name}: spent {spent}, quote {cost}, limit {args.max_credits}"); break
        bal = client.management.get_credit_balance(org)
        print(f"{name}: submitting (2q={n2}; {bal.spark_credits} credits left)", flush=True)
        t0 = time.time()
        try:
            job = s.submit_job(JobSubmissionConfig(
                backend_class_id=spec["backend"], name=f"qsched-e10-{name}", job_subcategory_id="phys:oth",
                shots=shots, configuration_data=cfg, auto_approve_quote=True, verbose=True),
                file_path=str(path))
            counts, err, jid = s.download_job_output(job), None, str(job.id)
        except Exception as e:
            counts, err, jid = None, f"{type(e).__name__}: {e}"[:400], None
        done[name] = dict(job_id=jid, shots=shots, actual_n1=n1, actual_n2=n2, actual_depth=nat.depth(),
                          raw_counts=counts, error=err, seconds=round(time.time() - t0, 1))
        out_p.write_text(json.dumps(done, indent=1))
        spent += cost or 0
        print(f"{name}: {'ok' if counts else 'ERROR ' + str(err)}", flush=True)
        if err:
            print("STOP: check this error before rerunning", flush=True)
            break
    if args.quote_only:
        ok = [q["credits"] for q in quotes.values() if "credits" in q]
        print(f"TOTAL quote: {sum(ok)} credits for {len(ok)} circuits ({len(quotes) - len(ok)} failed)")


if __name__ == "__main__":
    main()
