"""
E7 — run on Open Quantum devices (AQT IBEX Q1, Rigetti Cepheus). USES CREDITS.

Run with the Open Quantum venv:
  ../.venv-oq/bin/python e7_run_oq.py --device aqt_ibex_q1
  ../.venv-oq/bin/python e7_run_oq.py --device rigetti_cepheus

For each circuit in results/e7/manifest.json: transpile to the device's native basis
and coupling map (the "actual" gate counts used for ESP), submit one job, and record
counts. Results are appended to results/e7/run_<device>.json, so the script resumes
where it stopped. Before every submission it checks the credit balance and stops if
it is below the expected cost per circuit.
"""
import argparse
import json
import os
import time
from pathlib import Path

from qiskit import qasm2, transpile

E7 = Path(__file__).resolve().parent / "results" / "e7"
COST = {"aqt_ibex_q1": 2, "rigetti_cepheus": 1}          # credits per circuit (quoted 3 Nov)
os.environ.setdefault("OPENQUANTUM_SDK_KEY", os.path.expanduser("~/.config/qsched/openquantum-sdk-key.json"))


def counts_of(qc):
    ops = qc.count_ops()
    n2 = sum(v for k, v in ops.items() if k in ("cz", "cx", "rxx", "ms", "xx", "ecr"))
    n1 = sum(v for k, v in ops.items() if k not in ("cz", "cx", "rxx", "ms", "xx", "ecr",
                                                     "measure", "barrier", "delay"))
    return n1, n2, qc.depth()


def native_transpile(qc, dev):
    edges = dev["edges"] + [[b, a] for a, b in dev["edges"]]
    best = None
    for s in (11, 23, 37):
        try:
            out = transpile(qc, basis_gates=dev["basis"], coupling_map=edges,
                            optimization_level=1, seed_transpiler=s)
        except Exception:
            # Loading the Open Quantum plugin registers extra gate equivalences that can
            # make direct basis translation fail ("Error during parameter assignment").
            # Decomposing to {u, cx} first avoids it.
            pre = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
            out = transpile(pre, basis_gates=dev["basis"], coupling_map=edges,
                            optimization_level=1, seed_transpiler=s)
        if best is None or counts_of(out)[1] < counts_of(best)[1]:
            best = out
    return best


def take_lock(device):
    """One process per device, so no circuit is ever submitted twice."""
    lock = E7 / f".lock_{device}"
    if lock.exists():
        try:
            pid = int(lock.read_text())
            os.kill(pid, 0)
            return None                      # another live process holds it
        except (ValueError, ProcessLookupError, PermissionError):
            pass                             # stale lock
    lock.write_text(str(os.getpid()))
    return lock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=list(COST))
    args = ap.parse_args()
    dev = json.loads((E7 / f"device_{args.device}.json").read_text())
    man = json.loads((E7 / "manifest.json").read_text())["devices"][args.device]
    lock = take_lock(args.device)
    if lock is None:
        print(f"{args.device}: another process is already running it; exiting", flush=True)
        return
    out_path = E7 / f"run_{args.device}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}

    from openquantum_sdk.session import OpenQuantumClient
    from openquantum_sdk_qiskit.oq_service import OpenQuantumService
    client = OpenQuantumClient.from_env()
    org = client.scheduler._resolve_organization_id(None)
    backend = OpenQuantumService().return_backend(dev["backend_class"])

    for r in man["rows"]:
        name = r["circuit"]
        if name in done and done[name].get("counts"):
            continue
        bal = client.management.get_credit_balance(org)
        credits = (bal.spark_credits or 0) + (bal.full_credits or 0)
        if credits < COST[args.device]:
            print(f"STOP: {credits} credits left, need {COST[args.device]} for {name}", flush=True)
            break
        qc = qasm2.loads((E7 / "circuits" / f"{name}.qasm").read_text(),
                         custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        try:
            nat = native_transpile(qc, dev)
        except Exception as e:
            print(f"{name}: transpile failed, skipping ({type(e).__name__}: {e})", flush=True)
            continue
        n1a, n2a, da = counts_of(nat)
        # Submit the native circuit when the backend accepts all its gates; otherwise
        # (AQT: rxx is not an accepted QASM op) submit a version transpiled for the
        # backend's accepted gate set. All-to-all devices need no routing either way.
        ops = set(nat.count_ops()) - {"measure", "barrier"}
        to_run = nat if ops <= set(backend.operation_names) else \
            transpile(qc, backend=backend, optimization_level=1, seed_transpiler=11)
        t0 = time.time()
        print(f"{name}: submitting ({credits} credits left)", flush=True)
        try:
            job = backend.run(to_run, shots=man["shots"], name=f"qsched-e7-{name}")
            res = job.result()
            counts = res.get_counts()
            err = None
        except Exception as e:                       # keep going with the next circuit
            counts, err = None, f"{type(e).__name__}: {e}"[:400]
        done[name] = dict(actual_n1=n1a, actual_n2=n2a, actual_depth=da,
                          submitted_ops=sorted(set(to_run.count_ops()) - {"measure", "barrier"}),
                          counts=counts, error=err, seconds=round(time.time() - t0, 1))
        out_path.write_text(json.dumps(done, indent=1))
        print(f"{name}: {'ok' if counts else 'ERROR ' + str(err)} in {done[name]['seconds']} s", flush=True)

    bal = client.management.get_credit_balance(org)
    print("credits left:", bal, flush=True)
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
