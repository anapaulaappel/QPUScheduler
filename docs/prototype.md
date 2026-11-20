> This page describes the Python prototype (scheduler, probe, forecaster, demo). To reproduce the
> paper's experiments, see the [main README](../README.md).

# QPU-Aware Multi-Vendor Quantum Workload Orchestration for OpenShift

> Prototype of transpilation-aware placement of quantum jobs across QPUs from several vendors, with a
> fidelity requirement as a hard filter, for container orchestration platforms.

---

## Overview

This project demonstrates a Kubernetes scheduling system that routes quantum
computing workloads across multiple QPU vendors (IBM Quantum, IonQ, Rigetti, simulators)
based on **real-time physical quality properties** and **topology-aware post-transpilation
fidelity prediction**.

### The Central Problem

When a quantum circuit compiled for IBM's `heavy-hex` topology is routed to IonQ's
`all-to-all` (trapped-ion) hardware — which nominally has *higher* fidelity — the circuit
must be **retranspiled**: recompiled for the new hardware architecture. This process
introduces extra SWAP gates and increases circuit depth, **degrading result quality below
the user's declared fidelity requirement** even though the hardware itself is nominally
better.

No existing scheduling system predicts this degradation before routing.

### The Approach

Before routing a job to any QPU, the system:

1. Estimates **topology divergence** between the circuit's compiled topology and the
   candidate QPU's topology
2. Predicts **post-transpilation fidelity** using a physics-informed formula and a
   learned surrogate model (Gradient Boosted Regressor). *The surrogate is currently
   trained on synthetic labels generated from the physics formula; its R² measures
   agreement with that formula, not accuracy on real hardware.*
3. **Rejects** QPUs where predicted post-transpilation fidelity falls below the declared
   threshold — even if raw hardware fidelity would otherwise qualify
4. **Forecasts future quality windows** using per-backend time-series models, enabling
   deferral of latency-tolerant jobs to predicted calibration peaks (*quality surfing*)

---

## Key Results

**Paper experiments** (`experiments/`, noisy Aer simulation with real IBM calibration
snapshots; details in [experiments/README.md](../experiments/README.md)):

- **E1:** the graph-based divergence estimate ranks transpilation overhead with Spearman
  ρ ≈ 0.99. Calibrated, it predicts the 2Q gate count with a 7–12% median error
  (lookup table: 31%; ignoring overhead: 41%).
- **E1b:** the estimate is 4–20× faster than a level-1 transpilation per candidate QPU.
- **E2:** choosing the target by the estimated success probability, without transpiling,
  is as good as transpiling every candidate (mean regret 0.0129 for both). Raw device
  fidelity is a poor predictor (ρ = 0.45).
- **E3:** the estimate-based filter has 0 SLA violations for τ ≥ 0.6, like transpile-all,
  at about 25× lower scheduling latency. Naive policies violate the requirement for up to
  21 of 28 jobs. The trade-off is more conservative rejections.

**Demo notebook**: from `demo/QPUScheduler_Demo_executed.ipynb`. **All results are simulations**: QPU
telemetry is synthetic, and "execution fidelity" is computed with the physics-informed
formula plus noise. They illustrate the mechanism, not real-world accuracy. Numbers vary
with the time of day the notebook is run (simulated calibration phase).

### Cost-Fidelity Prediction (Section 2)

Circuit compiled for `heavy-hex` (depth 85, 42 two-qubit gates), min_fidelity=0.985:

| Backend | HW Fidelity | Topology | Divergence | Post-Transpile F | Decision |
|---|---|---|---|---|---|
| IBM Brisbane | 0.9970 | heavy-hex | 0.00 | **0.9968** | PASS |
| IonQ Forte | 0.9930 | all-to-all | 0.10 | 0.8521 | **REJECTED** |
| Rigetti Ankaa | 0.9700 | square-lattice | 0.20 | 0.2638 | **REJECTED** |

> A backend whose raw fidelity passes the threshold can still be rejected because
> retranspilation to its topology reduces the predicted fidelity below the threshold.

### Deferral Scheduling / Quality Surfing (Section 5)

When all backends are degraded (10h into their 12h calibration cycle):
- Immediate scheduling: all backends fail Filter → `UNSCHEDULABLE`
- Forecaster (anchored on live calibration age) predicts IBM Brisbane will enter a
  quality window after its next calibration
- Decision: **DEFERRED by 120 minutes** → `fake_ibm_brisbane` (predicted F=0.9989)

### Naive vs QPU-Aware Comparison (Section 6)

Over 50 random jobs; both policies are evaluated with the same simulated execution
outcome:

| Scheduler | Mean fidelity (executed jobs) | Below threshold | Unschedulable |
|---|---|---|---|
| Naive (highest raw fidelity) | 0.9461 | **38/50** | 0 |
| QPU-Aware (topology-aware) | **0.9949** | **0/30** | 20/50 |

The QPU-aware scheduler never runs a job below its fidelity requirement; when no QPU
can meet it, the job is reported as unschedulable (or deferred / sent to the simulator
when allowed) instead of silently producing low-quality results.

---

## Architecture

```
QPUScheduler/
├── crd/                    # QuantumProcessor Custom Resource Definition
│   ├── quantumprocessor_crd.yaml
│   └── sample_simulator.yaml
│
├── probe/                  # QPU Telemetry Probe (DaemonSet)
│   ├── probe.py            # Main polling loop
│   ├── config.py           # PROBE_MODE switch (simulator | real)
│   ├── crd_writer.py       # Telemetry (+ quality forecast) → CRD YAML → kubectl apply
│   ├── topology.py         # Coupling maps per topology family
│   ├── collect_calibration.py    # Read-only IBM calibration collector (E4/E5)
│   └── backends/
│       ├── simulator_backend.py  # Realistic drift simulation (no IBM quota)
│       ├── ibm_backend.py        # IBM Quantum API (calibration read-only, saved account)
│       └── target_telemetry.py   # Telemetry from any Qiskit BackendV2 Target
│
├── scheduler/              # Quantum-Aware Scheduler
│   ├── job_request.py      # Workload requirements (Pod annotation schema)
│   ├── scheduler.py        # PreFilter → Filter → Score → Select pipeline
│   ├── circuit_analysis.py # QASM → interaction graph; graph-based topology divergence
│   ├── cost_fidelity_model.py    # Post-transpilation fidelity prediction + acceptance threshold
│   ├── retranspilation_broker.py # Transpile for the reserved QPU (Qiskit)
│   ├── binding.py          # Two-stage estimate → reserve → transpile → re-validate → bind/rollback
│   ├── usage.py            # QPU-time, pay-per-use and effective cost estimates
│   ├── quota.py            # Namespace quotas in USD / QPU seconds / shots (optimistic concurrency)
│   ├── generate_training_data.py # Synthetic dataset generation
│   ├── train_model.py      # GBR model training (synthetic labels; see report)
│   └── cost_fidelity_model.pkl   # Trained model bundle (model + validation metrics)
│
├── experiments/            # Paper experiments E1–E5 (see experiments/README.md)
├── tests/                  # pytest suite
├── deploy/                 # launchd template for the calibration collector
│
├── forecaster/             # Predictive Quality Forecaster
│   ├── history_store.py    # Per-backend telemetry ring buffer + CSV
│   ├── forecaster.py       # Physics + Prophet time-series models
│   ├── deferral_scheduler.py     # Deferral / quality surfing logic
│   └── simulate_history.py       # Generate N hours of synthetic history
│
├── demo/                   # Jupyter notebook demo
│   ├── QPUScheduler_Demo.ipynb   # Full end-to-end demonstration
│   └── fig1–fig7_*.png     # Generated figures
│
├── operator/               # OpenShift Operator packaging (Phase 5 — future)
└── data/
    └── history/            # Persisted telemetry CSV (one file per backend)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- (Optional) IBM Quantum account with API token — only for `PROBE_MODE=real`
- (Optional) OpenShift or minikube — only for `K8S_APPLY=true`

### Install

```bash
git clone https://github.com/anapaulaappel/QPUScheduler
cd QPUScheduler
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Optional: Prophet for time-series forecasting
pip install prophet
```

### Run the Tests

```bash
python -m pytest
```

### Regenerate Data and Model

```bash
cd forecaster && python simulate_history.py --hours 24   # overwrites data/history/*.csv
cd ../scheduler && python generate_training_data.py && python train_model.py
```

### Run the Demo Notebook

```bash
source .venv/bin/activate
jupyter notebook demo/QPUScheduler_Demo.ipynb
```

The notebook runs entirely in **simulator mode** — no IBM Quantum quota is consumed.

---

## Component Usage

### 1. QPU Telemetry Probe

```bash
cd probe

# Simulator mode (safe, no IBM quota):
export PROBE_MODE=simulator
export POLLING_INTERVAL=30    # seconds between polls
python probe.py

# Real IBM Quantum mode (reads calibration data only — no QPU minutes).
# Uses IBM_QUANTUM_TOKEN if set, otherwise the saved Qiskit account
# (QiskitRuntimeService.save_account). IBM_BACKENDS=ibm_fez,... limits backends.
export PROBE_MODE=real
python probe.py

# Collect real calibration history for the forecasting experiments:
python collect_calibration.py --once          # test
python collect_calibration.py --interval 15   # or install deploy/*.plist (launchd)

# Apply CRD updates to Kubernetes/OpenShift:
export K8S_APPLY=true
python probe.py
```

**Important:** `PROBE_MODE=real` only calls `backend.properties()` and `backend.status()`
— read-only calibration data. QPU minutes are only consumed when you call `backend.run()`,
which this probe never does.

### 2. Scheduler + Cost-Fidelity Model

```bash
cd scheduler

# Generate training data (synthetic, no QPU quota):
python generate_training_data.py

# Train the model:
python train_model.py
# Output: cost_fidelity_model.pkl  (R²=0.995, MAE=0.008)
```

### 3. Forecaster + History Simulation

```bash
cd forecaster

# Simulate 24h of telemetry history (no polling wait):
python simulate_history.py --hours 24 --output ../data/history

# Options:
python simulate_history.py --hours 48 --step 30 --output ../data/history
```

### 4. Apply CRD to OpenShift

```bash
# Create the CRD schema:
kubectl apply -f crd/quantumprocessor_crd.yaml

# Create the quantum-system namespace:
kubectl create namespace quantum-system

# Start probe in simulator mode:
cd probe
export K8S_APPLY=true
export PROBE_MODE=simulator
python probe.py

# Watch CRDs update in real time:
watch kubectl get qpu -n quantum-system
```

---

## Design Decisions

### Two-Mode Architecture (simulator / real)

The probe supports two modes controlled by a single environment variable:

| Mode | IBM Quota Used | Use When |
|---|---|---|
| `simulator` | None | Development, testing, demo, CI |
| `real` | Zero (read-only) | Validating real hardware data |

QPU minutes are only consumed when `backend.run(circuit)` is called — which happens only
in the demo notebook's optional Section 7 (disabled by default).

### Physics vs ML for Fidelity Prediction

The Cost-Fidelity model supports two prediction modes:

| Mode | Requires | Accuracy | Use When |
|---|---|---|---|
| `physics` | Nothing | Good (physics-based formula) | No training data yet |
| `ml` | `training_data.csv` + `cost_fidelity_model.pkl` | R²=0.995 | After training |

The physics model is always available as a fallback and produces results consistent with
the ML model for the topology pairs covered in the divergence table.

### Forecaster: Physics vs Prophet

| Model | Requires | Accuracy | Use When |
|---|---|---|---|
| `physics` | Just calibration age | Good for stable cycles | Always available |
| `prophet` | ≥50 history records + `pip install prophet` | Better for irregular patterns | After accumulating history |

---

## Components


| Capability | Component | Demo Section |
|---|---|---|
| QuantumProcessor CRD with real-time physical properties | `probe/` + `crd/` | Section 1 |
| Post-transpilation fidelity prediction as hard filter | `scheduler/cost_fidelity_model.py` | Section 2 |
| Multi-dimensional Filter + Score scheduling | `scheduler/scheduler.py` | Section 3 |
| Per-backend quality window forecasting | `forecaster/forecaster.py` | Section 4 |
| Deferral to predicted calibration peak | `forecaster/deferral_scheduler.py` | Section 5 |
| Naive vs QPU-aware outcome comparison | `demo/notebook` | Section 6 |

### Related systems

| System | Covers | Not in scope |
|---|---|---|
| Qonductor (arXiv, IBM-only) | Cost-fidelity trade-off | Multi-vendor, topology-aware prediction, deferral |
| DRLQ / QFOR (RL scheduling) | RL-based routing | Forecasted quality windows as RL state, monetary cost |
| IBM QTS (Qiskit AI Transpiler) | Circuit-level transpilation | Scheduling, multi-vendor, topology prediction |
| Qubernetes / CERN arXiv:2603 | Kubernetes-native QPU jobs | Quality-based scheduling, cost-fidelity, deferral |
| AWS Braket / Azure Quantum | Multi-vendor access | Automatic routing, Kubernetes integration, forecasting |

---

## Roadmap

- [x] Phase 1: QuantumProcessor CRD + Telemetry Probe (simulator + IBM real)
- [x] Phase 2: Quantum-Aware Scheduler + Cost-Fidelity Prediction Model
- [x] Phase 3: Predictive Quality Forecaster + Deferral Scheduler
- [x] Phase 4: Demo Notebook with 7 visualizations
- [x] Graph-based topology divergence, model acceptance threshold
- [x] Retranspilation broker (Qiskit) + two-stage re-validation/rollback
- [x] Unified usage/cost model and quota in USD / QPU seconds, in-memory
- [x] Paper experiments E1–E3 (offline); E4/E5 scripts + real calibration collector
- [ ] Real IBM calibration history (≥ several weeks) for E4/E5
- [ ] Kubernetes Scheduling Framework plugin (Go) and OpenShift Operator packaging
- [ ] QuantumQuota CRD controller (cluster-side state for quota.py)
- [ ] RL Routing Agent (PPO, multi-vendor action space)
- [ ] Additional toolchains in the broker (tket, PyQuil, Cirq)

---

## References

1. arXiv:2603.24206 — CERN/DESY hybrid quantum-classical workflow on Kubernetes (Mar 2026)
2. arXiv:2407.02748 — DRLQ: Deep RL for quantum task placement (IEEE CLOUD 2024)
3. arXiv:2508.04974 — QFOR: Fidelity-aware RL orchestration (Aug 2025)
4. arXiv:2404.06535 — Hartnett et al.: ML for quantum layout ranking (Quantum, 2024)
5. arXiv:2303.17523 — Q-fid: LSTM fidelity prediction (2023)
6. Qonductor — Kubernetes-based QPU orchestrator with cost-fidelity scheduling (academic)
7. IBM Qiskit AI-Powered Transpiler Service — arXiv:2405.13196 (Kremer et al., 2024)

---

## License

Apache License 2.0 (SPDX: `Apache-2.0`) — see [LICENSE](../LICENSE).

## Contact

Ana Paula Appel — aappel@redhat.com
