# QPU-Aware Multi-Vendor Quantum Workload Orchestration — Architecture Diagram

## Full System Diagram (ASCII)

```
╔══════════════════╗   ╔══════════════════════════════════════════════════════════════════════════════╗   ╔══════════════════════╗
║   QPU PROVIDERS  ║   ║                         OPENSHIFT CLUSTER                                    ║   ║    OPENSHIFT AI      ║
╠══════════════════╣   ╠══════════════════════════════════════════════════════════════════════════════╣   ╠══════════════════════╣
║                  ║   ║                                                                              ║   ║                      ║
║  ┌────────────┐  ║   ║  ┌──────────────────┐      ┌───────────────────────────────────────────┐   ║   ║  ┌────────────────┐  ║
║  │ IBM Quantum│──╫──>║  │  QPU Telemetry   │      │           QuantumProcessor CRD            │   ║   ║  │ Kubeflow       │  ║
║  └────────────┘  ║   ║  │  Probe DaemonSet │─────>│                                           │   ║   ║  │ Pipelines      │  ║
║                  ║   ║  │                  │      │  fidelity, T1, T2, error rate             │   ║   ║  │ (PipelineRun)  │  ║
║  ┌────────────┐  ║   ║  │  (polls all      │      │  queue, costPerShot                       │   ║   ║  └───────┬────────┘  ║
║  │ AWS Braket │──╫──>║  │   providers      │      │  nativeGateSet, couplingMap               │   ║   ║          │           ║
║  └────────────┘  ║   ║  │   via REST API)  │      │  isSimulator, qualityForecast             │   ║   ║  ┌───────▼────────┐  ║
║                  ║   ║  └──────────────────┘      └────────────────────┬──────────────────────┘   ║   ║  │ MLflow         │  ║
║  ┌────────────┐  ║   ║           │                                      │                          ║   ║  │ Experiment     │  ║
║  │Azure Quant.│──╫──>║           │ PATCH status                         │ reads                   ║   ║  │ Tracking       │  ║
║  └────────────┘  ║   ║           │                                      │                          ║   ║  └───────▲────────┘  ║
║                  ║   ║  ┌────────▼─────────┐      ┌────────────────────▼──────────────────────┐   ║   ║          │           ║
║  ┌────────────┐  ║   ║  │  Multi-Provider  │      │           SCHEDULER PLUGIN                 │   ║   ║  ┌───────┴────────┐  ║
║  │ Quantinuum │──╫──>║  │  Normalization   │      │                                            │   ║   ║  │ OpenShift AI   │  ║
║  └────────────┘  ║   ║  │  Layer           │      │  ┌──────────────────────────────────────┐  │   ║   ║  │ ModelServer    │  ║
║                  ║   ║  │                  │      │  │ PreFilter (parse annotations)         │  │   ║   ║  │ (RL Agent      │  ║
║  ┌────────────┐  ║   ║  │  IBM / AWS /     │      │  └──────────────────────────────────────┘  │   ║   ║  │  via gRPC)     │  ║
║  │   IonQ     │──╫──>║  │  Azure / IonQ /  │      │  ┌──────────────────────────────────────┐  │   ║   ║  └───────▲────────┘  ║
║  └────────────┘  ║   ║  │  Rigetti adapters│      │  │ Filter (fidelity/cost/deadline)      │  │   ║   ║          │           ║
║                  ║   ║  └──────────────────┘      │  └──────────────────────────────────────┘  │   ║   ║          │ RL policy  ║
║  ┌────────────┐  ║   ║                            │  ┌──────────────────────────────────────┐  │   ║   ║          │ training   ║
║  │  Rigetti   │──╫──>║                            │  │ Score (cost-fidelity composite)      │  │   ║   ║          │ feedback   ║
║  └────────────┘  ║   ║                            │  └──────────────────────────────────────┘  │   ║   ╚══════════╪═══════════╝
║                  ║   ║                            │  ┌──────────────────────────────────────┐  │   ║             │
║  ┌────────────┐  ║   ║                            │  │ NormalizeScore [0-100]               │  │   ║             │
║  │ Simulator  │··╫··>║                            │  └──────────────────────────────────────┘  │   ║             │
║  │ (fallback) │  ║   ║                            └───────────┬───────────────────────────────┘   ║             │
║  └────────────┘  ║   ║                                        │                                   ║             │
╚══════════════════╝   ║                                        │ selected QPU                      ║             │
                        ║  ┌─────────────────────────┐          │                                   ║             │
                        ║  │                         │          ▼                                   ║             │
                        ║  │  Pod  (quantum job)     │──>  ┌──────────────────┐                    ║             │
                        ║  │                         │     │ Quantum Resource  │                   ║             │
                        ║  │  annotations:           │     │ Binding           │                   ║             │
                        ║  │  min-fidelity: 0.99     │     │ Controller        │                   ║             │
                        ║  │  max-cost: 0.001        │     └────────┬─────────┘                    ║             │
                        ║  │  deadline: 60min        │              │                               ║             │
                        ║  │  allow-fallback: true   │              ▼                               ║             │
                        ║  └─────────────────────────┘     ┌──────────────────┐                    ║             │
                        ║                                   │ Inject creds:    │                   ║             │
                        ║  ┌─────────────────────────┐     │ QUANTUM_BACKEND  │                   ║             │
                        ║  │  Cost-Fidelity          │<─── │ QUANTUM_ENDPOINT │                   ║             │
                        ║  │  Prediction Module      │     │ /var/run/quantum  │                   ║             │
                        ║  │                         │     └────────┬─────────┘                    ║             │
                        ║  │  - Topology divergence  │              │ credentials                   ║             │
                        ║  │  - Post-transpilation   │              │ injected                     ║             │
                        ║  │    fidelity prediction  │              ▼                               ║             │
                        ║  └─────────────────────────┘     ┌──────────────────┐                    ║             │
                        ║                                   │ Audit Annotation  │                  ║             │
                        ║  ┌─────────────────────────┐     │ scheduled-fidelity│──────────────────╫─────────────┘
                        ║  │  Circuit Retranspilation│     │ predicted-fidelity│  QPU state +
                        ║  │  Broker                 │     │ cost-incurred     │  cost recorded
                        ║  │                         │     │ simulator-fallback│
                        ║  │  OpenQASM/QIR input     │     └────────┬─────────┘
                        ║  │  Qiskit/tket/PyQuil     │              │
                        ║  │  Fidelity re-validation │              ▼
                        ║  └─────────────────────────┘     ┌──────────────────┐
                        ║                                   │ QPU Executes     │
                        ║  ┌─────────────────────────┐     │ circuit workload  │
                        ║  │  Financial Quota        │     └──────────────────┘
                        ║  │  Controller             │
                        ║  │                         │
                        ║  │  QuantumQuota CRD       │     ┌──────────────────┐
                        ║  │  maxCostPerNamespace    │     │ No real QPU meets │
                        ║  │  vendorBudgetAllocation │     │ SLA?             │
                        ║  └─────────────────────────┘     │ -> Simulator     │
                        ║                                   │    fallback /    │
                        ║  ┌─────────────────────────┐     │    Unschedulable │
                        ║  │  Predictive Quality     │     └──────────────────┘
                        ║  │  Forecaster             │
                        ║  │                         │
                        ║  │  Time-series model      │
                        ║  │  per QPU backend        │
                        ║  │  Future fidelity windows│
                        ║  │  Calibration peak pred. │
                        ║  └────────────┬────────────┘
                        ║               │ quality windows
                        ║               ▼
                        ║  ┌─────────────────────────┐
                        ║  │  RL Routing Agent (PPO) │
                        ║  │                         │
                        ║  │  State: current +       │
                        ║  │    forecast quality,    │
                        ║  │    cost, queue, deadline│
                        ║  │                         │
                        ║  │  Actions:               │
                        ║  │    route / defer /      │
                        ║  │    simulator fallback   │
                        ║  │                         │
                        ║  │  Reward: fidelity       │
                        ║  │    - cost - latency     │
                        ║  └─────────────────────────┘
                        ╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Arrow Legend

### Orange arrows — Provider REST API polling (no QPU minutes consumed)

| Arrow | From | To | Description |
|---|---|---|---|
| `a_ibm` | IBM Quantum | QPU Telemetry Probe | Probe polls IBM Quantum `/backends/{name}/properties` API |
| `a_aws` | AWS Braket | QPU Telemetry Probe | Probe polls AWS Braket `GetDevice` API |
| `a_azr` | Azure Quantum | QPU Telemetry Probe | Probe polls Azure Quantum REST API |
| `a_qnt` | Quantinuum | QPU Telemetry Probe | Probe polls Quantinuum H-series API |
| `a_ionq` | IonQ | QPU Telemetry Probe | Probe polls IonQ API |
| `a_rigt` | Rigetti | QPU Telemetry Probe | Probe polls Rigetti API |
| `a_sim` | Simulator | QPU Telemetry Probe | Aer/local simulator (dashed = fallback only) |

### Purple arrows — Telemetry normalization and CRD update

| Arrow | From | To | Description |
|---|---|---|---|
| `a_pr_nm` | QPU Telemetry Probe | Multi-Provider Normalization Layer | Raw API response passed to normalization |
| `a_nm_cr` | Multi-Provider Normalization Layer | QuantumProcessor CRD | Normalized properties written via `kubectl PATCH status` |
| `a_cr_fc` | QuantumProcessor CRD | Predictive Quality Forecaster | Historical telemetry fed to forecaster |
| `a_fc_cr` | Predictive Quality Forecaster | QuantumProcessor CRD | `qualityForecast` field written back into CRD (dashed) |

### Blue arrows — Scheduling data flow

| Arrow | From | To | Description |
|---|---|---|---|
| `a_cr_sc` | QuantumProcessor CRD | Scheduler Plugin | Scheduler reads current physical properties during Filter/Score |
| `a_pd_sc` | Pod (quantum job) | Scheduler Plugin | Pod annotations submitted to scheduler (min-fidelity, max-cost, deadline) |
| `a_in_ex` | Inject credentials | QPU Executes | Workload receives injected credentials and connects to assigned QPU |

### Green arrows — Routing and execution (solid)

| Arrow | From | To | Description |
|---|---|---|---|
| `a_sc_bd` | Scheduler Plugin | Quantum Resource Binding Controller | Selected QPU passed to binding phase |
| `a_bd_in` | Quantum Resource Binding Controller | Inject credentials | Controller triggers credential injection |
| `a_in_au` | Inject credentials | Audit Annotation | Scheduling metadata recorded (fidelity, cost, backend, timestamp) |
| `a_au_qe` | Audit Annotation | QPU Executes | Job proceeds to execution after audit record created |
| `a_qt_bd` | Financial Quota Controller | Binding Controller | Cost approved, quota deducted, binding proceeds |
| `a_sc_qt` | Scheduler Plugin | Financial Quota Controller | Pre-execution cost check against namespace budget |
| `a_rt_qt` | Circuit Retranspilation Broker | Financial Quota Controller | Estimated cost of retranspiled circuit submitted for approval |

### Green arrows — OpenShift AI integration (dashed)

| Arrow | From | To | Description |
|---|---|---|---|
| `a_kfp_p` | Kubeflow Pipelines | Pod (quantum job) | Pipeline submits annotated PipelineRun as Pod |
| `a_au_ml` | Audit Annotation | MLflow Experiment Tracking | QPU state, predicted fidelity, and cost recorded per experiment run |
| `a_ml_rl` | MLflow Experiment Tracking | RL Routing Agent | Execution outcomes fed back as RL training data |
| `a_rl_ms` | RL Routing Agent | OpenShift AI ModelServer | Agent served via gRPC for inference (dashed) |

### Purple arrows — RL / Forecaster integration (dashed)

| Arrow | From | To | Description |
|---|---|---|---|
| `a_fc_rl` | Predictive Quality Forecaster | RL Routing Agent | Quality windows passed as state features to RL agent |
| `a_rl_sc` | RL Routing Agent | Scheduler Plugin | Routing recommendation returned to Score phase (dashed) |

### Red arrows — Error / fallback paths (dashed)

| Arrow | From | To | Description |
|---|---|---|---|
| `a_sc_nq` | Scheduler Plugin | No QPU meets SLA? | All real QPUs failed Filter → evaluate fallback |
| `a_cf_rt` | Cost-Fidelity Prediction Module | Circuit Retranspilation Broker | If topology differs, trigger retranspilation |

---

## Simplified Flow (for presentations)

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                    HAPPY PATH                            │
                  └─────────────────────────────────────────────────────────┘

 IBM / IonQ / Rigetti                                            OpenShift AI
 AWS / Azure / Sim
       │
       │ REST API (read-only calibration)
       ▼
 QPU Telemetry Probe
       │
       │ PATCH status
       ▼
 QuantumProcessor CRD ──────────────────────────────> Predictive Forecaster
 (fidelity, T2, error,                                (quality windows 24h)
  cost, topology,                                              │
  qualityForecast)                                             │ quality windows
       │                                                       ▼
       │ reads                                         RL Routing Agent
       ▼                                               (route/defer/fallback)
 Scheduler Plugin  <───── Pod annotations ──────────────────  │
 (PreFilter)               min-fidelity: 0.99                  │ recommendation
 (Filter)  ──── Cost-Fidelity Prediction ──── Retranspilation  │
 (Score)        (topology divergence +         Broker          │
 (NormalizeScore) post-transpile fidelity)     (Qiskit/tket)   │
       │
       │ selected QPU
       ▼
 Binding Controller ──── Financial Quota Controller ──── MLflow
       │                  (QuantumQuota CRD)
       │ inject credentials
       ▼
 Pod executes on QPU
       │
       │ results + audit
       ▼
 Audit Annotation ──────────────────────────────────> MLflow Tracking
 (fidelity, cost,                                     (experiment record)
  backend, timestamp)


                  ┌─────────────────────────────────────────────────────────┐
                  │                  DEFERRAL PATH                           │
                  └─────────────────────────────────────────────────────────┘

 All backends degraded → Filter: all FAIL
       │
       │ deadline_minutes > 0?
       ▼
 Forecaster checks quality windows
       │
       │ window found within deadline?
       ├── YES → DEFERRED: "run at 14:30 UTC on IBM Brisbane (F=0.998)"
       └── NO  → UNSCHEDULABLE (with reason)
```

---

## Component Responsibility Summary

| Component | Zone | Responsibility |
|---|---|---|
| QPU Telemetry Probe | OpenShift Cluster | Polls provider APIs, writes CRDs |
| Multi-Provider Normalization Layer | OpenShift Cluster | Translates heterogeneous API formats to unified schema |
| QuantumProcessor CRD | OpenShift Cluster | Stores real-time QPU physical properties as Kubernetes resource |
| Scheduler Plugin (PreFilter/Filter/Score) | OpenShift Cluster | Routes workloads based on quality constraints |
| Cost-Fidelity Prediction Module | OpenShift Cluster | Predicts post-transpilation fidelity |
| Circuit Retranspilation Broker | OpenShift Cluster | Recompiles circuits for target vendor topology |
| Financial Quota Controller | OpenShift Cluster | Enforces cross-vendor cost limits per namespace |
| Predictive Quality Forecaster | OpenShift Cluster | Forecasts future fidelity windows (time-series) |
| RL Routing Agent | OpenShift Cluster | Learns optimal routing policy from execution history |
| Binding Controller | OpenShift Cluster | Injects QPU credentials into Pod, records audit annotation |
| Kubeflow Pipelines | OpenShift AI | Submits annotated quantum pipeline steps |
| MLflow | OpenShift AI | Records QPU state, fidelity, cost per experiment |
| OpenShift AI ModelServer | OpenShift AI | Serves RL agent via gRPC for inference |
