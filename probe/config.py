"""
QPU Probe Configuration
-----------------------
Set PROBE_MODE to control whether the probe uses real IBM Quantum hardware
or a local Qiskit Aer simulator.

PROBE_MODE = "simulator"  -> Qiskit Aer (free, unlimited, no IBM account needed)
PROBE_MODE = "real"       -> IBM Quantum real hardware (uses your monthly quota!)

To switch modes, set the environment variable:
    export PROBE_MODE=simulator   # safe for development
    export PROBE_MODE=real        # uses real QPU minutes — use sparingly!
"""

import os

# ── Mode ──────────────────────────────────────────────────────────────────────
# Override via env var: export PROBE_MODE=real
PROBE_MODE = os.environ.get("PROBE_MODE", "simulator")  # "simulator" | "real"

# ── IBM Quantum (only used when PROBE_MODE=real) ───────────────────────────────
# Credentials: if IBM_QUANTUM_TOKEN is empty, the account saved with
# QiskitRuntimeService.save_account() (~/.qiskit/qiskit-ibm.json) is used.
IBM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN", "")
IBM_INSTANCE = os.environ.get("IBM_QUANTUM_INSTANCE", "") or None
IBM_CHANNEL = os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform")

# Backends to monitor. Empty list = all operational backends of the account.
IBM_PREFERRED_BACKENDS = [b for b in os.environ.get("IBM_BACKENDS", "").split(",") if b]

# Example IBM pricing used for usage/cost estimates (illustrative list price,
# pay-as-you-go per QPU second; verify against your plan).
IBM_PRICING = {"scheme": "time",
               "price_per_second": float(os.environ.get("IBM_PRICE_PER_SECOND", "1.60"))}

# ── Simulator config (only used when PROBE_MODE=simulator) ────────────────────
# Simulate multiple "fake" backends to test multi-vendor scheduling logic
SIMULATOR_BACKENDS = [
    {
        "name": "fake_ibm_brisbane",
        "calibration_phase_hours": 2.0,  # offset of this backend's calibration cycle
        "provider": "simulator",
        "qubit_count": 127,
        "topology": "heavy-hex",
        "cost_per_shot": 0.0,
        # Realistic noise ranges — probe will randomize within these bounds
        # to simulate intra-day hardware drift
        "fidelity_range": (0.990, 0.999),
        "t1_range_us":    (120.0, 220.0),
        "t2_range_us":    (80.0,  160.0),
        "sq_error_range": (0.0005, 0.003),
        "tq_error_range": (0.003,  0.015),
        "ro_error_range": (0.005,  0.025),
        "queue_range":    (0, 5),
        "native_gates":   ["ecr", "id", "rz", "sx", "x"],
        # Example pricing/durations (illustrative values; verify with provider)
        "pricing":   {"scheme": "time", "price_per_second": 1.60},
        "durations": {"gate_1q_ns": 60, "gate_2q_ns": 660, "measure_ns": 1500, "reset_us": 250},
    },
    {
        "name": "fake_ionq_forte",
        "calibration_phase_hours": 6.0,  # offset of this backend's calibration cycle
        "provider": "simulator",
        "qubit_count": 36,
        "topology": "all-to-all",
        "cost_per_shot": 0.0,
        "fidelity_range": (0.993, 0.9995),
        "t1_range_us":    (10000.0, 50000.0),   # trapped-ion T1 >> superconducting
        "t2_range_us":    (1000.0,  5000.0),
        "sq_error_range": (0.0001, 0.001),
        "tq_error_range": (0.001,  0.008),
        "ro_error_range": (0.002,  0.010),
        "queue_range":    (0, 15),
        "native_gates":   ["gpi", "gpi2", "ms"],
        "pricing":   {"scheme": "per_gate_shot", "price_1q": 0.000220, "price_2q": 0.000975},
        "durations": {"gate_1q_ns": 63000, "gate_2q_ns": 650000, "measure_ns": 250000, "reset_us": 50},
    },
    {
        "name": "fake_rigetti_ankaa",
        "calibration_phase_hours": 9.0,  # offset of this backend's calibration cycle
        "provider": "simulator",
        "qubit_count": 84,
        "topology": "square-lattice",
        "cost_per_shot": 0.0009,
        "fidelity_range": (0.970, 0.992),
        "t1_range_us":    (30.0, 80.0),
        "t2_range_us":    (20.0, 60.0),
        "sq_error_range": (0.001,  0.005),
        "tq_error_range": (0.008,  0.030),
        "ro_error_range": (0.010,  0.040),
        "queue_range":    (0, 8),
        "native_gates":   ["cz", "rx", "rz"],
        "pricing":   {"scheme": "per_shot", "task_fee": 0.30, "price_per_shot": 0.0009},
        "durations": {"gate_1q_ns": 40, "gate_2q_ns": 200, "measure_ns": 1500, "reset_us": 100},
    },
    {
        "name": "aer_simulator",
        "calibration_phase_hours": 0.0,  # offset of this backend's calibration cycle
        "provider": "simulator",
        "qubit_count": 32,
        "topology": "all-to-all",
        "cost_per_shot": 0.0,
        # Perfect simulator — always available as fallback
        "fidelity_range": (1.0, 1.0),
        "t1_range_us":    (999999.0, 999999.0),
        "t2_range_us":    (999999.0, 999999.0),
        "sq_error_range": (0.0, 0.0),
        "tq_error_range": (0.0, 0.0),
        "ro_error_range": (0.0, 0.0),
        "queue_range":    (0, 0),
        "native_gates":   ["any"],
        "pricing":   {"scheme": "free"},
        "durations": {"gate_1q_ns": 0, "gate_2q_ns": 0, "measure_ns": 0, "reset_us": 0},
        "is_simulator_fallback": True,
    },
]

# ── Probe polling ─────────────────────────────────────────────────────────────
POLLING_INTERVAL_SECONDS = int(os.environ.get("POLLING_INTERVAL", "30"))  # faster for dev
# Telemetry older than this is treated as unknown quality and excluded by the
# scheduler Filter ("temporal safety"); also drives the TelemetryFresh CRD condition.
STALENESS_THRESHOLD_MINUTES = float(os.environ.get("STALENESS_THRESHOLD_MINUTES", "15"))

# ── Kubernetes / CRD ──────────────────────────────────────────────────────────
# Namespace where QuantumProcessor CRDs will be created/updated
K8S_NAMESPACE = os.environ.get("K8S_NAMESPACE", "quantum-system")

# Set to False to run in dry-run mode (prints CRD updates, does not apply to k8s)
K8S_APPLY = os.environ.get("K8S_APPLY", "false").lower() == "true"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
