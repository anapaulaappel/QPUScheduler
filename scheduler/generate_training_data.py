"""
Generate Training Data for Cost-Fidelity Prediction Model
----------------------------------------------------------
Creates a synthetic dataset of (circuit, hardware, result) tuples
using physics-based simulation. No QPU quota is consumed.

The dataset captures the key relationship the model needs to learn:
  "Given a circuit compiled for topology A, if I retranspile it for
   topology B with hardware properties X, what fidelity will I observe?"

Run:
    python generate_training_data.py

Output:
    training_data.csv  (used by train_model.py)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from cost_fidelity_model import (
    FEATURE_NAMES, build_feature_vector, physics_fidelity, topology_divergence,
)

# NOTE: labels are synthetic — they come from the same physics-informed formula
# used by the 'physics' mode, plus Gaussian noise. A model trained on this data
# learns to reproduce the formula; its accuracy says nothing about real hardware.
LABEL_NOISE_STD = 0.003

np.random.seed(42)
N_SAMPLES = 5000
OUTPUT = Path(__file__).parent / "training_data.csv"

TOPOLOGIES = ["heavy-hex", "all-to-all", "square-lattice", "trapped-ion"]

# Realistic hardware parameter ranges per topology
HARDWARE_RANGES = {
    "heavy-hex":     {"fidelity": (0.985, 0.999), "tq_error": (0.003, 0.020), "t2": (60, 200)},
    "all-to-all":    {"fidelity": (0.990, 0.9995),"tq_error": (0.001, 0.010), "t2": (500, 5000)},
    "square-lattice":{"fidelity": (0.970, 0.995), "tq_error": (0.005, 0.030), "t2": (20, 80)},
    "trapped-ion":   {"fidelity": (0.993, 0.9998),"tq_error": (0.001, 0.008), "t2": (1000, 50000)},
}

rows = []
for _ in range(N_SAMPLES):
    src_top = np.random.choice(TOPOLOGIES)
    dst_top = np.random.choice(TOPOLOGIES)

    hw = HARDWARE_RANGES[dst_top]
    hw_fidelity = np.random.uniform(*hw["fidelity"])
    tq_error    = np.random.uniform(*hw["tq_error"])
    t2_us       = np.random.uniform(*hw["t2"])

    circuit_depth   = np.random.randint(5, 200)
    num_tq_gates    = np.random.randint(1, circuit_depth // 2 + 1)

    divergence = topology_divergence(src_top, dst_top)

    clean_fidelity, *_ = physics_fidelity(
        divergence, circuit_depth, num_tq_gates,
        tq_error, t2_us, hw_fidelity,
    )
    target_fidelity = float(np.clip(
        clean_fidelity + np.random.normal(0, LABEL_NOISE_STD), 0.0, 1.0))

    features = build_feature_vector(
        src_top, dst_top, circuit_depth,
        num_tq_gates, tq_error, t2_us, hw_fidelity
    )

    row = {"src_topology": src_top, "dst_topology": dst_top}
    row.update(dict(zip(FEATURE_NAMES, features)))
    row["post_transpilation_fidelity"] = target_fidelity
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(f"Generated {len(df)} training samples -> {OUTPUT}")
print(df.describe().round(4))
