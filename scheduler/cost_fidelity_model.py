"""
Cost-Fidelity Prediction Model
-------------------------------
Predicts post-transpilation fidelity when routing a circuit to a QPU
whose topology differs from the one the circuit was originally compiled for.

Goal: predict, before routing, whether the circuit transpiled to another
hardware topology still delivers the declared fidelity requirement.

Model inputs:
  - topology_divergence_score  (how different are the two topologies)
  - circuit_depth              (deeper = more decoherence exposure)
  - num_two_qubit_gates        (more 2Q gates = more error accumulation)
  - target_mean_tq_error       (current 2Q error rate of the target QPU)
  - target_mean_t2_us          (current T2 of the target QPU)
  - target_mean_fidelity       (raw hardware fidelity of target QPU)

Model output:
  - predicted_post_transpilation_fidelity  (float in [0, 1])

Training data:
  Currently generated synthetically from the physics-informed formula below
  plus Gaussian noise (generate_training_data.py). Accuracy metrics obtained
  on this data measure how well the model reproduces the formula, NOT accuracy
  on real hardware. Real labels (noise-model simulation of transpiled circuits
  and observed executions) are planned.

Overhead model (shared by the physics formula and the ML features):
  The topology divergence D is read from a family-level lookup table and
  interpreted as estimated SWAP insertions per original two-qubit gate.
    added two-qubit gates   ΔN_2q       = SWAP_COST_2Q · D · N_2q
    added circuit layers    Δdepth      = D · depth
    added execution time    Δt          = Δdepth · LAYER_TIME_US
"""

import os
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Native cost of one SWAP in two-qubit gates (3 CX/ECR/CZ on superconducting devices)
SWAP_COST_2Q = 3.0
# Approximate duration of one circuit layer, in microseconds (~100 ns)
LAYER_TIME_US = 0.1

KNOWN_TOPOLOGIES = ["heavy-hex", "falcon", "all-to-all", "square-lattice", "trapped-ion"]

# Topology divergence scores: how different two topologies are
# (estimated SWAP insertions per original two-qubit gate). Simplified
# lookup-table version; a graph-based estimate is planned.
TOPOLOGY_DIVERGENCE = {
    # Same topology: no routing overhead
    **{(t, t): 0.00 for t in KNOWN_TOPOLOGIES},
    ("heavy-hex",     "falcon"):        0.05,
    ("heavy-hex",     "all-to-all"):    0.10,  # all-to-all needs fewer SWAPs
    ("heavy-hex",     "square-lattice"):0.20,
    ("heavy-hex",     "trapped-ion"):   0.35,  # very different connectivity
    ("all-to-all",    "heavy-hex"):     0.15,
    ("all-to-all",    "square-lattice"):0.10,
    ("all-to-all",    "trapped-ion"):   0.05,
    ("square-lattice","heavy-hex"):     0.20,
    ("square-lattice","trapped-ion"):   0.30,
    ("trapped-ion",   "heavy-hex"):     0.40,
    ("trapped-ion",   "all-to-all"):    0.05,
}

MODEL_PATH = Path(__file__).parent / "cost_fidelity_model.pkl"

# Default model-acceptance threshold (override per instance or via env vars).
# A high R² is typically required; MAE is optional.
DEFAULT_MIN_R2 = float(os.environ.get("QSCHED_MODEL_MIN_R2", "0.95"))
DEFAULT_MAX_MAE = float(os.environ["QSCHED_MODEL_MAX_MAE"]) \
    if os.environ.get("QSCHED_MODEL_MAX_MAE") else None


def topology_divergence(src_topology: str, dst_topology: str) -> float:
    """
    Returns a divergence score [0, 1] between two QPU topologies.
    0.0 = identical topology, no retranspilation overhead.
    1.0 = completely incompatible (theoretical maximum).
    """
    key = (src_topology.lower(), dst_topology.lower())
    if key in TOPOLOGY_DIVERGENCE:
        return TOPOLOGY_DIVERGENCE[key]
    # Symmetric fallback
    key_rev = (dst_topology.lower(), src_topology.lower())
    if key_rev in TOPOLOGY_DIVERGENCE:
        return TOPOLOGY_DIVERGENCE[key_rev]
    # Unknown pair: assume moderate divergence
    logger.warning(f"Unknown topology pair {key}, using default divergence 0.25")
    return 0.25


def estimate_overhead(divergence: float, circuit_depth: int,
                      num_two_qubit_gates: int) -> Tuple[float, float]:
    """
    Estimated transpilation overhead used consistently by the physics formula,
    the ML features and the training-data generator.

    Returns:
        (added_two_qubit_gates ΔN_2q, added_layers Δdepth)
    """
    added_2q = SWAP_COST_2Q * divergence * num_two_qubit_gates
    added_layers = divergence * circuit_depth
    return added_2q, added_layers


def physics_fidelity(
    divergence: float,
    circuit_depth: int,
    num_two_qubit_gates: int,
    target_mean_tq_error: float,
    target_mean_t2_us: float,
    target_mean_fidelity: float,
) -> Tuple[float, float, float, float]:
    """
    Physics-informed post-transpilation fidelity estimate:

        F_post = F_hw · (1 − e_2q)^ΔN_2q · exp(−Δt / T2)

    Returns:
        (F_post, ΔN_2q, gate_factor, t2_factor)
    """
    added_2q, added_layers = estimate_overhead(divergence, circuit_depth,
                                               num_two_qubit_gates)
    gate_factor = (1.0 - target_mean_tq_error) ** added_2q
    added_time_us = added_layers * LAYER_TIME_US
    t2_factor = float(np.exp(-added_time_us / max(target_mean_t2_us, 1.0)))
    predicted = target_mean_fidelity * gate_factor * t2_factor
    return float(np.clip(predicted, 0.0, 1.0)), added_2q, gate_factor, t2_factor


def build_feature_vector(
    src_topology: str,
    dst_topology: str,
    circuit_depth: int,
    num_two_qubit_gates: int,
    target_mean_tq_error: float,
    target_mean_t2_us: float,
    target_mean_fidelity: float,
    divergence: Optional[float] = None,
) -> np.ndarray:
    """Build the feature vector used by the ML model (see FEATURE_NAMES)."""
    div = topology_divergence(src_topology, dst_topology) if divergence is None \
        else float(divergence)
    added_2q, _ = estimate_overhead(div, circuit_depth, num_two_qubit_gates)

    # Expected number of failures among the added two-qubit gates
    added_error = added_2q * target_mean_tq_error

    # Information surviving the full (post-transpilation) execution time
    total_time_us = circuit_depth * (1 + div) * LAYER_TIME_US
    decoherence_factor = np.exp(-total_time_us / max(target_mean_t2_us, 1.0))

    features = np.array([
        div,                          # topology divergence score
        circuit_depth,                # original circuit depth
        num_two_qubit_gates,          # number of 2Q gates
        target_mean_tq_error,         # target QPU 2Q error rate
        target_mean_t2_us,            # target QPU T2 time
        target_mean_fidelity,         # target QPU raw fidelity
        added_2q,                     # ΔN_2q: estimated added 2Q gates
        added_error,                  # ΔN_2q · e_2q: estimated added error
        decoherence_factor,           # T2-based decoherence factor
    ], dtype=float)
    return features


FEATURE_NAMES = [
    "topology_divergence",
    "circuit_depth",
    "num_two_qubit_gates",
    "target_tq_error",
    "target_t2_us",
    "target_hw_fidelity",
    "added_two_qubit_gates",
    "added_error",
    "decoherence_factor",
]


class CostFidelityModel:
    """
    Predicts post-transpilation fidelity for routing decisions.

    Two modes:
    - 'physics': Uses the physics-based formula directly (no ML, always works)
    - 'ml':      Uses a trained sklearn model (more accurate with real data)

    Model acceptance threshold : the learned model is
    used only if its validation metrics meet an operator-configured criterion
    (minimum R² and/or maximum MAE). Otherwise — or after a failed
    re-validation on observed executions — the physics formula is used.

    Safety margin: optionally, a candidate passes only if
    F_pred − margin ≥ min_fidelity, where the margin is derived from the
    validation error ('mae', 'p95') or given as a number.
    """

    def __init__(self, mode: str = "physics",
                 min_r2: Optional[float] = None,
                 max_mae: Optional[float] = None,
                 safety_margin="none",
                 model_path: Path = MODEL_PATH):
        self.requested_mode = mode
        self.mode = mode
        self._model = None
        self.metrics: dict = {}
        self.label_source: str = "unknown"
        self.min_r2 = DEFAULT_MIN_R2 if min_r2 is None else min_r2
        self.max_mae = DEFAULT_MAX_MAE if max_mae is None else max_mae
        self.safety_margin_spec = safety_margin
        self.acceptance_status = "not_applicable"
        self.model_path = Path(model_path)

        if mode == "ml":
            self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            logger.warning("[CostFidelityModel] No trained model found. "
                           "Run generate_training_data.py + train_model.py first. "
                           "Falling back to physics mode.")
            self._fallback("model_missing")
            return
        with open(self.model_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "model" in obj:
            self._model = obj["model"]
            self.metrics = dict(obj.get("metrics", {}))
            self.label_source = obj.get("label_source", "unknown")
            names = obj.get("feature_names")
            if names and list(names) != list(FEATURE_NAMES):
                logger.warning("[CostFidelityModel] Feature names differ from this "
                               "code version; retrain the model.")
                self._fallback("feature_mismatch")
                return
        else:
            # Legacy artifact: a bare estimator without validation metrics
            self._model = obj
        logger.info(f"[CostFidelityModel] Loaded ML model from {self.model_path} "
                    f"(label_source={self.label_source}, metrics={self.metrics})")
        self._check_acceptance(self.metrics, source="training validation")

    def _fallback(self, status: str):
        self.mode = "physics"
        self.acceptance_status = status

    def _meets_threshold(self, metrics: dict) -> Tuple[bool, str]:
        if self.min_r2 is not None:
            r2 = metrics.get("r2")
            if r2 is None or r2 < self.min_r2:
                return False, f"r2={r2} < min_r2={self.min_r2}"
        if self.max_mae is not None:
            mae = metrics.get("mae")
            if mae is None or mae > self.max_mae:
                return False, f"mae={mae} > max_mae={self.max_mae}"
        return True, "ok"

    def _check_acceptance(self, metrics: dict, source: str) -> bool:
        ok, why = self._meets_threshold(metrics)
        if ok:
            self.mode = "ml"
            self.acceptance_status = "accepted"
        else:
            logger.warning(f"[CostFidelityModel] Learned model rejected on {source} "
                           f"({why}); using physics-informed estimate.")
            self._fallback(f"rejected: {why}")
        return ok

    def revalidate(self, predicted, observed, min_samples: int = 20) -> dict:
        """
        Re-validate against recent executions (predicted vs. observed fidelity,
        e.g. from the audit annotations / experiment tracking). Disables the
        learned model if the acceptance threshold is no longer met.
        """
        from sklearn.metrics import mean_absolute_error, r2_score
        predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
        if len(observed) < min_samples:
            return {"status": "insufficient_samples", "n": int(len(observed))}
        err = np.abs(predicted - observed)
        metrics = {"r2": float(r2_score(observed, predicted)),
                   "mae": float(err.mean()),
                   "p95_abs_error": float(np.percentile(err, 95)),
                   "n": int(len(observed))}
        if self._model is not None:
            self._check_acceptance(metrics, source="recent executions")
            if self.mode == "ml":
                self.metrics.update(metrics)
        return metrics

    @property
    def safety_margin(self) -> float:
        """Margin subtracted from the prediction before comparing to the threshold."""
        spec = self.safety_margin_spec
        if isinstance(spec, (int, float)):
            return float(spec)
        if spec in (None, "none") or self.mode != "ml":
            return 0.0
        key = {"mae": "mae", "p95": "p95_abs_error"}.get(spec)
        if key is None:
            raise ValueError(f"Unknown safety_margin: {spec!r}")
        return float(self.metrics.get(key, 0.0))

    def predict(
        self,
        src_topology: str,
        dst_topology: str,
        circuit_depth: int,
        num_two_qubit_gates: int,
        target_mean_tq_error: float,
        target_mean_t2_us: float,
        target_mean_fidelity: float,
        divergence: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        Predict post-transpilation fidelity.

        Args:
            divergence: SWAP insertions per original two-qubit gate. If None,
                        the family-level lookup table is used; otherwise e.g.
                        the graph-based estimate (circuit_analysis) or the
                        actual overhead measured after retranspilation.

        Returns:
            (predicted_fidelity, explanation_string)
        """
        table = divergence is None
        div = topology_divergence(src_topology, dst_topology) if table else float(divergence)

        if self.mode == "ml" and self._model is not None:
            features = build_feature_vector(
                src_topology, dst_topology, circuit_depth,
                num_two_qubit_gates, target_mean_tq_error,
                target_mean_t2_us, target_mean_fidelity,
                divergence=None if table else div,
            )
            predicted = float(self._model.predict(features.reshape(1, -1))[0])
            predicted = max(0.0, min(1.0, predicted))
            explanation = (
                f"ML model prediction: topology_divergence={div:.2f}, "
                f"predicted_fidelity={predicted:.4f}"
            )
        else:
            predicted, explanation = self._physics_prediction(
                div, circuit_depth, num_two_qubit_gates,
                target_mean_tq_error, target_mean_t2_us, target_mean_fidelity
            )

        return predicted, explanation

    def _physics_prediction(
        self,
        divergence: float,
        circuit_depth: int,
        num_two_qubit_gates: int,
        target_mean_tq_error: float,
        target_mean_t2_us: float,
        target_mean_fidelity: float,
    ) -> Tuple[float, str]:
        """
        Physics-informed fidelity estimate after retranspilation
        (see physics_fidelity()).
        """
        predicted, extra_2q, gate_fidelity_factor, t2_factor = physics_fidelity(
            divergence, circuit_depth, num_two_qubit_gates,
            target_mean_tq_error, target_mean_t2_us, target_mean_fidelity,
        )

        explanation = (
            f"Physics model: divergence={divergence:.2f}, "
            f"extra_2Q_gates={extra_2q:.1f}, "
            f"gate_factor={gate_fidelity_factor:.4f}, "
            f"t2_factor={t2_factor:.4f}, "
            f"predicted_fidelity={predicted:.4f}"
        )
        return predicted, explanation

    def will_meet_threshold(
        self,
        min_fidelity: float,
        src_topology: str,
        dst_topology: str,
        circuit_depth: int,
        num_two_qubit_gates: int,
        target_mean_tq_error: float,
        target_mean_t2_us: float,
        target_mean_fidelity: float,
        divergence: Optional[float] = None,
    ) -> Tuple[bool, float, str]:
        """
        Returns (passes, predicted_fidelity, explanation).
        Use this in the Filter phase to exclude topologically incompatible QPUs.
        """
        predicted, explanation = self.predict(
            src_topology, dst_topology, circuit_depth,
            num_two_qubit_gates, target_mean_tq_error,
            target_mean_t2_us, target_mean_fidelity,
            divergence=divergence,
        )
        margin = self.safety_margin
        passes = (predicted - margin) >= min_fidelity
        if margin > 0:
            explanation += f", safety_margin={margin:.4f}"
        return passes, predicted, explanation
