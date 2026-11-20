import pytest

from cost_fidelity_model import (
    KNOWN_TOPOLOGIES, SWAP_COST_2Q, CostFidelityModel, build_feature_vector,
    estimate_overhead, physics_fidelity, topology_divergence,
)


@pytest.mark.parametrize("topology", KNOWN_TOPOLOGIES)
def test_same_topology_has_zero_divergence(topology):
    # Bug 2: same-topology pairs used to fall back to the 0.25 default
    assert topology_divergence(topology, topology) == 0.0


def test_divergence_is_case_insensitive_and_symmetric_fallback():
    assert topology_divergence("Heavy-Hex", "SQUARE-LATTICE") == 0.20
    # ("falcon", "heavy-hex") is not in the table; reverse lookup applies
    assert topology_divergence("falcon", "heavy-hex") == 0.05


def test_physics_fidelity_equals_hw_fidelity_without_divergence():
    f, added_2q, gate_factor, t2_factor = physics_fidelity(0.0, 85, 42, 0.01, 100.0, 0.995)
    assert f == pytest.approx(0.995)
    assert added_2q == 0.0 and gate_factor == 1.0 and t2_factor == 1.0


def test_physics_fidelity_decreases_with_divergence():
    f_low_div, *_ = physics_fidelity(0.1, 85, 42, 0.01, 100.0, 0.995)
    f_high_div, *_ = physics_fidelity(0.3, 85, 42, 0.01, 100.0, 0.995)
    assert f_high_div < f_low_div < 0.995


def test_features_use_same_overhead_as_physics_formula():
    # Bug 4: physics used div*N_2q*3 while features used div*depth
    div = topology_divergence("heavy-hex", "square-lattice")
    feats = build_feature_vector("heavy-hex", "square-lattice", 85, 42, 0.01, 100.0, 0.995)
    added_2q, _ = estimate_overhead(div, 85, 42)
    assert added_2q == pytest.approx(SWAP_COST_2Q * div * 42)
    assert feats[6] == pytest.approx(added_2q)
    assert feats[7] == pytest.approx(added_2q * 0.01)
    _, physics_added_2q, _, _ = physics_fidelity(div, 85, 42, 0.01, 100.0, 0.995)
    assert feats[6] == pytest.approx(physics_added_2q)


def test_will_meet_threshold_physics_mode():
    model = CostFidelityModel(mode="physics")
    passes, f, _ = model.will_meet_threshold(
        min_fidelity=0.99, src_topology="heavy-hex", dst_topology="square-lattice",
        circuit_depth=85, num_two_qubit_gates=42, target_mean_tq_error=0.01,
        target_mean_t2_us=100.0, target_mean_fidelity=0.995)
    assert not passes and f < 0.99
