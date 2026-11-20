import threading
from dataclasses import replace

import pytest
from qiskit import QuantumCircuit, qasm2

from binding import SchedulingPipeline
from circuit_analysis import circuit_stats, estimate_divergence, load_circuit
from conftest import make_backend, make_job
from cost_fidelity_model import CostFidelityModel
from job_request import CircuitProfile
from quota import QuotaController, QuotaExceeded, QuotaSpec
from retranspilation_broker import RetranspilationBroker, TranspileResult, qiskit_basis
from scheduler import QuantumScheduler
from topology import family_coupling_map
from usage import estimate_usage


def line_circuit(n=4):
    qc = QuantumCircuit(n)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def star_circuit(n=6):
    """Qubit 0 interacts with every other qubit (needs SWAPs on sparse graphs)."""
    qc = QuantumCircuit(n)
    for i in range(1, n):
        qc.cx(0, i)
    qc.measure_all()
    return qc


# ── circuit analysis ──────────────────────────────────────────────────────

def test_circuit_stats_interaction_graph():
    st = circuit_stats(line_circuit(4))
    assert st.num_qubits == 4 and st.num_two_qubit_gates == 3
    assert st.interaction == {(0, 1): 1, (1, 2): 1, (2, 3): 1}


def test_line_embeds_without_swaps_on_heavy_hex():
    st = circuit_stats(line_circuit(5))
    est = estimate_divergence(st, family_coupling_map("heavy-hex", 27), 27, ["ecr"])
    assert est.exact_embedding and est.swaps == 0 and est.added_two_qubit_gates == 0


def test_star_needs_swaps_on_heavy_hex_but_not_all_to_all():
    st = circuit_stats(star_circuit(6))   # degree-5 node; heavy-hex max degree is 3
    sparse = estimate_divergence(st, family_coupling_map("heavy-hex", 27), 27, ["ecr"])
    dense = estimate_divergence(st, family_coupling_map("all-to-all", 8), 8, ["ms"])
    assert sparse.swaps > 0 and sparse.added_two_qubit_gates == 3 * sparse.swaps
    assert dense.swaps == 0 and dense.added_two_qubit_gates == 0
    assert sparse.score > dense.score


def test_profile_from_qasm():
    prof = CircuitProfile.from_qasm(qasm2.dumps(line_circuit(4)), "heavy-hex")
    assert prof.num_qubits == 4 and prof.num_two_qubit_gates == 3 and prof.interaction


def test_scheduler_uses_graph_divergence(now):
    prof = CircuitProfile.from_qasm(qasm2.dumps(star_circuit(6)), "all-to-all")
    job = make_job(circuit=prof, min_fidelity=0.5)
    b = make_backend(coupling_map=[list(e) for e in family_coupling_map("heavy-hex", 27)],
                     qubit_count=27)
    fr = QuantumScheduler().filter_backend(job, b, now=now)
    assert fr.divergence_estimate is not None and fr.divergence_estimate.swaps > 0
    assert fr.cost_fidelity_explanation.startswith("graph-divergence")


# ── model acceptance threshold ───────────────────────────────────────────────

def test_model_rejected_below_min_r2_falls_back_to_physics():
    m = CostFidelityModel("ml", min_r2=1.01)
    assert m.mode == "physics" and m.acceptance_status.startswith("rejected")


def test_model_revalidation_disables_ml():
    m = CostFidelityModel("ml", min_r2=0.9)
    assert m.mode == "ml"
    m.revalidate(predicted=[0.99] * 30, observed=[0.5 + 0.01 * i for i in range(30)])
    assert m.mode == "physics"


def test_safety_margin_blocks_borderline_candidate():
    m = CostFidelityModel("ml", safety_margin=0.01)
    passes, f, _ = m.will_meet_threshold(0.99, "heavy-hex", "heavy-hex", 10, 2,
                                         0.001, 200, 0.995)
    assert f >= 0.99 and not passes


# ── usage / cost model ───────────────────────────────────────────────────────

def test_usage_schemes():
    d = {"gate_1q_ns": 50, "gate_2q_ns": 500, "measure_ns": 1000, "reset_us": 100,
         "job_overhead_s": 0}
    time_based = estimate_usage({"scheme": "time", "price_per_second": 2.0}, d, 10, 10, 10, 1000)
    assert time_based.qpu_seconds == pytest.approx(1000 * ((10 * 275 + 1000) * 1e-9 + 100e-6))
    assert time_based.cost_usd == pytest.approx(2.0 * time_based.qpu_seconds)
    gate = estimate_usage({"scheme": "per_gate_shot", "price_1q": 0.001, "price_2q": 0.01},
                          d, 10, 10, 10, 100)
    assert gate.cost_usd == pytest.approx(100 * (0.01 + 0.1))
    prepaid = estimate_usage({"scheme": "prepaid_time", "plan_price": 100, "plan_seconds": 1000},
                             d, 10, 10, 10, 1000)
    assert prepaid.cost_usd == 0 and prepaid.effective_cost_usd == pytest.approx(
        prepaid.qpu_seconds * 0.1)


# ── quota ────────────────────────────────────────────────────────────────────

def test_quota_units_and_vendor_limits(now):
    q = QuotaController()
    q.set_quota("ns", QuotaSpec(limits={"usd": 10}, vendor_limits={"ibm": {"qpu_seconds": 60}}))
    r = q.reserve("ns", "ibm", {"usd": 0, "qpu_seconds": 50}, now=now)
    with pytest.raises(QuotaExceeded) as e:
        q.reserve("ns", "ibm", {"usd": 0, "qpu_seconds": 20}, now=now)
    assert e.value.unit == "qpu_seconds" and e.value.scope == "vendor:ibm"
    q.release(r)
    q.reserve("ns", "ibm", {"usd": 0, "qpu_seconds": 20}, now=now)


def test_quota_reconcile_and_daily_reset(now):
    from datetime import timedelta
    q = QuotaController()
    q.set_quota("ns", QuotaSpec(limits={"usd": 10}))
    r = q.reserve("ns", "v", {"usd": 4}, now=now)
    q.reconcile(r, {"usd": 6})
    assert q.usage("ns")["total"]["usd"] == pytest.approx(6)
    q.reserve("ns", "v", {"usd": 9}, now=now + timedelta(days=1))   # new day
    assert q.usage("ns")["total"]["usd"] == pytest.approx(9)


def test_quota_concurrent_reservations_never_overspend(now):
    q = QuotaController()
    q.set_quota("ns", QuotaSpec(limits={"usd": 100}))
    ok = []

    def worker():
        for _ in range(50):
            try:
                q.reserve("ns", "v", {"usd": 1}, now=now)
                ok.append(1)
            except QuotaExceeded:
                pass

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(ok) == 100 and q.usage("ns")["total"]["usd"] == pytest.approx(100)


# ── broker + two-stage binding ───────────────────────────────────────────────

def test_qiskit_basis_mapping():
    assert qiskit_basis(["gpi", "gpi2", "ms"])[0] == "rxx"
    assert qiskit_basis(["cz", "rx", "rz"]) == ["cz", "rx", "rz"]
    assert qiskit_basis(["ecr", "rz", "sx", "x"])[0] == "ecr"


def test_broker_transpiles_to_target():
    b = make_backend(coupling_map=[list(e) for e in family_coupling_map("heavy-hex", 27)],
                     qubit_count=27, native_gates=["ecr", "rz", "sx", "x"])
    prof = CircuitProfile.from_qasm(qasm2.dumps(star_circuit(6)), "all-to-all")
    res = RetranspilationBroker(optimization_levels=(1,), seeds=(1,)).transpile(prof, b)
    assert res.num_two_qubit_gates >= prof.num_two_qubit_gates
    assert res.added_two_qubit_gates > 0 and res.swap_ratio > 0


class _BadBroker(RetranspilationBroker):
    """Pretends transpilation added far more SWAPs than estimated."""
    def transpile(self, circuit_profile, backend):
        return TranspileResult(qasm=None, depth=500, num_single_qubit_gates=100,
                               num_two_qubit_gates=400, added_two_qubit_gates=390,
                               swap_ratio=50.0, seconds=0.0, toolchain="stub")


def test_revalidation_failure_rolls_back_to_next_candidate(now):
    prof = CircuitProfile.from_qasm(qasm2.dumps(line_circuit(4)), "all-to-all")
    job = make_job(circuit=prof, min_fidelity=0.9)
    foreign = make_backend(backend_name="foreign", mean_qubit_fidelity=0.999)   # heavy-hex
    native = make_backend(backend_name="native", topology="all-to-all",
                          mean_qubit_fidelity=0.95)
    pipe = SchedulingPipeline(QuantumScheduler(), _BadBroker(), max_retries=2)
    d = pipe.schedule_and_bind(job, [foreign, native], now=now)
    assert [a.outcome for a in d.attempts] == ["revalidation_failed", "bound"]
    assert d.bound and d.backend_name == "native"


def test_budget_failure_does_not_fall_back_to_simulator(now):
    job = make_job(allow_simulator_fallback=True)
    b = make_backend(pricing={"scheme": "per_shot", "price_per_shot": 1.0})
    sim = make_backend(backend_name="sim", is_simulator_fallback=True)
    q = QuotaController()
    q.set_quota("default", QuotaSpec(limits={"usd": 10}))
    d = SchedulingPipeline(QuantumScheduler(), None, q).schedule_and_bind(job, [b, sim], now=now)
    assert not d.bound and "quota_exceeded" in d.unschedulable_reason


@pytest.mark.parametrize("family,n", [("heavy-hex", 27), ("heavy-hex", 127),
                                      ("square-lattice", 84), ("all-to-all", 12)])
def test_family_coupling_maps_are_connected(family, n):
    import rustworkx as rx
    g = rx.PyGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from_no_data(family_coupling_map(family, n))
    assert rx.is_connected(g)
    assert max(max(e) for e in family_coupling_map(family, n)) == n - 1


def test_crd_contains_forecast_and_topology(now):
    from crd_writer import telemetry_to_crd
    from forecaster import QualityWindow
    w = QualityWindow(now, now, 0.999, 0.9, 30.0, 0.995)
    b = make_backend(coupling_map=[[0, 1]], pricing={"scheme": "time", "price_per_second": 1.6})
    st = telemetry_to_crd(b, [w])["status"]
    assert st["qualityForecast"][0]["minLowerBound"] == 0.995
    assert st["physicalProperties"]["couplingMap"] == [[0, 1]]
    assert st["pricing"]["scheme"] == "time"


def test_prophet_windows_start_in_the_future(now):
    pytest.importorskip("prophet")
    from test_forecaster_and_deferral import make_history
    from forecaster import QualityForecaster
    hist = make_history(now - __import__("datetime").timedelta(hours=1))
    res = QualityForecaster(hist, model="prophet").forecast(
        "qpu_a", min_fidelity=0.99, horizon_hours=6, now=now)
    assert res.model_used == "prophet"
    assert res.time_steps[0] == now
    assert all(w.minutes_from_now >= 0 for w in res.quality_windows)


def test_broker_uses_tket_for_trapped_ion():
    pytest.importorskip("pytket")
    from topology import family_coupling_map
    ion = make_backend(topology="all-to-all", native_gates=["gpi", "gpi2", "ms"],
                       qubit_count=8,
                       coupling_map=[list(e) for e in family_coupling_map("all-to-all", 8)])
    prof = CircuitProfile.from_qasm(qasm2.dumps(star_circuit(6)), "heavy-hex")
    res = RetranspilationBroker().transpile(prof, ion)
    assert res.toolchain == "tket"
    assert res.added_two_qubit_gates == 0          # all-to-all: no SWAPs
    assert res.num_two_qubit_gates == prof.num_two_qubit_gates


def test_tket_routes_on_sparse_architecture():
    pytest.importorskip("pytket")
    from retranspilation_broker import tket_toolchain
    b = make_backend(coupling_map=[list(e) for e in family_coupling_map("heavy-hex", 27)],
                     qubit_count=27, native_gates=["ecr", "rz", "sx", "x"])
    out = tket_toolchain(star_circuit(6), b)
    names = {i.operation.name for i in out.data}
    assert "ecr" in names and "swap" not in names and "cx" not in names
