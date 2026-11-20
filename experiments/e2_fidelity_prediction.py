"""
E2 — Can circuit fidelity on a candidate QPU be predicted WITHOUT transpiling?

Circuits: MIRROR versions of the benchmark circuits (Proctor et al., 2021):
a random X layer, the circuit C, a barrier, and C†. The ideal output is a
single known bitstring, so every family yields a well-defined label. (Raw
Hellinger fidelity of the plain circuits is not usable: e.g. QFT|0…0⟩ has a
uniform ideal output, and noise pushes near-uniform outputs towards uniform
without lowering the fidelity.) The barrier prevents the transpiler from
cancelling C·C†.

Label: success probability of the ideal bitstring in a noisy Aer simulation of
the circuit transpiled for the target, using the target's calibration snapshot
(gate/readout errors, T1/T2, durations).

Predictors (no transpilation unless stated):
  raw_fidelity    device mean qubit fidelity (what a naive scheduler uses)
  esp_logical     estimated success probability of the LOGICAL circuit
                  (ignores transpilation overhead)
  esp_estimated   ESP with the topology overhead estimate (static placement, the
                  scheduler default; calibrated on E1), device-mean errors
  esp_dynamic     same with the routing-replay estimate (variant)
  esp_per_qubit   ESP on the qubits/couplers of the noise-aware placement and
                  routing replay (SWAP term calibrated on E1) (variant)
  esp_actual      ESP of the ACTUALLY transpiled circuit (oracle: needs transpile)
  learned         gradient-boosted regressor on estimate-based features,
                  evaluated with leave-one-family-out cross-validation

Run:  python e2_fidelity_prediction.py
Out:  results/e2_dataset.csv, results/e2_summary.json, results/e2_prediction.png
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS, benchmark_suite, count_ops, esp, load_targets, transpile_best

warnings.filterwarnings("ignore")
SHOTS = 2000


def e1_alpha(col="static_added_2q"):
    """Global SWAP-overhead calibration factor fitted on E1 (sparse targets)."""
    df = pd.read_csv(RESULTS / "e1_divergence.csv")
    d = df[df[col] > 0]
    return float((d[col] * d.actual_added_2q).sum() / (d[col] ** 2).sum())


def mirror(circuit, seed):
    """X-layer + C + barrier + C†; returns (mirror circuit, ideal bitstring)."""
    rng = np.random.default_rng(seed)
    n = circuit.num_qubits
    bits = rng.integers(0, 2, n)
    from qiskit import QuantumCircuit
    m = QuantumCircuit(n)
    for q, b in enumerate(bits):
        if b:
            m.x(q)
    m.compose(circuit, inplace=True)
    m.barrier()
    m.compose(circuit.inverse(), inplace=True)
    key = "".join(str(b) for b in reversed(bits))      # Qiskit little-endian
    return m, key


def build_dataset(sizes=(4, 6, 8, 10), instances=1, ckpt=None, ckpt_every=1,
                  threads=None):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from circuit_analysis import ErrorModel, circuit_stats, estimate_divergence
    from usage import estimate_qpu_seconds

    alpha = e1_alpha("static_added_2q")
    alpha_dyn = e1_alpha("est_added_2q")
    targets = load_targets()
    # Nestes tamanhos o vetor de estado e pequeno (2^14 amplitudes), entao os
    # threads do OpenMP passam mais tempo na barreira do que calculando.
    # Limitar custa pouco tempo de relogio e devolve a maquina para voce.
    opts = {"seed_simulator": 5}
    if threads:
        opts["max_parallel_threads"] = threads
    sims = {t.name: AerSimulator(noise_model=NoiseModel.from_backend(t.backend), **opts)
            for t in targets}

    # Checkpoint: as linhas ja calculadas voltam do disco e os circuitos
    # correspondentes sao pulados. E o que transforma uma interrupcao em
    # "recomeca de onde parou" em vez de "perdeu tudo".
    rows, feitos = [], set()
    if ckpt is not None and ckpt.exists():
        anterior = pd.read_csv(ckpt)
        rows = anterior.to_dict("records")
        feitos = set(anterior.circuit)
        print(f"  retomando de {ckpt.name}: {len(feitos)} circuitos ja prontos",
              flush=True)

    bench = benchmark_suite(sizes=sizes, instances=instances)
    total = len(bench)
    for i, b in enumerate(bench):
        if b.name in feitos:
            continue
        circ, key = mirror(b.circuit, seed=100 + i)
        st = circuit_stats(circ)
        qc = circ.copy()
        qc.measure_all()
        for t in targets:
            tel = t.telemetry
            est = estimate_divergence(st, tel.coupling_map, tel.qubit_count, tel.native_gates,
                                      routing="static")
            est_dyn = estimate_divergence(st, tel.coupling_map, tel.qubit_count, tel.native_gates,
                                          routing="dynamic")
            em = ErrorModel(tel.error_map, tel.mean_sq_error, tel.mean_tq_error, tel.mean_ro_error)
            est_pq = estimate_divergence(st, tel.coupling_map, tel.qubit_count, tel.native_gates,
                                         routing="dynamic", errors=em)
            added = alpha * est.added_two_qubit_gates
            n1_est = st.num_single_qubit_gates * est.gate_overhead
            n2_est = st.num_two_qubit_gates + added
            depth_est = st.depth * (1 + est.swap_ratio * alpha)
            dur_est = estimate_qpu_seconds(depth_est, n1_est, n2_est, 1,
                                           {**tel.durations, "reset_us": 0, "job_overhead_s": 0})
            n2_dyn = st.num_two_qubit_gates + alpha_dyn * est_dyn.added_two_qubit_gates
            dur_dyn = estimate_qpu_seconds(st.depth * (1 + est_dyn.swap_ratio * alpha_dyn),
                                           n1_est, n2_dyn, 1,
                                           {**tel.durations, "reset_us": 0, "job_overhead_s": 0})
            out, _ = transpile_best(qc, t.backend)
            n1a, n2a, deptha = count_ops(out)
            dur_act = estimate_qpu_seconds(deptha, n1a, n2a, 1,
                                           {**tel.durations, "reset_us": 0, "job_overhead_s": 0})
            counts = sims[t.name].run(out, shots=SHOTS).result().get_counts()
            label = counts.get(key, 0) / SHOTS
            e1, e2, ero, T2 = tel.mean_sq_error, tel.mean_tq_error, tel.mean_ro_error, tel.mean_t2_us
            rows.append(dict(
                circuit=b.name, family=b.family, n=b.num_qubits, target=t.name,
                topology=t.topology, depth=st.depth, n1=st.num_single_qubit_gates,
                n2=st.num_two_qubit_gates, est_added_2q=added, gate_overhead=est.gate_overhead,
                e1=e1, e2=e2, ero=ero, t2_us=T2, raw_fidelity=tel.mean_qubit_fidelity,
                esp_logical=esp(st.num_single_qubit_gates, st.num_two_qubit_gates, b.num_qubits,
                                e1, e2, ero),
                esp_estimated=esp(n1_est, n2_est, b.num_qubits, e1, e2, ero, dur_est, T2),
                esp_actual=esp(n1a, n2a, b.num_qubits, e1, e2, ero, dur_act, T2),
                esp_dynamic=esp(n1_est, n2_dyn, b.num_qubits, e1, e2, ero, dur_dyn, T2),
                esp_per_qubit=est_pq.esp_calibrated(alpha_dyn) * float(np.exp(-dur_dyn / (T2 * 1e-6))),
                actual_n2=n2a, success_probability=label,
            ))
        # flush=True porque um `grep` no meio do pipe rebufferiza a saida e
        # o progresso some ate o processo terminar.
        print(f"  {b.name:<24} done  ({i + 1}/{total})", flush=True)
        if ckpt is not None and (i + 1) % ckpt_every == 0:
            pd.DataFrame(rows).to_csv(ckpt, index=False)

    if ckpt is not None:
        pd.DataFrame(rows).to_csv(ckpt, index=False)
    return pd.DataFrame(rows), alpha


FEATURES = ["n", "depth", "n1", "n2", "est_added_2q", "gate_overhead",
            "e1", "e2", "ero", "t2_us", "raw_fidelity", "esp_estimated", "esp_per_qubit"]


def learned_lofo(df):
    """Leave-one-family-out predictions of a gradient-boosted regressor."""
    from sklearn.ensemble import GradientBoostingRegressor
    pred = pd.Series(np.nan, index=df.index)
    for fam in df.family.unique():
        tr, te = df.family != fam, df.family == fam
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=0)
        m.fit(df.loc[tr, FEATURES], df.loc[tr, LABEL])
        pred[te] = m.predict(df.loc[te, FEATURES])
    return pred.clip(0, 1)


LABEL = "success_probability"
PREDICTORS = ("raw_fidelity", "esp_logical", "esp_estimated", "esp_dynamic",
              "esp_per_qubit", "esp_actual", "learned")


def add_labels(df):
    return df.copy()


def summarize(df):
    from sklearn.metrics import mean_absolute_error, r2_score
    df = add_labels(df)
    df["learned"] = learned_lofo(df)
    out = {}
    for label in (LABEL,):
        y = df[label]
        res_l = {}
        for col in PREDICTORS:
            if col == "learned" and label != LABEL:
                continue
            res = {"spearman": float(spearmanr(df[col], y).correlation)}
            # Per circuit: is the predicted best target the truly best one, and
            # how much label is lost by picking it (regret)?
            hits, regret = [], []
            for _, g in df.groupby("circuit"):
                pick, best = g.loc[g[col].idxmax()], g.loc[g[label].idxmax()]
                hits.append(pick.target == best.target)
                regret.append(best[label] - pick[label])
            res["top1_target_accuracy"] = float(np.mean(hits))
            res["mean_regret"] = float(np.mean(regret))
            if col == "learned":
                res["mae"] = float(mean_absolute_error(y, df[col]))
                res["r2"] = float(r2_score(y, df[col]))
            res_l[col] = res
        out[label] = res_l
    return df, out


def plot(df, tag=""):
    import matplotlib.pyplot as plt
    cols = ["raw_fidelity", "esp_logical", "esp_estimated", "esp_per_qubit", "esp_actual", "learned"]
    titles = ["Raw device fidelity", "ESP, logical circuit", "ESP, topology estimate",
              "ESP, per-qubit rates", "ESP, actual transpilation (oracle)", "Learned (LOFO CV)"]
    colors = {"heavy-hex": "#4a9eed", "square-lattice": "#ef4444", "all-to-all": "#22c55e"}
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    axes = axes.ravel()
    for k, (ax, c, title) in enumerate(zip(axes, cols, titles)):
        for topo, d in df.groupby("topology"):
            ax.scatter(d[c], d[LABEL], s=12, alpha=0.7, color=colors[topo], label=topo)
        rho = spearmanr(df[c], df[LABEL]).correlation
        ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}", fontsize=11)
        ax.set_xlabel("Predicted value")
        if k % 3 == 0:
            ax.set_ylabel("Simulated success probability")
        if c == "raw_fidelity":
            from matplotlib.ticker import MaxNLocator, FormatStrFormatter
            ax.xaxis.set_major_locator(MaxNLocator(3)); ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS / f"e2_prediction{tag}.png", dpi=200)
    plt.savefig(RESULTS / f"e2_prediction{tag}.pdf")


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse the saved dataset")
    ap.add_argument("--sizes", default="4,6,8,10")
    ap.add_argument("--instances", type=int, default=1,
                    help="random instances per (family, size)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _large")
    ap.add_argument("--checkpoint-every", type=int, default=1, metavar="N",
                    help="save partial results every N circuits (0 disables)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing checkpoint and start over")
    ap.add_argument("--threads", type=int, default=0, metavar="N",
                    help="cap Aer threads (0 = all cores, -1 = half)")
    ap.add_argument("--nice", type=int, default=0, metavar="N",
                    help="lower process priority by N (10 keeps the Mac usable)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_csv = RESULTS / f"e2_dataset{args.tag}.csv"
    ckpt = RESULTS / f"e2_dataset{args.tag}.parcial.csv"

    if args.nice:
        os.nice(args.nice)
    threads = args.threads
    if threads == -1:
        threads = max(1, (os.cpu_count() or 2) // 2)

    if args.fresh and ckpt.exists():
        ckpt.unlink()

    if args.reuse and out_csv.exists():
        df = pd.read_csv(out_csv)
        alpha = e1_alpha()
    else:
        df, alpha = build_dataset(tuple(int(x) for x in args.sizes.split(",")),
                                  args.instances,
                                  ckpt=ckpt if args.checkpoint_every else None,
                                  ckpt_every=max(1, args.checkpoint_every),
                                  threads=threads or None)
    df, summary = summarize(df)
    summary["e1_alpha_static"] = alpha
    summary["e1_alpha_dynamic"] = e1_alpha("est_added_2q")
    summary["n_samples"] = int(len(df))
    summary["n_circuits"] = int(df.circuit.nunique())
    df.to_csv(out_csv, index=False)
    (RESULTS / f"e2_summary{args.tag}.json").write_text(json.dumps(summary, indent=2))
    # Chegou ao fim: o parcial ja nao serve para nada e so confunde a
    # proxima execucao, que o leria como trabalho pronto.
    if ckpt.exists():
        ckpt.unlink()
    plot(df, args.tag)
    print(json.dumps(summary, indent=2))
