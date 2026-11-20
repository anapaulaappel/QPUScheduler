# Experiments (paper E1–E10)

All experiments run offline. Heavy-hex targets (Fez, Torino, Brisbane,
Sherbrooke, Kyiv) and the square-lattice target (Nighthawk) are IBM fake
backends carrying **real calibration snapshots**. All-to-all targets are
trapped-ion-like backends with error rates and durations of the order reported
for current devices. Labels come from **noisy simulation** (Qiskit Aer), not
hardware runs.

```bash
cd experiments
python e1_divergence.py            # ~10 s
python e1b_latency.py              # ~1 min
python e2_fidelity_prediction.py   # ~4 min (Aer simulations)
python e3_routing_policies.py      # ~10 s (needs E2 output)
python e4_e5_forecast_deferral.py  # real history if ≥3 days collected, else simulated
```

Outputs go to `results/` (CSV, JSON summaries, PNG figures).

## Results (current run)

### E1: topology-divergence estimate vs actual transpilation
35 circuits (7 families, 4–12 qubits) × 8 targets = 280 placements.

| Predictor of the 2Q gate count | Spearman ρ | MAE (2Q gates) | Median rel. error |
|---|---|---|---|
| Ignore transpilation | 0.946 | 40.9 | 41% |
| Family lookup table | 0.970 | 30.5 | 31% |
| Graph estimate, static placement (scheduler default), raw | **0.989** | 93.1 | 51% |
| Graph estimate, static placement, calibrated (LOFO) | 0.979 | 20.9 | 12% |
| Graph estimate, routing replay, raw | 0.986 | 83.6 | 54% |
| Graph estimate, routing replay, calibrated (LOFO) | 0.981 | **12.4** | **7%** |

The graph estimate ranks circuits best but overestimates SWAPs by about 3.5×
(α ≈ 0.28–0.29), since it is weaker than Qiskit's SABRE router. One
calibration factor fixes the magnitude. The routing replay gives the most
accurate calibrated gate count.

### E1b: latency per candidate QPU (FakeFez, 156 qubits)
The estimate is 4–20× faster than a level-1 transpilation and 13–115× faster
than level 3. At 80 qubits: 26–40 ms vs 0.25–0.56 s (level 1) and 0.55–3.0 s
(level 3). Timings vary by about ±30% between runs.

### E2: predicting success without transpiling
28 mirror circuits × 8 targets = 224 noisy simulations.

| Predictor | Spearman ρ | Mean regret of target choice |
|---|---|---|
| Raw device fidelity | 0.45 | 0.021 |
| ESP of the logical circuit | 0.91 | 0.028 |
| **ESP, topology estimate, static placement (no transpilation)** | 0.90 | **0.0129** |
| ESP, topology estimate, routing replay (variant) | 0.91 | 0.031 |
| ESP on the placed qubits, per-qubit errors (variant) | 0.75 | 0.030 |
| ESP of the actual transpiled circuit (oracle) | 0.91 | 0.0129 |
| Learned, leave-one-family-out | 0.93 (R² 0.88, MAE 0.07) | 0.014 |

Choosing the target with the static-placement estimate is as good as
transpiling every candidate.

**Negative results:**
- The routing-replay variant, although more accurate for gate counts, picks
  worse targets.
- Per-qubit errors make the estimate worse, probably because Qiskit chooses
  different physical qubits than our placement.
- With 28 circuits, regret differences of ~0.02 depend on a few circuits.

### E2 (large run, 3 Nov 2026): 102 circuits, 4–14 qubits
`python e2_fidelity_prediction.py --sizes 4,6,8,10,12,14 --instances 3 --tag _large`
(816 simulations; results in `results/e2_summary_large.json`, `e2_prediction_large.png`).

| Predictor | Spearman ρ | Mean regret | Best target picked |
|---|---|---|---|
| Raw device fidelity | 0.37 | 0.019 | 19% |
| ESP of the logical circuit | 0.94 | 0.041 | 25% |
| **ESP, topology estimate, static placement (no transpilation)** | **0.95** | 0.016 | 32% |
| ESP, topology estimate, routing replay (variant) | 0.95 | 0.026 | 24% |
| ESP, per-qubit errors (variant) | 0.81 | 0.036 | 24% |
| ESP of the actual transpiled circuit (oracle) | 0.94 | 0.015 | 41% |
| Learned, leave-one-family-out | 0.96 (R² 0.91, MAE 0.07) | 0.014 | 38% |

With more circuits the ranking result holds, but target selection by the
estimate is now slightly behind transpile-all (regret 0.016 vs 0.015, best
target 32% vs 41%), so "as good as" should read "nearly as good as".

### E3: routing policies (28 jobs × 8 candidates)

| τ = 0.8 | SLA violations | Unschedulable (false) | Latency / job |
|---|---|---|---|
| Highest raw fidelity | 10 | 0 | — |
| Lowest cost | 12 | 0 | — |
| **Estimate filter (ours)** | **0** | 15 (5) | **6 ms** |
| Estimate filter, isotonic-calibrated + margin | 0 | 18 (8) | 6 ms |
| Learned filter | 2 | 10 (2) | 6 ms |
| Transpile every candidate | 0 | 10 (0) | 154 ms |
| Oracle | 0 | 10 (0) | — |

Ours matches transpile-all on SLA violations (0 for τ ≥ 0.6) at about 25×
lower scheduling latency. The price is more conservative rejections, because
ESP underestimates success. Neither isotonic calibration nor the per-qubit
variant reduced those rejections. The learned filter rejects less but lets a
few violations through.

### E3 (large run, 3 Nov 2026): 102 jobs × 8 candidates
`python e3_routing_policies.py --tag _large --instances 3` (uses `e2_dataset_large.csv`).

| τ | Estimate filter: violations / false rejections | Transpile-all | Raw-best violations | Lowest-cost violations |
|---|---|---|---|---|
| 0.5 | 1 / 8 | 1 / 2 | 26 | 32 |
| 0.6 | 0 / 13 | 0 / 3 | 31 | 39 |
| 0.7 | 0 / 17 | 2 / 3 | 43 | 48 |
| 0.8 | 1 / 14 | 0 / 3 | 52 | 60 |
| 0.9 | 0 / 20 | 0 / 12 | 79 | 81 |

Latency per job: about 7 ms (estimate) vs 174 ms (transpile-all). With more jobs the
"zero violations" result of the 28-job run no longer holds exactly: both filters have
0–2 violations per threshold, against 26–81 for the naive policies.

### E4/E5: forecast and deferral
On **simulated** history:
- Lower-bound windows: precision 1.00, recall 0.57.
- Mean-based windows: precision 0.995, recall 0.95.
- Persistence baseline: 0.58 / 0.58.
- Deferral meets the SLA for 39% of jobs that failed at arrival (oracle 44%, run-now 0%).

These numbers are only a sanity check, because the forecaster's model matches
the simulator. The real evaluation needs the IBM history from
`probe/collect_calibration.py`. Early real data show IBM calibration ages of
18–63 min, i.e. much more frequent updates than the simulated 12 h cycle.

**On real IBM history (17 Nov 2026; negative result).** `python e4_e5_forecast_deferral.py --source real
--until 2026-11-13T17:30:00+00:00 --metric tq --tag _real_tq` (also `--metric qubit_fidelity`). 11.6 days of
coverage after removing collection gaps (grid points > 30 min from a poll are not scored). The reported mean
qubit fidelity takes only 3–6 values, so quality = 1 − mean 2Q error. Persistence 0.95/0.97 (precision/recall);
calibration-cycle lower bound 1.00/0.01; Prophet lower bound 0.87/0.54. Deferral success 0% vs oracle 6/16/38%
for 3/12/24 h deadlines (`results/e5_deadlines_real.json`). IBM stopped publishing calibrations on 13 Nov
(~17:20 UTC) and resumed on 18 Nov. Figure: `fig_e4_real_history.py`.

### E8: training-free estimate vs a learned predictor
`python e8_learned_baseline.py --boot 500 --seed 0` (offline). Gradient-boosted models on circuit + device
features vs our estimate, same out-of-fold rows; random, leave-one-device-out (strict) and leave-one-topology-out
splits, sim → hardware. Learned wins on seen devices; ours is more robust on unseen IBM devices (regret 0.008 vs
0.031) and topologies (ρ 0.95 vs 0.81). Outputs `results/e8_*`.

### E9: placement policies replayed on measured outcomes
`python e9_hardware_policies.py --dataset ibm|crossvendor [--split circuit]` (offline). On IBM, estimate filter
0–6 violations of 102 jobs vs 42–74 raw fidelity; split-conformal margin (δ = 0.1) reaches 0.91 coverage on
held-out circuits (0.87 on held-out families). Outputs `results/e9_*`.

### E10: application circuits on hardware
`e10_prepare.py` (circuits, ideal distributions), `e10_run_ibm.py` (IBM ×3), `e10_run_oq.py` (Rigetti, AQT,
verbatim via Open Quantum; `--quote-only` first), `e10_analyze.py`. 19 circuits (BV, GHZ, Grover, adder, QAOA,
VQE), 93 executions, normalized Hellinger fidelity. IBM selection regret 0.009 vs 0.063 raw fidelity; rankings
within a device ρ 0.48–0.77; cross-vendor selection fails with published error rates. Outputs `results/e10/`.

### Stale calibration (17 Nov 2026)
`python e6_hardware_validation.py prepare|submit|collect --backends ibm_fez,ibm_marrakesh,ibm_kingston
--sizes 4,6,8,10,12,14 --instances 3 --shots 2000 --tag _stale`: the 102 E6 circuits rerun while the provider
served a four-day-old calibration. Estimate ρ 0.951 (fresh 0.945), selection regret 0.003; raw fidelity top-1
fell to 8%. The first kingston job failed with a control-instrument error (not charged) and was resubmitted.

## Limitations
- Labels are noisy simulations from calibration snapshots: no crosstalk,
  drift during a job, or leakage.
- Mean error rates per device, not per qubit.
- Mirror circuits are used for well-defined labels; application circuits with
  non-unique outputs need other metrics.
- Prices are illustrative list prices.

### E6: validation on real IBM hardware (28 Oct 2026)
The 28 mirror circuits of E2 were run on **ibm_fez** and **ibm_marrakesh** (Heron r2), 4000 shots
each, using 65 s of the free Open Plan quota (job IDs in `results/e6_jobs.json`; the
`ibm_kingston` job was still queued). Results are in `results/e6_summary.json` and `e6_hardware.png`.

| Predictor vs MEASURED success | Spearman ρ | MAE |
|---|---|---|
| Noisy simulation (same calibration) | 0.90 | 0.19 |
| Raw device fidelity | 0.05 | — |
| **ESP, topology estimate (no transpilation)** | **0.88** | **0.12** |
| ESP, actual transpilation | 0.91 | 0.13 |

- On hardware, the transpilation-free estimate ranks circuits almost as well as the
  estimate from the actually transpiled circuit, and far better than raw fidelity.
- The simulation overestimates hardware success by about 0.19 on average, which is
  consistent with noise the model does not include (crosstalk, drift).
- **Negative result:** choosing between two near-identical devices of the same
  family is not resolved by device-mean ESP. It picked the better device in 39% of
  circuits, with mean regret 0.048, against 0.034 for always picking fez.


#### E6 with the third device (collected 3 Nov 2026)
The `ibm_kingston` job finished. With 3 devices (84 executions, ~98 s of QPU in total):

| Predictor vs MEASURED success | Spearman ρ | MAE | Mean regret | Best device picked |
|---|---|---|---|---|
| Noisy simulation | 0.93 | 0.18 | 0.015 | 68% |
| Raw device fidelity | 0.16 | — | 0.010 | 89% |
| **ESP, topology estimate (no transpilation)** | **0.91** | **0.11** | 0.010 | 89% |
| ESP, actual transpilation | 0.93 | 0.12 | 0.010 | 89% |

`ibm_kingston` was clearly the best device and every device-level predictor picks it.
The fez-vs-marrakesh negative result above still holds. The simulated column keeps the
29 Oct values for fez/marrakesh (calibration close to execution); kingston is simulated
with the 3 Nov calibration. The 2-device files are in `results/e6_2devices_backup/`.

#### E6 large run (3 Nov 2026): 102 circuits × 3 devices, 2000 shots
`e6_hardware_validation.py prepare|submit|collect --backends ibm_fez,ibm_marrakesh,ibm_kingston
--sizes 4,6,8,10,12,14 --instances 3 --shots 2000 --tag _large`. 306 executions, 177 s of QPU
(estimate 178 s). Job IDs in `results/e6_jobs_large.json`; results in `e6_summary_large.json`,
`e6_hardware_large.csv`, `e6_hardware_large.png`.

| Predictor vs MEASURED success | Spearman ρ | MAE | Mean regret | Best device picked |
|---|---|---|---|---|
| Noisy simulation | 0.97 | 0.17 | 0.014 | 76% |
| Raw device fidelity | 0.17 | — | 0.008 | 86% |
| **ESP, topology estimate (no transpilation)** | **0.95** | **0.10** | 0.008 | 86% |
| ESP, actual transpilation | 0.95 | 0.10 | 0.008 | 86% |

- With 3.6× more circuits the hardware result is stronger than the 28-circuit run
  (ρ 0.91 → 0.95) and the estimate now matches the transpiled-circuit estimate on both
  ranking and MAE.
- `ibm_kingston` was again clearly the best device (mean success 0.45 vs 0.38 fez, 0.31
  marrakesh); every device-level predictor picks it, regret 0.008 vs 0.008 for always
  choosing kingston.
- **Negative result, stronger than before:** between fez and marrakesh alone, the ESP
  estimates pick the better device for 44% of circuits (regret 0.045), while raw device
  fidelity picks it for 75% (regret 0.015). Device-mean ESP does not separate devices of
  the same family.
- 30% of executions have measured success below 0.05 (deep 12–14-qubit circuits), so part
  of the ranking comes from separating feasible from infeasible circuits.
- The simulated column uses the calibration at collection time, about 2 h after execution.
