# Experiments for the major revision (20 Nov 2026)

All offline (no QPU). Seeds fixed in each script. Qiskit 2.4.1, same targets as E2.

- **X3 margin trade-off** (`x3_margin_tradeoff.py`): additive margin on ESP(estimate), totals over
  tau = 0.3..0.8. IBM measured: margin 0 -> 16 violations / 52 false rejections (matches E9); -0.1 -> 37/13;
  +0.1 -> 6/104; +0.3 -> 2/201. Raw fidelity: 341 violations. Conformal delta 0.05..0.5 lies on the same
  curve. Simulation: margin 0 -> 2/72. Figure `fig_margin_tradeoff.{png,pdf}`.
- **X5-A E10 by output type** (`x5_e10_por_familia.py`): IBM concentrated outputs (BV, GHZ, adder, Grover)
  rho 0.71 (transpiled 0.74, n=36); broad outputs (QAOA, VQE) 0.55 (n=21); all vendors broad 0.07.
- **X4 cost/quota** (`x4_cost_quota.py`, illustrative list prices): raw fidelity always picks the most
  expensive device (USD 12.2k for 102 jobs) with 26-79 violations; estimate filter 0-1 violations and
  3-10 jobs meeting tau per 100 USD (raw fidelity 0.2-0.6); with a quota of 25% of raw-fidelity spend,
  raw fidelity has 74 jobs blocked, the estimate filter none (it runs fewer jobs, rejecting infeasible ones).
- **X1 contention** (`x1_contention.py`, 400 Poisson jobs, 8 devices, tau = 0.7): best-average-fidelity
  sends every job to one device (mean wait ~2.9-3.0 ks, 40% violations); shortest queue waits ~0 but violates
  64-67%; estimate filter + earliest completion: 0% violations, mean wait 0.25-8 s, decision 4 ms;
  transpile-all + earliest completion: 4.8% violations, 3% false rejections, wait 54-92 s, decision 125 ms.
  Cost: estimate filter rejects 18.5% feasible jobs in simulation (known ESP conservatism; see X3).
- **X2 bootstrap** (`x2_bootstrap.py`): paired cluster bootstrap over circuits, B=2000 — see x2_bootstrap.csv.
- **X8 layout-consistent ESP** (`x8_layout_esp.py`; IBM historical calibration of 3 Nov 2026 via
  properties(datetime), read-only): ESP from each transpiled gate on its physical qubits + readout of the
  measured qubits. rho vs measured 0.967 (E6 ESP 0.945/0.952). fez vs marrakesh: top-1 77% / regret 0.013
  (E6 ESP 48% / 0.046). Three devices: 84% / 0.009 (E6 88% / 0.008). A plain mean including dead elements
  (error = 1, excluded by the E6 telemetry) also separates fez/marrakesh (76%) but fails on three devices
  (10%). Layout ESP needs the transpiled circuit: diagnostic upper bound, not a cheap-stage predictor.
- **X7 controlled cross-vendor** (`x7_prepare.py`, `x7_analyze.py`; 20 Nov 2026, same day, 1000 shots on
  every device, E7 seeds): IBM fez/marrakesh/kingston (28 circuits), IQM Garnet (28 = 12 in x7 + 16 in x7b,
  IQM Resonance), Rigetti Cepheus (12, verbatim, Open Quantum; first ghz_4 job lost to a client connection
  reset and charged, all 12 rerun: 13 credits). AQT not run (12 x 1000 shots quoted at 180 credits).
  Within-device rho (estimate / transpiled): fez 0.82/0.87, marrakesh 0.83/0.88, kingston 0.91/0.95,
  IQM 0.90/0.96, Rigetti 0.57/0.59 (MAE 0.32; E7 was 0.90). Selection IBM x3 + IQM (28 circuits):
  regret 0.005 [0.000, 0.012], top-1 82% (transpiled 0.025, 71%); with Rigetti (12 circuits): 0.007
  [0.000, 0.020], 83% (transpiled 0.032, 67%).
