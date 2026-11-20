# E7 run notes (3 Nov 2026)

- **IQM Garnet** (IQM Resonance, 28 circuits × 500 shots, one batch job
  `01a1a2a6-0019-7366-bfc1-1bb411364412`): valid. Error rates from IQM calibration metrics.
- **Rigetti Cepheus-1-108Q** (Open Quantum, 12 circuits × 1000 shots, 12 credits): **invalid
  for this benchmark.** The provider's compiler optimized across the mirror barrier: deep circuits
  that simplify (real_amplitudes_8, 256 CZ: 0.74 success; qaoa_ring_8: 0.72) beat every other
  device, while QFT (hard to simplify) behaves as expected (qft_8: 0.004). Raw results kept in
  `run_rigetti_cepheus.json`; excluded in `e7_analyze.py`. A rerun needs a verbatim-box
  submission (native gates, physical qubits; Open Quantum lists `verbatim_boxes` as a feature)
  or circuits that cannot be simplified (e.g. randomized compiling of C and C†).
- **AQT IBEX Q1** (Open Quantum): only `ghz_6` was submitted (2 credits), job
  `88573c2a-b8bc-455e-94a8-ef788706b224`, queued (AQT seems to run in European business hours).
  The runner was stopped before the other 5 circuits to avoid spending 10 credits on a run that
  may be simplified in the same way. Fetch the ghz_6 result later with the scheduler API.
- Open Quantum credits left: 11 of 25 (renew every 90 days).
- `e7_run_oq.py` now has a per-device lock, a basis-translation fallback (the Open Quantum
  plugin registers gate equivalences that break direct translation), and per-circuit error
  handling.

## Update (3 Nov 2026, evening): Rigetti verbatim
- Verbatim submission (`e7_verbatim_oq.py`): native {rz, rx(±π/2), rx(π), cz} on physical qubits
  inside `#pragma braket verbatim`; Rigetti requires rx angles with 16-digit precision. Rejected
  submissions are not charged.
- Check: real_amplitudes_8 (256 CZ) went from 0.74 (non-verbatim) to 0.005 (verbatim), in line with
  IQM Garnet (0.008). So the non-verbatim run was simplified by the provider compiler.
- 11 of 12 circuits done (bernstein_vazirani_8 not run: credits exhausted). Bit order of the raw
  provider bitstrings chosen from the data (as-is: 1.17 vs reversed 0.22 summed success).
- Rigetti error rates: published medians (1Q 99.9%, CZ 99.1%), readout assumed 3% (no job
  calibration available). Measured success is well below ESP (MAE 0.30): the device performs worse
  than the published medians suggest, so absolute predictions for Rigetti are poor; ranking is fine.
- Open Quantum credits: 0 left (renew in 90 days).

## Update (4 Nov 2026): second Open Quantum account, AQT verbatim
- Second account (separate organization, 25 Spark credits): key at
  `~/.config/qsched/openquantum-sdk-key-2.json`. The scripts use it with
  `OPENQUANTUM_SDK_KEY=~/.config/qsched/openquantum-sdk-key-2.json`.
- The non-verbatim AQT ghz_6 job from 3 Nov completed with 100/100 on the ideal output. Kept in
  `run_aqt_ghz6_nonverbatim.json`, excluded from the analysis (likely simplified by the provider
  compiler, as on Rigetti; not proven, a clean 100/100 on 10 two-qubit gates is the only evidence).
- AQT verbatim (`e7_verbatim_aqt.py`), 100 shots, 2 credits per circuit, 6 of 6 done:
  ghz_6 0.81, bernstein_vazirani_6 0.83, random_6 0.28, qaoa_ring_6 0.17, real_amplitudes_6 0.24,
  qft_6 0.06. With 100 shots the standard error is up to 0.05. Bit order as is (2.33 vs 0.03).
- First verbatim attempt (job `a4777a81-9f3d-4fe9-8ab0-025806c52cf6`) failed on the device with
  "Execution failed after 3 attempts", not charged. The file had prx with phi < 0. After mapping phi
  to [0, 2π) the same circuit ran. This fits a phi-range rule but is not confirmed: the error
  gives no reason and no documentation of the range was found. A failure unrelated to phi cannot be
  excluded. phi values that rounded to 2π are written as 0.
- Local statevector simulation of all 6 verbatim files gives the expected key with p = 1.
- AQT error rates are published figures with readout assumed 0.5%. Measured success falls below ESP
  on the four deeper circuits (0.06 to 0.28 vs 0.36 to 0.63, MAE 0.22), so as on Rigetti the
  ranking holds (Spearman 0.94, n = 6) and absolute predictions do not.
- In `e7_summary.json`, `always_device_regret` for AQT covers only the 6 circuits it ran, and
  Rigetti only 11 or 12, so those values are not directly comparable with the 28-circuit devices.
- Job IDs: ghz_6 `74561892-68c0-4ca9-8232-c79d7ddfd92d`, bernstein_vazirani_6
  `cd5b24a2-17d0-4339-9d42-12d97bda2c34`, random_6 `c38ffa5f-57be-476c-8af9-bb25bf6603af`,
  qaoa_ring_6 `630faed4-cadd-4634-a6a7-f4d5891313a7`, real_amplitudes_6
  `ad5e99b8-b27d-41f3-b1f0-1185a01e9cd4`, qft_6 `a7fc4b95-2f72-4288-96d0-78b98b10f598`.
- Rigetti bernstein_vazirani_8 (account 2): first job `51de66bc-6070-410e-9121-7c3c6db48de1` failed
  with `Invalid qubit(s): $8` (the transpile had used a qubit without couplers; fixed in
  `e7_verbatim_oq.py`, job not charged). Second job `b00fef8c-9f5c-44a1-8729-d9e3c9b858d8` completed:
  1000 shots, measured 0.422 vs ESP 0.655, 8 CZ, 0 estimated SWAPs. Rigetti is now 12 of 12. Balance
  on account 2: 12 credits. Results: Rigetti n = 12, Spearman 0.90, MAE 0.30; all devices n = 130,
  Spearman 0.85; selection unchanged (top-1 75%, regret 0.051), since Rigetti never has the highest
  ESP on any circuit.
- The SDK waits by polling and can drop the connection (seen twice); the job keeps running. Never
  resubmit after a ConnectionError: read the job ID from the log and wait for that job.

## Update (4 Nov 2026, afternoon): learned error rates (no QPU time)
- `e7_learned_rates.py` -> `e7_learned_rates.json`. One factor gamma per device, predicted success =
  ESP**gamma (scales all of the device's error rates), least squares on the linear scale, fitted on
  circuits other than the one predicted.
- All data (topology estimate): gamma 2.2 AQT, 3.4 Rigetti, 0.7-0.8 IBM, 1.1 IQM Garnet.
- Leave-one-circuit-out: regret 0.010, best device 89% over 6 devices (both predictors), equal to
  always ibm_kingston (the learned rates pick kingston for all 28). MAE AQT 0.21 -> 0.08,
  Rigetti 0.30 -> 0.07.
- k random probes per device (200 reps, seed 0): regret 0.059 / 0.055 / 0.048 / 0.040 for
  k = 1 / 2 / 3 / 5 (estimate); best device 46% to 62%. AQT is avoided already at k = 1; the
  remaining regret comes from choosing among the three IBM devices (gamma noisy: kingston k = 3,
  p10-p90 0.1-3.3).
- Same 28 circuits used to fit and test (leave-one-out), only 6 on AQT. No shrinkage or other model
  was tried, to avoid tuning on these data.

## Update (4 Nov 2026, night): Rigetti bernstein_vazirani_8 done, 12 of 12
- First job `51de66bc-6070-410e-9121-7c3c6db48de1` (09:24) Failed on the device: "[line 9] Invalid
  qubit(s): $8". A qubit with only single-qubit gates (the random X layer) had been placed on Cepheus
  qubit 8, which has no couplers. The device rejects it only after the job has waited in the queue.
- `e7_verbatim_oq.py` now transpiles on the connected qubits only when that happens (`to_native`).
  Resubmitted as job `b00fef8c-9f5c-44a1-8729-d9e3c9b858d8` (13:03, account 2), Completed:
  success 0.422 (1000 shots). Account 2 has 12 credits left.
- Rerun of `e7_analyze.py` and `e7_learned_rates.py`: Rigetti n = 12, rho 0.90 (estimate) / 0.94
  (actual), MAE 0.30 / 0.28, 2Q-count rho 0.993. Cross-vendor selection unchanged (6 devices: 0.051 /
  0.096). Learned gamma Rigetti 3.1; k = 5 probes regret 0.043 / 0.037.
