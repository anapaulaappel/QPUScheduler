# Transpilation-aware QPU placement: experiments

Code, data and results for the paper on letting a cluster scheduler choose the QPU for a quantum job:
**[Transpilation-Aware Multi-Vendor QPU Scheduling with Predicted Post-Transpilation Fidelity](https://arxiv.org/abs/2611.27927)**
(arXiv:2611.27927).
A training-free estimate predicts, without compiling, the success probability of a circuit on each
candidate QPU and filters out the devices that would miss the job's requirement; only the chosen device
then compiles the circuit and re-checks it before the job is bound (or rolled back).

This repository is organised so that every number, table and figure in the paper can be regenerated.
Most experiments run **offline** from the data already in `experiments/results/`; the hardware runs are
kept as recorded (job IDs and measured counts) and can be re-analysed without QPU access.

## Setup

Python 3.11 or later (the paper used 3.13; Qiskit 2.4.1, Qiskit Aer 0.17.2, qiskit-ibm-runtime 0.51.0,
scikit-learn 1.9.0, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1).

```bash
git clone https://github.com/anapaulaappel/QPUScheduler
cd QPUScheduler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # 64 tests, about 10 s
cd experiments
```

Hardware runs on other vendors use their own environments, because their SDKs pin different Qiskit
versions: `requirements-iqm.txt` (IQM Resonance, venv `.venv-iqm`) and `requirements-oq.txt` (Open
Quantum: Rigetti and AQT, venv `.venv-oq`).

## Reproducing the paper

Run from `experiments/`. **Offline** = no account needed. **Account (no QPU)** = reads public
calibration through the provider API. **QPU** = submits jobs and spends QPU time or credits; the
recorded results are already in the repository, so these steps are only needed to collect new data.

| Paper | What | Command | Needs |
|---|---|---|---|
| Fig. 1 | Two-stage pipeline diagram | `python fig_architecture.py` | Offline |
| E1, Fig. 2, Table 2 | Topology estimate vs. actual routing overhead | `python e1_divergence.py` | Offline |
| E1b, Fig. 3 | Latency: estimate vs. transpilation | `python e1b_latency.py` | Offline |
| E2, Fig. 4, Table 3 | Success prediction without transpiling (102 mirror circuits, 8 simulated devices) | `python e2_fidelity_prediction.py --sizes 4,6,8,10,12,14 --instances 3 --tag _large` | Offline |
| E3, Fig. 5 | Placement policies: SLA violations, unschedulable jobs, latency | `python e3_routing_policies.py --tag _large --instances 3` | Offline (uses E2) |
| E11, Fig. 6 | Contention among jobs (discrete-event simulation) | `python x1_contention.py` | Offline |
| E11, Table 4 | Cost and quota under the E3 policies | `python x4_cost_quota.py` | Offline |
| E6, Fig. 7, Table 5 | IBM hardware: 102 circuits × 3 Heron devices, 2000 shots | analysis: `python e6_hardware_validation.py collect --backends ibm_fez,ibm_marrakesh,ibm_kingston --sizes 4,6,8,10,12,14 --instances 3 --shots 2000 --tag _large` | QPU (IBM, ~3 min) for `prepare`/`submit` |
| E6, layout diagnostic | ESP with the per-qubit calibration of the run date | `python x8_layout_esp.py` | Account (no QPU) + output of E6 `prepare` |
| Stale-calibration rerun | E6 with four-day-old calibration | same as E6 with `--tag _stale` | QPU (IBM) |
| E7, Fig. 8, Table 6 | Other vendors: IQM Garnet, Rigetti Cepheus, AQT IBEX Q1 | analysis: `python e7_analyze.py` then `python e7_learned_rates.py` | Offline; runs: `e7_prepare.py`, `e7_run_iqm.py` (`.venv-iqm`), `e7_verbatim_oq.py` / `e7_verbatim_aqt.py` (`.venv-oq`) |
| E12, Table 7 | Controlled same-day cross-vendor run | analysis: `python x7_analyze.py` | Offline; runs: `x7_prepare.py`, E6 with `--tag _x7`, `QSCHED_RUN_DIR=x7 e7_run_iqm.py`, `QSCHED_RUN_DIR=x7 e10_run_oq.py` |
| E8, Table 8 | Training-free estimate vs. learned predictor | `python e8_learned_baseline.py --boot 500 --seed 0` | Offline |
| E9, Fig. 9, Table 9 | Policies replayed on measured outcomes, conformal margins | `python e9_hardware_policies.py --dataset ibm` (also `crossvendor`, `stale`) | Offline |
| E9, Fig. 10 | Operating curve of the safety margin | `python x3_margin_tradeoff.py` | Offline |
| E4/E5, Fig. 11 | Calibration forecasting and deferral on real IBM history | `python e4_e5_forecast_deferral.py --source real --until 2026-11-13T17:30:00+00:00 --metric tq --tag _real_tq`; figure: `python fig_e4_real_history.py` | Offline (history in `data/real_history/`) |
| E10, Table 10 | Application circuits (Hellinger fidelity) on five devices | analysis: `python e10_analyze.py` then `python x5_e10_por_familia.py` | Offline; runs: `e10_prepare.py`, `e10_run_ibm.py`, `e10_run_oq.py` (`.venv-oq`) |
| Appendix A | 95% bootstrap intervals for the main claims | `python x2_bootstrap.py` | Offline |

Each script states its inputs and outputs in its header. Results are written to
`experiments/results/` (CSV/JSON plus the figure as PNG and PDF). The full log of every run, with
dates, shot counts, QPU time and intermediate findings, is in [`experiments/README.md`](experiments/README.md);
provider notes are in [`experiments/PROVIDERS.md`](experiments/PROVIDERS.md).

**Notes on the hardware data.**
- Measured counts and job IDs are in `experiments/results/` (`e6_jobs_*.json`, `e7/run_*.json`,
  `e10/run_*.json`, `x7/`, `x7b/`).
- The error rates used for IQM, Rigetti and AQT are in `results/e7/device_*.json` and
  `results/x7/device_*.json`. IBM calibration is read through the IBM Quantum API; historical
  calibration for a given date is available with `backend.properties(datetime=...)`.
- The transpiled IBM circuits (`results/e6_prepared_*.pkl`, 28 MB each) are not in the repository;
  `e6_hardware_validation.py prepare` regenerates them with the calibration current at run time.
- Credentials are never stored in the repository. IBM uses the account saved with
  `QiskitRuntimeService.save_account`; IQM reads its token from the macOS Keychain (service
  `qsched-iqm`); Open Quantum reads the key file given by `OPENQUANTUM_SDK_KEY` (see the script headers).

## The scheduler prototype

The estimate, the placement policies, the two-stage binding with rollback and the usage/quota model are
implemented as a Python prototype under `scheduler/`, `probe/` and `forecaster/`, used by the
experiments above. It is not a Kubernetes scheduler plugin or a Slurm integration. See
[`docs/prototype.md`](docs/prototype.md) for its components and the demo notebook.

## Citation

```bibtex
@misc{appel2026qsched,
  title         = {Transpilation-Aware Multi-Vendor {QPU} Scheduling with Predicted Post-Transpilation Fidelity},
  author        = {Appel, Ana Paula},
  year          = {2026},
  eprint        = {2611.27927},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = {https://arxiv.org/abs/2611.27927}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Contact

Ana Paula Appel — aappel@redhat.com
