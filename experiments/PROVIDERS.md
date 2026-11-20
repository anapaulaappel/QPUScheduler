# Hardware providers: access setup (no secrets in this file)

| Provider | Hardware | Credential location | Python env |
|---|---|---|---|
| IBM Quantum (Open Plan, 10 min / 28 days) | Heron r2: fez, marrakesh, kingston | `QiskitRuntimeService` saved account | `.venv` |
| Open Quantum (Quantum Rings; $50 / 90 days = 25 Spark credits) | AQT IBEX Q1 (trapped ion), IQM Garnet, Rigetti Cepheus-1-108Q, IonQ Forte (often offline) | SDK key JSON at `~/.config/qsched/openquantum-sdk-key.json`, second account at `~/.config/qsched/openquantum-sdk-key-2.json` (both chmod 600); use `OPENQUANTUM_SDK_KEY` + `OpenQuantumClient.from_env()` | `.venv-oq` |
| IQM Resonance (Starter: up to 30 credits / month) | Garnet (20q), Emerald (54q); native `r`, `cz` | macOS Keychain, service `qsched-iqm` (`security find-generic-password -s qsched-iqm -w`); `IQMProvider("https://resonance.iqm.tech", quantum_computer=..., token=...)` | `.venv-iqm` |
| Quantum Inspire (QuTech; free, non-commercial, account approved manually) | Tuna-5, Tuna-9, Tuna-17 (native CZ) | `~/.quantuminspire/config.json` (chmod 600), written by `qi login` | `.venv-qi` |

Notes:
- Separate venvs because the SDKs pin different Qiskit versions (IQM 2.1, QI 2.3, Open Quantum 2.6; main env 2.4).
- Quantum Inspire: the python.org build of Python lacks CA certificates, so run QI commands with
  `SSL_CERT_FILE=$PWD/.venv-qi/lib/python3.13/site-packages/certifi/cacert.pem`.
- Quantum Inspire terms say non-commercial use; check before any commercial use of the results.
- Always estimate cost before submitting; never submit from scheduled tasks.
