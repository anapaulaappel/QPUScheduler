"""Figure for E4: real IBM 2Q-error series (quality = 1 - mean 2Q error), with tau and the
calibration-publication freeze. Data: data/real_history (until the freeze is shaded)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS, ROOT

FREEZE = pd.Timestamp("2026-11-13T17:22:56+00:00")
fig, axes = plt.subplots(3, 1, figsize=(7, 5.2), sharex=True)
for ax, b in zip(axes, ["ibm_fez", "ibm_marrakesh", "ibm_kingston"]):
    df = pd.read_csv(ROOT / "data" / "real_history" / f"{b}.csv", parse_dates=["timestamp"])
    q = 1 - df.mean_tq_error
    tau = q[df.timestamp <= FREEZE].quantile(0.5)
    ax.step(df.timestamp, q, where="post", color="#1f4e79", lw=1.2)
    ax.axhline(tau, color="#888", ls="--", lw=0.8)
    ax.axvspan(FREEZE, df.timestamp.max(), color="#ddd", lw=0)
    ax.set_ylabel(b.replace("ibm_", ""), fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].set_title("1 − mean two-qubit error (15-min polls); dashed: median τ; grey: no new calibrations published", fontsize=8)
fig.autofmt_xdate()
fig.tight_layout()
out = RESULTS / "e4_real_history.png"
fig.savefig(out, dpi=200)
print(out)
