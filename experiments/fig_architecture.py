"""
Paper figure: two-stage, transpilation-aware scheduling pipeline.

Run:  python fig_architecture.py
Out:  results/fig_architecture.pdf, results/fig_architecture.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import RESULTS

FAST, SLOW, DATA, FAIL, NEUTRAL = "#dbeafe", "#fde68a", "#e5e7eb", "#fecaca", "#ffffff"


def box(ax, x, y, w, h, text, color, bold=False, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#374151", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)
    return (x, y, w, h)


def arrow(ax, p, q, text="", color="#374151", rad=0.0, ls="-", fs=7.5, offset=(0, 0.08)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.1,
                                 color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1], text,
                ha="center", va="bottom", fontsize=fs, color=color)


def main():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    # Inputs / state
    box(ax, 0.2, 4.4, 2.2, 0.8, "QPU providers\n(calibration APIs, read-only)", DATA)
    box(ax, 0.2, 3.0, 2.2, 0.9, "Telemetry agent\n+ normalization\n(errors, T1/T2, coupling,\nprice, durations)", DATA, fs=7.5)
    box(ax, 3.0, 3.0, 2.3, 0.9, "QPU resource records\n(platform API)\n+ quality forecast", DATA, fs=7.8)
    box(ax, 0.2, 1.3, 2.2, 1.0, "Job\ncircuit + min fidelity,\ncost, deadline", NEUTRAL, bold=False)

    # Fast path
    box(ax, 3.0, 1.3, 2.3, 1.0, "FILTER (all candidates)\ntopology-divergence estimate\n→ predicted fidelity", FAST, fs=7.8)
    box(ax, 5.8, 1.3, 1.6, 1.0, "SCORE\n+ RESERVE\nbest QPU", FAST, fs=7.8)

    # Slow path
    box(ax, 7.9, 3.0, 2.8, 0.9, "RETRANSPILE\nreserved QPU only\n(Qiskit / tket)", SLOW, fs=7.8)
    box(ax, 7.9, 1.3, 2.8, 1.0, "RE-VALIDATE\nactual gate counts →\nfidelity, cost; QUOTA\n(USD / QPU-seconds)", SLOW, fs=7.5)
    box(ax, 7.9, 0.1, 2.8, 0.7, "BIND + audit record", "#bbf7d0", bold=True)

    # Fallback
    box(ax, 3.0, 0.1, 4.4, 0.7, "No candidate: defer to forecast window (live re-check)\n| simulator | unschedulable (reason)", FAIL, fs=7.5)

    # Arrows
    arrow(ax, (1.3, 4.4), (1.3, 3.9))
    arrow(ax, (2.4, 3.45), (3.0, 3.45))
    arrow(ax, (4.15, 3.0), (4.15, 2.3), "telemetry", offset=(0.42, 0.05))
    arrow(ax, (2.4, 1.8), (3.0, 1.8))
    arrow(ax, (5.3, 1.8), (5.8, 1.8))
    arrow(ax, (6.6, 2.3), (7.9, 3.3), rad=-0.2)
    arrow(ax, (9.3, 3.0), (9.3, 2.3))
    arrow(ax, (9.3, 1.3), (9.3, 0.8), "pass", offset=(0.25, -0.12))
    # Re-queue loop (fail), routed below the fast-path boxes
    red = "#b91c1c"
    ax.plot([7.9, 7.65, 7.65, 4.75], [1.45, 1.45, 1.05, 1.05], color=red, lw=1.1)
    arrow(ax, (4.75, 1.05), (4.75, 1.3), color=red)
    ax.text(6.2, 1.08, "fail: unreserve, exclude candidate, re-queue",
            fontsize=7.5, color=red, ha="center", va="bottom")
    # No candidate left → fallback
    arrow(ax, (3.55, 1.3), (3.55, 0.8), color=red)
    ax.text(3.62, 0.98, "none left", fontsize=7.5, color=red, ha="left")

    # Legend / path labels
    ax.text(3.05, 2.42, "fast path: milliseconds per candidate", fontsize=8, color="#1d4ed8",
            style="italic", ha="left")
    ax.text(7.95, 4.05, "slow path: seconds, one candidate", fontsize=8, color="#b45309",
            style="italic", ha="left")

    plt.tight_layout()
    plt.savefig(RESULTS / "fig_architecture.pdf")
    plt.savefig(RESULTS / "fig_architecture.png", dpi=220)


if __name__ == "__main__":
    main()
