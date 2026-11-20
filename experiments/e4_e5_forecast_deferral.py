"""
E4 — Are quality windows reliable?   E5 — Does deferral help under deadlines?

Data: real IBM calibration history collected by probe/collect_calibration.py
(data/real_history) when it spans at least --min-days; otherwise a simulated
history (calibration-cycle model) is generated, and the results are labeled
as simulated.

E4 (rolling origin): at each origin, forecast the next H hours from the data
up to the origin; compare predicted quality windows (lower bound ≥ τ) with
the realized fidelity. Metrics: window precision (share of predicted-window
time where the realized fidelity ≥ τ), recall (share of realized-good time
covered by predicted windows). Models: physics (lower bound), physics
(mean prediction, ablation), prophet (lower bound), persistence baseline.

E5 (trace-driven): jobs arrive at random times with a deadline. If the QPU
does not meet τ at arrival, policies are:
  run_now       run anyway (degraded)
  give_up       unschedulable / simulator
  defer         OURS: wait for the earliest forecast window within the deadline
                (re-check live at window start; run degraded at the deadline)
  oracle_wait   wait for the first realized-good time within the deadline

Run:  python e4_e5_forecast_deferral.py [--source auto|real|sim] [--min-days 3]
Out:  results/e4_forecast.csv, results/e5_deferral.csv, results/e4_e5_summary.json

NOTE: on simulated history the forecaster's physics model matches the data
generator, so E4/E5 on simulated data are a sanity check, not evidence.
"""

import argparse
import json
import tempfile
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, ROOT

warnings.filterwarnings("ignore")


def load_history(source: str, min_days: float):
    from history_store import HistoryStore
    real_dir = ROOT / "data" / "real_history"
    if source in ("auto", "real") and real_dir.exists() and any(real_dir.glob("*.csv")):
        store = HistoryStore.from_csv_dir(real_dir)
        spans = [(r[-1].timestamp - r[0].timestamp).total_seconds() / 86400
                 for r in (store.get(b) for b in store.all_backends()) if len(r) > 1]
        if source == "real" or (spans and max(spans) >= min_days):
            return store, "real"
    # Simulated: 4 days at 5-minute resolution
    import simulate_history
    tmp = Path(tempfile.mkdtemp())
    simulate_history.simulate(hours=96, output_dir=tmp, step_seconds=300)
    import history_store
    history_store.MAX_HISTORY_PER_BACKEND = 10**6
    return HistoryStore.from_csv_dir(tmp), "simulated"


def _series(records):
    return pd.Series([r.mean_qubit_fidelity for r in records],
                     index=pd.DatetimeIndex([r.timestamp for r in records])).sort_index()


def realized_at(series: pd.Series, t):
    i = series.index.searchsorted(t)
    i = min(max(i, 0), len(series) - 1)
    return float(series.iloc[i])


MAX_GAP = timedelta(minutes=30)   # a grid point counts only if a real poll is this close


def has_data(series: pd.Series, t) -> bool:
    i = series.index.searchsorted(t)
    near = [series.index[j] for j in (i - 1, i) if 0 <= j < len(series)]
    return any(abs(x - t) <= MAX_GAP for x in near)


def e4(store, tau_quantile=0.5, horizon_h=6, origin_step_h=2, models=("physics", "prophet")):
    from forecaster import QualityForecaster
    from history_store import HistoryStore
    rows = []
    for backend in store.all_backends():
        recs = store.get(backend)
        if len(recs) < 50 or backend == "aer_simulator":
            continue
        s = _series(recs)
        tau = float(s.quantile(tau_quantile))   # threshold inside the observed range
        t0, t_end = s.index[0] + timedelta(hours=24), s.index[-1] - timedelta(hours=horizon_h)
        origin = t0
        while origin <= t_end:
            past = HistoryStore()
            for r in recs:
                if r.timestamp <= origin:
                    past.add(r)
            grid = [origin + timedelta(minutes=15 * k) for k in range(horizon_h * 4 + 1)]
            if not has_data(s, origin):
                origin += timedelta(hours=origin_step_h)
                continue
            valid = np.array([has_data(s, t) for t in grid])
            truth = np.array([realized_at(s, t) for t in grid])
            good = (truth >= tau) & valid
            preds = {}
            for m in models:
                try:
                    fc = QualityForecaster(past, model=m).forecast(
                        backend, min_fidelity=tau, horizon_hours=horizon_h, now=origin)
                except Exception:
                    continue
                n = min(len(grid), len(fc.confidence_lower))
                preds[f"{m}_lower"] = np.array(fc.confidence_lower[:n]) >= tau
                if m == "physics":
                    preds["physics_mean"] = np.array(fc.predicted_fidelity[:n]) >= tau
            preds["persistence"] = np.full(len(grid), realized_at(s, origin) >= tau)
            for name, p in preds.items():
                p = p & valid[:len(p)]          # score only grid points with real data
                g = good[:len(p)]
                tp = int((p & g).sum())
                rows.append(dict(backend=backend, origin=origin, model=name, tau=tau,
                                 predicted_good=int(p.sum()), realized_good=int(g.sum()), tp=tp))
            origin += timedelta(hours=origin_step_h)
    df = pd.DataFrame(rows)
    agg = df.groupby("model")[["predicted_good", "realized_good", "tp"]].sum()
    agg["precision"] = agg.tp / agg.predicted_good.clip(lower=1)
    agg["recall"] = agg.tp / agg.realized_good.clip(lower=1)
    return df, agg


def e5(store, n_jobs=200, deadline_min=180, tau_quantile=0.6, seed=0):
    from forecaster import QualityForecaster
    from history_store import HistoryStore
    rng = np.random.default_rng(seed)
    rows = []
    backends = [b for b in store.all_backends() if b != "aer_simulator" and len(store.get(b)) > 50]
    for _ in range(n_jobs):
        backend = backends[rng.integers(len(backends))]
        recs = store.get(backend)
        s = _series(recs)
        tau = float(s.quantile(tau_quantile))
        lo, hi = s.index[0] + timedelta(hours=24), s.index[-1] - timedelta(minutes=deadline_min)
        if hi <= lo:
            continue
        arrival = lo + (hi - lo) * float(rng.uniform())
        f_now = realized_at(s, arrival)
        if f_now >= tau:
            continue                  # deferral only matters when "now" fails
        grid = pd.date_range(arrival, arrival + timedelta(minutes=deadline_min), freq="5min")
        if not all(has_data(s, t) for t in grid[::3]):
            continue                  # skip jobs whose deadline window overlaps a collection gap
        truth = np.array([realized_at(s, t) for t in grid])
        # ours: forecast from data up to arrival, anchored on the live calibration age
        past = HistoryStore()
        for r in recs:
            if r.timestamp <= arrival:
                past.add(r)
        live = past.get(backend)[-1]
        elapsed = (arrival - live.timestamp).total_seconds() / 60
        fc = QualityForecaster(past).forecast(
            backend, min_fidelity=tau, horizon_hours=deadline_min / 60, now=arrival,
            current_calibration_age_minutes=live.calibration_age_minutes + elapsed)
        win = next((w for w in fc.quality_windows if w.minutes_from_now <= deadline_min), None)
        if win is not None:
            t_run = arrival + timedelta(minutes=win.minutes_from_now)
            f_defer = realized_at(s, t_run)
            if f_defer < tau:           # live re-check failed: wait until deadline
                f_defer, wait_defer = truth[-1], deadline_min
            else:
                wait_defer = win.minutes_from_now
        else:
            f_defer, wait_defer = f_now, 0.0          # no window: run now (degraded)
        ok_idx = np.flatnonzero(truth >= tau)
        f_oracle = truth[ok_idx[0]] if len(ok_idx) else f_now
        wait_oracle = ok_idx[0] * 5 if len(ok_idx) else 0
        rows.append(dict(backend=backend, tau=tau,
                         run_now_ok=f_now >= tau, give_up_ok=False,
                         defer_ok=f_defer >= tau, defer_wait_min=wait_defer,
                         oracle_ok=f_oracle >= tau, oracle_wait_min=wait_oracle))
    df = pd.DataFrame(rows)
    summary = {"jobs_needing_deferral": int(len(df))}
    for p in ("run_now", "give_up", "defer", "oracle"):
        summary[f"{p}_sla_met_rate"] = float(df[f"{p}_ok"].mean()) if len(df) else None
    if len(df):
        d = df[df.defer_ok & (df.defer_wait_min > 0)]
        o = df[df.oracle_ok & (df.oracle_wait_min > 0)]
        summary["defer_median_wait_min_when_deferred"] = float(d.defer_wait_min.median()) if len(d) else None
        summary["oracle_median_wait_min_when_waited"] = float(o.oracle_wait_min.median()) if len(o) else None
    return df, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", choices=["auto", "real", "sim"])
    ap.add_argument("--min-days", type=float, default=3.0)
    ap.add_argument("--until", default=None,
                    help="ISO timestamp; drop records after it (e.g. when the provider "
                         "stops publishing new calibrations)")
    ap.add_argument("--metric", default="qubit_fidelity", choices=["qubit_fidelity", "tq"],
                    help="quality series: reported mean qubit fidelity, or 1 - mean 2Q error")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    store, source = load_history(args.source, args.min_days)
    if args.until:
        from history_store import HistoryStore
        cutoff = pd.Timestamp(args.until)
        cut = HistoryStore()
        for b in store.all_backends():
            for r in store.get(b):
                if pd.Timestamp(r.timestamp) <= cutoff:
                    cut.add(r)
        store, source = cut, f"{source} (until {args.until})"
    if args.metric == "tq":
        # Quality := 1 - mean two-qubit error. On IBM Heron the reported mean qubit
        # fidelity takes only 3-6 distinct values over weeks, while the 2Q error varies
        # (CV 5-12%) and dominates the estimated success probability.
        from dataclasses import replace
        from history_store import HistoryStore
        q = HistoryStore()
        for b in store.all_backends():
            for r in store.get(b):
                q.add(replace(r, mean_qubit_fidelity=1.0 - r.mean_tq_error))
        store, source = q, f"{source}; metric 1-mean_tq_error"
    print(f"History source: {source}; backends: {store.all_backends()}")
    e4_df, e4_agg = e4(store)
    e5_df, e5_sum = e5(store)
    e4_df.to_csv(RESULTS / f"e4_forecast{args.tag}.csv", index=False)
    e5_df.to_csv(RESULTS / f"e5_deferral{args.tag}.csv", index=False)
    summary = {"source": source, "e4": e4_agg.round(4).to_dict(orient="index"), "e5": e5_sum}
    (RESULTS / f"e4_e5_summary{args.tag}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
