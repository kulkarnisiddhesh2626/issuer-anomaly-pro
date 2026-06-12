"""Anomaly detection layer.

Design choices (defended in WRITEUP.md):

* **Primary signal — seasonal robust z-score.** Card-transaction metrics have
  strong hour-of-day and day-of-week seasonality. For each metric series we build
  a baseline from the median of matching ``(weekday, hour)`` buckets and measure
  how far each point deviates in MAD (median-absolute-deviation) units. MAD is
  robust to the very outliers we are hunting, so the baseline is not dragged
  around by the anomaly itself. The score is directly interpretable: "approval
  rate was 6.2 MADs below its usual Monday-09:00 level."

* **Low false positives by construction.** We require a minimum hourly volume per
  bucket (tiny samples are noisy), use a conservative threshold (~4 MADs), and
  merge contiguous flagged hours into a single *event* so a 6-hour outage is one
  alert, not six.

* **Secondary signal — Isolation Forest.** A multivariate cross-check over
  engineered features (approval/decline/fraud rates, log-volume, seasonal
  residuals) catches odd *combinations* the per-metric view might miss. It is
  used only to raise confidence, never as the sole reason to alert, because its
  score is not human-explainable on its own.

The detector emits structured ``AnomalyEvent`` records. These — not the raw
rows — are what the GenAI layer is grounded in.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config

METRICS = ("approval_rate", "decline_rate", "fraud_rate", "txn_count")
# Which direction is "bad" for each metric (the side we alert on).
BAD_DIRECTION = {
    "approval_rate": "down",
    "decline_rate": "up",
    "fraud_rate": "up",
    "txn_count": "both",
}


@dataclass
class AnomalyEvent:
    metric: str
    grain: str                 # e.g. "global" or "country=GB|channel=ecom"
    scope: dict                # parsed grain as a dict of dimension filters
    direction: str             # "up" or "down"
    start: str
    end: str
    duration_hours: int
    peak_z: float              # max |robust z| during the event
    observed: float            # representative observed metric value (worst hour)
    expected: float            # seasonal baseline at that hour
    observed_pct: float | None # observed as % where applicable
    expected_pct: float | None
    volume: int                # transactions involved during the event
    iso_forest_corroborated: bool
    severity: str              # low / medium / high
    top_contributors: dict = field(default_factory=dict)  # filled by context layer


# --------------------------------------------------------------------------- #
# Series construction
# --------------------------------------------------------------------------- #
def _hourly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a (already-filtered) frame to one row per hour with metrics."""
    g = df.groupby("timestamp", as_index=False).agg(
        txn_count=("txn_count", "sum"),
        approved_count=("approved_count", "sum"),
        declined_count=("declined_count", "sum"),
        fraud_count=("fraud_count", "sum"),
    )
    g["approval_rate"] = g.approved_count / g.txn_count
    g["decline_rate"] = g.declined_count / g.txn_count
    g["fraud_rate"] = g.fraud_count / g.txn_count
    g["weekday"] = g.timestamp.dt.weekday
    g["hour"] = g.timestamp.dt.hour
    return g


def _robust_z(series: pd.Series, weekday: pd.Series, hour: pd.Series) -> pd.Series:
    """Seasonal robust z-score: deviation from (weekday,hour) median in MAD units."""
    frame = pd.DataFrame({"v": series, "wd": weekday, "hr": hour})
    med = frame.groupby(["wd", "hr"])["v"].transform("median")
    abs_dev = (frame["v"] - med).abs()
    # MAD per seasonal bucket; scale 1.4826 makes it comparable to std for normals.
    mad = abs_dev.groupby([frame.wd, frame.hr]).transform("median") * 1.4826
    # Guard against zero MAD (flat buckets): fall back to global MAD.
    global_mad = (series - series.median()).abs().median() * 1.4826 or 1e-9
    mad = mad.replace(0, np.nan).fillna(global_mad).clip(lower=global_mad * 0.25)
    return (frame["v"] - med) / mad


# Effect-size floors keep statistically-significant-but-trivial blips quiet.
_RATE_NUMERATOR = {
    "approval_rate": "approved_count",
    "decline_rate": "declined_count",
    "fraud_rate": "fraud_count",
}
_MIN_EFFECT_PP = {          # minimum absolute change vs baseline (in rate points)
    "approval_rate": 0.03,
    "decline_rate": 0.03,
    "fraud_rate": 0.004,
}
_MIN_FRAUD_COUNT = 8        # absolute fraud floor so 1-2 frauds never alert


def _proportion_scores(hm: pd.DataFrame, num_col: str) -> tuple[pd.Series, pd.Series]:
    """Volume-aware z-score for a rate metric against its seasonal baseline.

    Baseline rate p0 is pooled over matching (weekday, hour) buckets. The
    standard error sqrt(p0(1-p0)/n) shrinks with volume, so a wild rate on a
    handful of transactions scores low while the same rate on heavy volume
    scores high. This is what makes the detector resistant to small-sample noise.
    """
    n = hm["txn_count"].clip(lower=1)
    num = hm[num_col]
    frame = pd.DataFrame({"num": num, "den": n, "wd": hm.weekday, "hr": hm.hour})
    pooled_num = frame.groupby(["wd", "hr"])["num"].transform("sum")
    pooled_den = frame.groupby(["wd", "hr"])["den"].transform("sum")
    p0 = (pooled_num / pooled_den).clip(1e-4, 1 - 1e-4)
    p = (num / n)
    se = np.sqrt(p0 * (1 - p0) / n)
    z = (p - p0) / se
    return z, p0


def _poisson_scores(hm: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Poisson-style z-score for transaction volume vs seasonal expectation."""
    frame = pd.DataFrame({"v": hm["txn_count"], "wd": hm.weekday, "hr": hm.hour})
    mu = frame.groupby(["wd", "hr"])["v"].transform("median").clip(lower=1)
    z = (hm["txn_count"] - mu) / np.sqrt(mu)
    return z, mu


def _flag_series(hm: pd.DataFrame, metric: str, grain: str, scope: dict,
                 z_threshold: float, min_volume: int) -> list[AnomalyEvent]:
    if metric == "txn_count":
        z, expected_series = _poisson_scores(hm)
    else:
        z, expected_series = _proportion_scores(hm, _RATE_NUMERATOR[metric])

    hm = hm.assign(_z=z, _expected=expected_series)
    direction = BAD_DIRECTION[metric]

    if direction == "down":
        flagged = (hm._z <= -z_threshold)
    elif direction == "up":
        flagged = (hm._z >= z_threshold)
    else:
        flagged = (hm._z.abs() >= z_threshold)
    flagged = flagged & (hm.txn_count >= min_volume)

    # Effect-size floors -- significance is necessary but not sufficient.
    if metric in _MIN_EFFECT_PP:
        flagged = flagged & ((hm[metric] - hm._expected).abs() >= _MIN_EFFECT_PP[metric])
    if metric == "fraud_rate":
        flagged = flagged & (hm.fraud_count >= _MIN_FRAUD_COUNT) & (hm[metric] >= 2 * hm._expected)
    if metric == "txn_count":
        flagged = flagged & ((hm.txn_count - hm._expected).abs() >= 0.5 * hm._expected)

    if not flagged.any():
        return []

    # Merge contiguous flagged hours into events.
    times = hm.loc[flagged, "timestamp"].sort_values().tolist()
    events_idx: list[list[pd.Timestamp]] = []
    cur = [times[0]]
    for t in times[1:]:
        if t - cur[-1] <= timedelta(hours=config.EVENT_MERGE_GAP_HOURS):
            cur.append(t)
        else:
            events_idx.append(cur)
            cur = [t]
    events_idx.append(cur)

    out: list[AnomalyEvent] = []
    for grp in events_idx:
        win = hm[(hm.timestamp >= grp[0]) & (hm.timestamp <= grp[-1])]
        # worst hour = max |z| in the window
        worst = win.loc[win._z.abs().idxmax()]
        peak_z = float(abs(worst._z))
        dirn = "down" if worst._z < 0 else "up"
        is_rate = metric.endswith("_rate")
        expected = float(worst._expected)
        out.append(AnomalyEvent(
            metric=metric, grain=grain, scope=scope, direction=dirn,
            start=str(grp[0]), end=str(grp[-1]),
            duration_hours=int((grp[-1] - grp[0]).total_seconds() // 3600) + 1,
            peak_z=round(peak_z, 2),
            observed=float(round(worst[metric], 6)),
            expected=float(round(expected, 6)),
            observed_pct=round(float(worst[metric]) * 100, 2) if is_rate else None,
            expected_pct=round(expected * 100, 2) if is_rate else None,
            volume=int(win.txn_count.sum()),
            iso_forest_corroborated=False,
            severity=_severity(peak_z),
        ))
    return out


def _severity(peak_z: float) -> str:
    if peak_z >= 8:
        return "high"
    if peak_z >= 5.5:
        return "medium"
    return "low"


# --------------------------------------------------------------------------- #
# Isolation Forest cross-check
# --------------------------------------------------------------------------- #
def _iso_forest_hours(df: pd.DataFrame) -> set:
    """Return the set of hour-timestamps flagged by a global Isolation Forest."""
    from sklearn.ensemble import IsolationForest

    hm = _hourly_metrics(df)
    feats = hm[["approval_rate", "decline_rate", "fraud_rate"]].copy()
    feats["log_vol"] = np.log1p(hm.txn_count)
    for m in ("approval_rate", "decline_rate", "fraud_rate", "txn_count"):
        feats[f"{m}_resid"] = _robust_z(hm[m], hm.weekday, hm.hour).clip(-15, 15)
    feats = feats.fillna(0.0)
    model = IsolationForest(contamination=0.02, random_state=0, n_estimators=200)
    pred = model.fit_predict(feats.values)
    return set(hm.loc[pred == -1, "timestamp"].tolist())


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def detect(df: pd.DataFrame,
           z_threshold: float = config.ROBUST_Z_THRESHOLD,
           min_volume: int = config.MIN_HOURLY_VOLUME) -> list[AnomalyEvent]:
    """Run detection across global + single-dimension + (country,channel) grains."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    events: list[AnomalyEvent] = []

    # 1) Global
    hm = _hourly_metrics(df)
    for metric in METRICS:
        events += _flag_series(hm, metric, "global", {}, z_threshold, min_volume)

    # 2) Single-dimension slices
    for dim in ("mcc", "country", "channel", "auth_type"):
        for val in sorted(df[dim].unique()):
            sub = df[df[dim] == val]
            hsub = _hourly_metrics(sub)
            scope = {dim: val}
            grain = f"{dim}={val}"
            for metric in METRICS:
                events += _flag_series(hsub, metric, grain, scope,
                                       z_threshold, min_volume)

    # 3) (country, channel) corridors — catches cross-border ecom issues
    for country in sorted(df.country.unique()):
        for channel in sorted(df.channel.unique()):
            sub = df[(df.country == country) & (df.channel == channel)]
            if sub.empty:
                continue
            hsub = _hourly_metrics(sub)
            scope = {"country": country, "channel": channel}
            grain = f"country={country}|channel={channel}"
            for metric in ("approval_rate", "fraud_rate"):
                events += _flag_series(hsub, metric, grain, scope,
                                       z_threshold, min_volume)

    # 4) Isolation Forest corroboration (global view)
    try:
        iso_hours = _iso_forest_hours(df)
        for e in events:
            start = pd.to_datetime(e.start)
            end = pd.to_datetime(e.end)
            if any(start <= h <= end for h in iso_hours):
                e.iso_forest_corroborated = True
    except Exception:
        pass  # IF is a nice-to-have; never let it break detection

    # De-duplicate near-identical events, keep the most severe / specific.
    events = _dedupe(events)
    events.sort(key=lambda e: e.peak_z, reverse=True)
    return events


def _dedupe(events: list[AnomalyEvent]) -> list[AnomalyEvent]:
    seen: dict[tuple, AnomalyEvent] = {}
    for e in events:
        key = (e.metric, e.start[:13], e.grain)  # hour-level key per grain
        if key not in seen or e.peak_z > seen[key].peak_z:
            seen[key] = e
    return list(seen.values())


def events_to_frame(events: list[AnomalyEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in events])


# --------------------------------------------------------------------------- #
# Incident consolidation
# --------------------------------------------------------------------------- #
@dataclass
class Incident:
    """A cluster of time-overlapping anomaly events = one operational incident."""
    incident_id: str
    title: str
    start: str
    end: str
    severity: str
    peak_z: float
    primary_metric: str
    primary_scope: dict
    primary_grain: str
    n_member_events: int
    n_grains_affected: int
    iso_forest_corroborated: bool
    members: list[AnomalyEvent] = field(default_factory=list)


_METRIC_LABEL = {
    "approval_rate": "Approval-rate drop",
    "decline_rate": "Decline-rate spike",
    "fraud_rate": "Fraud spike",
    "txn_count": "Volume anomaly",
}


def consolidate(events: list[AnomalyEvent],
                merge_gap_hours: int = 6) -> list[Incident]:
    """Group events whose time windows overlap (within a small gap) into incidents.

    The representative ("primary") event of an incident is the one with the
    highest robust score; its metric/scope name the incident. In production you
    would also cluster on scope similarity -- see WRITEUP.md 'What I'd improve'.
    """
    if not events:
        return []
    ev = sorted(events, key=lambda e: pd.to_datetime(e.start))
    clusters: list[list[AnomalyEvent]] = [[ev[0]]]
    for e in ev[1:]:
        cur_end = max(pd.to_datetime(m.end) for m in clusters[-1])
        if pd.to_datetime(e.start) <= cur_end + timedelta(hours=merge_gap_hours):
            clusters[-1].append(e)
        else:
            clusters.append([e])

    incidents: list[Incident] = []
    for i, members in enumerate(clusters, start=1):
        primary = max(members, key=lambda m: m.peak_z)
        grains = {m.grain for m in members}
        start = min(pd.to_datetime(m.start) for m in members)
        end = max(pd.to_datetime(m.end) for m in members)
        sev = "high" if any(m.severity == "high" for m in members) else \
              "medium" if any(m.severity == "medium" for m in members) else "low"
        scope_txt = ("global" if not primary.scope
                     else ", ".join(f"{k}={v}" for k, v in primary.scope.items()))
        incidents.append(Incident(
            incident_id=f"INC-{i:03d}",
            title=f"{_METRIC_LABEL.get(primary.metric, primary.metric)} ({scope_txt})",
            start=str(start), end=str(end), severity=sev,
            peak_z=primary.peak_z, primary_metric=primary.metric,
            primary_scope=primary.scope, primary_grain=primary.grain,
            n_member_events=len(members), n_grains_affected=len(grains),
            iso_forest_corroborated=any(m.iso_forest_corroborated for m in members),
            members=sorted(members, key=lambda m: m.peak_z, reverse=True),
        ))
    incidents.sort(key=lambda inc: pd.to_datetime(inc.start))
    return incidents
