"""Context builder — the grounding layer.

The GenAI layer never sees raw transaction rows. Instead, for each incident we
compute a compact, fact-only JSON payload: the metric trajectory before / during
/ after the event, and the dimension breakdowns (decline reason, MCC, country,
channel) that explain *where* the anomaly concentrates. The LLM's job is to turn
these verified facts into prose and recommendations — not to compute or recall
them. This is what keeps diagnoses grounded and auditable.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from . import config
from .detection import Incident


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _apply_scope(df: pd.DataFrame, scope: dict) -> pd.DataFrame:
    out = df
    for k, v in scope.items():
        out = out[out[k] == v]
    return out


def _rates(frame: pd.DataFrame) -> dict:
    tot = int(frame.txn_count.sum())
    if tot == 0:
        return {"txn_count": 0, "approval_rate_pct": None,
                "decline_rate_pct": None, "fraud_rate_pct": None}
    return {
        "txn_count": tot,
        "approval_rate_pct": round(100 * frame.approved_count.sum() / tot, 2),
        "decline_rate_pct": round(100 * frame.declined_count.sum() / tot, 2),
        "fraud_rate_pct": round(100 * frame.fraud_count.sum() / tot, 3),
    }


def _share_shift(during: pd.DataFrame, baseline: pd.DataFrame,
                 dim: str, top: int = 5) -> list[dict]:
    """How each value's share of *declines* shifted during the window vs baseline."""
    def share(frame):
        d = frame[frame.declined_count > 0]
        if d.declined_count.sum() == 0:
            return {}
        s = d.groupby(dim).declined_count.sum()
        return (s / s.sum()).to_dict()

    bs = share(baseline)
    ds = share(during)
    rows = []
    for val in sorted(set(bs) | set(ds), key=lambda v: ds.get(v, 0), reverse=True):
        rows.append({
            dim: val,
            "baseline_share_pct": round(100 * bs.get(val, 0.0), 1),
            "during_share_pct": round(100 * ds.get(val, 0.0), 1),
            "delta_pp": round(100 * (ds.get(val, 0.0) - bs.get(val, 0.0)), 1),
        })
    rows.sort(key=lambda r: abs(r["delta_pp"]), reverse=True)
    return rows[:top]


# --------------------------------------------------------------------------- #
# Per-incident context
# --------------------------------------------------------------------------- #
def incident_context(df: pd.DataFrame, inc: Incident) -> dict:
    """Compact, fact-only payload describing one incident for the LLM."""
    start = pd.to_datetime(inc.start)
    end = pd.to_datetime(inc.end)
    dur = max((end - start), timedelta(hours=1))

    scoped = _apply_scope(df, inc.primary_scope)
    during = scoped[(scoped.timestamp >= start) & (scoped.timestamp <= end)]
    before = scoped[(scoped.timestamp >= start - dur - timedelta(hours=1)) &
                    (scoped.timestamp < start)]
    after = scoped[(scoped.timestamp > end) &
                   (scoped.timestamp <= end + dur + timedelta(hours=1))]
    # baseline = same scope, comparable hours-of-day, outside the window
    outside = scoped[(scoped.timestamp < start) | (scoped.timestamp > end)]

    primary = inc.members[0]
    facts = {
        "incident_id": inc.incident_id,
        "title": inc.title,
        "severity": inc.severity,
        "window": {"start": inc.start, "end": inc.end,
                   "duration_hours": int(dur.total_seconds() // 3600) or 1},
        "scope": {k: str(v) for k, v in inc.primary_scope.items()},
        "scope_label": ("global" if not inc.primary_scope
                        else ", ".join(f"{k}={v}" for k, v in inc.primary_scope.items())),
        "primary_signal": {
            "metric": primary.metric,
            "direction": primary.direction,
            "robust_z_score": primary.peak_z,
            "observed": primary.observed_pct if primary.observed_pct is not None else primary.observed,
            "expected_baseline": primary.expected_pct if primary.expected_pct is not None else primary.expected,
            "unit": "percent" if primary.metric.endswith("_rate") else "transactions/hour",
        },
        "isolation_forest_corroborated": inc.iso_forest_corroborated,
        "grains_affected": inc.n_grains_affected,
        "metrics_window": {
            "before": _rates(before),
            "during": _rates(during),
            "after": _rates(after),
        },
        "decline_reason_shift": _share_shift(during, outside, "decline_reason_code"),
    }
    # add dimension breakdowns only where they add information (not already pinned)
    for dim in ("mcc", "country", "channel", "auth_type"):
        if dim not in inc.primary_scope:
            shift = _share_shift(during, outside, dim)
            if shift:
                facts.setdefault("dimension_shift", {})[dim] = shift

    # decode codes to labels so the model has plain-language anchors
    facts["code_glossary"] = {
        c: config.DECLINE_REASON_CODES.get(c, c)
        for c in {r["decline_reason_code"] for r in facts["decline_reason_shift"]}
        if c in config.DECLINE_REASON_CODES
    }
    if "mcc" in inc.primary_scope:
        label = config.MCCS.get(str(inc.primary_scope["mcc"]), "?")
        facts["scope_label"] += f" ({label})"
    return facts


# --------------------------------------------------------------------------- #
# Dataset + chat context
# --------------------------------------------------------------------------- #
def dataset_overview(df: pd.DataFrame) -> dict:
    tot = int(df.txn_count.sum())
    return {
        "rows": int(len(df)),
        "date_range": {"start": str(df.timestamp.min()), "end": str(df.timestamp.max())},
        "total_transactions": tot,
        "overall_approval_rate_pct": round(100 * df.approved_count.sum() / tot, 2),
        "overall_decline_rate_pct": round(100 * df.declined_count.sum() / tot, 2),
        "overall_fraud_rate_pct": round(100 * df.fraud_count.sum() / tot, 3),
        "dimensions": {
            "mcc": {k: v for k, v in config.MCCS.items()},
            "country": config.COUNTRIES,
            "channel": list(config.CHANNELS),
            "auth_type": ["3DS", "non-3DS"],
            "decline_reason_code": config.DECLINE_REASON_CODES,
        },
        "schema_note": ("Each row is one outcome bucket within a slice-hour. "
                        "decline_reason_code='00_APPROVED' marks the approved bucket; "
                        "rates are computed by aggregating txn_count / approved_count / "
                        "declined_count / fraud_count."),
    }


def daily_metric_table(df: pd.DataFrame, by: str | None = None) -> list[dict]:
    """Compact daily metric series (optionally split by one dimension)."""
    d = df.copy()
    d["date"] = d.timestamp.dt.date.astype(str)
    keys = ["date"] + ([by] if by else [])
    g = d.groupby(keys).agg(
        txn=("txn_count", "sum"), appr=("approved_count", "sum"),
        dec=("declined_count", "sum"), fr=("fraud_count", "sum")).reset_index()
    g["approval_rate_pct"] = (100 * g.appr / g.txn).round(2)
    g["decline_rate_pct"] = (100 * g.dec / g.txn).round(2)
    g["fraud_rate_pct"] = (100 * g.fr / g.txn).round(3)
    cols = keys + ["txn", "approval_rate_pct", "decline_rate_pct", "fraud_rate_pct"]
    return g[cols].to_dict("records")


def decline_reason_daily(df: pd.DataFrame) -> list[dict]:
    d = df[df.declined_count > 0].copy()
    d["date"] = d.timestamp.dt.date.astype(str)
    g = (d.groupby(["date", "decline_reason_code"]).declined_count.sum()
         .reset_index().rename(columns={"declined_count": "count"}))
    return g.to_dict("records")


def build_chat_context(df: pd.DataFrame, incidents: list[Incident],
                       max_incidents: int = 12) -> dict:
    """Compact context the conversational layer reasons over.

    Kept deliberately small (well under free-tier token-per-minute limits): the
    full per-incident fact sheets and the daily decline-reason table are large,
    so for chat we send a trimmed incident view plus the dataset overview and the
    daily metric table. The Incidents tab still shows each incident's complete
    fact sheet separately (it calls incident_context directly).
    """
    keep = ("incident_id", "title", "severity", "window", "scope", "scope_label",
            "primary_signal", "metrics_window", "code_glossary")
    slim = []
    for f in (incident_context(df, inc) for inc in incidents[:max_incidents]):
        s = {k: f[k] for k in keep if k in f}
        s["decline_reason_shift"] = f.get("decline_reason_shift", [])[:5]
        slim.append(s)
    return {
        "dataset_overview": dataset_overview(df),
        "detected_incidents": slim,
        "daily_metrics_overall": daily_metric_table(df),
    }
