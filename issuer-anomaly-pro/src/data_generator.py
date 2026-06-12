"""Generate a synthetic issuer-side authorization dataset with realistic
seasonality/noise and a handful of deliberately injected anomalies.

Grain of the output table (one row per outcome bucket within a slice-hour):

    timestamp, mcc, country, channel, card_present_flag, auth_type,
    decline_reason_code, txn_count, approved_count, declined_count,
    fraud_count, txn_amount

`decline_reason_code` carries the synthetic value ``00_APPROVED`` for the
approved bucket and a real ISO-style code for each decline bucket. This keeps a
single tidy table that matches the suggested schema while still letting us answer
"which decline reason codes drove the spike?" by a simple group-by.

The ground-truth list of injected anomalies is written next to the CSV so the
detector's recall can be measured during the demo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Ground-truth anomaly description
# --------------------------------------------------------------------------- #
@dataclass
class InjectedAnomaly:
    name: str
    kind: str
    start: str
    end: str
    scope: dict          # which dimension values are affected
    description: str


# --------------------------------------------------------------------------- #
# Baseline behaviour per slice
# --------------------------------------------------------------------------- #
def _slice_base_volume(mcc: str, country: str, channel: str, auth_type: str) -> float:
    """Mean hourly transaction count for a slice (before time-of-day shaping)."""
    base = {
        "5411": 90, "5812": 70, "5999": 55, "4829": 25, "7995": 18,
    }[mcc]
    base *= {"US": 1.0, "GB": 0.55, "IN": 0.7, "DE": 0.45, "BR": 0.5}[country]
    base *= {"ecom": 1.0, "pos": 0.8, "contactless": 0.6}[channel]
    base *= 0.65 if auth_type == "3DS" else 1.0
    return max(base, 4.0)


def _baseline_approval_rate(mcc: str, country: str, channel: str, auth_type: str) -> float:
    """Healthy steady-state approval rate for a slice."""
    rate = 0.93
    if country != config.ISSUER_COUNTRY:        # cross-border is approved less often
        rate -= 0.04
    if channel == "ecom":                        # CNP a touch riskier
        rate -= 0.02
    if mcc in ("7995", "4829"):                  # gambling / money transfer
        rate -= 0.05
    if auth_type == "3DS":                        # step-up auth lifts approvals
        rate += 0.02
    return float(np.clip(rate, 0.70, 0.985))


def _baseline_fraud_rate(mcc: str, channel: str) -> float:
    rate = 0.0010
    if channel == "ecom":
        rate += 0.0015
    if mcc in ("7995", "4829", "5999"):
        rate += 0.0020
    return rate


def _diurnal_factor(hour: int) -> float:
    """Smooth day/night demand curve, peaking late afternoon."""
    return 0.55 + 0.45 * np.sin((hour - 3) / 24.0 * 2 * np.pi - np.pi / 2) ** 2


def _weekly_factor(dow: int) -> float:
    # Mon=0 .. Sun=6; weekends slightly busier for retail/leisure.
    return [0.95, 0.96, 0.98, 1.0, 1.08, 1.18, 1.10][dow]


def _decline_mix(channel: str, auth_type: str) -> dict[str, float]:
    """Probability split of declines across reason codes in steady state."""
    mix = {
        "05_DO_NOT_HONOR": 0.34,
        "51_INSUFFICIENT_FUNDS": 0.40,
        "14_INVALID_CARD": 0.10,
        "59_SUSPECTED_FRAUD": 0.10,
        "91_ISSUER_UNAVAILABLE": 0.02,
        "N7_3DS_FAILURE": 0.04 if auth_type == "3DS" else 0.0,
    }
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


# --------------------------------------------------------------------------- #
# Anomaly injection
# --------------------------------------------------------------------------- #
def _build_injected(start_date: datetime, n_days: int) -> list[InjectedAnomaly]:
    """Place anomalies at sensible points inside the window."""
    d = lambda day, hour=0: (start_date + timedelta(days=day, hours=hour))
    fmt = "%Y-%m-%d %H:%M"
    return [
        InjectedAnomaly(
            name="Issuer processor outage",
            kind="technical_decline_spike",
            start=d(12, 9).strftime(fmt), end=d(12, 15).strftime(fmt),
            scope={"all": True},
            description=("6-hour issuer/processor outage: 91_ISSUER_UNAVAILABLE "
                         "technical declines surge across all slices, approval "
                         "rate collapses."),
        ),
        InjectedAnomaly(
            name="Card-testing fraud attack",
            kind="fraud_spike",
            start=d(20, 0).strftime(fmt), end=d(21, 23).strftime(fmt),
            scope={"mcc": "5999", "country": "BR", "channel": "ecom"},
            description=("2-day fraud attack on Misc-Retail e-commerce in BR: "
                         "fraud_count and 59_SUSPECTED_FRAUD declines spike."),
        ),
        InjectedAnomaly(
            name="3DS authentication failure",
            kind="auth_failure_spike",
            start=d(28, 6).strftime(fmt), end=d(29, 6).strftime(fmt),
            scope={"country": "GB", "channel": "ecom", "auth_type": "3DS"},
            description=("24-hour 3DS outage in GB e-commerce: N7_3DS_FAILURE "
                         "declines spike, 3DS approval rate drops sharply."),
        ),
        InjectedAnomaly(
            name="Cross-border approval drop",
            kind="approval_drop",
            start=d(35, 0).strftime(fmt), end=d(37, 23).strftime(fmt),
            scope={"country": "IN", "channel": "ecom"},
            description=("3-day approval-rate drop on cross-border IN e-commerce "
                         "driven by elevated 05_DO_NOT_HONOR risk declines."),
        ),
        InjectedAnomaly(
            name="Gambling volume surge",
            kind="volume_spike",
            start=d(40, 18).strftime(fmt), end=d(41, 2).strftime(fmt),
            scope={"mcc": "7995"},
            description=("8-hour transaction-volume surge on Gambling MCC "
                         "(~3x normal) with mildly depressed approvals."),
        ),
    ]


def _anomaly_effect(ts: datetime, mcc: str, country: str, channel: str,
                    auth_type: str, injected: list[InjectedAnomaly]) -> dict:
    """Return multiplicative/additive effects active at this slice-hour."""
    eff = {"vol_mult": 1.0, "appr_delta": 0.0, "fraud_mult": 1.0,
           "decline_boost": {}}  # code -> extra probability mass

    def active(a: InjectedAnomaly) -> bool:
        s = datetime.strptime(a.start, "%Y-%m-%d %H:%M")
        e = datetime.strptime(a.end, "%Y-%m-%d %H:%M")
        return s <= ts <= e

    for a in injected:
        if not active(a):
            continue
        sc = a.scope
        if "mcc" in sc and sc["mcc"] != mcc:
            continue
        if "country" in sc and sc["country"] != country:
            continue
        if "channel" in sc and sc["channel"] != channel:
            continue
        if "auth_type" in sc and sc["auth_type"] != auth_type:
            continue
        # matched
        if a.kind == "technical_decline_spike":
            eff["appr_delta"] -= 0.55
            eff["decline_boost"]["91_ISSUER_UNAVAILABLE"] = \
                eff["decline_boost"].get("91_ISSUER_UNAVAILABLE", 0) + 0.85
        elif a.kind == "fraud_spike":
            eff["fraud_mult"] *= 28.0
            eff["vol_mult"] *= 1.6
            eff["appr_delta"] -= 0.10
            eff["decline_boost"]["59_SUSPECTED_FRAUD"] = \
                eff["decline_boost"].get("59_SUSPECTED_FRAUD", 0) + 0.45
        elif a.kind == "auth_failure_spike":
            eff["appr_delta"] -= 0.30
            eff["decline_boost"]["N7_3DS_FAILURE"] = \
                eff["decline_boost"].get("N7_3DS_FAILURE", 0) + 0.65
        elif a.kind == "approval_drop":
            eff["appr_delta"] -= 0.18
            eff["decline_boost"]["05_DO_NOT_HONOR"] = \
                eff["decline_boost"].get("05_DO_NOT_HONOR", 0) + 0.40
        elif a.kind == "volume_spike":
            eff["vol_mult"] *= 3.0
            eff["appr_delta"] -= 0.05
    return eff


# --------------------------------------------------------------------------- #
# Main generation routine
# --------------------------------------------------------------------------- #
def generate(n_days: int = 45, seed: int = 7,
             start_date: datetime | None = None) -> tuple[pd.DataFrame, list[InjectedAnomaly]]:
    rng = np.random.default_rng(seed)
    if start_date is None:
        start_date = (datetime.utcnow() - timedelta(days=n_days)).replace(
            hour=0, minute=0, second=0, microsecond=0)

    injected = _build_injected(start_date, n_days)
    rows: list[dict] = []

    slices = [
        (mcc, country, channel, cp_flag, auth)
        for mcc in config.MCCS
        for country in config.COUNTRIES
        for channel, (cp_flag, auths) in config.CHANNELS.items()
        for auth in auths
    ]

    total_hours = n_days * 24
    for h in range(total_hours):
        ts = start_date + timedelta(hours=h)
        tf = _diurnal_factor(ts.hour) * _weekly_factor(ts.weekday())
        for (mcc, country, channel, cp_flag, auth) in slices:
            lam = _slice_base_volume(mcc, country, channel, auth) * tf
            eff = _anomaly_effect(ts, mcc, country, channel, auth, injected)
            lam *= eff["vol_mult"]
            total = int(rng.poisson(max(lam, 0.1)))
            if total == 0:
                continue

            appr_rate = _baseline_approval_rate(mcc, country, channel, auth)
            appr_rate += eff["appr_delta"]
            # small idiosyncratic noise on the rate
            appr_rate += rng.normal(0, 0.008)
            appr_rate = float(np.clip(appr_rate, 0.02, 0.995))

            approved = int(rng.binomial(total, appr_rate))
            declined = total - approved

            # fraud occurs predominantly on approved transactions
            fr = _baseline_fraud_rate(mcc, channel) * eff["fraud_mult"]
            fraud_on_appr = int(rng.binomial(approved, float(np.clip(fr, 0, 0.6))))

            # amounts: lognormal scaled per MCC
            mu = {"5411": 3.4, "5812": 3.2, "5999": 3.8,
                  "4829": 5.0, "7995": 4.2}[mcc]
            avg_amt = float(np.exp(mu))

            # ---- approved bucket row ----
            rows.append({
                "timestamp": ts, "mcc": mcc, "country": country,
                "channel": channel, "card_present_flag": cp_flag,
                "auth_type": auth, "decline_reason_code": config.APPROVED_CODE,
                "txn_count": approved, "approved_count": approved,
                "declined_count": 0, "fraud_count": fraud_on_appr,
                "txn_amount": round(approved * avg_amt * rng.uniform(0.9, 1.1), 2),
            })

            # ---- decline buckets ----
            if declined > 0:
                mix = _decline_mix(channel, auth).copy()
                for code, boost in eff["decline_boost"].items():
                    mix[code] = mix.get(code, 0.0) + boost
                codes = list(mix.keys())
                probs = np.array([mix[c] for c in codes], dtype=float)
                probs = probs / probs.sum()
                counts = rng.multinomial(declined, probs)
                for code, cnt in zip(codes, counts):
                    if cnt == 0:
                        continue
                    rows.append({
                        "timestamp": ts, "mcc": mcc, "country": country,
                        "channel": channel, "card_present_flag": cp_flag,
                        "auth_type": auth, "decline_reason_code": code,
                        "txn_count": int(cnt), "approved_count": 0,
                        "declined_count": int(cnt), "fraud_count": 0,
                        "txn_amount": round(cnt * avg_amt * rng.uniform(0.9, 1.1), 2),
                    })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, injected


def generate_and_save(n_days: int = 45, seed: int = 7) -> pd.DataFrame:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df, injected = generate(n_days=n_days, seed=seed)
    df.to_csv(config.TRANSACTIONS_CSV, index=False)
    with open(config.INJECTED_TRUTH_JSON, "w") as f:
        json.dump([asdict(a) for a in injected], f, indent=2)
    return df


def load_transactions(path=None) -> pd.DataFrame:
    """Load the transactions CSV with correct dtypes (codes are strings)."""
    path = path or config.TRANSACTIONS_CSV
    return pd.read_csv(
        path,
        dtype={"mcc": str, "country": str, "channel": str,
               "auth_type": str, "decline_reason_code": str},
        parse_dates=["timestamp"],
    )


if __name__ == "__main__":
    out = generate_and_save()
    print(f"Wrote {len(out):,} rows to {config.TRANSACTIONS_CSV}")
    print(f"Date range: {out.timestamp.min()} -> {out.timestamp.max()}")
    print(f"Total transactions: {out.txn_count.sum():,}")
