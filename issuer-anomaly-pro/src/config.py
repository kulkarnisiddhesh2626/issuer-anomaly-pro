"""Central configuration: dimension vocabularies, file paths, and LLM settings.

Everything that another module might want to tweak lives here so the rest of the
codebase stays declarative and easy to reason about.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load a local .env (if present) before reading any environment variables, so a
# user's API keys / provider choice take effect. No-op if python-dotenv isn't
# installed or there's no .env file.
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except Exception:
    pass
DATA_DIR = ROOT_DIR / "data"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
INJECTED_TRUTH_JSON = DATA_DIR / "injected_anomalies.json"

# --------------------------------------------------------------------------- #
# Synthetic-data dimensions
# --------------------------------------------------------------------------- #
# Merchant Category Codes (issuer-relevant subset) with human labels.
MCCS: dict[str, str] = {
    "5411": "Grocery Stores",
    "5812": "Restaurants",
    "5999": "Misc Retail",
    "4829": "Money Transfer",
    "7995": "Gambling",
}

# Issuing/acquiring countries. "domestic" is defined relative to ISSUER_COUNTRY.
COUNTRIES: list[str] = ["US", "GB", "IN", "DE", "BR"]
ISSUER_COUNTRY = "US"  # cards are issued here; everything else is cross-border

# (channel, card_present_flag, allowed auth_types)
CHANNELS: dict[str, tuple[bool, list[str]]] = {
    "ecom": (False, ["3DS", "non-3DS"]),       # card-not-present
    "pos": (True, ["non-3DS"]),                # card-present, chip/swipe
    "contactless": (True, ["non-3DS"]),        # card-present, tap
}

# Decline reason codes (ISO-8583-flavoured) plus the synthetic APPROVED bucket.
APPROVED_CODE = "00_APPROVED"
DECLINE_REASON_CODES: dict[str, str] = {
    "05_DO_NOT_HONOR": "Do Not Honor (issuer risk decline)",
    "51_INSUFFICIENT_FUNDS": "Insufficient Funds",
    "14_INVALID_CARD": "Invalid Card Number",
    "59_SUSPECTED_FRAUD": "Suspected Fraud",
    "91_ISSUER_UNAVAILABLE": "Issuer/Processor Unavailable (technical)",
    "N7_3DS_FAILURE": "3DS Authentication Failure",
}
ALL_OUTCOME_CODES: list[str] = [APPROVED_CODE] + list(DECLINE_REASON_CODES)

# --------------------------------------------------------------------------- #
# LLM settings (read from environment; safe defaults)
# --------------------------------------------------------------------------- #
# Provider is pluggable. Supported: "anthropic" (default) or "groq".
# Groq runs open models (e.g. Llama 3.3 70B) on very fast inference with a
# generous free tier — a good fit for a widely shared demo, because the LLM here
# only *narrates pre-computed facts*, so the grounding (not raw model size) does
# the heavy lifting. Switching providers touches only this file + llm_client.py.
LLM_PROVIDER = os.environ.get("ANOMALY_LLM_PROVIDER", "anthropic").strip().lower()

# Per-provider API keys.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

# Per-provider models (override with the matching env var).
# Anthropic cheaper option for high volume: "claude-haiku-4-5".
ANTHROPIC_MODEL = os.environ.get("ANOMALY_LLM_MODEL", "claude-sonnet-4-6")
GROQ_MODEL = os.environ.get("ANOMALY_GROQ_MODEL", "llama-3.3-70b-versatile")

# Resolve the active model/key for the chosen provider.
DEFAULT_MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else ANTHROPIC_MODEL
ACTIVE_API_KEY = GROQ_API_KEY if LLM_PROVIDER == "groq" else ANTHROPIC_API_KEY

LLM_MAX_TOKENS = int(os.environ.get("ANOMALY_LLM_MAX_TOKENS", "1200"))
# Low temperature: we want grounded, repeatable diagnostics, not creativity.
LLM_TEMPERATURE = float(os.environ.get("ANOMALY_LLM_TEMPERATURE", "0.2"))

# --------------------------------------------------------------------------- #
# Detection defaults (overridable from the UI)
# --------------------------------------------------------------------------- #
ROBUST_Z_THRESHOLD = 5.0     # MAD/proportion-scaled z above which a point is anomalous
MIN_HOURLY_VOLUME = 30       # ignore buckets too small to be statistically meaningful
EVENT_MERGE_GAP_HOURS = 2    # contiguous flagged hours within this gap = one event
