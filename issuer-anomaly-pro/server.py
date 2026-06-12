"""FastAPI backend for the Issuer Anomaly Console (pro web UI).

Wraps the unchanged detection + GenAI engine in src/ as a small JSON API and
serves the single-page frontend in static/index.html.

Run:  python server.py        (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src import config, detection, diagnosis, llm_client
from src.context_builder import build_chat_context, incident_context
from src.data_generator import generate_and_save, load_transactions

# --------------------------------------------------------------------------- #
# One-time compute at startup
# --------------------------------------------------------------------------- #
if not config.TRANSACTIONS_CSV.exists():
    generate_and_save(n_days=45, seed=7)

DF = load_transactions()
EVENTS = detection.detect(DF)
INCIDENTS = detection.consolidate(EVENTS)
INC_BY_ID = {i.incident_id: i for i in INCIDENTS}
CHAT_CTX = build_chat_context(DF, INCIDENTS)
ONLINE = llm_client.is_online()
DIAG_CACHE: dict[str, str] = {}

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Issuer Anomaly Console")


def _overall_hourly() -> pd.DataFrame:
    g = DF.groupby("timestamp").agg(
        txn=("txn_count", "sum"), appr=("approved_count", "sum"),
        dec=("declined_count", "sum"), fr=("fraud_count", "sum")).reset_index()
    g["Approval rate %"] = (100 * g.appr / g.txn).round(3)
    g["Decline rate %"] = (100 * g.dec / g.txn).round(3)
    g["Fraud rate %"] = (100 * g.fr / g.txn).round(3)
    g["Volume (txns/hr)"] = g.txn
    return g


GH = _overall_hourly()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/overview")
def overview() -> dict:
    tot = int(DF.txn_count.sum())
    appr = 100 * DF.approved_count.sum() / tot
    fraud = 100 * DF.fraud_count.sum() / tot
    sev = {"high": 0, "medium": 0, "low": 0}
    for i in INCIDENTS:
        sev[i.severity] = sev.get(i.severity, 0) + 1
    recent = sorted(INCIDENTS, key=lambda i: i.start, reverse=True)[:6]
    return {
        "online": ONLINE,
        "provider": llm_client.provider_label() if ONLINE else None,
        "window": f"{DF.timestamp.min():%d %b} – {DF.timestamp.max():%d %b}",
        "kpis": {"txn_m": f"{tot/1e6:.2f}M", "transactions": tot,
                 "approval": round(appr, 1), "fraud": round(fraud, 2),
                 "incidents": len(INCIDENTS), "alerts": len(EVENTS), "sev": sev},
        "steps": {"rows": len(DF), "events": len(EVENTS), "incidents": len(INCIDENTS)},
        "series": {
            "timestamps": [t.isoformat() for t in GH.timestamp],
            "Approval rate %": GH["Approval rate %"].tolist(),
            "Decline rate %": GH["Decline rate %"].tolist(),
            "Fraud rate %": GH["Fraud rate %"].tolist(),
            "Volume (txns/hr)": GH["Volume (txns/hr)"].tolist()},
        "bands": [{"start": pd.to_datetime(i.start).isoformat(),
                   "end": pd.to_datetime(i.end).isoformat(), "title": i.title}
                  for i in INCIDENTS],
        "recent": [{"id": i.incident_id, "severity": i.severity, "title": i.title,
                    "start": i.start, "end": i.end, "peak_z": round(i.peak_z, 1),
                    "metric": i.primary_metric} for i in recent]}


@app.get("/api/incidents")
def incidents() -> list:
    return [{"id": i.incident_id, "severity": i.severity, "title": i.title,
             "label": f"{i.incident_id} · {i.severity.upper()} · {i.title}"}
            for i in INCIDENTS]


@app.get("/api/incident/{incident_id}")
def incident(incident_id: str) -> dict:
    inc = INC_BY_ID.get(incident_id)
    if inc is None:
        return {"error": "not found"}
    facts = incident_context(DF, inc)
    if inc.incident_id not in DIAG_CACHE:
        DIAG_CACHE[inc.incident_id] = diagnosis.diagnose_incident(DF, inc)[0]
    scope = " · ".join(f"{k}={v}" for k, v in inc.primary_scope.items()) or "global"
    shift = facts.get("decline_reason_shift")
    decline = None
    if shift:
        decline = [{"code": r["decline_reason_code"],
                    "baseline": r["baseline_share_pct"],
                    "during": r["during_share_pct"]} for r in shift]
    return {
        "detail": {"id": inc.incident_id, "severity": inc.severity, "title": inc.title,
                   "start": inc.start, "end": inc.end, "peak_z": round(inc.peak_z, 1),
                   "metric": inc.primary_metric, "grain": inc.primary_grain,
                   "scope": scope, "slices": inc.n_grains_affected,
                   "iso": inc.iso_forest_corroborated},
        "diagnosis": DIAG_CACHE[inc.incident_id],
        "source": llm_client.provider_label() if ONLINE else "offline template",
        "decline": decline, "facts": facts}


class ChatIn(BaseModel):
    message: str
    history: list = []


@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    answer = diagnosis.answer_question(DF, INCIDENTS, body.message,
                                       history=body.history, _cached_context=CHAT_CTX)
    return {"answer": answer}


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
