"""GenAI diagnostic layer.

Responsibilities:
* Build the diagnosis prompt from an incident's grounded context and call the LLM.
* Answer analyst follow-up questions grounded ONLY in the structured context.
* Provide deterministic offline fallbacks so the POC always works.

Hallucination control (see WRITEUP.md) is implemented here through:
1. The LLM is given pre-computed *facts*, never raw rows, and is told to treat
   them as the only source of truth.
2. The system prompt forbids inventing numbers and requires it to say when the
   data does not contain an answer.
3. Low temperature + explicit "cite the numbers you used" instruction.
4. The offline path is 100% template-driven from the same facts.
"""
from __future__ import annotations

import json

from . import config, llm_client
from .context_builder import incident_context, build_chat_context
from .detection import Incident
import pandas as pd


DIAGNOSIS_SYSTEM = """You are a payments risk analyst assistant for a card-issuing bank.
You write concise, factual diagnostic narratives for anomalies that an automated
detection system has ALREADY found and quantified.

Strict grounding rules:
- The JSON facts provided are the ONLY source of truth. Do not invent numbers,
  causes, MCCs, countries, or codes that are not present in the facts.
- Every figure you state must come from the facts. Quote the actual numbers.
- The detector found the anomaly; your job is to EXPLAIN it, not re-detect it.
- If the facts are insufficient to support a probable cause, say so plainly.
- Decline reason codes map to meanings via the provided glossary; use them.

Write four short sections with these exact headers:
**What happened** — one or two sentences with the key metric move and window.
**Where it's concentrated** — the slices / dimensions carrying the anomaly.
**Probable root cause(s)** — ranked, each tied to specific evidence in the facts.
**Recommended next steps** — 2-4 concrete actions for the analyst/ops team.
Keep it under ~220 words. Plain language, no preamble."""


CHAT_SYSTEM = """You are a payments risk analyst assistant for a card-issuing bank.
Answer the analyst's questions about transaction health using ONLY the JSON
context provided (dataset overview, detected incidents, daily metric tables, and
daily decline-reason counts).

Rules:
- Ground every claim in the context. Quote the specific numbers you used.
- If the answer is not derivable from the context, say "The data I have doesn't
  cover that" and suggest what would be needed — do not guess.
- The detection layer already found the incidents; rely on those facts rather
  than re-deriving anomalies yourself.
- Be concise and specific. Reference incident IDs, dates, codes, and dimensions."""


# --------------------------------------------------------------------------- #
# Incident diagnosis
# --------------------------------------------------------------------------- #
def diagnose_incident(df: pd.DataFrame, inc: Incident,
                      model: str | None = None) -> tuple[str, dict]:
    """Return (narrative, facts). Uses LLM when online, else deterministic text."""
    facts = incident_context(df, inc)
    if llm_client.is_online():
        user = ("Diagnose this anomaly. Facts (JSON):\n\n"
                + json.dumps(facts, indent=2, default=str))
        try:
            narrative = llm_client.complete(
                DIAGNOSIS_SYSTEM, [{"role": "user", "content": user}], model=model)
        except Exception as e:
            # Any LLM error (bad key, rate limit, retired model) -> safe fallback.
            print(f"[LLM fallback · diagnose] {type(e).__name__}: {e}", flush=True)
            narrative = _offline_diagnosis(facts)
    else:
        narrative = _offline_diagnosis(facts)
    return narrative, facts


def _offline_diagnosis(f: dict) -> str:
    sig = f["primary_signal"]
    w = f["window"]
    scope = f.get("scope_label", "global")
    top_decline = f["decline_reason_shift"][0] if f["decline_reason_shift"] else None
    gloss = f.get("code_glossary", {})

    metric_h = {"approval_rate": "Approval rate", "decline_rate": "Decline rate",
                "fraud_rate": "Fraud rate", "txn_count": "Transaction volume"}.get(
                    sig["metric"], sig["metric"])
    move = "fell" if sig["direction"] == "down" else "spiked"
    unit = "%" if sig["unit"] == "percent" else ""

    lines = [
        "**What happened**",
        f"{metric_h} {move} to {sig['observed']}{unit} versus an expected "
        f"~{sig['expected_baseline']}{unit} ({scope}) between {w['start']} and "
        f"{w['end']} ({w['duration_hours']}h), a {sig['robust_z_score']}-sigma move.",
        "",
        "**Where it's concentrated**",
        f"Scope: {scope}. The anomaly was visible across "
        f"{f.get('grains_affected', 1)} metric slice(s)"
        + (" and was corroborated by the Isolation Forest cross-check."
           if f.get("isolation_forest_corroborated") else ".")
    ]

    mw = f["metrics_window"]
    lines += ["", "**Probable root cause(s)**"]
    if top_decline and top_decline["delta_pp"] > 5:
        code = top_decline["decline_reason_code"]
        lines.append(
            f"1. Decline mix shifted sharply toward {code} "
            f"({gloss.get(code, code)}): {top_decline['baseline_share_pct']}% → "
            f"{top_decline['during_share_pct']}% of declines (+{top_decline['delta_pp']}pp). "
            "This is the dominant driver.")
    if sig["metric"] == "fraud_rate":
        lines.append(
            f"1. Fraud rate rose to {mw['during']['fraud_rate_pct']}% during the "
            f"window vs {mw['before']['fraud_rate_pct']}% before — consistent with "
            "a targeted fraud/card-testing attack on this slice.")
    if sig["metric"] == "txn_count":
        lines.append(
            f"1. Volume surged to {mw['during']['txn_count']:,} txns in-window "
            f"vs {mw['before']['txn_count']:,} before — a demand/traffic spike; "
            "check for promotions, bot traffic, or a BIN-attack pattern.")
    lines.append(
        f"2. Metric recovered after the window (approval {mw['after']['approval_rate_pct']}%), "
        "suggesting a transient/operational cause rather than a permanent shift.")

    lines += ["", "**Recommended next steps**"]
    if top_decline and "91_ISSUER" in top_decline["decline_reason_code"]:
        lines += ["- Page the issuer-processor on-call; check auth-platform health for the window.",
                  "- Confirm whether a downstream auth host or network link was degraded."]
    elif sig["metric"] == "fraud_rate":
        lines += ["- Tighten risk rules / step-up auth on the affected slice immediately.",
                  "- Pull the offending BINs/merchants for the window and review for chargebacks."]
    elif "N7_3DS" in (top_decline or {}).get("decline_reason_code", ""):
        lines += ["- Check the 3DS/ACS provider status for the affected country.",
                  "- Consider a temporary 3DS soft-decline fallback to limit approval loss."]
    else:
        lines += ["- Drill into the concentrated slice and confirm with the issuing/ops team.",
                  "- Compare against any deploys, rule changes, or partner incidents in-window."]
    lines.append("- Continue monitoring; alert is auto-resolved once the metric returns to baseline.")
    lines.append("")
    lines.append("_(Generated offline from detector facts — add an Anthropic or Groq key for LLM narratives.)_")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Conversational Q&A
# --------------------------------------------------------------------------- #
def answer_question(df: pd.DataFrame, incidents: list[Incident],
                    question: str, history: list[dict] | None = None,
                    model: str | None = None,
                    _cached_context: dict | None = None) -> str:
    context = _cached_context or build_chat_context(df, incidents)
    if not llm_client.is_online():
        return _offline_answer(question, context, incidents)

    history = history or []
    ctx_msg = ("Context (JSON) — your only source of truth:\n\n"
               + json.dumps(context, default=str))
    messages = ([{"role": "user", "content": ctx_msg},
                 {"role": "assistant",
                  "content": "Understood. I'll answer only from this context."}]
                + history
                + [{"role": "user", "content": question}])
    try:
        return llm_client.complete(CHAT_SYSTEM, messages, model=model)
    except Exception as e:
        # Any LLM error -> fall back to the grounded deterministic answer.
        print(f"[LLM fallback · chat] {type(e).__name__}: {e}", flush=True)
        return _offline_answer(question, context, incidents)


def _offline_answer(question: str, context: dict, incidents: list[Incident]) -> str:
    """Keyword-routed deterministic answers so chat works without a key."""
    q = question.lower()
    incs = context["detected_incidents"]
    ov = context.get("dataset_overview", {})

    def _one(i):
        s = i["primary_signal"]
        drv = (i["decline_reason_shift"][0]["decline_reason_code"]
               if i.get("decline_reason_shift") else None)
        line = (f"{i['incident_id']} [{i['severity']}] {i['title']} — {s['metric']} "
                f"{s['direction']} to {s['observed']} vs ~{s['expected_baseline']} "
                f"expected (z={s['robust_z_score']})")
        return line + (f"; top driver {drv}." if drv else ".")

    # greeting / help
    if q.strip() in ("hi", "hello", "hey", "yo", "hi there", "hello there") \
            or "what can you" in q or q.strip() in ("help", "?"):
        return ("Hi — I'm the analyst assistant for this issuer portfolio. I can summarise "
                f"the {len(incs)} detected incident(s), explain which decline-reason codes "
                "drove an incident, cover fraud activity, or tell you what happened on a given "
                "day. Try: \u201cwhat are the main concerns?\u201d, \u201cwhich codes drove "
                "the biggest incident?\u201d, or \u201cwhy did approvals drop on the worst "
                "day?\u201d")

    # main concerns / risks
    if any(k in q for k in ("concern", "risk", "important", "notable", "worry",
                            "critical", "attention", "priorit")):
        flagged = [i for i in incs if i["severity"] in ("high", "medium")] or incs
        return "The main concerns right now:\n" + "\n".join("- " + _one(i) for i in flagged[:5])

    # fraud
    if "fraud" in q:
        fr = [i for i in incs if "fraud" in (i["primary_signal"]["metric"] + i["title"]).lower()]
        if fr:
            return "Fraud-related activity detected:\n" + "\n".join("- " + _one(i) for i in fr)
        return (f"No incident was driven primarily by fraud; the overall fraud rate is "
                f"{ov.get('overall_fraud_rate_pct', '?')}% of transactions.")

    # approval / decline focus
    if "approv" in q or "declin" in q:
        rel = [i for i in incs if any(w in i["primary_signal"]["metric"].lower()
                                      for w in ("approval", "decline"))]
        if rel and "reason" not in q and "code" not in q:
            return "Approval / decline incidents:\n" + "\n".join("- " + _one(i) for i in rel[:5])

    # worst / biggest / most severe
    if any(w in q for w in ("worst", "biggest", "largest", "most severe", "severe")) and incs:
        t = max(incs, key=lambda i: i["primary_signal"]["robust_z_score"])
        return "Most severe incident: " + _one(t)

    # overall health / status
    if any(w in q for w in ("overview", "overall", "status", "health", "how is", "portfolio")):
        tx = ov.get("total_transactions")
        txs = f"{tx:,}" if isinstance(tx, int) else "?"
        dr = ov.get("date_range", {})
        return (f"Across {str(dr.get('start',''))[:10]} → {str(dr.get('end',''))[:10]}: "
                f"{txs} transactions, {ov.get('overall_approval_rate_pct','?')}% approved, "
                f"{ov.get('overall_fraud_rate_pct','?')}% fraud. "
                f"{len(incs)} incident(s) detected.")

    if any(k in q for k in ("how many", "list", "what anomal", "incidents", "summary")):
        out = [f"Detected {len(incs)} incident(s):"]
        for i in incs:
            out.append(f"- {i['incident_id']} [{i['severity']}] {i['title']} "
                       f"({i['window']['start']} → {i['window']['end']})")
        return "\n".join(out)

    if "decline reason" in q or "reason code" in q:
        target = incs[0]
        if any(w in q for w in ("worst", "biggest", "largest", "top", "spike")):
            target = max(incs, key=lambda i: i["primary_signal"]["robust_z_score"])
        for i in incs:
            if any(t and t in q for t in (i["scope"].get("country", "").lower(),
                                          i["window"]["start"][:10])):
                target = i
        rows = target["decline_reason_shift"][:3]
        out = [f"For {target['incident_id']} ({target['title']}), the decline mix shifted toward:"]
        for r in rows:
            out.append(f"- {r['decline_reason_code']}: "
                       f"{r['baseline_share_pct']}% → {r['during_share_pct']}% "
                       f"({r['delta_pp']:+}pp)")
        return "\n".join(out)

    # date-based ("why did X drop on <date>")
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", q) or re.search(r"(jun|may|apr)\w*\s*\d{1,2}", q)
    if m:
        frag = m.group(1) if m.lastindex else m.group(0)
        for i in incs:
            if i["window"]["start"].startswith(frag) or frag in i["window"]["start"]:
                s = i["primary_signal"]
                return (f"{i['incident_id']} — {i['title']}. {s['metric']} {s['direction']} "
                        f"to {s['observed']} vs ~{s['expected_baseline']} expected "
                        f"(z={s['robust_z_score']}). Top driver: "
                        f"{(i['decline_reason_shift'][0]['decline_reason_code'] if i['decline_reason_shift'] else 'n/a')}.")

    # default — still useful and grounded, never a dead end
    if incs:
        top = sorted(incs, key=lambda i: i["primary_signal"]["robust_z_score"],
                     reverse=True)[:3]
        body = "\n".join("- " + _one(i) for i in top)
        return ("Here's what stands out from the detector outputs:\n" + body
                + f"\n\n{len(incs)} incident(s) in total. You can ask about a specific "
                "incident, its decline-reason drivers, fraud activity, or what happened on "
                "a particular day.")
    return ("No anomalies were detected in the current data. You can ask about the overall "
            "approval, decline, or fraud rates, or regenerate the data to explore other "
            "scenarios.")
