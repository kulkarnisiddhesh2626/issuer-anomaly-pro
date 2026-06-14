# Design Write-up — Anomaly Detection & Diagnostic Assistant

**Scope:** a working POC for an issuer-side risk/ops team — detect anomalies in card
transaction health metrics and explain them in plain language, grounded in the detector's
own output rather than the LLM's imagination.

Click: https://issuer-anomaly-pro-1.onrender.com/

---

## 1. Approach & problem framing

Issuer analysts watch a handful of **health metrics** — approval rate, decline rate, fraud
rate, transaction volume — sliced by **MCC, country, channel, auth type, and
card-present**. Problems show up as a metric moving in a *specific slice* (cross-border
e-com approvals drop; fraud concentrates in one MCC; a decline code spikes). The pain is
that finding the slice is slow manual work.

So I split the system the way the brief asks, and kept a hard wall between the halves:

- **Detection layer** answers *"what moved, where, and how much?"* — pure, explainable
  statistics. Every alert is a concrete number with a before/during/after context.
- **GenAI layer** answers *"what does this mean and what do I do?"* — it consumes only the
  detector's structured output and narrates it. **It never sees raw rows and never
  detects.**

This separation is also the foundation of hallucination control: if the model is only ever
handed verified numbers and told they are the sole source of truth, it has nothing to make
up.

The work is surfaced through a product-style single-page web app (FastAPI + a custom
Tailwind/ECharts frontend) with an app shell — a sidebar, a top bar, and a bento dashboard:
**Overview** (portfolio health + detection pipeline), **Incidents** (grounded diagnosis +
the exact fact sheet the model was given), **Assistant** (grounded chat), and **How it
works** (method in-app). The detection + GenAI engine is framework-independent and is reused
unchanged from the earlier prototypes; only the presentation changed — here the engine is
exposed as a small JSON API.

---

## 2. Architecture

```
┌──────────────────┐   45d hourly aggregates (~275k rows)   ┌──────────────────────┐
│ data_generator   │ ─────────────────────────────────────▶ │ transactions.csv      │
│  + 5 injected     │   ground truth → injected_anomalies.json│ (one row = one        │
│    anomalies      │                                         │  outcome bucket/hour) │
└──────────────────┘                                         └──────────┬───────────┘
                                                                         │
                              ┌──────────────────────────────────────────▼─────────────┐
                              │ DETECTION LAYER  (detection.py)                          │
                              │  • seasonal two-proportion z-test (rate metrics)         │
                              │  • Poisson deviation (volume)                            │
                              │  • effect-size floors (kill trivial-but-significant)     │
                              │  • Isolation Forest = corroboration only                 │
                              │  → merge hours into EVENTS → consolidate into INCIDENTS  │
                              └──────────────────────────────────────┬───────────────────┘
                                                                      │ incidents (numbers)
                              ┌────────────────────────────────────────▼─────────────────┐
                              │ CONTEXT BUILDER (context_builder.py)                       │
                              │  incident → compact FACT SHEET (JSON):                     │
                              │  window · metric before/during/after · decline-mix shift · │
                              │  dimension share-shift · code glossary · scope label       │
                              └────────────────────────────────────────┬─────────────────┘
                                                                        │ facts only (no raw rows)
                              ┌──────────────────────────────────────────▼───────────────┐
                              │ GENAI LAYER (diagnosis.py + llm_client.py)                 │
                              │  • diagnose_incident() → 4-section narrative               │
                              │  • answer_question()   → grounded chat Q&A                 │
                              │  • online = Anthropic Messages API (temp 0.2)              │
                              │  • offline = deterministic templates over the same facts   │
                              └──────────────────────────────────────────┬───────────────┘
                                                                          │
                                                            ┌─────────────▼─────────────┐
                                                            │ Streamlit UI (app.py)      │
                                                            │ Overview · Incidents ·     │
                                                            │ Ask-the-data chat          │
                                                            └────────────────────────────┘
```

---

## 3. Detection method & why

**Choice: seasonal, volume-aware statistical tests, with an Isolation Forest as a
cross-check only.** Rationale, mapped to the evaluation criteria:

**Why not a plain z-score / MAD on the raw series?** I tried it; it is a false-positive
machine on issuer data. Fraud-rate and decline-rate buckets are *sparse* — a single fraud
in a low-volume hour is a huge relative swing but operationally meaningless. A naive z-score
fired thousands of alerts. The fix is to make the test **volume-aware**.

- **Rate metrics → two-proportion z-test vs. a pooled seasonal baseline.**
  For approval/decline/fraud rate I compare the observed proportion *p* in a bucket against
  the pooled baseline *p₀* for the *same weekday × hour*, using
  `z = (p − p₀) / sqrt(p₀(1 − p₀)/n)`. The `n` in the denominator means low-volume noise
  scores low automatically — exactly the low-false-positive behaviour we want — while a
  genuine shift over thousands of transactions scores high.
- **Volume → Poisson deviation.** Counts are tested as `z = (x − μ)/sqrt(μ)` against the
  seasonal mean μ, the natural model for event counts.
- **Seasonality** is handled by building baselines per **weekday × hour** slice, so the
  detector compares Tuesday-9am to other Tuesday-9ams, not to 3am. (This is an STL-style
  seasonal decomposition done with grouped robust statistics; I avoided a hard `statsmodels`
  dependency for the POC, but the spirit — strip seasonality, test the residual — is the
  same.)
- **Effect-size floors** sit on top of significance: a rate has to move by a minimum
  absolute amount (~3pp) and a fraud incident needs a minimum *absolute* fraud count and a
  multiple of baseline. This is the standard "statistically significant ≠ worth paging
  someone" guard.
- **Isolation Forest** runs over the multivariate slice features as a **corroborating**
  signal. It can raise confidence on an already-flagged incident, but it can **never be the
  sole trigger**, because a black-box outlier score with no explanation defeats the
  purpose. This keeps the whole detector explainable.

**From points to incidents.** Contiguous flagged hours merge into **events**; time
overlapping events across slices consolidate into **incidents**, so the analyst sees "one
issuer outage", not 200 hourly alerts. On the synthetic set this yields **100% recall on
all 5 injected anomalies** (peak z from 6.3 for the subtle fraud attack up to 61 for the
outage).

---

## 4. LLM design choices

**Prompting strategy.** The diagnosis prompt pins the model into a fixed role (issuer risk
analyst) and a fixed four-section output: *What happened / Where it's concentrated /
Probable root cause(s) / Recommended next steps*. Structure makes the output skimmable for
an on-call analyst and makes the model's job extraction-and-explanation rather than
open-ended generation. Temperature is **0.2** — we want repeatable, grounded diagnoses, not
creativity.

**How the model is grounded in data.** This is the heart of the design. The LLM **never
receives raw transactions.** The context builder turns each incident into a compact
**fact sheet**: the metric's before/during/after values, the decline-reason mix shift, the
dimension share-shift that localises the anomaly, a glossary mapping codes to meanings, and
a human scope label. The model's only job is to narrate numbers that are already true. For
chat, the same principle applies: questions are answered over a bundle of pre-computed
aggregates (dataset overview, incident list, daily metric tables, daily decline-reason
counts) — structured context, not a data dump.

**Hallucination control** is layered:

1. **Architectural** — the model can't hallucinate a number it was never in a position to
   read; it only gets verified facts.
2. **Instructional** — the system prompt names the JSON facts as the *only* source of
   truth, forbids inventing numbers/codes/causes, and **requires the model to say "the data
   I have doesn't cover that"** instead of guessing.
3. **Parametric** — low temperature.
4. **Fallback as proof-of-concept** — with no API key the app uses a deterministic,
   template-driven generator over the *same* fact sheet. It is both a zero-dependency demo
   path and a concrete illustration of the grounding principle: prose derived strictly from
   verified numbers. The online path is held to that same standard, just with better
   language.

**Provider independence.** All model I/O is isolated in `llm_client.py`, which ships with
two interchangeable providers — **Anthropic** (Claude) and **Groq** (open models such as
Llama 3.3 70B) — selected by a single environment variable; adding OpenAI/Gemini/local is a
one-file change. Because the model only narrates pre-computed facts, a fast open model on
Groq's free tier gives comparable results to a frontier model here, which makes the app
cheap to share widely. Model and parameters are environment-configurable.

---

## 5. What I'd improve with more time

- **Scope-aware incident clustering.** Today consolidation is primarily temporal; a richer
  version would cluster on shared dimensions so two unrelated same-hour events in different
  corridors don't merge.
- **Tool-calling / SQL-backed chat (RAG over the warehouse).** Instead of pre-bundling
  aggregates, let the chat layer issue *constrained, read-only* aggregate queries on demand,
  so analysts can ask arbitrary slice questions while staying grounded.
- **Proper seasonal models.** Swap the grouped-robust baseline for STL or Prophet to handle
  trend, holidays, and multiple seasonalities; add change-point detection for slow drifts.
- **An evaluation harness.** Sweep injected-anomaly scenarios and report precision/recall
  and detection latency vs. detector settings, plus an automated grounding check that every
  number the LLM emits exists in the fact sheet.
- **Streaming / alerting.** Move from batch CSV to a windowed online detector with
  paging/Slack integration and severity-based routing.
- **Richer fact sheets.** Add baseline confidence intervals and co-movement across metrics
  so the narrative can reason about correlation (e.g. fraud-rate up *and* approval-rate
  down ⇒ tightened risk rules).

---

*POC philosophy: a focused, working end-to-end flow — data → explainable detection →
grounded LLM diagnosis → conversational Q&A — over a breadth of half-built features.*
