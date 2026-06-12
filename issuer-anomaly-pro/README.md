# Issuer Anomaly Console — web app (FastAPI + custom UI)

A proof-of-concept **Transaction Anomaly Detection & Diagnostic Assistant** for a
card-issuing bank, presented as a product-style single-page web app. It detects anomalies
in issuer transaction-health metrics with an explainable statistical detector, then uses an
**LLM to turn the detector's output into a plain-language diagnosis** plus a grounded
conversational assistant.

This is the same two-layer engine as the earlier prototypes (`src/` is reused unchanged),
now behind a small **FastAPI** JSON API with a custom **Tailwind + ECharts** frontend — a
real app shell (sidebar, top bar, bento dashboard) instead of a framework's stacked layout.
The detector finds anomalies; **the LLM only explains them and never sees raw transactions.**

```
synthetic data  →  detection layer  →  fact sheet  →  LLM diagnosis  →  chat assistant
        (src/, framework-independent)            (served as a JSON API by server.py)
```

---

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

Open <http://127.0.0.1:8000>. On first run it generates the synthetic dataset automatically.

### Optional: live LLM

Runs fully offline (deterministic templates) with no key. To enable real narratives, copy
`.env.example` to `.env` and set a key for one provider:
- **Anthropic**: `ANTHROPIC_API_KEY=sk-ant-...`
- **Groq**: `ANOMALY_LLM_PROVIDER=groq` + `GROQ_API_KEY=...` (free, fast — good for a shared link).

The sidebar shows the active provider, or "offline · templates".

---

## Deploy a public link

**Option A — Render (connects to your GitHub repo):**
1. Push this project to a GitHub repo.
2. At <https://render.com> → **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt`
   Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. (Optional) add `ANTHROPIC_API_KEY` or `GROQ_API_KEY` as an environment variable.
5. Deploy → Render gives a public `https://…onrender.com` URL.

**Option B — Hugging Face Spaces (Docker):**
1. New Space → SDK **Docker**. The included `Dockerfile` serves on port 7860.
2. Upload the project (or connect the repo).
3. Optional: add the API key under **Settings → Secrets**.

---

## What the app shows

- **Overview** — KPI tiles, a detection-pipeline panel, an interactive ECharts health chart
  with detected incidents shaded, and recent incidents, in a bento grid.
- **Incidents** — a selectable incident list with a detail panel: grounded diagnosis, a
  decline-reason mix-shift chart, and the exact **fact sheet** the LLM was given.
- **Assistant** — a chat (with suggested questions) grounded only in detector outputs.
- **How it works** — method, design choices, and a mapping to the assessment criteria.

---

## Layout

```
issuer-anomaly-pro/
├── server.py              # FastAPI: serves the UI + JSON API over the engine
├── static/index.html      # single-page frontend (Tailwind + ECharts + vanilla JS)
├── requirements.txt
├── Dockerfile             # for Hugging Face Spaces (Docker SDK)
├── .env.example           # copy to .env for the live LLM (Anthropic or Groq)
├── README.md
├── WRITEUP.md             # design write-up + architecture
├── data/                  # generated CSV + ground truth (git-ignored, auto-created)
└── src/                   # detection + GenAI engine (unchanged across prototypes)
    ├── config.py · data_generator.py · detection.py
    ├── context_builder.py · llm_client.py · diagnosis.py
```

The detector is seasonal and volume-aware (two-proportion z-test on rates vs a weekday×hour
baseline; Poisson on volume; effect-size floors; Isolation-Forest cross-check) and recovers
all five injected anomalies. See `WRITEUP.md` for the full rationale.
