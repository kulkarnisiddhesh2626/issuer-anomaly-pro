"""Thin, pluggable LLM client.

Providers (set ``ANOMALY_LLM_PROVIDER``):
* **anthropic** (default) — Claude via the Anthropic Messages API.
* **groq** — open models (e.g. Llama 3.3 70B) via Groq's OpenAI-compatible
  Chat Completions API. Fast and free-tier friendly; a good choice when the
  app's link is shared widely, since the LLM only narrates pre-computed facts.

Modes:
* **Online** — a real API call is made when the active provider's key (and SDK)
  is present.
* **Offline** — a deterministic, template-based generator (see diagnosis.py) is
  used when no key is present, so the whole POC runs end-to-end for grading and
  never hard-fails during a demo. The offline path only restates detector facts,
  so it also illustrates the hallucination-control principle: prose is derived
  strictly from verified numbers.

Adding another provider (OpenAI, Gemini, local) means editing only this file.
"""
from __future__ import annotations

from . import config


def is_online() -> bool:
    """True when a real LLM call can be made with the active provider."""
    if config.LLM_PROVIDER == "groq":
        if not config.GROQ_API_KEY:
            return False
        try:
            import groq  # noqa: F401
            return True
        except ImportError:
            return False
    # default: anthropic
    if not config.ANTHROPIC_API_KEY:
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def provider_label() -> str:
    """Human-readable 'provider · model' for status display."""
    return f"{config.LLM_PROVIDER} · {config.DEFAULT_MODEL}"


def complete(system: str, messages: list[dict],
             model: str | None = None,
             max_tokens: int | None = None,
             temperature: float | None = None) -> str:
    """Call the active provider's chat API and return the text content.

    `messages` is a list of {"role": "user"|"assistant", "content": str}.
    Raises RuntimeError if called while offline (callers should check is_online).
    """
    if not is_online():
        raise RuntimeError("LLM offline: no API key / SDK for the active provider.")

    temp = config.LLM_TEMPERATURE if temperature is None else temperature
    max_tok = max_tokens or config.LLM_MAX_TOKENS

    if config.LLM_PROVIDER == "groq":
        import groq
        client = groq.Groq(api_key=config.GROQ_API_KEY)
        # Groq is OpenAI-compatible: the system prompt is the first message.
        resp = client.chat.completions.create(
            model=model or config.GROQ_MODEL,
            max_tokens=max_tok,
            temperature=temp,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return (resp.choices[0].message.content or "").strip()

    # default: anthropic
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=model or config.ANTHROPIC_MODEL,
        max_tokens=max_tok,
        temperature=temp,
        system=system,
        messages=messages,
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()
