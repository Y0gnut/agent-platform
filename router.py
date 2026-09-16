"""
router.py – Local LLM Gateway (Week 1 Portfolio Project)
=========================================================
This module is the heart of the gateway. It:
  1. Receives a prompt via a POST /chat/completions endpoint.
  2. Classifies the prompt as "simple" or "complex" using a rule-based heuristic.
  3. Routes simple prompts to the small model and complex prompts to the large model.
  4. Falls back to the small model if the large model times out or errors.
  5. Returns a structured JSON response with routing metadata.

Design philosophy
-----------------
Clarity > cleverness. Every non-obvious decision has a comment explaining WHY,
not just WHAT, so you can walk through the logic in an interview without notes.
"""

import os
import time
import logging
import re
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# ── Langfuse observability (optional — degrades gracefully if not configured) ──
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
from gateway.langfuse_client import flush, get_langfuse, start_trace, timed_span
_lf = get_langfuse()  # None if LANGFUSE_PUBLIC_KEY not set

# ── Logging ──────────────────────────────────────────────────────────────────
# Structured logging is critical in a gateway: it gives you an audit trail of
# every routing and fallback decision without requiring a full APM solution.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("router")

# ── Configuration ─────────────────────────────────────────────────────────────
# Centralised constants make the system easy to tune without hunting through code.

LITELLM_BASE_URL = "http://localhost:4000"  # LiteLLM proxy address
LITELLM_API_KEY = "sk-local-dev"           # Must match docker-compose env var

# Model aliases – these match the model_name values in litellm_config.yaml.
# Using aliases means we never hard-code "llama3.1:8b" in routing logic;
# if we upgrade the model, we change litellm_config.yaml in one place.
SMALL_MODEL = "small-model"
LARGE_MODEL = "large-model"

# How long (seconds) to wait for the large model before declaring failure.
# Configurable via environment variable; defaults to 45s to allow for local
# hardware cold-starts (loading 8B model into memory on consumer machines).
LARGE_MODEL_TIMEOUT = float(os.getenv("LARGE_MODEL_TIMEOUT", "45"))

# ── Complexity classifier ──────────────────────────────────────────────────────
# Heuristic signals used to decide "simple" vs "complex".
# These are deliberately simple and readable; see README for honest limitations.

# Keywords that suggest the user wants multi-step reasoning or comparison.
# Single-hop factual lookups rarely use these words.
COMPLEXITY_KEYWORDS = {
    "explain", "compare", "analyze", "analyse", "evaluate", "summarize",
    "summarise", "contrast", "discuss", "elaborate", "critique", "assess",
    "differentiate", "synthesize", "synthesise", "justify", "outline",
}

# If the prompt is longer than this many words, it likely carries enough context
# to warrant a more capable model. Threshold chosen empirically: short questions
# ("What is Python?") are almost always under 20 words; nuanced ones are usually
# longer.
LONG_PROMPT_WORD_THRESHOLD = 30

# A prompt that poses multiple questions at once is complex because the model
# must track several independent sub-tasks simultaneously.
MULTI_QUESTION_THRESHOLD = 2  # ≥ this many "?" → complex


def classify_prompt(prompt: str) -> str:
    """
    Return "complex" or "simple" based on heuristic analysis of the prompt.

    Why a heuristic instead of a trained classifier?
    -------------------------------------------------
    A trained classifier (e.g. a fine-tuned BERT on query intent data) would
    generalise better, but it adds a third model dependency, a training
    pipeline, and ongoing maintenance.  For a routing gateway that sits in
    front of already-capable LLMs, a fast, interpretable heuristic is a
    reasonable first step.  The README section "Routing Strategy" discusses
    this trade-off honestly.

    Parameters
    ----------
    prompt : str
        The raw user prompt text.

    Returns
    -------
    str
        "complex" or "simple"
    """
    # Normalise for comparison (lowercase, but keep punctuation for "?" count)
    lower = prompt.lower()

    # Signal 1 – complexity keyword present?
    # Using set intersection for O(k) lookup where k = keyword count.
    words = set(re.findall(r"\b\w+\b", lower))
    keyword_hit = bool(words & COMPLEXITY_KEYWORDS)

    # Signal 2 – prompt is long?
    # Word count is a crude but cheap proxy for information density.
    word_count = len(prompt.split())
    length_hit = word_count >= LONG_PROMPT_WORD_THRESHOLD

    # Signal 3 – multiple questions?
    # Counting "?" handles "What is X? How does Y work?" style prompts.
    question_count = prompt.count("?")
    multi_question_hit = question_count >= MULTI_QUESTION_THRESHOLD

    # Decision: any single signal is sufficient to route to the large model.
    # This errs on the side of quality (over-routing to large is safer than
    # under-routing to small for a genuinely hard question).
    is_complex = keyword_hit or length_hit or multi_question_hit

    logger.info(
        "Classifier | words=%d keyword=%s length=%s multi_q=%s → %s",
        word_count,
        keyword_hit,
        length_hit,
        multi_question_hit,
        "complex" if is_complex else "simple",
    )
    return "complex" if is_complex else "simple"


# ── LiteLLM caller ────────────────────────────────────────────────────────────

async def call_litellm(prompt: str, model: str, timeout: float, base_url: str = LITELLM_BASE_URL) -> str:
    """
    Send a prompt to LiteLLM and return the assistant's reply text.

    We use the OpenAI /chat/completions format because LiteLLM normalises
    every backend (Ollama, OpenAI, Anthropic, ...) to this single interface.
    That means this function never changes even if we swap the underlying model.

    Parameters
    ----------
    prompt   : str   - User message text
    model    : str   - LiteLLM model alias (e.g. "small-model")
    timeout  : float - Max seconds to wait before raising httpx.TimeoutException
    base_url : str   - Base URL for LiteLLM (overridable for simulating failure)

    Returns
    -------
    str - The model's reply text

    Raises
    ------
    httpx.TimeoutException  - if the model doesn't respond within `timeout`
    httpx.HTTPStatusError   - if LiteLLM returns a 4xx/5xx status
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    # httpx is used instead of the openai SDK because it gives us fine-grained
    # timeout control and doesn't require a real API key object to be set up.
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
        timeout=timeout,
    ) as client:
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()  # surface 4xx/5xx as exceptions

    data = response.json()
    # Navigate the OpenAI response schema: choices[0].message.content
    return data["choices"][0]["message"]["content"]


# -- FastAPI app ---------------------------------------------------------------

app = FastAPI(
    title="Local LLM Gateway",
    description=(
        "Routes prompts between llama3.2:3b (small) and llama3.1:8b (large) "
        "based on query complexity, with automatic fallback."
    ),
    version="1.0.0",
)


class ChatRequest(BaseModel):
    """Schema for the incoming request body."""
    prompt: str
    simulate_large_failure: bool = False  # Allows simulating failure (e.g. wrong port) for testing


class ChatResponse(BaseModel):
    """Schema for the structured response we return to callers."""
    response: str
    model_used: str
    fell_back: bool
    latency_ms: float


@app.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest, http_request: Request = None) -> ChatResponse:
    """
    Main routing endpoint.

    Flow
    ----
    1. Classify the prompt -> simple or complex
    2. If complex -> try large model (with timeout)
       2a. On timeout or error -> fall back to small model (log clearly)
    3. If simple -> go straight to small model
    4. Return response + routing metadata

    Why expose routing metadata (model_used, fell_back, latency_ms)?
    ---------------------------------------------------------------
    Observability is a first-class concern in any gateway. Callers and
    operators need to know which model actually answered and whether the
    system degraded gracefully. Without this metadata the system is a black
    box - hard to debug, impossible to tune.
    """
    start_time = time.monotonic()
    fell_back = False
    model_used = SMALL_MODEL  # default; overwritten below

    # ── Langfuse: create or continue trace ────────────────────────────────────
    # mcp_client passes a trace ID via header so all router spans attach to
    # the same trace as the outer orchestration call.
    incoming_trace_id = None
    if http_request is not None:
        incoming_trace_id = http_request.headers.get("x-langfuse-trace-id") or None
    lf_trace = start_trace(
        _lf,
        name="router:chat_completions",
        user_query=request.prompt[:500],
        trace_id=incoming_trace_id,
    )

    complexity = classify_prompt(request.prompt)

    # Record routing decision as a span
    with timed_span(lf_trace, "classify_prompt", input_data=request.prompt[:200]) as cls_span:
        cls_span.update(metadata={"complexity": complexity})

    if complexity == "complex":
        # Attempt the large model first.
        target_base_url = "http://localhost:9999" if request.simulate_large_failure else LITELLM_BASE_URL
        if request.simulate_large_failure:
            logger.info("Routing to LARGE model (simulating failure via wrong port 9999)")
        else:
            logger.info("Routing to LARGE model (complexity=complex)")

        try:
            with timed_span(lf_trace, "litellm:large_model", input_data=request.prompt[:200]) as span:
                reply = await call_litellm(
                    prompt=request.prompt,
                    model=LARGE_MODEL,
                    timeout=LARGE_MODEL_TIMEOUT,
                    base_url=target_base_url,
                )
                model_used = LARGE_MODEL
                span.update(
                    output=reply[:500],
                    metadata={
                        "model": LARGE_MODEL,
                        "estimated_tokens": len(reply.split()),
                        "response_length_chars": len(reply),
                        "simulate_failure": request.simulate_large_failure,
                    },
                )

        except (httpx.TimeoutException, httpx.HTTPStatusError, Exception) as exc:
            # -- FALLBACK ------------------------------------------------------
            # The large model failed (timeout, crash, wrong port, etc.).
            # Rather than returning an error to the user, we degrade gracefully
            # by retrying with the small model, which is more resilient.
            #
            # Why not retry the large model?
            #   If it timed out once, it's likely still slow/unavailable.
            #   Retrying would just double the user's wait time.
            #   The small model can answer most questions adequately.
            #
            # IMPORTANT: log clearly so operators can detect degraded mode.
            logger.warning(
                "[FALLBACK] triggered | Large model failed: %s: %s | "
                "Retrying with small model.",
                type(exc).__name__,
                exc,
            )
            fell_back = True
            lf_trace.span(name="fallback_triggered").update(
                input=str(exc)[:300],
                metadata={"fallback_reason": type(exc).__name__},
            )
            try:
                with timed_span(lf_trace, "litellm:small_model_fallback", input_data=request.prompt[:200]) as span:
                    reply = await call_litellm(
                        prompt=request.prompt,
                        model=SMALL_MODEL,
                        timeout=30,
                        base_url=LITELLM_BASE_URL,
                    )
                    model_used = SMALL_MODEL
                    span.update(
                        output=reply[:500],
                        metadata={"model": SMALL_MODEL, "fell_back": True},
                    )
            except Exception as fallback_exc:
                # Both models failed - surface the error rather than silently
                # swallowing it. A 503 tells the caller to retry later.
                logger.error("[ERROR] Both models failed. Small model error: %s", fallback_exc)
                raise HTTPException(
                    status_code=503,
                    detail="Both large and small models are unavailable.",
                ) from fallback_exc

    else:
        # Simple prompt -> go directly to small model.
        logger.info("Routing to SMALL model (complexity=simple)")
        try:
            with timed_span(lf_trace, "litellm:small_model", input_data=request.prompt[:200]) as span:
                reply = await call_litellm(
                    prompt=request.prompt,
                    model=SMALL_MODEL,
                    timeout=30,
                    base_url=LITELLM_BASE_URL,
                )
                model_used = SMALL_MODEL
                span.update(
                    output=reply[:500],
                    metadata={
                        "model": SMALL_MODEL,
                        "estimated_tokens": len(reply.split()),
                        "response_length_chars": len(reply),
                    },
                )
        except Exception as exc:
            logger.error("Small model failed for simple query: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"Small model unavailable: {exc}",
            ) from exc

    latency_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "[DONE] model=%s fell_back=%s latency=%.0f ms",
        model_used,
        fell_back,
        latency_ms,
    )

    # ── Langfuse: finalise trace ───────────────────────────────────────────────
    lf_trace.update(
        output=reply[:500],
        metadata={
            "model_used": model_used,
            "fell_back": fell_back,
            "complexity": complexity,
            "latency_ms": round(latency_ms, 1),
            "response_length_chars": len(reply),
            "estimated_output_tokens": len(reply.split()),
        },
    )
    flush(_lf)

    return ChatResponse(
        response=reply,
        model_used=model_used,
        fell_back=fell_back,
        latency_ms=round(latency_ms, 2),
    )


# ── Entry point ───────────────────────────────────────────────────────────────
# Run with: uvicorn router:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
