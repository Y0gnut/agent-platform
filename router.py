"""
router.py - Local Model Routing Gateway
========================================
Exposes a /chat/completions endpoint that:
1. Receives chat completion requests.
2. Evaluates query complexity using a deterministic heuristic.
3. Routes simple queries to small-model and complex queries to large-model.
4. Falls back to small-model upon timeout or upstream failure.
5. Emits structured routing telemetry and traces via Langfuse.
"""

import logging
import os
import pathlib
import re
import sys
import time

from fastapi import FastAPI, HTTPException, Request
import httpx
from pydantic import BaseModel

# Langfuse observability integration (optional; runs in no-op mode if unconfigured)
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
from gateway.langfuse_client import flush, get_langfuse, start_trace, timed_span

_lf = get_langfuse()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("router")

# Upstream proxy configuration
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-local-dev")

# Model aliases mapping to litellm_config.yaml definitions
SMALL_MODEL = "small-model"
LARGE_MODEL = "large-model"

# Execution timeout for the large model. Defaults to 45s to accommodate
# initial weights cold-loading on local host hardware.
LARGE_MODEL_TIMEOUT = float(os.getenv("LARGE_MODEL_TIMEOUT", "45"))

# Complexity heuristics
COMPLEXITY_KEYWORDS = {
    "explain", "compare", "analyze", "analyse", "evaluate", "summarize",
    "summarise", "contrast", "discuss", "elaborate", "critique", "assess",
    "differentiate", "synthesize", "synthesise", "justify", "outline",
}
LONG_PROMPT_WORD_THRESHOLD = 30
MULTI_QUESTION_THRESHOLD = 2


def classify_prompt(prompt: str) -> str:
    """
    Classify a prompt as 'complex' or 'simple' via deterministic heuristics.

    Heuristics evaluate keyword presence, word density, and multi-question counts.
    A rule-based classifier avoids introducing additional model inference latency
    and dependencies in the critical routing path.
    """
    lower = prompt.lower()
    words = set(re.findall(r"\b\w+\b", lower))
    keyword_hit = bool(words & COMPLEXITY_KEYWORDS)

    word_count = len(prompt.split())
    length_hit = word_count >= LONG_PROMPT_WORD_THRESHOLD

    question_count = prompt.count("?")
    multi_question_hit = question_count >= MULTI_QUESTION_THRESHOLD

    is_complex = keyword_hit or length_hit or multi_question_hit

    logger.info(
        "Classifier decision: words=%d keyword=%s length=%s multi_q=%s -> %s",
        word_count,
        keyword_hit,
        length_hit,
        multi_question_hit,
        "complex" if is_complex else "simple",
    )
    return "complex" if is_complex else "simple"


async def call_litellm(
    prompt: str,
    model: str,
    timeout: float,
    base_url: str = LITELLM_BASE_URL,
) -> str:
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


@app.get("/health")
async def health_check():
    """Health check endpoint for liveness probes."""
    return {"status": "ok", "service": "gateway"}


@app.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest, http_request: Request = None) -> ChatResponse:
    """
    Main model routing endpoint.

    Execution Flow:
    1. Heuristically classify query complexity.
    2. Dispatch complex requests to LARGE_MODEL.
       - Upon timeout, HTTP status failure, or connection error, fall back to SMALL_MODEL.
    3. Dispatch simple requests directly to SMALL_MODEL.
    4. Emit tracing telemetry and return response payload with routing metadata.
    """
    start_time = time.monotonic()
    fell_back = False
    model_used = SMALL_MODEL

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

    with timed_span(lf_trace, "classify_prompt", input_data=request.prompt[:200]) as cls_span:
        cls_span.update(metadata={"complexity": complexity})

    if complexity == "complex":
        target_base_url = (
            "http://localhost:9999"
            if request.simulate_large_failure
            else LITELLM_BASE_URL
        )
        if request.simulate_large_failure:
            logger.info("Routing to large model with simulated failure target")
        else:
            logger.info("Routing to large model (complexity=complex)")

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
            # Fallback path: Upon large model timeout or failure, degrade to small-model
            # rather than returning an immediate 5xx error to the client.
            logger.warning(
                "Fallback triggered: %s (%s). Retrying with %s.",
                type(exc).__name__,
                exc,
                SMALL_MODEL,
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
                logger.error("Primary and fallback models failed: %s", fallback_exc)
                raise HTTPException(
                    status_code=503,
                    detail="Both large and small models are unavailable.",
                ) from fallback_exc

    else:
        logger.info("Routing to small model (complexity=simple)")
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
            logger.error("Small model invocation failed for simple query: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"Small model unavailable: {exc}",
            ) from exc

    latency_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "Request complete: model=%s fell_back=%s latency=%.1f ms",
        model_used,
        fell_back,
        latency_ms,
    )

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
