# ADR 005 — Langfuse Self-Hosted for Observability

**Date:** 2026-09  
**Status:** Accepted  
**Deciders:** Project author

---

## Context

The production-ready version of this platform needs observability: the ability to trace individual requests end-to-end, see which model was used, detect fallbacks, measure latency, and understand tool usage patterns without reading log files.

Options considered:

| Option | Cost | Setup | Persistent? | LLM-aware? |
|---|---|---|---|---|
| **Print/logging** | Free | Zero | No | No |
| **Prometheus + Grafana** | Free | Medium | Yes (in Docker) | No — metrics only |
| **Langfuse cloud** | Free tier (limited) | Minimal | Yes (SaaS) | Yes |
| **Langfuse self-hosted OSS** | Free | Medium (Docker) | Yes | Yes |
| **Weights & Biases** | Paid for teams | Minimal | Yes (SaaS) | Partial |
| **OpenTelemetry + Jaeger** | Free | High | Yes | No |

---

## Decision

Use **Langfuse self-hosted OSS (v3)** via Docker Compose.

Langfuse is purpose-built for LLM observability: it natively understands the concept of traces with nested spans (routing decision → tool call → LLM answer), captures inputs/outputs/latency at each span, and renders a timeline view of the full request flow. The self-hosted version is MIT-licensed, runs entirely locally (Postgres + worker + web server), and requires no API key or account with an external service.

Instrumentation is via the official Python SDK (`langfuse>=2.0.0`), wrapped in a thin no-op fallback layer (`gateway/langfuse_client.py`) so the system works identically whether or not Langfuse is running.

---

## Consequences

**Pros:**
- **Purpose-built for LLMs**: span-level trace view maps directly to our orchestration flow (routing → MCP tool → final answer), which Prometheus/Jaeger timelines do not represent well.
- **Free and local**: no SaaS account, no data leaving the machine, no cloud spend.
- **No-op fallback**: if Langfuse is not running, the gateway degrades gracefully — no `None` pointer errors, no changed behavior.
- **Trace linking**: mcp_client creates the root trace, passes the trace ID to the gateway via `X-Langfuse-Trace-Id` header, so router spans attach to the same trace. One request → one coherent timeline.

**Cons / Honest Limitations:**
- **Docker overhead**: adds three more containers (Postgres, langfuse-server, langfuse-worker). On a laptop, `docker compose up` takes longer and uses additional RAM (~500 MB for Postgres + Langfuse).
- **v3 requires a worker process**: Langfuse v3 changed the architecture — a separate worker container is required for trace processing. This is a non-obvious operational detail.
- **No alerting**: Langfuse OSS is a dashboard, not an alerting system. For anomaly detection (e.g. fallback rate spike), you'd need to integrate Prometheus or set up Langfuse's cloud alerts.
- **SDK version coupling**: the `langfuse>=2.0.0` Python SDK works with both v2 and v3 servers, but its API may diverge in future major versions.

**What we'd do at scale:** Add structured OpenTelemetry spans alongside Langfuse so that the same trace data flows into an existing APM system (Datadog, Grafana Cloud) if the organisation already uses one. Use Langfuse for LLM-specific dashboards and OpenTelemetry for infrastructure-level metrics.
