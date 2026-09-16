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

Use **Langfuse self-hosted OSS (v2/v3)** via Docker Compose.

Langfuse provides dedicated LLM observability: natively modeling traces with nested spans (routing decision -> tool invocation -> final generation), recording inputs, outputs, and latencies across execution nodes, and rendering chronological timelines of request lifecycles. The self-hosted edition is open-source, operates fully within local infrastructure (PostgreSQL, server, and background worker), and requires no external API keys or cloud dependencies.

Instrumentation utilizes the official Python SDK (`langfuse>=2.0.0`), abstracted behind a lightweight client layer (`gateway/langfuse_client.py`) providing graceful no-op behavior when observability services are offline.

---

## Consequences

**Advantages:**
- **Tailored trace visualization**: Span-level hierarchical traces directly reflect multi-step agent orchestration (prompt classification -> tool dispatch -> answer synthesis).
- **Self-contained and secure**: No telemetry leaves the local execution boundary; zero operational cloud expenses.
- **Resilient fallback**: When Langfuse services are unreachable, tracing silently falls back to no-op wrappers without impeding core gateway functions.
- **Trace propagation**: The agent orchestrator initiates the root trace and forwards the trace identifier to downstream services via the `X-Langfuse-Trace-Id` HTTP header, correlating all spans into a unified timeline.

**Limitations:**
- **Container footprint**: Requires additional supporting infrastructure (PostgreSQL database and Langfuse container services), consuming extra system memory.
- **Alerting configuration**: Open-source self-hosted instances do not bundle native alerting engines; alerting requires pairing with monitoring stacks like Prometheus or Alertmanager.
- **SDK lifecycle**: Client libraries must be maintained in sync with containerized server schema versions.

**Production Considerations:**
Integrate OpenTelemetry exporters alongside Langfuse to mirror trace spans into enterprise APM platforms (such as Grafana Tempo or Datadog) while maintaining specialized LLM diagnostic dashboards.
