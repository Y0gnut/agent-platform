"""
gateway/langfuse_client.py — Thin Langfuse Observability Wrapper
=================================================================
Provides a minimal, no-crash interface around the Langfuse Python SDK
for use by router.py and mcp_client.py.

Design goals
------------
1. OPTIONAL by default — if LANGFUSE_PUBLIC_KEY is not set, all operations
   are no-ops. This means the gateway works identically with and without
   Langfuse configured; no test or local run requires Docker to be up.

2. ONE trace per user request — mcp_client.py creates the root trace,
   router.py adds child spans by receiving the trace_id as a header.

3. Thread-safe & async-safe — the Langfuse SDK handles its own background
   flushing; we don't need to await anything.

Configuration (via environment variables):
    LANGFUSE_PUBLIC_KEY   — from Langfuse UI Settings → API Keys
    LANGFUSE_SECRET_KEY   — from Langfuse UI Settings → API Keys
    LANGFUSE_HOST         — default: http://localhost:3000

Usage example (mcp_client.py):
    from gateway.langfuse_client import get_langfuse, start_trace

    lf = get_langfuse()
    trace = start_trace(lf, name="run_query", user_query=query)
    with trace.span("decide_tool") as span:
        ...
        span.update(output=decision)
    trace.update(output=final_response)
    lf.flush()
"""

import getpass
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger("langfuse_client")

# ── Lazy singleton & state ────────────────────────────────────────────────────
_langfuse_instance = None
_langfuse_disabled = False  # set True if SDK import fails or keys are missing
_credentials_prompted = False


def prompt_langfuse_credentials():
    """
    交互式提示用户输入 Langfuse 凭据并注入到 os.environ。
    - 使用 input 获取 Public Key
    - 使用 getpass 获取 Secret Key（避免终端明文显示）
    - 强制设置 LANGFUSE_HOST 为 http://localhost:3000 并写入 os.environ
    - 容错机制：用户回车跳过或空值时不报错，平滑进入 no-op 模式
    """
    global _credentials_prompted
    if _credentials_prompted:
        return
    _credentials_prompted = True

    # 1. 强行将 LANGFUSE_HOST 设为 http://localhost:3000 写入 os.environ
    os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

    # 如果是非交互环境（例如 pytest / 管道无 tty），避免阻塞
    if not (sys.stdin and sys.stdin.isatty()):
        return

    print("\n" + "=" * 60)
    print("【Langfuse V2 可观测性配置 (本地模式)】")
    print("已设置环境变量 LANGFUSE_HOST = http://localhost:3000")
    print("提示：直接按回车可跳过输入，网关将以无追踪 (no-op) 模式正常启动")
    print("=" * 60)

    try:
        # 2. 使用 Python 原生 input 获取 Public Key
        public_key = input("请输入 Langfuse Public Key (pk-lf-...): ").strip()

        # 3. 使用 Python 原生 getpass 获取 Secret Key 避免明文显示
        secret_key = ""
        if public_key:
            secret_key = getpass.getpass("请输入 Langfuse Secret Key (sk-lf-...) [输入不可见]: ").strip()
    except (EOFError, OSError):
        public_key = ""
        secret_key = ""

    # 4. 注入环境变量与容错判断
    if public_key and secret_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        print(" Langfuse 凭证已成功注入环境变量 (os.environ)，开始初始化客户端...\n")
    else:
        # 如果跳过或未输全，清理可能残留的空值
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)
        print("⏭️  已跳过 Langfuse 凭证输入，网关将保持 no-op 模式平滑启动。\n")


def get_langfuse():
    """
    Return a Langfuse client instance (singleton).

    Returns None if:
      - langfuse package is not installed
      - LANGFUSE_PUBLIC_KEY env var is not set / skipped
    In either case, all subsequent calls are no-ops — the gateway continues
    working without observability.
    """
    global _langfuse_instance, _langfuse_disabled

    if _langfuse_disabled:
        return None
    if _langfuse_instance is not None:
        return _langfuse_instance

    # 触发交互式提示并将凭证注入 os.environ
    prompt_langfuse_credentials()

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000").strip()

    if not public_key or not secret_key:
        logger.info(
            "Langfuse 未启用: 未检测到有效密钥，系统已平滑切换为 no-op 模式，主业务正常运行。"
        )
        _langfuse_disabled = True
        return None

    try:
        from langfuse import Langfuse

        _langfuse_instance = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("Langfuse client initialised successfully (host=%s)", host)
        return _langfuse_instance

    except ImportError:
        logger.warning(
            "langfuse package not installed. Run: pip install langfuse>=2.0.0"
        )
        _langfuse_disabled = True
        return None
    except Exception as exc:
        logger.warning("Langfuse init failed (will continue without tracing): %s", exc)
        _langfuse_disabled = True
        return None


# ── Trace helpers ──────────────────────────────────────────────────────────────

def start_trace(lf, name: str, user_query: str = "", trace_id: Optional[str] = None):
    """
    Create a new Langfuse trace for one user request.

    Args:
        lf: Langfuse client from get_langfuse(). If None, returns a _NoopTrace.
        name: Human-readable trace name (e.g. "run_query").
        user_query: The raw user input — captured as trace input.
        trace_id: Optional existing trace ID to continue a trace started elsewhere.

    Returns:
        A Langfuse Trace object, or a _NoopTrace if Langfuse is disabled.
    """
    if lf is None:
        return _NoopTrace()
    try:
        kwargs = dict(name=name, input=user_query)
        if trace_id:
            kwargs["id"] = trace_id
        return lf.trace(**kwargs)
    except Exception as exc:
        logger.debug("Langfuse start_trace failed: %s", exc)
        return _NoopTrace()


def flush(lf) -> None:
    """Flush pending Langfuse events to the server (call at end of request)."""
    if lf is None:
        return
    try:
        lf.flush()
    except Exception as exc:
        logger.debug("Langfuse flush failed: %s", exc)


# ── Span helpers ───────────────────────────────────────────────────────────────

@contextmanager
def timed_span(trace_or_span, name: str, input_data=None):
    """
    Context manager that wraps a block of code in a Langfuse span with
    automatic timing and error capture.

    Usage:
        with timed_span(trace, "call_litellm", input_data=prompt) as span:
            result = await call_litellm(...)
            span.update(output=result, metadata={"model": model})
    """
    if isinstance(trace_or_span, _NoopTrace):
        yield _NoopSpan()
        return

    span = None
    start = time.monotonic()
    try:
        kwargs = dict(name=name)
        if input_data is not None:
            kwargs["input"] = str(input_data)[:1000]  # truncate large inputs
        span = trace_or_span.span(**kwargs)
        yield span
    except Exception as exc:
        if span:
            try:
                span.update(
                    level="ERROR",
                    status_message=str(exc)[:500],
                    metadata={"error_type": type(exc).__name__},
                )
            except Exception:
                pass
        raise
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000
        if span:
            try:
                span.end()
                span.update(metadata={"latency_ms": round(elapsed_ms, 1)})
            except Exception:
                pass


# ── No-op stubs (when Langfuse is disabled) ────────────────────────────────────

class _NoopTrace:
    """
    Drop-in replacement for a Langfuse Trace when the SDK is unavailable.
    Every method is a no-op that returns self or a _NoopSpan.
    This avoids scattered `if lf is not None:` checks throughout the codebase.
    """

    def span(self, *args, **kwargs) -> "_NoopSpan":
        return _NoopSpan()

    def generation(self, *args, **kwargs) -> "_NoopSpan":
        return _NoopSpan()

    def event(self, *args, **kwargs) -> "_NoopSpan":
        return _NoopSpan()

    def update(self, *args, **kwargs) -> "_NoopTrace":
        return self

    def end(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def id(self) -> str:
        return ""


class _NoopSpan:
    """Drop-in replacement for a Langfuse Span."""

    def update(self, *args, **kwargs) -> "_NoopSpan":
        return self

    def end(self, *args, **kwargs) -> None:
        pass

    def span(self, *args, **kwargs) -> "_NoopSpan":
        return _NoopSpan()

    def generation(self, *args, **kwargs) -> "_NoopSpan":
        return _NoopSpan()

    def event(self, *args, **kwargs) -> "_NoopSpan":
        return _NoopSpan()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def id(self) -> str:
        return ""

