"""
Optional OpenTelemetry tracing + Prometheus metrics wiring.

Design goal: ZERO behaviour change unless explicitly enabled. Everything here is
guarded so that:
  * if the observability packages are not installed (local dev / CI), it no-ops;
  * if the enabling env vars are unset, tracing stays off.

Enable in production by setting (see deploy/observability/):
    OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy:4318   # local Grafana Alloy collector
    OTEL_SERVICE_NAME=bot-service                    # optional, falls back to arg

Metrics: a Prometheus /metrics endpoint is exposed by default (harmless; just an
extra route). Disable with METRICS_ENABLED=false.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("observability")

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _enabled(var: str, default: bool) -> bool:
    val = os.getenv(var)
    if val is None:
        return default
    return val.strip().lower() in _TRUTHY


def setup_fastapi_observability(app, service_name: str) -> None:
    """Instrument a FastAPI app. Safe to call unconditionally."""
    _setup_metrics(app, service_name)
    _setup_tracing(service_name, fastapi_app=app)


def setup_worker_observability(service_name: str) -> None:
    """Instrument a non-HTTP worker (the LiveKit agent).

    Starts a standalone Prometheus metrics server (so Alloy can scrape the
    worker) and wires OTLP tracing for outbound HTTP calls. No-op unless enabled.
    """
    _start_metrics_server(service_name)
    _setup_tracing(service_name, fastapi_app=None)


# ── Metrics ──────────────────────────────────────────────────────────────────


def _setup_metrics(app, service_name: str) -> None:
    if not _enabled("METRICS_ENABLED", default=True):
        return
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except Exception as exc:  # package not installed → skip silently in dev
        logger.info("prometheus-fastapi-instrumentator unavailable; /metrics off (%s)", exc)
        return
    Instrumentator().instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )
    logger.info("Prometheus /metrics enabled for %s", service_name)


def _start_metrics_server(service_name: str) -> None:
    if not _enabled("METRICS_ENABLED", default=True):
        return
    port = int(os.getenv("METRICS_PORT", "9464"))
    try:
        from prometheus_client import start_http_server
    except Exception as exc:
        logger.info("prometheus-client unavailable; worker metrics off (%s)", exc)
        return
    try:
        start_http_server(port)
        logger.info("Prometheus metrics for %s on :%d/metrics", service_name, port)
    except OSError as exc:  # port already bound (e.g. multiple job procs) → fine
        logger.info("worker metrics server not started on :%d (%s)", port, exc)


# ── Tracing ──────────────────────────────────────────────────────────────────


def _setup_tracing(service_name: str, fastapi_app=None) -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    )
    if not endpoint and not _enabled("OTEL_ENABLED", default=False):
        return  # tracing not configured → stay off
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
    except Exception as exc:
        logger.warning("OpenTelemetry libs unavailable; tracing disabled (%s)", exc)
        return

    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", service_name),
            "service.namespace": "jarvis",
            "deployment.environment": os.getenv("DEPLOY_ENV", "production"),
        }
    )
    provider = TracerProvider(resource=resource)
    # OTLPSpanExporter reads OTEL_EXPORTER_OTLP_* env vars for endpoint/headers.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    RequestsInstrumentor().instrument()
    if fastapi_app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(fastapi_app)
        except Exception as exc:
            logger.warning("FastAPI instrumentation failed (%s)", exc)

    logger.info("OpenTelemetry tracing enabled for %s → %s", service_name, endpoint)
