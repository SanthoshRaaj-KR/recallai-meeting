"""
Optional OpenTelemetry tracing + Prometheus metrics wiring for the org service.

Mirrors my-agent/src/observability.py. ZERO behaviour change unless enabled:
  * no-ops if the observability packages aren't installed (local dev / CI);
  * tracing stays off unless OTEL_EXPORTER_OTLP_ENDPOINT (or OTEL_ENABLED) is set.

Production env (see deploy/observability/):
    OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy:4318
    OTEL_SERVICE_NAME=org-service
Metrics /metrics is exposed by default; disable with METRICS_ENABLED=false.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("observability")

_TRUTHY = {"1", "true", "yes", "on"}


def _enabled(var: str, default: bool) -> bool:
    val = os.getenv(var)
    if val is None:
        return default
    return val.strip().lower() in _TRUTHY


def setup_fastapi_observability(app, service_name: str) -> None:
    _setup_metrics(app, service_name)
    _setup_tracing(service_name, app)


def _setup_metrics(app, service_name: str) -> None:
    if not _enabled("METRICS_ENABLED", default=True):
        return
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except Exception as exc:
        logger.info("prometheus-fastapi-instrumentator unavailable; /metrics off (%s)", exc)
        return
    Instrumentator().instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )
    logger.info("Prometheus /metrics enabled for %s", service_name)


def _setup_tracing(service_name: str, app) -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    )
    if not endpoint and not _enabled("OTEL_ENABLED", default=False):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
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
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    RequestsInstrumentor().instrument()
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry tracing enabled for %s → %s", service_name, endpoint)
