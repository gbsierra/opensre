"""Reporting helpers for integration broad-exception sites.

Every ``except Exception`` block in ``app/integrations/<vendor>.py`` validators
should call :func:`report_validation_failure` *before* returning the degraded
``ValidationResult``. This keeps vendor-level failure trends visible in Sentry
without changing operator-visible output.

Runtime diagnostic helpers should call :func:`report_integration_runtime_failure`.
Those failures usually describe downstream service availability, credentials, or
permissions, so they are logged for local debugging but are not captured to
OpenSRE's Sentry as product errors.
"""

from __future__ import annotations

import logging
from typing import Any

from app.utils.errors import report_exception


def _log_integration_failure(
    exc: BaseException,
    *,
    logger: logging.Logger,
    message: str,
    severity: str,
    extras: dict[str, Any] | None = None,
) -> None:
    log_fn = getattr(logger, severity, logger.warning)
    if extras:
        log_fn("%s extras=%s", message, extras, exc_info=exc)
        return
    log_fn("%s", message, exc_info=exc)


def report_validation_failure(
    exc: BaseException,
    *,
    logger: logging.Logger,
    integration: str,
    method: str,
    severity: str = "warning",
    extras: dict[str, Any] | None = None,
) -> None:
    """Log + Sentry-capture a validator broad-except failure with vendor tags.

    Args:
        exc: The exception caught in the broad-except block.
        logger: The caller's module-level logger.
        integration: Vendor identifier (e.g. ``"postgresql"``, ``"kafka"``).
        method: Function or method name where the failure happened. Use
            ``"<outer>.<inner>"`` for nested probes (e.g.
            ``"get_replication_status.statement_probe"``).
        severity: ``logging``-compatible level name; defaults to ``"warning"``
            since most validator failures are vendor/config issues rather
            than bugs in OpenSRE.
        extras: Optional structured fields (DAG id, statement name, etc.).
            Merged into Sentry ``extra`` without becoming Sentry tags, so
            they don't inflate Sentry's tag cardinality.
    """
    report_exception(
        exc,
        logger=logger,
        message=f"[{integration}] {method} validation failed",
        severity=severity,
        tags={
            "surface": "integration",
            "integration": integration,
            "event": "validation_failed",
            "method": method,
        },
        extras=extras,
    )


def report_integration_runtime_failure(
    exc: BaseException,
    *,
    logger: logging.Logger,
    integration: str,
    method: str,
    severity: str = "warning",
    extras: dict[str, Any] | None = None,
) -> None:
    """Log a runtime integration failure without Sentry capture.

    Runtime integration functions are diagnostic probes against customer-owned
    or third-party systems. Their broad-except paths normally return degraded
    tool output, so downstream outages should be visible in logs and tool
    results without becoming OpenSRE product-error issues in Sentry.
    """
    _log_integration_failure(
        exc,
        logger=logger,
        message=f"[{integration}] {method} runtime failed",
        severity=severity,
        extras=extras,
    )
