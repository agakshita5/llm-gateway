"""Small helpers shared by the http-based adapters."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After is either seconds ("7") or an HTTP date. Returns seconds, or None if missing/unparseable."""
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
