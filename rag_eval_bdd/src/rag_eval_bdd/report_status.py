from __future__ import annotations

from datetime import datetime


def format_timestamp(timestamp: str, output_format: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime(output_format)
    except Exception:  # noqa: BLE001
        return timestamp


def clamp_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def required_pass_rate(
    threshold: float | None,
    pass_rate_rule: str,
    min_pass_rate: float,
) -> float | None:
    if pass_rate_rule == "none":
        return None
    if pass_rate_rule == "threshold_based":
        if threshold is None:
            return None
        return clamp_score(threshold) * 100.0
    return max(0.0, min(100.0, float(min_pass_rate)))


def status_with_class(
    avg_score: float | None,
    threshold: float | None,
    pass_rate: float | None = None,
    pass_rate_rule: str = "min_pass_rate",
    min_pass_rate: float = 100.0,
) -> tuple[str, str]:
    if avg_score is None or threshold is None:
        return "N/A", "status-na"
    if avg_score < threshold:
        return "FAIL", "status-fail"

    required = required_pass_rate(
        threshold=threshold,
        pass_rate_rule=pass_rate_rule,
        min_pass_rate=min_pass_rate,
    )
    if required is not None and pass_rate is not None and pass_rate < required:
        return "FAIL", "status-fail"
    return "PASS", "status-pass"


def status_text(
    avg_score: float | None,
    threshold: float | None,
    pass_rate: float | None = None,
    pass_rate_rule: str = "min_pass_rate",
    min_pass_rate: float = 100.0,
) -> str:
    text, _ = status_with_class(
        avg_score=avg_score,
        threshold=threshold,
        pass_rate=pass_rate,
        pass_rate_rule=pass_rate_rule,
        min_pass_rate=min_pass_rate,
    )
    return text
