from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import html
import json
import re
from pathlib import Path
import shutil
from typing import Dict, Iterable, List

from rag_eval_bdd.models import RunResult, TrendSummary


def _short_timestamp(timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        return timestamp


def _safe_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _required_pass_rate(threshold: float | None, pass_rate_rule: str, min_pass_rate: float) -> float | None:
    if pass_rate_rule == "none":
        return None
    if pass_rate_rule == "threshold_based":
        if threshold is None:
            return None
        return _safe_score(threshold) * 100.0
    return max(0.0, min(100.0, float(min_pass_rate)))


def _status(
    avg_score: float | None,
    threshold: float | None,
    pass_rate: float | None = None,
    pass_rate_rule: str = "min_pass_rate",
    min_pass_rate: float = 100.0,
) -> str:
    if avg_score is None or threshold is None:
        return "N/A"
    if avg_score < threshold:
        return "FAIL"
    required_pass_rate = _required_pass_rate(
        threshold=threshold,
        pass_rate_rule=pass_rate_rule,
        min_pass_rate=min_pass_rate,
    )
    if required_pass_rate is not None and pass_rate is not None and pass_rate < required_pass_rate:
        return "FAIL"
    return "PASS"


def _row_status(score: float | None, threshold: float | None, passed: bool | None) -> str:
    if isinstance(passed, bool):
        return "PASS" if passed else "FAIL"
    if score is None or threshold is None:
        return "N/A"
    return "PASS" if score >= threshold else "FAIL"


def _badge_class(status: str) -> str:
    if status == "PASS":
        return "badge-pass"
    if status == "FAIL":
        return "badge-fail"
    return "badge-na"


def _normalize_metric(name: str) -> str:
    return name.replace("_", " ").title()


def _infer_data_source(scenario_name: str) -> str:
    lowered = scenario_name.lower()
    if "live" in lowered or "unseen" in lowered:
        return "Live Data"
    if "inline" in lowered:
        return "Inline Data"
    if "external" in lowered:
        return "External Data"
    return "Mixed/Other"


def _truncate(text: str, limit: int = 180) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


def _collect_rows(run_results: Iterable[RunResult]) -> list[dict]:
    rows: list[dict] = []
    for run in run_results:
        source_type = _infer_data_source(run.scenario)
        for question in run.question_results:
            for metric in question.metrics:
                status = _row_status(metric.score, metric.threshold, metric.passed)
                reason = metric.reason or metric.error or ""
                row_id = f"{run.run_id}__{question.question_id}__{metric.metric_name}"
                rows.append(
                    {
                        "row_id": row_id,
                        "metric": metric.metric_name,
                        "metric_label": _normalize_metric(metric.metric_name),
                        "run_id": run.run_id,
                        "timestamp": run.timestamp,
                        "timestamp_short": _short_timestamp(run.timestamp),
                        "type": source_type,
                        "threshold": metric.threshold,
                        "score": metric.score,
                        "question_id": question.question_id,
                        "question": question.question,
                        "expected_output": question.expected_answer or "",
                        "actual_output": question.actual_answer,
                        "retrieval_context": [str(chunk) for chunk in question.retrieval_context],
                        "status": status,
                        "reason": reason,
                        "scenario": run.scenario,
                        "feature": run.feature,
                        "evaluation_model": metric.evaluation_model or "",
                    }
                )
    return rows


def _format_score(score: float | None) -> str:
    if score is None:
        return "N/A"
    return f"{score:.4f}"


def _format_threshold(threshold: float | None) -> str:
    if threshold is None:
        return "N/A"
    return f"{threshold:.2f}"


def _normalize_reason_for_grouping(reason: str) -> str:
    text = reason.strip()
    # Remove volatile IDs and collapse whitespace so similar errors group together.
    text = re.sub(r"0x[0-9a-fA-F]+", "0x...", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _is_transient_infra_reason(reason: str) -> bool:
    lowered = reason.lower()
    infra_markers = (
        "retryerror",
        "max retries exceeded",
        "timeouterror",
        "readtimeout",
        "connecttimeout",
        "apiconnectionerror",
        "connection error",
        "temporarily unavailable",
        "service unavailable",
        "http 429",
        "rate limit",
    )
    return any(marker in lowered for marker in infra_markers)


def _top_failure_reasons_text(rows: list[dict]) -> str:
    failed_reasons = [
        _normalize_reason_for_grouping(row["reason"])
        for row in rows
        if row["status"] == "FAIL" and row["reason"].strip()
    ]
    if not failed_reasons:
        return "No failed reasons captured."

    quality_counts: Counter[str] = Counter()
    transient_count = 0
    for reason in failed_reasons:
        if _is_transient_infra_reason(reason):
            transient_count += 1
        else:
            quality_counts[reason] += 1

    summary_parts: list[str] = []
    for reason, count in quality_counts.most_common(3):
        summary_parts.append(f"{count}x {_truncate(reason, 90)}")

    if transient_count > 0:
        summary_parts.append(f"{transient_count}x Transient backend/API retry errors")

    return "; ".join(summary_parts) if summary_parts else "No failed reasons captured."


def _summary_cards(rows: list[dict]) -> dict[str, str]:
    total = len(rows)
    pass_count = sum(1 for row in rows if row["status"] == "PASS")
    fail_count = sum(1 for row in rows if row["status"] == "FAIL")
    na_count = total - pass_count - fail_count
    overall_pass_rate = (pass_count / total * 100.0) if total else 0.0

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    inline_rows = by_type.get("Inline Data", [])
    external_rows = by_type.get("External Data", [])
    live_rows = by_type.get("Live Data", [])
    inline_rate = (sum(1 for r in inline_rows if r["status"] == "PASS") / len(inline_rows) * 100.0) if inline_rows else 0.0
    external_rate = (
        sum(1 for r in external_rows if r["status"] == "PASS") / len(external_rows) * 100.0
    ) if external_rows else 0.0
    live_rate = (
        sum(1 for r in live_rows if r["status"] == "PASS") / len(live_rows) * 100.0
    ) if live_rows else 0.0

    top_reasons = _top_failure_reasons_text(rows)

    return {
        "total_rows": str(total),
        "pass_count": str(pass_count),
        "fail_count": str(fail_count),
        "na_count": str(na_count),
        "overall_pass_rate": f"{overall_pass_rate:.2f}%",
        "inline_rate": f"{inline_rate:.2f}%",
        "external_rate": f"{external_rate:.2f}%",
        "live_rate": f"{live_rate:.2f}%",
        "top_reasons": top_reasons,
    }


def _metric_status_counts(rows: list[dict]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        key = row["metric"]
        if key not in counts:
            counts[key] = {"PASS": 0, "FAIL": 0, "N/A": 0}
        status = row["status"] if row["status"] in {"PASS", "FAIL", "N/A"} else "N/A"
        counts[key][status] += 1
    return counts


def _status_count_html(pass_count: int, fail_count: int, na_count: int) -> str:
    return (
        "<div class='status-counts'>"
        f"<span class='count-pill count-pass'>{pass_count}</span>"
        "<span class='count-sep'>/</span>"
        f"<span class='count-pill count-fail'>{fail_count}</span>"
        "<span class='count-sep'>/</span>"
        f"<span class='count-pill count-na'>{na_count}</span>"
        "</div>"
    )


def _metric_health_status_from_counts(
    pass_count: int,
    fail_count: int,
    na_count: int,
) -> str:
    if fail_count > 0:
        return "FAIL"
    if pass_count > 0:
        return "PASS"
    if na_count > 0:
        return "N/A"
    return "N/A"


def _metric_health_rows(
    eval_rows: list[dict],
    trend_summary: TrendSummary,
    pass_rate_rule: str,
    min_pass_rate: float,
) -> str:
    metric_counts = _metric_status_counts(eval_rows)
    metric_rows: list[str] = []
    for metric in sorted(trend_summary.metrics, key=lambda metric: metric.metric_name):
        if not metric.points:
            continue
        latest = metric.points[-1]
        count_bucket = metric_counts.get(metric.metric_name, {"PASS": 0, "FAIL": 0, "N/A": 0})
        status = _metric_health_status_from_counts(
            pass_count=count_bucket["PASS"],
            fail_count=count_bucket["FAIL"],
            na_count=count_bucket["N/A"],
        )
        metric_rows.append(
            "<tr>"
            f"<td>{html.escape(_normalize_metric(metric.metric_name))}</td>"
            f"<td>{_format_score(latest.avg_score)}</td>"
            f"<td>{'N/A' if latest.pass_rate is None else f'{latest.pass_rate:.2f}%'}"
            "</td>"
            f"<td>{_format_threshold(latest.threshold)}</td>"
            f"<td>{_status_count_html(count_bucket['PASS'], count_bucket['FAIL'], count_bucket['N/A'])}</td>"
            f"<td><span class='badge {_badge_class(status)}'>{status}</span></td>"
            "</tr>"
        )
    return "".join(metric_rows)


def _snapshot_executive_report(
    output_path: Path,
    generated_at: str,
    keep_last_n: int = 5,
) -> None:
    if keep_last_n <= 1:
        return
    if not output_path.exists():
        return

    report_dir = output_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        dt = datetime.utcnow()

    timestamp_token = dt.strftime("%Y%m%dT%H%M%S")
    snapshot_path = report_dir / f"index_{timestamp_token}.html"
    suffix = 1
    while snapshot_path.exists():
        snapshot_path = report_dir / f"index_{timestamp_token}_{suffix}.html"
        suffix += 1

    shutil.copy2(output_path, snapshot_path)

    snapshots = sorted(report_dir.glob("index_*.html"), reverse=True)
    for stale_snapshot in snapshots[max(0, keep_last_n - 1):]:
        stale_snapshot.unlink(missing_ok=True)


def write_executive_html(
    run_results: List[RunResult],
    trend_summary: TrendSummary,
    output_path: Path,
    pass_rate_rule: str = "min_pass_rate",
    min_pass_rate: float = 100.0,
) -> Path:
    rows = _collect_rows(run_results)
    summary = _summary_cards(rows)
    generated_at = _short_timestamp(trend_summary.generated_at)
    unique_runs = len({row["run_id"] for row in rows})
    metric_names = sorted({row["metric_label"] for row in rows})
    source_types = sorted({row["type"] for row in rows})
    metric_options = "".join(
        f"<option value='{html.escape(name)}'>{html.escape(name)}</option>"
        for name in metric_names
    )
    source_options = "".join(
        f"<option value='{html.escape(name)}'>{html.escape(name)}</option>"
        for name in source_types
    )

    log_json_path = output_path.parent / "technical_logs.json"
    logs_payload = {
        "generated_at": trend_summary.generated_at,
        "run_files": [
            {
                "run_id": run.run_id,
                "timestamp": run.timestamp,
                "scenario": run.scenario,
                "path": f"../runs/{run.run_id}/results.json",
            }
            for run in run_results
        ],
        "rows": [
            {
                "metric": row["metric"],
                "run_id": row["run_id"],
                "timestamp": row["timestamp"],
                "type": row["type"],
                "threshold": row["threshold"],
                "score": row["score"],
                "question_id": row["question_id"],
                "question": row["question"],
                "expected_output": row["expected_output"],
                "actual_output": row["actual_output"],
                "retrieval_context": row["retrieval_context"],
                "result": row["status"],
                "reason_for_score": row["reason"],
                "scenario": row["scenario"],
                "feature": row["feature"],
                "evaluation_model": row["evaluation_model"],
            }
            for row in rows
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_json_path.write_text(json.dumps(logs_payload, indent=2))
    context_payload = {
        row["row_id"]: row["retrieval_context"]
        for row in rows
    }
    context_payload_json = json.dumps(context_payload).replace("</", "<\\/")

    table_rows = []
    run_log_panels = []
    for run in run_results:
        run_log_panels.append(
            "<details class='run-log'>"
            f"<summary>{html.escape(run.run_id)} · {html.escape(_short_timestamp(run.timestamp))} · "
            f"{html.escape(_infer_data_source(run.scenario))}</summary>"
            f"<iframe class='run-log-frame' src='../runs/{html.escape(run.run_id)}/results.json' "
            f"title='Run log {html.escape(run.run_id)}'></iframe>"
            "</details>"
        )

    for row in rows:
        status_class = _badge_class(row["status"])
        reason = row["reason"] or "N/A"
        context_chunks = row["retrieval_context"]
        context_count = len(context_chunks)
        context_preview_text = _truncate(" ".join(context_chunks), 140) if context_chunks else "No retrieval context captured."
        context_cell = (
            f"<button type='button' class='context-link' data-row-id='{html.escape(row['row_id'])}' "
            f"data-metric='{html.escape(row['metric_label'])}' "
            f"data-question='{html.escape(row['question'])}' "
            f"data-run='{html.escape(row['run_id'])}'>"
            f"{context_count} chunk{'s' if context_count != 1 else ''} · View</button>"
            f"<div class='context-preview' title='{html.escape(context_preview_text)}'>{html.escape(context_preview_text)}</div>"
        ) if context_chunks else "<span class='context-empty'>N/A</span>"
        table_rows.append(
            "<tr "
            f"data-metric='{html.escape(row['metric_label'])}' "
            f"data-type='{html.escape(row['type'])}' "
            f"data-status='{html.escape(row['status'])}' "
            ">"
            f"<td>{html.escape(row['metric_label'])}</td>"
            f"<td>{html.escape(row['run_id'])}</td>"
            f"<td>{html.escape(row['timestamp_short'])}</td>"
            f"<td>{html.escape(row['type'])}</td>"
            f"<td>{_format_threshold(row['threshold'])}</td>"
            f"<td title='{html.escape(row['question'])}'>{html.escape(_truncate(row['question'], 90))}</td>"
            f"<td title='{html.escape(row['expected_output'])}'>{html.escape(_truncate(row['expected_output'], 90))}</td>"
            f"<td title='{html.escape(row['actual_output'])}'>{html.escape(_truncate(row['actual_output'], 90))}</td>"
            f"<td>{context_cell}</td>"
            f"<td>{_format_score(row['score'])}</td>"
            f"<td><span class='badge {status_class}'>{html.escape(row['status'])}</span></td>"
            f"<td title='{html.escape(reason)}'>{html.escape(_truncate(reason, 95))}</td>"
            f"<td><a class='tech-link' href='#technical-logs'>View Log</a></td>"
            "</tr>"
        )

    html_doc = f"""
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Evaluation Executive Report</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-soft: #f2f5fb;
      --text: #0f172a;
      --muted: #556277;
      --border: #d7deea;
      --accent: #0f5ea8;
      --accent-soft: #e7f2ff;
      --pass: #0f8a4c;
      --fail: #b42318;
      --na: #6b7280;
      --shadow: 0 14px 42px rgba(15, 23, 42, 0.08);
      --hero-a: #e6f0ff;
      --hero-b: #fef7e9;
      --hero-c: #f4ecff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 15% 12%, var(--hero-a), transparent 46%),
        radial-gradient(circle at 86% 4%, var(--hero-b), transparent 42%),
        radial-gradient(circle at 82% 92%, var(--hero-c), transparent 45%),
        var(--bg);
      color: var(--text);
      font-family: "SF Pro Display", "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      line-height: 1.4;
    }}
    .container {{
      max-width: 1450px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    .hero {{
      background: linear-gradient(140deg, #0f172a, #132a56 45%, #18417f);
      color: #f8fbff;
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 28px;
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      width: 320px;
      height: 320px;
      right: -120px;
      top: -140px;
      background: radial-gradient(circle, rgba(255,255,255,0.2), transparent 60%);
      pointer-events: none;
    }}
    .hero h1 {{
      margin: 0 0 8px 0;
      font-size: clamp(24px, 2.8vw, 36px);
      letter-spacing: 0.2px;
    }}
    .hero p {{
      margin: 0;
      color: #dce8ff;
      max-width: 920px;
    }}
    .hero-meta {{
      margin-top: 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      font-size: 13px;
      color: #cadcff;
    }}
    .hero-meta span {{
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 999px;
      padding: 6px 10px;
    }}
    .summary-grid {{
      margin-top: 18px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
      gap: 12px;
    }}
    .summary-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
    }}
    .summary-card .label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }}
    .summary-card .value {{
      font-size: 22px;
      font-weight: 700;
    }}
    .summary-card .hint {{
      margin-top: 4px;
      font-size: 12px;
      color: var(--muted);
    }}
    .section {{
      margin-top: 18px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 8px 26px rgba(15, 23, 42, 0.05);
      overflow: hidden;
    }}
    .section-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, #ffffff, var(--surface-soft));
    }}
    .section-header h2 {{
      margin: 0;
      font-size: 18px;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .controls input, .controls select {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fff;
      height: 36px;
      font-size: 13px;
      padding: 0 10px;
      min-width: 150px;
      color: var(--text);
    }}
    .kicker {{
      font-size: 12px;
      color: var(--muted);
      margin-left: 2px;
    }}
    .metric-health, .report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .metric-health th, .metric-health td, .report-table th, .report-table td {{
      border-bottom: 1px solid var(--border);
      padding: 10px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .metric-health th, .report-table th {{
      background: #f8fbff;
      font-weight: 700;
      color: #1a2a45;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .table-wrap {{
      overflow: auto;
      max-height: 62vh;
    }}
    .table-wrap table {{
      min-width: 1500px;
    }}
    .badge {{
      border-radius: 999px;
      font-weight: 700;
      font-size: 11px;
      letter-spacing: 0.2px;
      padding: 3px 9px;
      display: inline-block;
    }}
    .badge-pass {{
      background: #dcfce7;
      color: var(--pass);
      border: 1px solid #9ce8ba;
    }}
    .badge-fail {{
      background: #fee4e2;
      color: var(--fail);
      border: 1px solid #f9b4af;
    }}
    .badge-na {{
      background: #e5e7eb;
      color: var(--na);
      border: 1px solid #d1d5db;
    }}
    .status-counts {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 700;
    }}
    .count-pill {{
      min-width: 26px;
      text-align: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      border: 1px solid transparent;
    }}
    .count-pass {{
      color: #0f8a4c;
      background: #dcfce7;
      border-color: #9ce8ba;
    }}
    .count-fail {{
      color: #b42318;
      background: #fee4e2;
      border-color: #f9b4af;
    }}
    .count-na {{
      color: #854d0e;
      background: #fef9c3;
      border-color: #fde68a;
    }}
    .count-sep {{
      color: var(--muted);
      font-weight: 600;
    }}
    .tech-link {{
      color: var(--accent);
      font-weight: 600;
      text-decoration: none;
    }}
    .tech-link:hover {{
      text-decoration: underline;
    }}
    .context-link {{
      border: 1px solid #bfd4f6;
      background: #edf4ff;
      color: var(--accent);
      font-weight: 700;
      font-size: 11px;
      letter-spacing: 0.2px;
      padding: 5px 10px;
      border-radius: 999px;
      cursor: pointer;
      white-space: nowrap;
    }}
    .context-link:hover {{
      background: #e3eeff;
      border-color: #9fc0ee;
    }}
    .context-preview {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      max-width: 260px;
      display: -webkit-box;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .context-empty {{
      color: var(--muted);
      font-size: 12px;
      font-style: italic;
    }}
    .context-modal[hidden] {{
      display: none;
    }}
    .context-modal {{
      position: fixed;
      inset: 0;
      z-index: 9999;
      background: rgba(15, 23, 42, 0.52);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 18px;
      backdrop-filter: blur(2px);
    }}
    .context-modal-card {{
      width: min(980px, 96vw);
      max-height: 88vh;
      display: flex;
      flex-direction: column;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: #fff;
      box-shadow: 0 24px 50px rgba(15, 23, 42, 0.22);
      overflow: hidden;
    }}
    .context-modal-header {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, #ffffff, var(--surface-soft));
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }}
    .context-modal-title {{
      margin: 0;
      font-size: 16px;
    }}
    .context-modal-meta {{
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    .context-modal-close {{
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }}
    .context-modal-close:hover {{
      background: #f8fbff;
    }}
    .context-modal-body {{
      padding: 14px;
      overflow: auto;
      display: grid;
      gap: 10px;
      background: #fbfdff;
    }}
    .context-modal-empty {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .context-chunk {{
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
      padding: 10px 12px;
    }}
    .context-chunk-label {{
      margin: 0;
      font-size: 11px;
      font-weight: 700;
      color: #334155;
      letter-spacing: 0.35px;
      text-transform: uppercase;
    }}
    .context-chunk-text {{
      margin: 6px 0 0 0;
      font-size: 12.5px;
      line-height: 1.5;
      color: #1f2937;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .technical {{
      margin-top: 22px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: #fff;
      box-shadow: 0 8px 26px rgba(15, 23, 42, 0.05);
      overflow: hidden;
    }}
    .technical > summary {{
      list-style: none;
      cursor: pointer;
      padding: 16px 18px;
      font-weight: 700;
      background: linear-gradient(180deg, #ffffff, var(--surface-soft));
      border-bottom: 1px solid var(--border);
    }}
    .technical > summary::-webkit-details-marker {{
      display: none;
    }}
    .technical-body {{
      padding: 14px 14px 22px;
      display: grid;
      gap: 10px;
    }}
    .technical-body p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .run-log {{
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
      overflow: hidden;
    }}
    .run-log summary {{
      cursor: pointer;
      font-weight: 600;
      padding: 10px 12px;
      background: #f8fbff;
    }}
    .run-log-frame {{
      width: 100%;
      height: 340px;
      border: 0;
      border-top: 1px solid var(--border);
      background: #fff;
    }}
    .log-frame {{
      width: 100%;
      height: 420px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fbfdff;
    }}
    .report-footer {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 16px;
      color: var(--muted);
      font-size: 12px;
    }}
    .footer-links {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .footer-link {{
      color: var(--accent);
      font-weight: 600;
      text-decoration: none;
    }}
    .footer-link:hover {{
      text-decoration: underline;
    }}
    @media (max-width: 780px) {{
      .container {{
        padding: 16px 12px 30px;
      }}
      .hero {{
        padding: 20px 16px;
        border-radius: 16px;
      }}
      .section-header {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .controls {{
        width: 100%;
      }}
      .controls input, .controls select {{
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <header class="hero">
      <h1>RAG Evaluation Executive Report</h1>
      <p>Business-oriented quality report with scenario outcomes, metric health, and a technical appendix for full traceability.</p>
      <div class="hero-meta">
        <span>Generated: {html.escape(generated_at)}</span>
        <span>Recent Runs: {unique_runs}</span>
        <span>Rows: {summary["total_rows"]}</span>
        <span>Status Rule: {html.escape(pass_rate_rule)} (min pass rate: {min_pass_rate:.2f}%)</span>
      </div>
    </header>

    <section class="summary-grid">
      <article class="summary-card"><span class="label">Overall Pass Rate</span><span class="value">{summary["overall_pass_rate"]}</span></article>
      <article class="summary-card"><span class="label">Pass / Fail / N/A</span><span class="value">{summary["pass_count"]} / {summary["fail_count"]} / {summary["na_count"]}</span></article>
      <article class="summary-card"><span class="label">Inline Data Pass Rate</span><span class="value">{summary["inline_rate"]}</span></article>
      <article class="summary-card"><span class="label">External Data Pass Rate</span><span class="value">{summary["external_rate"]}</span></article>
      <article class="summary-card"><span class="label">Live Data Pass Rate</span><span class="value">{summary["live_rate"]}</span></article>
      <article class="summary-card"><span class="label">Top Failure Reasons</span><span class="value" style="font-size: 14px;">{html.escape(summary["top_reasons"])}</span></article>
    </section>

    <section class="section">
      <div class="section-header">
        <div>
          <h2>Metric Health (Latest Trend Point)</h2>
          <div class="kicker">Each metric's latest score, pass rate, and threshold using current report status rule.</div>
        </div>
      </div>
      <div class="table-wrap" style="max-height: 260px;">
        <table class="metric-health">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Latest Score</th>
              <th>Pass Rate</th>
              <th>Threshold</th>
              <th>PASS / FAIL / N/A</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {_metric_health_rows(rows, trend_summary, pass_rate_rule=pass_rate_rule, min_pass_rate=min_pass_rate)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <div>
          <h2>Evaluation Results</h2>
          <div class="kicker">Columns aligned to management review needs with direct links to full technical logs.</div>
        </div>
        <div class="controls">
          <input id="searchInput" type="search" placeholder="Search question / output / run..." />
          <select id="metricFilter">
            <option value="">All Metrics</option>
            {metric_options}
          </select>
          <select id="typeFilter">
            <option value="">All Types</option>
            {source_options}
          </select>
          <select id="statusFilter">
            <option value="">All Results</option>
            <option value="PASS">PASS</option>
            <option value="FAIL">FAIL</option>
            <option value="N/A">N/A</option>
          </select>
        </div>
      </div>
      <div class="table-wrap">
        <table class="report-table" id="reportTable">
          <thead>
            <tr>
              <th>Metric</th>
              <th>RunID</th>
              <th>TimeStamp</th>
              <th>Type</th>
              <th>Threshold</th>
              <th>Question</th>
              <th>Expected Output</th>
              <th>Actual Output</th>
              <th>Retrieved Context</th>
              <th>Score</th>
              <th>Result</th>
              <th>Reason For Score</th>
              <th>Technical Logs</th>
            </tr>
          </thead>
          <tbody id="reportBody">
            {"".join(table_rows)}
          </tbody>
        </table>
      </div>
      <div class="section-header" style="padding-top: 10px; padding-bottom: 10px;">
        <div class="kicker" id="visibleCount">Visible rows: {len(rows)}</div>
        <a class="footer-link" href="#technical-logs" id="jumpToLogs">Go To Complete Logs</a>
      </div>
    </section>

    <details class="technical" id="technical-logs">
      <summary>Complete Technical Logs (Click To Expand)</summary>
      <div class="technical-body">
        <p>Summary logs are available in <code>technical_logs.json</code>. Complete per-run logs are embedded below.</p>
        <iframe class="log-frame" src="./technical_logs.json" title="Technical Logs"></iframe>
        {"".join(run_log_panels)}
      </div>
    </details>

    <div class="report-footer">
      <span>HTML report is generated from saved run artifacts under <code>results/runs/</code>.</span>
      <div class="footer-links">
        <a class="footer-link" href="../trends/last5.html">Open Trend Dashboard (Last 5 Runs)</a>
        <a class="footer-link" href="./technical_logs.json">Download Full Technical Logs (JSON)</a>
      </div>
    </div>
  </div>
  <div class="context-modal" id="contextModal" hidden>
    <div class="context-modal-card" role="dialog" aria-modal="true" aria-labelledby="contextModalTitle">
      <div class="context-modal-header">
        <div>
          <h3 class="context-modal-title" id="contextModalTitle">Retrieved Context</h3>
          <div class="context-modal-meta" id="contextModalMeta"></div>
        </div>
        <button type="button" class="context-modal-close" id="contextModalClose">Close</button>
      </div>
      <div class="context-modal-body" id="contextModalBody"></div>
    </div>
  </div>
  <script type="application/json" id="contextPayload">{context_payload_json}</script>
  <script>
    (function () {{
      const searchInput = document.getElementById("searchInput");
      const metricFilter = document.getElementById("metricFilter");
      const typeFilter = document.getElementById("typeFilter");
      const statusFilter = document.getElementById("statusFilter");
      const rows = Array.from(document.querySelectorAll("#reportBody tr"));
      const visibleCount = document.getElementById("visibleCount");
      const techDetails = document.getElementById("technical-logs");
      const jumpToLogs = document.getElementById("jumpToLogs");
      const inlineTechLinks = Array.from(document.querySelectorAll(".tech-link"));
      const contextLinks = Array.from(document.querySelectorAll(".context-link"));
      const contextModal = document.getElementById("contextModal");
      const contextModalBody = document.getElementById("contextModalBody");
      const contextModalMeta = document.getElementById("contextModalMeta");
      const contextModalClose = document.getElementById("contextModalClose");
      const contextPayloadNode = document.getElementById("contextPayload");
      const contextPayload = contextPayloadNode ? JSON.parse(contextPayloadNode.textContent || "{{}}") : {{}};

      function normalize(text) {{
        return (text || "").toLowerCase();
      }}

      function applyFilters() {{
        const query = normalize(searchInput.value);
        const metric = metricFilter.value;
        const type = typeFilter.value;
        const status = statusFilter.value;
        let visible = 0;

        rows.forEach((row) => {{
          const metricMatch = !metric || row.dataset.metric === metric;
          const typeMatch = !type || row.dataset.type === type;
          const statusMatch = !status || row.dataset.status === status;
          const textMatch = !query || normalize(row.innerText).includes(query);
          const show = metricMatch && typeMatch && statusMatch && textMatch;
          row.style.display = show ? "" : "none";
          if (show) visible += 1;
        }});

        visibleCount.textContent = `Visible rows: ${{visible}}`;
      }}

      [searchInput, metricFilter, typeFilter, statusFilter].forEach((el) => {{
        el.addEventListener("input", applyFilters);
        el.addEventListener("change", applyFilters);
      }});

      jumpToLogs.addEventListener("click", () => {{
        techDetails.open = true;
      }});

      inlineTechLinks.forEach((link) => {{
        link.addEventListener("click", () => {{
          techDetails.open = true;
        }});
      }});

      function closeContextModal() {{
        contextModal.setAttribute("hidden", "hidden");
        contextModalBody.innerHTML = "";
        contextModalMeta.textContent = "";
        document.body.style.overflow = "";
      }}

      function openContextModal(link) {{
        const rowId = link.dataset.rowId || "";
        const chunks = Array.isArray(contextPayload[rowId]) ? contextPayload[rowId] : [];
        const metric = link.dataset.metric || "Metric";
        const runId = link.dataset.run || "Run";
        const question = link.dataset.question || "";

        contextModalMeta.textContent = `${{metric}} · ${{runId}} · ${{chunks.length}} chunk${{chunks.length === 1 ? "" : "s"}}`;
        contextModalBody.innerHTML = "";

        if (question) {{
          const questionCard = document.createElement("article");
          questionCard.className = "context-chunk";

          const questionLabel = document.createElement("p");
          questionLabel.className = "context-chunk-label";
          questionLabel.textContent = "Question";
          questionCard.appendChild(questionLabel);

          const questionText = document.createElement("p");
          questionText.className = "context-chunk-text";
          questionText.textContent = question;
          questionCard.appendChild(questionText);

          contextModalBody.appendChild(questionCard);
        }}

        if (!chunks.length) {{
          const empty = document.createElement("p");
          empty.className = "context-modal-empty";
          empty.textContent = "No retrieval context captured for this row.";
          contextModalBody.appendChild(empty);
        }} else {{
          chunks.forEach((chunk, idx) => {{
            const card = document.createElement("article");
            card.className = "context-chunk";

            const label = document.createElement("p");
            label.className = "context-chunk-label";
            label.textContent = `Chunk ${{idx + 1}}`;
            card.appendChild(label);

            const body = document.createElement("p");
            body.className = "context-chunk-text";
            body.textContent = chunk || "";
            card.appendChild(body);

            contextModalBody.appendChild(card);
          }});
        }}

        contextModal.removeAttribute("hidden");
        document.body.style.overflow = "hidden";
      }}

      contextLinks.forEach((link) => {{
        link.addEventListener("click", () => openContextModal(link));
      }});
      contextModalClose.addEventListener("click", closeContextModal);
      contextModal.addEventListener("click", (event) => {{
        if (event.target === contextModal) {{
          closeContextModal();
        }}
      }});
      document.addEventListener("keydown", (event) => {{
        if (event.key === "Escape" && !contextModal.hasAttribute("hidden")) {{
          closeContextModal();
        }}
      }});
    }})();
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc)
    _snapshot_executive_report(
        output_path=output_path,
        generated_at=trend_summary.generated_at,
        keep_last_n=5,
    )
    return output_path
