from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    inline_rate = (sum(1 for r in inline_rows if r["status"] == "PASS") / len(inline_rows) * 100.0) if inline_rows else 0.0
    external_rate = (
        sum(1 for r in external_rows if r["status"] == "PASS") / len(external_rows) * 100.0
    ) if external_rows else 0.0

    failed_reasons = [
        row["reason"].strip()
        for row in rows
        if row["status"] == "FAIL" and row["reason"].strip()
    ]
    reason_counts = Counter(failed_reasons)
    top_reasons = ", ".join(
        _truncate(reason, 90) for reason, _ in reason_counts.most_common(3)
    ) or "No failed reasons captured."

    return {
        "total_rows": str(total),
        "pass_count": str(pass_count),
        "fail_count": str(fail_count),
        "na_count": str(na_count),
        "overall_pass_rate": f"{overall_pass_rate:.2f}%",
        "inline_rate": f"{inline_rate:.2f}%",
        "external_rate": f"{external_rate:.2f}%",
        "top_reasons": top_reasons,
    }


def _metric_latest_status_counts(rows: list[dict]) -> Dict[Tuple[str, str], Dict[str, int]]:
    counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    for row in rows:
        key = (row["run_id"], row["metric"])
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


def _metric_health_rows(
    eval_rows: list[dict],
    trend_summary: TrendSummary,
    pass_rate_rule: str,
    min_pass_rate: float,
) -> str:
    latest_counts = _metric_latest_status_counts(eval_rows)
    metric_rows: list[str] = []
    for metric in sorted(trend_summary.metrics, key=lambda metric: metric.metric_name):
        if not metric.points:
            continue
        latest = metric.points[-1]
        count_bucket = latest_counts.get((latest.run_id, metric.metric_name), {"PASS": 0, "FAIL": 0, "N/A": 0})
        status = _status(
            latest.avg_score,
            latest.threshold,
            pass_rate=latest.pass_rate,
            pass_rate_rule=pass_rate_rule,
            min_pass_rate=min_pass_rate,
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
      min-width: 1320px;
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
      <a class="footer-link" href="./technical_logs.json">Download Full Technical Logs (JSON)</a>
    </div>
  </div>
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
    }})();
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc)
    return output_path
