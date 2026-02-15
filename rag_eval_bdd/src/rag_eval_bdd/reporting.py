from __future__ import annotations

from datetime import datetime
import html
from pathlib import Path
from typing import Any, Iterable, List

from rag_eval_bdd.models import RunResult, TrendSummary

def attach_json(name: str, payload: Any) -> None:
    del name, payload
    return


def attach_text(name: str, payload: str) -> None:
    del name, payload
    return


def attach_file(name: str, path: Path, attachment_type: Any = None) -> None:
    del name, path, attachment_type
    return


def generate_trend_charts(trend_summary: TrendSummary, output_dir: Path) -> List[Path]:
    import os

    # Ensure matplotlib can write cache files in restricted environments.
    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = Path.cwd() / ".cache" / "matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    for metric_trend in trend_summary.metrics:
        if not metric_trend.points:
            continue

        x_labels = [point.run_id.split("_")[0] for point in metric_trend.points]
        avg_scores = [point.avg_score if point.avg_score is not None else 0.0 for point in metric_trend.points]
        pass_rates = [(point.pass_rate or 0.0) / 100.0 for point in metric_trend.points]
        threshold = metric_trend.points[-1].threshold or 0.0

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_labels, avg_scores, marker="o", linewidth=2, label="avg_score")
        ax.plot(x_labels, pass_rates, marker="s", linewidth=2, label="pass_rate")
        ax.axhline(y=threshold, linestyle="--", linewidth=1.5, label=f"threshold={threshold:.2f}")

        ax.set_ylim(0, 1)
        ax.set_ylabel("Score / Pass Rate")
        ax.set_xlabel("Run")
        ax.set_title(f"Trend for {metric_trend.metric_name}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        chart_path = output_dir / f"{metric_trend.metric_name}_trend.png"
        fig.savefig(chart_path)
        plt.close(fig)
        generated.append(chart_path)

    return generated


def _short_timestamp(timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        return timestamp


def _safe_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _required_pass_rate(
    threshold: float | None,
    pass_rate_rule: str,
    min_pass_rate: float,
) -> float | None:
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
) -> tuple[str, str]:
    if avg_score is None or threshold is None:
        return "N/A", "status-na"
    if avg_score < threshold:
        return "FAIL", "status-fail"

    required_pass_rate = _required_pass_rate(
        threshold=threshold,
        pass_rate_rule=pass_rate_rule,
        min_pass_rate=min_pass_rate,
    )
    if required_pass_rate is not None and pass_rate is not None and pass_rate < required_pass_rate:
        return "FAIL", "status-fail"
    return "PASS", "status-pass"


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "N/A"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}"


def _metric_display_name(metric_name: str) -> str:
    return metric_name.replace("_", " ").title()


def _svg_line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    if len(points) == 1:
        x, y = points[0]
        return f"M{x:.2f},{y:.2f}"

    commands = [f"M{points[0][0]:.2f},{points[0][1]:.2f}"]
    for idx in range(len(points) - 1):
        p0 = points[idx - 1] if idx > 0 else points[idx]
        p1 = points[idx]
        p2 = points[idx + 1]
        p3 = points[idx + 2] if idx + 2 < len(points) else p2

        c1x = p1[0] + (p2[0] - p0[0]) / 6.0
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0

        commands.append(
            f"C{c1x:.2f},{c1y:.2f} {c2x:.2f},{c2y:.2f} {p2[0]:.2f},{p2[1]:.2f}"
        )
    return " ".join(commands)


def _build_metric_svg(
    metric_name: str,
    points: list[Any],
    pass_rate_rule: str,
    min_pass_rate: float,
) -> str:
    width, height = 900, 250
    left, right, top, bottom = 58, 20, 24, 44
    plot_w = width - left - right
    plot_h = height - top - bottom

    if not points:
        return "<svg viewBox='0 0 900 250'></svg>"

    n = len(points)

    def x_pos(i: int) -> float:
        if n == 1:
            return left + (plot_w / 2.0)
        return left + (plot_w * i / (n - 1))

    def y_pos(v: float) -> float:
        return top + (1.0 - v) * plot_h

    avg_values = [_safe_score(p.avg_score) for p in points]
    pass_values = [_safe_score((p.pass_rate or 0.0) / 100.0) for p in points]
    threshold = _safe_score(points[-1].threshold)

    avg_points = [(x_pos(i), y_pos(v)) for i, v in enumerate(avg_values)]
    pass_points = [(x_pos(i), y_pos(v)) for i, v in enumerate(pass_values)]

    grid_lines = []
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(tick)
        grid_lines.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' y2='{y:.2f}' class='grid-line' />"
        )
        grid_lines.append(
            f"<text x='10' y='{y + 4:.2f}' class='axis-label'>{tick:.2f}</text>"
        )

    threshold_y = y_pos(threshold)
    threshold_line = (
        f"<line x1='{left}' y1='{threshold_y:.2f}' x2='{left + plot_w}' y2='{threshold_y:.2f}' "
        f"class='threshold-line' />"
    )

    circles = []
    for i, point in enumerate(points):
        score = point.avg_score
        threshold_value = point.threshold if point.threshold is not None else threshold
        _, klass = _status(
            score,
            threshold_value,
            pass_rate=point.pass_rate,
            pass_rate_rule=pass_rate_rule,
            min_pass_rate=min_pass_rate,
        )
        cx, cy = avg_points[i]
        circles.append(f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='4.5' class='dot {klass}' />")
        circles.append(
            f"<text x='{cx:.2f}' y='{height - 14}' text-anchor='middle' class='axis-label'>"
            f"{html.escape(_short_timestamp(point.timestamp))}</text>"
        )

    metric_label = html.escape(_metric_display_name(metric_name))
    return f"""
    <svg viewBox="0 0 {width} {height}" class="trend-svg" role="img" aria-label="Trend for {metric_label}">
      <rect x="0" y="0" width="{width}" height="{height}" class="plot-bg"></rect>
      {''.join(grid_lines)}
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis-line"></line>
      <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis-line"></line>
      {threshold_line}
      <path d="{_svg_line_path(avg_points)}" class="avg-line smooth-line"></path>
      <path d="{_svg_line_path(pass_points)}" class="pass-line smooth-line"></path>
      {''.join(circles)}
      <g transform="translate({left + 8}, {top + 16})">
        <rect x="0" y="-10" width="12" height="3" class="avg-line"></rect>
        <text x="18" y="-7" class="legend-label">Avg Score</text>
        <rect x="110" y="-10" width="12" height="3" class="pass-line"></rect>
        <text x="128" y="-7" class="legend-label">Pass Rate</text>
        <rect x="228" y="-10" width="12" height="3" class="threshold-line-fill"></rect>
        <text x="246" y="-7" class="legend-label">Threshold ({threshold:.2f})</text>
      </g>
    </svg>
    """


def _derive_shared_threshold(trend_summary: TrendSummary) -> float:
    rounded_thresholds: list[float] = []
    for metric in trend_summary.metrics:
        if not metric.points:
            continue
        latest_threshold = metric.points[-1].threshold
        if latest_threshold is None:
            continue
        rounded_thresholds.append(round(_safe_score(latest_threshold), 2))

    if not rounded_thresholds:
        return 0.70

    frequency: dict[float, int] = {}
    for threshold in rounded_thresholds:
        frequency[threshold] = frequency.get(threshold, 0) + 1

    ranked = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
    return float(ranked[0][0])


def _build_combined_trend_card(
    trend_summary: TrendSummary,
    pass_rate_rule: str,
    min_pass_rate: float,
) -> str:
    timeline_entries: dict[tuple[str, str], str] = {}
    for metric in trend_summary.metrics:
        for point in metric.points:
            timeline_entries[(point.run_id, point.timestamp)] = point.timestamp

    ordered_timeline = sorted(timeline_entries.keys(), key=lambda key: key[1])
    if trend_summary.keep_last_n > 0:
        ordered_timeline = ordered_timeline[-trend_summary.keep_last_n :]

    if not ordered_timeline:
        return ""

    width, height = 1080, 300
    left, right, top, bottom = 58, 24, 30, 48
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(i: int) -> float:
        if len(ordered_timeline) == 1:
            return left + (plot_w / 2.0)
        return left + (plot_w * i / (len(ordered_timeline) - 1))

    def y_pos(v: float) -> float:
        return top + (1.0 - v) * plot_h

    grid_lines: list[str] = []
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(tick)
        grid_lines.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' y2='{y:.2f}' class='grid-line' />"
        )
        grid_lines.append(f"<text x='10' y='{y + 4:.2f}' class='axis-label'>{tick:.2f}</text>")

    x_labels = []
    for idx, (_, timestamp) in enumerate(ordered_timeline):
        x = x_pos(idx)
        x_labels.append(
            f"<text x='{x:.2f}' y='{height - 14}' text-anchor='middle' class='axis-label'>"
            f"{html.escape(_short_timestamp(timestamp))}</text>"
        )

    shared_threshold = _derive_shared_threshold(trend_summary)
    threshold_y = y_pos(shared_threshold)
    threshold_line = (
        f"<line x1='{left}' y1='{threshold_y:.2f}' x2='{left + plot_w}' y2='{threshold_y:.2f}' "
        f"class='threshold-line' />"
    )

    palette = [
        "#7c83ff",
        "#34d399",
        "#f59e0b",
        "#f472b6",
        "#22d3ee",
        "#f97316",
        "#a78bfa",
        "#eab308",
        "#fb7185",
        "#60a5fa",
    ]
    metric_lines: list[str] = []
    metric_dots: list[str] = []
    legend_items: list[str] = []
    visible_metric_count = 0

    for idx, metric in enumerate(sorted(trend_summary.metrics, key=lambda item: item.metric_name)):
        color = palette[idx % len(palette)]
        point_lookup = {(point.run_id, point.timestamp): point for point in metric.points}
        coordinates: list[tuple[float, float]] = []
        points_for_status = []
        for run_idx, timeline_key in enumerate(ordered_timeline):
            point = point_lookup.get(timeline_key)
            if point is None or point.avg_score is None:
                continue
            x = x_pos(run_idx)
            y = y_pos(_safe_score(point.avg_score))
            coordinates.append((x, y))
            points_for_status.append((point, x, y))

        if len(coordinates) < 1:
            continue

        visible_metric_count += 1
        metric_lines.append(
            f"<path d='{_svg_line_path(coordinates)}' class='combined-line smooth-line' style='stroke: {color};'></path>"
        )
        for point, x, y in points_for_status:
            _, status_class = _status(
                point.avg_score,
                shared_threshold,
                pass_rate=point.pass_rate,
                pass_rate_rule=pass_rate_rule,
                min_pass_rate=min_pass_rate,
            )
            metric_dots.append(
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4.3' class='dot {status_class}' style='fill: {color};'></circle>"
            )

        legend_items.append(
            "<span class='legend-item'>"
            f"<span class='legend-swatch' style='background: {color};'></span>"
            f"{html.escape(_metric_display_name(metric.metric_name))}"
            "</span>"
        )

    if visible_metric_count == 0:
        return ""

    svg = f"""
    <svg viewBox="0 0 {width} {height}" class="trend-svg combined-trend-svg" role="img" aria-label="All metrics trend over last runs">
      <rect x="0" y="0" width="{width}" height="{height}" class="plot-bg"></rect>
      {''.join(grid_lines)}
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis-line"></line>
      <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis-line"></line>
      {threshold_line}
      {''.join(metric_lines)}
      {''.join(metric_dots)}
      {''.join(x_labels)}
    </svg>
    """

    return f"""
    <section class="metric-card combined-card">
      <div class="metric-header">
        <h3>All Metrics (Last {len(ordered_timeline)} Runs)</h3>
        <span class="status-pill status-na">Shared Threshold: {shared_threshold:.2f}</span>
      </div>
      <div class="combined-legend">
        {''.join(legend_items)}
        <span class='legend-item'>
          <span class='legend-swatch threshold-swatch'></span>
          Shared Threshold ({shared_threshold:.2f})
        </span>
      </div>
      <div class="chart-wrap">
        {svg}
      </div>
    </section>
    """


def write_trend_html(
    trend_summary: TrendSummary,
    output_path: Path,
    pass_rate_rule: str = "min_pass_rate",
    min_pass_rate: float = 100.0,
) -> Path:
    combined_trend_card = _build_combined_trend_card(
        trend_summary=trend_summary,
        pass_rate_rule=pass_rate_rule,
        min_pass_rate=min_pass_rate,
    )
    metric_cards: List[str] = []
    for metric in trend_summary.metrics:
        points = metric.points
        if not points:
            continue

        first = points[0]
        latest = points[-1]
        latest_score = latest.avg_score
        latest_threshold = latest.threshold
        latest_pass_rate = latest.pass_rate
        status_text, status_class = _status(
            latest_score,
            latest_threshold,
            pass_rate=latest_pass_rate,
            pass_rate_rule=pass_rate_rule,
            min_pass_rate=min_pass_rate,
        )

        delta = None
        if first.avg_score is not None and latest.avg_score is not None:
            delta = latest.avg_score - first.avg_score

        score_values = [p.avg_score for p in points if p.avg_score is not None]
        if len(score_values) > 1:
            avg_mean = sum(score_values) / len(score_values)
            variance = sum((s - avg_mean) ** 2 for s in score_values) / len(score_values)
            std_dev = variance**0.5
        else:
            std_dev = 0.0
        consistency = "Stable" if std_dev <= 0.05 else "Variable"
        consistency_class = "consistency-stable" if consistency == "Stable" else "consistency-variable"

        run_rows: List[str] = []
        for point in points:
            score = point.avg_score
            threshold = point.threshold
            row_status, row_class = _status(
                score,
                threshold,
                pass_rate=point.pass_rate,
                pass_rate_rule=pass_rate_rule,
                min_pass_rate=min_pass_rate,
            )
            score_text = "N/A" if score is None else f"{score:.4f}"
            pass_text = "N/A" if point.pass_rate is None else f"{point.pass_rate:.2f}%"
            threshold_text = "N/A" if threshold is None else f"{threshold:.2f}"
            run_rows.append(
                "<tr>"
                f"<td>{html.escape(point.run_id)}</td>"
                f"<td>{html.escape(_short_timestamp(point.timestamp))}</td>"
                f"<td class='{row_class}'>{score_text}</td>"
                f"<td>{pass_text}</td>"
                f"<td>{threshold_text}</td>"
                f"<td><span class='status-pill {row_class}'>{row_status}</span></td>"
                "</tr>"
            )

        metric_name_text = html.escape(_metric_display_name(metric.metric_name))
        latest_score_text = "N/A" if latest_score is None else f"{latest_score:.4f}"
        latest_pass_rate_text = "N/A" if latest_pass_rate is None else f"{latest_pass_rate:.2f}%"
        latest_threshold_text = "N/A" if latest_threshold is None else f"{latest_threshold:.2f}"

        metric_cards.append(
            f"""
            <section class="metric-card">
              <div class="metric-header">
                <h3>{metric_name_text}</h3>
                <span class="status-pill {status_class}">{status_text}</span>
              </div>
              <div class="metric-kpis">
                <div class="kpi"><span class="label">Latest Score</span><span class="value {status_class}">{latest_score_text}</span></div>
                <div class="kpi"><span class="label">Threshold</span><span class="value">{latest_threshold_text}</span></div>
                <div class="kpi"><span class="label">Pass Rate</span><span class="value">{latest_pass_rate_text}</span></div>
                <div class="kpi"><span class="label">Delta (first to latest)</span><span class="value">{_format_delta(delta)}</span></div>
                <div class="kpi"><span class="label">Consistency (1 SD)</span><span class="value {consistency_class}">{consistency}</span></div>
              </div>
              <div class="chart-wrap">
                {_build_metric_svg(metric.metric_name, points, pass_rate_rule, min_pass_rate)}
              </div>
              <table class="runs-table">
                <thead>
                  <tr><th>Run ID</th><th>Timestamp</th><th>Avg Score</th><th>Pass Rate</th><th>Threshold</th><th>Status</th></tr>
                </thead>
                <tbody>
                  {"".join(run_rows)}
                </tbody>
              </table>
            </section>
            """
        )

    generated = html.escape(_short_timestamp(trend_summary.generated_at))
    metric_count = len(trend_summary.metrics)
    runs_count = trend_summary.keep_last_n
    html_doc = f"""
<html>
<head>
  <meta charset="utf-8" />
  <title>RAG Eval Trend Dashboard</title>
  <style>
    :root {{
      --bg: #0b1020;
      --card: #131a2a;
      --card-border: #24314a;
      --text: #e5edf7;
      --muted: #9fb0c9;
      --ok: #22c55e;
      --bad: #ef4444;
      --warn: #f59e0b;
      --line-avg: #60a5fa;
      --line-pass: #f59e0b;
      --line-threshold: #f87171;
      --grid: #26334c;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background: var(--bg); color: var(--text); }}
    h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
    .subtitle {{ margin: 0 0 20px 0; color: var(--muted); }}
    .meta {{ display: flex; gap: 16px; margin-bottom: 20px; color: var(--muted); font-size: 14px; }}
    .metric-card {{ background: var(--card); border: 1px solid var(--card-border); border-radius: 12px; padding: 16px; margin-bottom: 18px; }}
    .metric-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
    .metric-header h3 {{ margin: 0; font-size: 18px; }}
    .status-pill {{ padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: 0.2px; }}
    .status-pass {{ color: #052e12; background: #86efac; }}
    .status-fail {{ color: #450a0a; background: #fca5a5; }}
    .status-na {{ color: #111827; background: #d1d5db; }}
    .metric-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-bottom: 12px; }}
    .kpi {{ border: 1px solid var(--card-border); border-radius: 10px; padding: 10px; background: #0f1525; }}
    .kpi .label {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 3px; }}
    .kpi .value {{ font-size: 16px; font-weight: 700; }}
    .consistency-stable {{ color: var(--ok); }}
    .consistency-variable {{ color: var(--warn); }}
    .chart-wrap {{ width: 100%; overflow-x: auto; margin-bottom: 12px; border: 1px solid var(--card-border); border-radius: 10px; background: #0f1525; }}
    .trend-svg {{ width: 100%; min-width: 780px; height: auto; display: block; }}
    .plot-bg {{ fill: #0f1525; }}
    .grid-line {{ stroke: var(--grid); stroke-width: 1; }}
    .axis-line {{ stroke: #4b5f82; stroke-width: 1; }}
    .axis-label {{ fill: #94a3b8; font-size: 11px; }}
    .legend-label {{ fill: #cbd5e1; font-size: 11px; }}
    .avg-line {{ fill: none; stroke: var(--line-avg); stroke-width: 2.5; }}
    .pass-line {{ fill: none; stroke: var(--line-pass); stroke-width: 2; }}
    .smooth-line {{ stroke-linecap: round; stroke-linejoin: round; }}
    .combined-line {{ fill: none; stroke-width: 2.6; opacity: 0.96; }}
    .threshold-line {{ stroke: var(--line-threshold); stroke-width: 1.5; stroke-dasharray: 5 5; }}
    .threshold-line-fill {{ fill: var(--line-threshold); }}
    .combined-legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 7px; font-size: 12px; color: #dbe7ff; }}
    .legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
    .threshold-swatch {{ background: var(--line-threshold); border: 1px dashed #fda4af; }}
    .dot.status-pass {{ fill: var(--ok); stroke: #0f172a; stroke-width: 1.5; }}
    .dot.status-fail {{ fill: var(--bad); stroke: #0f172a; stroke-width: 1.5; }}
    .dot.status-na {{ fill: #9ca3af; stroke: #0f172a; stroke-width: 1.5; }}
    .runs-table {{ width: 100%; border-collapse: collapse; }}
    .runs-table th, .runs-table td {{ border-bottom: 1px solid var(--card-border); padding: 8px 10px; font-size: 13px; }}
    .runs-table th {{ color: #cbd5e1; font-weight: 600; text-align: left; background: #111a2f; }}
    .runs-table td.status-pass {{ color: #86efac; font-weight: 700; background: rgba(34, 197, 94, 0.08); }}
    .runs-table td.status-fail {{ color: #fca5a5; font-weight: 700; background: rgba(239, 68, 68, 0.08); }}
    .runs-table td.status-na {{ color: #cbd5e1; }}
  </style>
</head>
<body>
  <h1>RAG Eval Trend Dashboard</h1>
  <p class="subtitle">Performance trend and pass/fail history for the most recent runs.</p>
  <div class="meta">
    <span>Generated: {generated}</span>
    <span>Metrics: {metric_count}</span>
    <span>Window: last {runs_count} runs</span>
  </div>
  {combined_trend_card}
  {"".join(metric_cards)}
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc)
    return output_path


def attach_run_artifacts(run_result: RunResult, trend_summary: TrendSummary, trend_charts: Iterable[Path], trend_html: Path) -> None:
    del run_result, trend_summary, trend_charts, trend_html
    return
