from __future__ import annotations

from datetime import datetime
import html
from pathlib import Path
from typing import Any, Iterable, List

from rag_eval_bdd.models import RunResult, TrendSummary
from rag_eval_bdd.report_status import clamp_score, format_timestamp, status_with_class

RUN_CLUSTER_MAX_GAP_MINUTES = 5

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
    return format_timestamp(timestamp, "%m-%d %H:%M")


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "N/A"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}"


def _metric_display_name(metric_name: str) -> str:
    return metric_name.replace("_", " ").title()


def _truncate(text: str, limit: int = 180) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


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

    avg_values = [clamp_score(p.avg_score) for p in points]
    pass_values = [clamp_score((p.pass_rate or 0.0) / 100.0) for p in points]
    threshold = clamp_score(points[-1].threshold)

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
        _, klass = status_with_class(
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
        rounded_thresholds.append(round(clamp_score(latest_threshold), 2))

    if not rounded_thresholds:
        return 0.70

    frequency: dict[float, int] = {}
    for threshold in rounded_thresholds:
        frequency[threshold] = frequency.get(threshold, 0) + 1

    ranked = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
    return float(ranked[0][0])


def _parse_timestamp(timestamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def _build_timeline_clusters(
    trend_summary: TrendSummary,
    max_gap_minutes: int = RUN_CLUSTER_MAX_GAP_MINUTES,
) -> list[list[tuple[str, str]]]:
    timeline_entries: dict[tuple[str, str], str] = {}
    for metric in trend_summary.metrics:
        for point in metric.points:
            timeline_entries[(point.run_id, point.timestamp)] = point.timestamp

    ordered_timeline = sorted(timeline_entries.keys(), key=lambda key: key[1])
    if not ordered_timeline:
        return []

    clusters: list[list[tuple[str, str]]] = []
    cluster_start_dt: list[datetime | None] = []
    max_gap_seconds = max_gap_minutes * 60

    for timeline_key in ordered_timeline:
        _, timestamp = timeline_key
        current_dt = _parse_timestamp(timestamp)

        if not clusters:
            clusters.append([timeline_key])
            cluster_start_dt.append(current_dt)
            continue

        start_dt = cluster_start_dt[-1]
        same_cluster = False
        if current_dt is not None and start_dt is not None:
            delta_seconds = (current_dt - start_dt).total_seconds()
            same_cluster = 0 <= delta_seconds <= max_gap_seconds

        if same_cluster:
            clusters[-1].append(timeline_key)
        else:
            clusters.append([timeline_key])
            cluster_start_dt.append(current_dt)

    if trend_summary.keep_last_n > 0:
        clusters = clusters[-trend_summary.keep_last_n :]

    return clusters


def _point_map_for_clusters(
    points: list[Any],
    timeline_clusters: list[list[tuple[str, str]]],
) -> dict[int, Any]:
    point_lookup = {(point.run_id, point.timestamp): point for point in points}
    point_map: dict[int, Any] = {}

    for cluster_idx, cluster in enumerate(timeline_clusters):
        selected = None
        selected_dt = None
        for timeline_key in cluster:
            point = point_lookup.get(timeline_key)
            if point is None:
                continue
            point_dt = _parse_timestamp(point.timestamp)
            if selected is None:
                selected = point
                selected_dt = point_dt
                continue
            if selected_dt is None or (point_dt is not None and point_dt >= selected_dt):
                selected = point
                selected_dt = point_dt

        if selected is not None:
            point_map[cluster_idx] = selected

    return point_map


def _points_for_clusters(
    points: list[Any],
    timeline_clusters: list[list[tuple[str, str]]],
) -> list[Any]:
    point_map = _point_map_for_clusters(points, timeline_clusters)
    return [point_map[idx] for idx in range(len(timeline_clusters)) if idx in point_map]


def _build_combined_trend_card(
    trend_summary: TrendSummary,
    pass_rate_rule: str,
    min_pass_rate: float,
    timeline_clusters: list[list[tuple[str, str]]],
) -> str:
    if not timeline_clusters:
        return ""

    width, height = 1080, 300
    left, right, top, bottom = 58, 24, 30, 48
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(i: int) -> float:
        if len(timeline_clusters) == 1:
            return left + (plot_w / 2.0)
        return left + (plot_w * i / (len(timeline_clusters) - 1))

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
    for idx, cluster in enumerate(timeline_clusters):
        _, timestamp = cluster[-1]
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
        point_map = _point_map_for_clusters(metric.points, timeline_clusters)
        coordinates: list[tuple[float, float]] = []
        points_for_status = []
        for run_idx in range(len(timeline_clusters)):
            point = point_map.get(run_idx)
            if point is None or point.avg_score is None:
                continue
            x = x_pos(run_idx)
            y = y_pos(clamp_score(point.avg_score))
            coordinates.append((x, y))
            points_for_status.append((point, x, y))

        if len(coordinates) < 1:
            continue

        visible_metric_count += 1
        metric_lines.append(
            f"<path d='{_svg_line_path(coordinates)}' class='combined-line smooth-line' style='stroke: {color};'></path>"
        )
        for point, x, y in points_for_status:
            _, status_class = status_with_class(
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
        <h3>All Metrics (Last {len(timeline_clusters)} Runs)</h3>
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


def _format_perf_number(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return format(float(value), f".{decimals}f")


def _format_perf_int(value: int | None) -> str:
    if value is None:
        return "N/A"
    return str(int(value))


def _perf_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.10, 1.0)
        return max(0.0, minimum - padding), maximum + padding
    padding = (maximum - minimum) * 0.08
    return max(0.0, minimum - padding), maximum + padding


def _build_performance_svg(clustered_runs: list[RunResult]) -> str:
    if not clustered_runs:
        return ""

    width, height = 1080, 280
    left, right, top, bottom = 66, 72, 24, 48
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(i: int) -> float:
        if len(clustered_runs) == 1:
            return left + (plot_w / 2.0)
        return left + (plot_w * i / (len(clustered_runs) - 1))

    def y_pos(value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return top + (plot_h / 2.0)
        normalized = (value - lower) / (upper - lower)
        return top + (1.0 - normalized) * plot_h

    latency_points = [
        (idx, float(run.performance.p95_latency_ms))
        for idx, run in enumerate(clustered_runs)
        if run.performance.p95_latency_ms is not None
    ]
    token_points = [
        (idx, float(run.performance.avg_total_tokens_per_request))
        for idx, run in enumerate(clustered_runs)
        if run.performance.avg_total_tokens_per_request is not None
    ]
    if not latency_points and not token_points:
        return ""

    latency_min, latency_max = _perf_bounds([point[1] for point in latency_points]) if latency_points else (0.0, 1.0)
    token_min, token_max = _perf_bounds([point[1] for point in token_points]) if token_points else (0.0, 1.0)

    grid_lines: list[str] = []
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + tick * plot_h
        latency_value = latency_max - tick * (latency_max - latency_min)
        token_value = token_max - tick * (token_max - token_min)
        grid_lines.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' y2='{y:.2f}' class='grid-line' />"
        )
        grid_lines.append(
            f"<text x='8' y='{y + 4:.2f}' class='axis-label'>{_format_perf_number(latency_value, 0)}</text>"
        )
        grid_lines.append(
            f"<text x='{width - 8}' y='{y + 4:.2f}' class='axis-label axis-label-right'>{_format_perf_number(token_value, 0)}</text>"
        )

    x_labels: list[str] = []
    for idx, run in enumerate(clustered_runs):
        x = x_pos(idx)
        x_labels.append(
            f"<text x='{x:.2f}' y='{height - 14}' text-anchor='middle' class='axis-label'>"
            f"{html.escape(_short_timestamp(run.timestamp))}</text>"
        )

    latency_coords = [(x_pos(idx), y_pos(value, latency_min, latency_max)) for idx, value in latency_points]
    token_coords = [(x_pos(idx), y_pos(value, token_min, token_max)) for idx, value in token_points]

    latency_path = (
        f"<path d='{_svg_line_path(latency_coords)}' class='perf-line-latency smooth-line'></path>"
        if latency_coords
        else ""
    )
    token_path = (
        f"<path d='{_svg_line_path(token_coords)}' class='perf-line-tokens smooth-line'></path>"
        if token_coords
        else ""
    )

    latency_dots = "".join(
        f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4.2' class='perf-dot-latency'></circle>"
        for x, y in latency_coords
    )
    token_dots = "".join(
        f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4.0' class='perf-dot-tokens'></circle>"
        for x, y in token_coords
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" class="trend-svg performance-trend-svg" role="img" aria-label="Performance trend over last runs">
      <rect x="0" y="0" width="{width}" height="{height}" class="plot-bg"></rect>
      {''.join(grid_lines)}
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis-line"></line>
      <line x1="{left + plot_w}" y1="{top}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis-line"></line>
      <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis-line"></line>
      <text x="{left + 2}" y="{top - 6}" class="legend-label">P95 Latency (ms)</text>
      <text x="{left + plot_w - 2}" y="{top - 6}" class="legend-label" text-anchor="end">Avg Tokens / Request</text>
      {latency_path}
      {token_path}
      {latency_dots}
      {token_dots}
      {''.join(x_labels)}
    </svg>
    """


def _build_performance_trend_card(
    run_results: list[RunResult],
    keep_last_n: int,
) -> str:
    if not run_results:
        return ""

    # Performance trend should reflect each run directly (no timeline clustering).
    # Sort oldest -> newest, then apply dashboard window size.
    ordered_runs = sorted(run_results, key=lambda run: run.timestamp)
    if keep_last_n > 0:
        ordered_runs = ordered_runs[-keep_last_n:]

    if not ordered_runs:
        return ""

    first = ordered_runs[0]
    latest = ordered_runs[-1]

    p95_delta = None
    if (
        latest.performance.p95_latency_ms is not None
        and first.performance.p95_latency_ms is not None
    ):
        p95_delta = float(latest.performance.p95_latency_ms) - float(first.performance.p95_latency_ms)

    avg_tokens_delta = None
    if (
        latest.performance.avg_total_tokens_per_request is not None
        and first.performance.avg_total_tokens_per_request is not None
    ):
        avg_tokens_delta = (
            float(latest.performance.avg_total_tokens_per_request)
            - float(first.performance.avg_total_tokens_per_request)
        )

    run_rows: list[str] = []
    for run in ordered_runs:
        perf = run.performance
        run_rows.append(
            "<tr>"
            f"<td title='{html.escape(run.run_id)}'>{html.escape(_truncate(run.run_id, 22))}</td>"
            f"<td>{html.escape(_short_timestamp(run.timestamp))}</td>"
            f"<td>{_format_perf_number(perf.p95_latency_ms)}</td>"
            f"<td>{_format_perf_number(perf.avg_total_tokens_per_request)}</td>"
            f"<td>{_format_perf_int(perf.total_tokens)}</td>"
            f"<td>{_format_perf_int(perf.request_count)}</td>"
            "</tr>"
        )

    performance_svg = _build_performance_svg(ordered_runs)

    return f"""
    <section class="metric-card performance-card">
      <div class="metric-header">
        <h3>Performance Parameters (Last {len(ordered_runs)} Runs)</h3>
        <span class="status-pill status-na">Operational Trend</span>
      </div>
      <div class="metric-kpis">
        <div class="kpi"><span class="label">Latest P95 Latency (ms)</span><span class="value">{_format_perf_number(latest.performance.p95_latency_ms)}</span></div>
        <div class="kpi"><span class="label">Latest Avg Tokens / Request</span><span class="value">{_format_perf_number(latest.performance.avg_total_tokens_per_request)}</span></div>
        <div class="kpi"><span class="label">Latest Total Tokens</span><span class="value">{_format_perf_int(latest.performance.total_tokens)}</span></div>
        <div class="kpi"><span class="label">P95 Delta (first to latest)</span><span class="value">{_format_delta(p95_delta)}</span></div>
        <div class="kpi"><span class="label">Avg Tokens Delta</span><span class="value">{_format_delta(avg_tokens_delta)}</span></div>
      </div>
      <div class="perf-legend">
        <span class="perf-legend-item"><span class="perf-swatch perf-swatch-latency"></span>P95 Latency (ms)</span>
        <span class="perf-legend-item"><span class="perf-swatch perf-swatch-tokens"></span>Avg Tokens / Request</span>
      </div>
      <div class="chart-wrap">
        {performance_svg}
      </div>
      <table class="runs-table">
        <thead>
          <tr><th>Run ID</th><th>Timestamp</th><th>P95 Latency (ms)</th><th>Avg Tokens / Request</th><th>Total Tokens</th><th>Requests</th></tr>
        </thead>
        <tbody>
          {"".join(run_rows)}
        </tbody>
      </table>
    </section>
    """


def write_trend_html(
    trend_summary: TrendSummary,
    output_path: Path,
    pass_rate_rule: str = "min_pass_rate",
    min_pass_rate: float = 100.0,
    run_results: Iterable[RunResult] | None = None,
) -> Path:
    # Trend dashboard status is intentionally threshold-only.
    # Executive report continues to apply pass-rate rules independently.
    trend_status_rule = "none"

    timeline_clusters = _build_timeline_clusters(trend_summary)
    run_results_list = list(run_results) if run_results is not None else []
    combined_trend_card = _build_combined_trend_card(
        trend_summary=trend_summary,
        pass_rate_rule=trend_status_rule,
        min_pass_rate=min_pass_rate,
        timeline_clusters=timeline_clusters,
    )
    performance_trend_card = _build_performance_trend_card(
        run_results=run_results_list,
        keep_last_n=trend_summary.keep_last_n,
    )
    metric_cards: List[str] = []
    for metric in trend_summary.metrics:
        points = _points_for_clusters(metric.points, timeline_clusters)
        if not points:
            continue

        first = points[0]
        latest = points[-1]
        latest_score = latest.avg_score
        latest_threshold = latest.threshold
        latest_pass_rate = latest.pass_rate
        status_text, status_class = status_with_class(
            latest_score,
            latest_threshold,
            pass_rate=latest_pass_rate,
            pass_rate_rule=trend_status_rule,
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
            row_status, row_class = status_with_class(
                score,
                threshold,
                pass_rate=point.pass_rate,
                pass_rate_rule=trend_status_rule,
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
                {_build_metric_svg(metric.metric_name, points, trend_status_rule, min_pass_rate)}
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
      --line-latency: #38bdf8;
      --line-tokens: #f59e0b;
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
    .axis-label-right {{ text-anchor: end; }}
    .avg-line {{ fill: none; stroke: var(--line-avg); stroke-width: 2.5; }}
    .pass-line {{ fill: none; stroke: var(--line-pass); stroke-width: 2; }}
    .smooth-line {{ stroke-linecap: round; stroke-linejoin: round; }}
    .combined-line {{ fill: none; stroke-width: 2.6; opacity: 0.96; }}
    .performance-trend-svg {{ min-width: 900px; }}
    .perf-line-latency {{ fill: none; stroke: var(--line-latency); stroke-width: 2.7; }}
    .perf-line-tokens {{ fill: none; stroke: var(--line-tokens); stroke-width: 2.7; }}
    .perf-dot-latency {{ fill: var(--line-latency); stroke: #0f172a; stroke-width: 1.3; }}
    .perf-dot-tokens {{ fill: var(--line-tokens); stroke: #0f172a; stroke-width: 1.3; }}
    .threshold-line {{ stroke: var(--line-threshold); stroke-width: 1.5; stroke-dasharray: 5 5; }}
    .threshold-line-fill {{ fill: var(--line-threshold); }}
    .combined-legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }}
    .perf-legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 10px; }}
    .perf-legend-item {{ display: inline-flex; align-items: center; gap: 8px; font-size: 12px; color: #dbe7ff; }}
    .perf-swatch {{ width: 14px; height: 3px; border-radius: 3px; display: inline-block; }}
    .perf-swatch-latency {{ background: var(--line-latency); }}
    .perf-swatch-tokens {{ background: var(--line-tokens); }}
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
  {performance_trend_card}
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
