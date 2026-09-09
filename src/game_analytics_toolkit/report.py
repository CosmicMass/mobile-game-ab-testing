"""Summary tables and Plotly figures for a finished analysis.

Two conventions here that are deliberate rather than stylistic.

**Every rate is drawn with its interval.** A bar chart of two retention rates
side by side invites the reader to treat any visible height difference as a
finding. Drawing the Wilson interval as an error bar puts the uncertainty in
the same picture as the estimate, so a difference that is inside the noise
*looks* like one.

**The effect chart is anchored on zero.** The question a stakeholder actually
has is not "which bar is taller" but "is the difference distinguishable from
no difference, and how big could it plausibly be". A point estimate with its
interval against a zero line answers exactly that: if the interval crosses
zero, the chart says so without a caption.

Figures are returned rather than shown or written, so the caller decides what
to do with them; :func:`build_html_report` is the one function that touches
the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

from .ab_test import ABTestResult
from .loader import GROUP_COLUMN, ROUNDS_COLUMN
from .retention import RetentionRate

Theme = Literal["light", "dark"]


@dataclass(frozen=True)
class Palette:
    """Colors for one theme.

    The two series slots are the first two of a categorical scale whose
    ordering is the colorblind-safety mechanism, not a cosmetic choice; the
    pair clears CVD and normal-vision separation on every pairing in both
    themes. The dark values are the same two hues re-stepped for a dark
    surface, not an automatic inversion of the light ones.

    Attributes:
        surface: The plotting area's background.
        plane: The page behind the plotting area.
        series: Categorical slots, assigned to groups in a fixed order.
        primary_ink: Titles and value labels.
        secondary_ink: Subtitles and legend text.
        muted_ink: Axis ticks and axis titles.
        gridline: Hairline grid.
        baseline: The axis line and the zero reference line.
    """

    surface: str
    plane: str
    series: tuple[str, str]
    primary_ink: str
    secondary_ink: str
    muted_ink: str
    gridline: str
    baseline: str


LIGHT_PALETTE = Palette(
    surface="#fcfcfb",
    plane="#f9f9f7",
    series=("#2a78d6", "#eb6834"),
    primary_ink="#0b0b0b",
    secondary_ink="#52514e",
    muted_ink="#898781",
    gridline="#e1e0d9",
    baseline="#c3c2b7",
)

DARK_PALETTE = Palette(
    surface="#1a1a19",
    plane="#0d0d0d",
    series=("#3987e5", "#d95926"),
    primary_ink="#ffffff",
    secondary_ink="#c3c2b7",
    muted_ink="#898781",
    gridline="#2c2c2a",
    baseline="#383835",
)

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Fraction of each category slot left empty between bar groups. Kept as a
# module constant because retention_figure's value-label placement has to do
# the same arithmetic Plotly does to find a bar's centre.
BAR_GAP = 0.45

# Friendlier axis labels for the dataset's column names.
METRIC_LABELS = {"retention_1": "D1 retention", "retention_7": "D7 retention"}


def palette_for(theme: Theme) -> Palette:
    """Returns the :class:`Palette` for a theme name.

    Args:
        theme: ``"light"`` or ``"dark"``.

    Returns:
        The matching palette.
    """
    return LIGHT_PALETTE if theme == "light" else DARK_PALETTE


def _metric_label(metric: str) -> str:
    """Human-readable name for a retention column."""
    return METRIC_LABELS.get(metric, metric)


def _apply_layout(
    figure: go.Figure,
    *,
    palette: Palette,
    title: str,
    subtitle: Optional[str] = None,
    height: int = 420,
    show_legend: bool = True,
    x_grid: bool = False,
) -> go.Figure:
    """Applies the shared chrome: recessive axes, surface colors, typography.

    Called last by each figure builder, so it is also the single place axis
    styling is decided -- which is why ``x_grid`` is a parameter rather than
    each builder setting its own grid beforehand and having it silently
    overwritten here.

    Args:
        figure: The figure to style, modified in place.
        palette: Theme colors.
        title: Chart title.
        subtitle: Optional second line, rendered in secondary ink.
        height: Figure height in pixels.
        show_legend: Whether the legend box is drawn.
        x_grid: Draw a grid along x. On only where x is a value scale; a grid
            along a category axis measures nothing.

    Returns:
        The same figure, for chaining.
    """
    heading = title
    if subtitle:
        heading = (
            f"{title}<br><span style='font-size:12px;color:{palette.secondary_ink}'>"
            f"{subtitle}</span>"
        )

    figure.update_layout(
        title={"text": heading, "font": {"size": 16, "color": palette.primary_ink}},
        paper_bgcolor=palette.plane,
        plot_bgcolor=palette.surface,
        font={"family": FONT_FAMILY, "size": 12, "color": palette.secondary_ink},
        height=height,
        margin={"l": 70, "r": 30, "t": 80 if subtitle else 60, "b": 60},
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "right",
            "x": 1.0,
            "font": {"color": palette.secondary_ink},
        },
        hoverlabel={"font": {"family": FONT_FAMILY}},
    )
    axis_style = {
        "gridcolor": palette.gridline,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": palette.baseline,
        "tickfont": {"color": palette.muted_ink},
        "title": {"font": {"color": palette.muted_ink, "size": 12}},
    }
    figure.update_xaxes(showgrid=x_grid, **axis_style)
    figure.update_yaxes(showgrid=not x_grid, **axis_style)
    return figure


def retention_figure(
    rates: Sequence[RetentionRate],
    *,
    theme: Theme = "light",
    title: str = "Retention by experiment group",
) -> go.Figure:
    """Grouped bars of retention rate per group, with Wilson intervals.

    One bar group per metric, one bar per experiment group. Values are
    labelled directly on the bars so identity and magnitude never rest on
    color alone.

    Args:
        rates: Rates to plot. Rows with group ``"overall"`` are dropped --
            an aggregate bar beside the arms it is made of invites a
            comparison that is not meaningful.
        theme: Which palette to draw with.
        title: Chart title.

    Returns:
        The figure.

    Raises:
        ValueError: If ``rates`` contains no per-group rows.
    """
    palette = palette_for(theme)
    per_group = [r for r in rates if r.group != "overall"]
    if not per_group:
        raise ValueError("no per-group rates to plot")

    groups = list(dict.fromkeys(r.group for r in per_group))
    metrics = list(dict.fromkeys(r.metric for r in per_group))
    figure = go.Figure()

    for index, group in enumerate(groups):
        by_metric = {r.metric: r for r in per_group if r.group == group}
        present = [m for m in metrics if m in by_metric]
        rows = [by_metric[m] for m in present]
        figure.add_trace(
            go.Bar(
                name=group,
                x=[_metric_label(m) for m in present],
                y=[r.rate * 100 for r in rows],
                marker={
                    "color": palette.series[index % len(palette.series)],
                    "line": {"width": 2, "color": palette.surface},
                },
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [(r.ci_high - r.rate) * 100 for r in rows],
                    "arrayminus": [(r.rate - r.ci_low) * 100 for r in rows],
                    "color": palette.secondary_ink,
                    "thickness": 1.5,
                    "width": 5,
                },
                customdata=[
                    [r.n_users, r.retained, r.ci_low * 100, r.ci_high * 100] for r in rows
                ],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>%{x}: %{y:.2f}%"
                    "<br>95% CI %{customdata[2]:.2f}% – %{customdata[3]:.2f}%"
                    "<br>%{customdata[1]:,} of %{customdata[0]:,} users<extra></extra>"
                ),
            )
        )

        # Value labels are annotations rather than the bar trace's own `text`,
        # because Plotly places `textposition="outside"` at the bar top and
        # ignores the error bar, so every label landed on top of its own
        # whisker. Anchoring to ci_high instead guarantees clearance.
        #
        # On a categorical axis a bar's centre is its category index plus an
        # in-group offset: the group occupies (1 - BAR_GAP) of the slot,
        # divided into len(groups) bars.
        span = 1.0 - BAR_GAP
        width = span / len(groups)
        offset = -span / 2 + (index + 0.5) * width
        for position, row in enumerate(rows):
            figure.add_annotation(
                x=position + offset,
                y=row.ci_high * 100,
                yshift=8,
                text=f"{row.rate * 100:.2f}%",
                showarrow=False,
                font={"color": palette.primary_ink, "size": 11},
            )

    figure.update_yaxes(title_text="Retention rate (%)", rangemode="tozero")
    figure.update_layout(bargap=BAR_GAP, bargroupgap=0.04, barcornerradius=4)
    return _apply_layout(
        figure,
        palette=palette,
        title=title,
        subtitle="Error bars are 95% Wilson intervals",
    )


def effect_figure(
    results: Sequence[ABTestResult],
    *,
    theme: Theme = "light",
    title: str = "Treatment effect, with confidence interval",
) -> go.Figure:
    """Point estimate and interval of the absolute effect, anchored on zero.

    A horizontal interval per metric against a zero reference line: an
    interval that crosses zero is a difference the data cannot distinguish
    from none. One series, so no legend -- the axis names each row.

    Args:
        results: Completed tests, one per metric.
        theme: Which palette to draw with.
        title: Chart title.

    Returns:
        The figure.

    Raises:
        ValueError: If ``results`` is empty.
    """
    palette = palette_for(theme)
    if not results:
        raise ValueError("no results to plot")

    labels = [_metric_label(r.metric) for r in results]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[r.effect.absolute * 100 for r in results],
            y=labels,
            mode="markers+text",
            marker={
                "size": 11,
                "color": palette.series[0],
                "line": {"width": 2, "color": palette.surface},
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [
                    (r.effect.absolute_ci_high - r.effect.absolute) * 100 for r in results
                ],
                "arrayminus": [
                    (r.effect.absolute - r.effect.absolute_ci_low) * 100 for r in results
                ],
                "color": palette.series[0],
                "thickness": 2,
                "width": 6,
            },
            text=[f"{r.effect.absolute * 100:+.2f} pp" for r in results],
            textposition="top center",
            textfont={"color": palette.primary_ink, "size": 11},
            # Without this the label on the topmost row is sliced off by the
            # plot area's edge.
            cliponaxis=False,
            customdata=[
                [r.chi_square.p_value, r.effect.relative * 100, r.power.achieved_power * 100]
                for r in results
            ],
            hovertemplate=(
                "<b>%{y}</b><br>absolute %{x:+.3f} pp"
                "<br>relative %{customdata[1]:+.2f}%"
                "<br>p = %{customdata[0]:.4g}"
                "<br>power %{customdata[2]:.0f}%<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.add_vline(
        x=0,
        line_width=2,
        line_color=palette.baseline,
        annotation_text="no effect",
        annotation_position="top",
        annotation_font={"color": palette.muted_ink, "size": 11},
    )
    figure.update_xaxes(title_text="Treatment minus control (percentage points)")
    figure.update_yaxes(autorange="reversed")
    return _apply_layout(
        figure,
        palette=palette,
        title=title,
        subtitle="An interval crossing the zero line is not distinguishable from no effect",
        height=320,
        show_legend=False,
        x_grid=True,
    )


def playtime_figure(
    df: pd.DataFrame,
    *,
    rounds_column: str = ROUNDS_COLUMN,
    group_column: str = GROUP_COLUMN,
    clip_quantile: float = 0.99,
    theme: Theme = "light",
    title: str = "Game rounds played, by experiment group",
) -> go.Figure:
    """Distribution of rounds played per group, as a box per group.

    ``sum_gamerounds`` is extremely right-skewed -- in the published data a
    single user logged around 49,000 rounds against a median near 16 -- so
    the top ``1 - clip_quantile`` of users are excluded from the view and the
    exclusion is stated in the subtitle.

    The heavy tail is *excluded from the data*, not hidden by narrowing the
    axis. Clipping the axis instead leaves each box's whisker running off the
    edge of the plot, which reads as a rendering fault and, worse, draws a
    box whose statistics belong to a range the reader cannot see. Dropping
    the rows makes the quartiles in the picture the quartiles of the data in
    the picture.

    Args:
        df: A loaded per-user frame.
        rounds_column: The play-count column.
        group_column: The A/B group column.
        clip_quantile: Upper quantile of the pooled distribution to keep, in
            ``(0, 1]``. Pass 1.0 to plot every row.
        theme: Which palette to draw with.
        title: Chart title.

    Returns:
        The figure.

    Raises:
        ValueError: If ``clip_quantile`` is outside ``(0, 1]``.
    """
    if not 0.0 < clip_quantile <= 1.0:
        raise ValueError("clip_quantile must be in (0, 1]")

    palette = palette_for(theme)
    groups = list(dict.fromkeys(df[group_column].tolist()))
    upper = float(df[rounds_column].quantile(clip_quantile))
    shown = df if clip_quantile == 1.0 else df[df[rounds_column] <= upper]
    excluded = len(df) - len(shown)

    figure = go.Figure()
    # Reversed so the first group lands at the TOP of the y axis, matching the
    # left-to-right order it has in every other figure.
    for index, group in reversed(list(enumerate(groups))):
        values = shown.loc[shown[group_column] == group, rounds_column]
        figure.add_trace(
            go.Box(
                name=str(group),
                x=values,
                marker={"color": palette.series[index % len(palette.series)]},
                line={"width": 2},
                fillcolor=palette.surface,
                boxpoints=False,
                # Traces are added bottom-up; legendrank keeps the legend in
                # the groups' canonical order rather than the drawing order.
                legendrank=index,
                hovertemplate="<b>%{fullData.name}</b><br>%{x:,.0f} rounds<extra></extra>",
            )
        )

    subtitle = (
        f"Top {1 - clip_quantile:.0%} by rounds played excluded from this view "
        f"({excluded:,} users above {upper:,.0f} rounds); quartiles are of the data shown"
        if clip_quantile < 1.0
        else "Full range shown"
    )
    figure.update_xaxes(title_text="Rounds played", rangemode="tozero")
    return _apply_layout(
        figure,
        palette=palette,
        title=title,
        subtitle=subtitle,
        height=320,
        x_grid=True,
    )


def retention_summary_frame(rates: Sequence[RetentionRate]) -> pd.DataFrame:
    """Formats retention rates as a presentation table.

    Args:
        rates: Rates to tabulate.

    Returns:
        A frame with the rate and interval rendered as percentage strings
        alongside the raw counts.
    """
    return pd.DataFrame(
        [
            {
                "group": r.group,
                "metric": _metric_label(r.metric),
                "users": f"{r.n_users:,}",
                "retained": f"{r.retained:,}",
                "rate": f"{r.rate * 100:.2f}%",
                f"{r.confidence:.0%} CI": f"{r.ci_low * 100:.2f}% – {r.ci_high * 100:.2f}%",
            }
            for r in rates
        ]
    )


def _table_html(df: pd.DataFrame, caption: str) -> str:
    """Renders a frame as an HTML table styled to match the figures."""
    header_cells = "".join(f"<th>{column}</th>" for column in df.columns)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in df.astype(str).itertuples(index=False, name=None)
    )
    return f"""
<section class="panel">
  <h2>{caption}</h2>
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{body_rows}</tbody>
  </table>
</section>
"""


def build_html_report(
    output_path: str | Path,
    *,
    figures: Sequence[go.Figure],
    tables: Sequence[tuple[str, pd.DataFrame]] = (),
    verdicts: Sequence[str] = (),
    theme: Theme = "light",
    title: str = "Cookie Cats A/B test readout",
    subtitle: str = "",
    inline_plotlyjs: bool = True,
) -> Path:
    """Writes one self-contained HTML page with the verdicts, tables and figures.

    The verdicts go at the top on purpose. A reader who scrolls no further
    should still leave with the conclusion and its caveat rather than with an
    impression formed from bar heights.

    Args:
        output_path: File to write. Parent directories are created.
        figures: Figures to embed, in order.
        tables: ``(caption, frame)`` pairs rendered as HTML tables.
        verdicts: Verdict sentences to show in a callout at the top.
        theme: Which palette to style the page with. Should match the theme
            the figures were built with.
        title: Page heading.
        subtitle: Optional line under the heading -- the place to record data
            provenance.
        inline_plotlyjs: Embed plotly.js in the file (larger, works offline)
            rather than loading it from a CDN.

    Returns:
        The path written.
    """
    palette = palette_for(theme)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure_html: List[str] = []
    for index, figure in enumerate(figures):
        include: bool | str = False
        if index == 0:
            include = True if inline_plotlyjs else "cdn"
        figure_html.append(
            f'<section class="panel">'
            f"{figure.to_html(full_html=False, include_plotlyjs=include, config={'displayModeBar': False})}"
            f"</section>"
        )

    verdict_html = ""
    if verdicts:
        items = "".join(f"<li>{verdict}</li>" for verdict in verdicts)
        verdict_html = f'<section class="panel callout"><h2>Verdict</h2><ul>{items}</ul></section>'

    tables_html = "".join(_table_html(frame, caption) for caption, frame in tables)

    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: {theme}; }}
  body {{
    margin: 0; padding: 32px 20px 64px;
    background: {palette.plane}; color: {palette.primary_ink};
    font: 14px/1.55 {FONT_FAMILY};
  }}
  .wrap {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .lede {{ color: {palette.secondary_ink}; margin: 0 0 28px; }}
  .panel {{
    background: {palette.surface}; border: 1px solid {palette.gridline};
    border-radius: 10px; padding: 18px; margin-bottom: 20px; overflow-x: auto;
  }}
  .panel h2 {{ font-size: 14px; margin: 0 0 12px; color: {palette.secondary_ink};
    text-transform: uppercase; letter-spacing: .06em; }}
  .callout ul {{ margin: 0; padding-left: 20px; }}
  .callout li {{ margin-bottom: 10px; }}
  table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: right; padding: 7px 10px;
    border-bottom: 1px solid {palette.gridline}; white-space: nowrap; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  th {{ color: {palette.muted_ink}; font-weight: 600; font-size: 12px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{title}</h1>
  <p class="lede">{subtitle}</p>
  {verdict_html}
  {tables_html}
  {"".join(figure_html)}
</div>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path
