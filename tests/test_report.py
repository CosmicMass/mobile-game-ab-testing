"""
Unit tests for game_analytics_toolkit.report

Figures are checked structurally -- trace counts, plotted values, the presence
of the zero line and the intervals -- rather than by image comparison. What
these guard is that the numbers reaching the chart are the numbers the
analysis produced, and that the annotations a reader relies on are actually
there.

Run this file only:
    python -m pytest tests/test_report.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from game_analytics_toolkit.ab_test import analyze_metric
from game_analytics_toolkit.report import (
    DARK_PALETTE,
    LIGHT_PALETTE,
    build_html_report,
    effect_figure,
    palette_for,
    playtime_figure,
    retention_figure,
    retention_summary_frame,
)
from game_analytics_toolkit.retention import retention_by_group, retention_table

SEED = 20260909


def frame() -> pd.DataFrame:
    """A synthetic two-arm frame big enough for stable figures."""
    rng = np.random.default_rng(SEED)
    n = 2_000
    return pd.DataFrame(
        {
            "userid": range(2 * n),
            "version": ["gate_30"] * n + ["gate_40"] * n,
            "sum_gamerounds": np.concatenate(
                [rng.geometric(0.05, n), rng.geometric(0.05, n)]
            ),
            "retention_1": np.concatenate([rng.random(n) < 0.45, rng.random(n) < 0.44]),
            "retention_7": np.concatenate([rng.random(n) < 0.19, rng.random(n) < 0.16]),
        }
    )


def rates(df: pd.DataFrame):
    """Per-group rates for both metrics."""
    return retention_by_group(df, "retention_1") + retention_by_group(df, "retention_7")


class TestPalette:

    def test_light_and_dark_are_distinct(self):
        assert palette_for("light") is LIGHT_PALETTE
        assert palette_for("dark") is DARK_PALETTE
        assert LIGHT_PALETTE.surface != DARK_PALETTE.surface

    def test_dark_series_are_restepped_not_reused(self):
        # The dark palette is the same two hues stepped for a dark surface.
        # Identical hexes would mean the dark theme was never selected.
        assert LIGHT_PALETTE.series != DARK_PALETTE.series
        assert len(DARK_PALETTE.series) == len(LIGHT_PALETTE.series) == 2


class TestRetentionFigure:

    def test_one_trace_per_group(self):
        figure = retention_figure(rates(frame()))
        assert len(figure.data) == 2
        assert {trace.name for trace in figure.data} == {"gate_30", "gate_40"}

    def test_plots_the_measured_rates(self):
        df = frame()
        figure = retention_figure(rates(df))
        gate_30 = next(t for t in figure.data if t.name == "gate_30")
        expected = [
            r.rate * 100
            for r in rates(df)
            if r.group == "gate_30"
        ]
        assert list(gate_30.y) == pytest.approx(expected)

    def test_error_bars_carry_the_wilson_interval(self):
        df = frame()
        source = [r for r in rates(df) if r.group == "gate_30"]
        gate_30 = next(t for t in retention_figure(rates(df)).data if t.name == "gate_30")
        assert list(gate_30.error_y.array) == pytest.approx(
            [(r.ci_high - r.rate) * 100 for r in source]
        )
        assert list(gate_30.error_y.arrayminus) == pytest.approx(
            [(r.rate - r.ci_low) * 100 for r in source]
        )

    def test_every_bar_is_directly_labelled(self):
        # 2 groups x 2 metrics; identity and magnitude must not rest on the
        # colour key alone.
        figure = retention_figure(rates(frame()))
        assert len(figure.layout.annotations) == 4

    def test_overall_rows_are_dropped(self):
        # An aggregate bar beside the arms it is made from invites a
        # comparison that means nothing.
        table_rows = retention_table(frame(), include_overall=True)
        assert "overall" in set(table_rows["group"])
        figure = retention_figure(
            retention_by_group(frame(), "retention_1")
            + [
                r
                for r in [
                    *retention_by_group(frame(), "retention_7"),
                ]
            ]
        )
        assert "overall" not in {trace.name for trace in figure.data}

    def test_raises_when_only_overall_rows_are_given(self):
        from game_analytics_toolkit.retention import retention_rate

        with pytest.raises(ValueError, match="no per-group rates"):
            retention_figure([retention_rate(frame(), "retention_1")])

    def test_both_themes_render(self):
        for theme in ("light", "dark"):
            figure = retention_figure(rates(frame()), theme=theme)
            assert figure.layout.plot_bgcolor == palette_for(theme).surface


class TestEffectFigure:

    def results(self):
        df = frame()
        return [analyze_metric(df, "retention_1"), analyze_metric(df, "retention_7")]

    def test_single_trace_no_legend(self):
        figure = effect_figure(self.results())
        assert len(figure.data) == 1
        assert figure.layout.showlegend is False

    def test_plots_the_absolute_effects(self):
        results = self.results()
        figure = effect_figure(results)
        assert list(figure.data[0].x) == pytest.approx(
            [r.effect.absolute * 100 for r in results]
        )

    def test_error_bars_carry_the_confidence_interval(self):
        results = self.results()
        trace = effect_figure(results).data[0]
        assert list(trace.error_x.array) == pytest.approx(
            [(r.effect.absolute_ci_high - r.effect.absolute) * 100 for r in results]
        )

    def test_zero_reference_line_is_present(self):
        # The whole point of the chart: an interval crossing zero is visibly
        # a non-finding. Without the line the reader has to infer it.
        figure = effect_figure(self.results())
        vlines = [s for s in figure.layout.shapes if s.x0 == 0 and s.x1 == 0]
        assert len(vlines) == 1

    def test_labels_are_not_clipped_at_the_plot_edge(self):
        assert effect_figure(self.results()).data[0].cliponaxis is False

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="no results"):
            effect_figure([])


class TestPlaytimeFigure:

    def test_one_trace_per_group(self):
        figure = playtime_figure(frame())
        assert len(figure.data) == 2
        assert {trace.name for trace in figure.data} == {"gate_30", "gate_40"}

    def test_legend_keeps_the_canonical_group_order(self):
        # Traces are added bottom-up so the first group sits at the top of the
        # y axis; the legend must not inherit that reversal.
        figure = playtime_figure(frame())
        by_rank = sorted(figure.data, key=lambda t: t.legendrank)
        assert [t.name for t in by_rank] == ["gate_30", "gate_40"]

    def test_clipping_excludes_rows_rather_than_narrowing_the_axis(self):
        # Narrowing the axis instead would leave whiskers running off the
        # plot and quartiles describing data the reader cannot see.
        df = frame()
        figure = playtime_figure(df, clip_quantile=0.90)
        cutoff = float(df["sum_gamerounds"].quantile(0.90))
        plotted = sum(len(trace.x) for trace in figure.data)
        assert plotted == int((df["sum_gamerounds"] <= cutoff).sum())
        assert plotted < len(df)
        assert max(max(trace.x) for trace in figure.data) <= cutoff

    def test_no_clipping_plots_every_row(self):
        df = frame()
        figure = playtime_figure(df, clip_quantile=1.0)
        assert sum(len(trace.x) for trace in figure.data) == len(df)

    def test_subtitle_states_the_exclusion(self):
        title = str(playtime_figure(frame(), clip_quantile=0.99).layout.title.text)
        assert "excluded" in title

    def test_rejects_bad_clip_quantile(self):
        for bad in (0.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="clip_quantile"):
                playtime_figure(frame(), clip_quantile=bad)


class TestRetentionSummaryFrame:

    def test_columns(self):
        table = retention_summary_frame(rates(frame()))
        assert list(table.columns) == [
            "group",
            "metric",
            "users",
            "retained",
            "rate",
            "95% CI",
        ]

    def test_one_row_per_rate(self):
        source = rates(frame())
        assert len(retention_summary_frame(source)) == len(source)

    def test_rates_are_formatted_as_percentages(self):
        table = retention_summary_frame(rates(frame()))
        assert all(value.endswith("%") for value in table["rate"])

    def test_metric_names_are_humanized(self):
        table = retention_summary_frame(rates(frame()))
        assert set(table["metric"]) == {"D1 retention", "D7 retention"}


class TestBuildHtmlReport:

    def artifacts(self):
        df = frame()
        results = [analyze_metric(df, "retention_1"), analyze_metric(df, "retention_7")]
        figures = [retention_figure(rates(df)), effect_figure(results)]
        return df, results, figures

    def test_writes_a_file(self, tmp_path: Path):
        _, results, figures = self.artifacts()
        path = build_html_report(
            tmp_path / "report.html",
            figures=figures,
            verdicts=[r.verdict for r in results],
            inline_plotlyjs=False,
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_creates_missing_parent_directories(self, tmp_path: Path):
        _, _, figures = self.artifacts()
        path = build_html_report(
            tmp_path / "nested" / "deeper" / "report.html",
            figures=figures,
            inline_plotlyjs=False,
        )
        assert path.exists()

    def test_contains_the_verdicts(self, tmp_path: Path):
        _, results, figures = self.artifacts()
        path = build_html_report(
            tmp_path / "report.html",
            figures=figures,
            verdicts=[r.verdict for r in results],
            inline_plotlyjs=False,
        )
        html = path.read_text(encoding="utf-8")
        for result in results:
            assert result.verdict in html

    def test_contains_the_tables(self, tmp_path: Path):
        df, _, figures = self.artifacts()
        table = retention_summary_frame(rates(df))
        path = build_html_report(
            tmp_path / "report.html",
            figures=figures,
            tables=[("Retention", table)],
            inline_plotlyjs=False,
        )
        html = path.read_text(encoding="utf-8")
        assert "Retention" in html
        assert str(table["rate"].iloc[0]) in html

    def test_embeds_every_figure(self, tmp_path: Path):
        _, _, figures = self.artifacts()
        path = build_html_report(
            tmp_path / "report.html", figures=figures, inline_plotlyjs=False
        )
        assert path.read_text(encoding="utf-8").count("plotly-graph-div") == len(figures)

    def test_records_provenance_in_the_subtitle(self, tmp_path: Path):
        _, _, figures = self.artifacts()
        path = build_html_report(
            tmp_path / "report.html",
            figures=figures,
            subtitle="Synthetic data, not the Kaggle download",
            inline_plotlyjs=False,
        )
        assert "Synthetic data" in path.read_text(encoding="utf-8")

    def test_inlining_plotlyjs_makes_a_self_contained_file(self, tmp_path: Path):
        _, _, figures = self.artifacts()
        linked = build_html_report(
            tmp_path / "linked.html", figures=figures, inline_plotlyjs=False
        )
        inlined = build_html_report(
            tmp_path / "inlined.html", figures=figures, inline_plotlyjs=True
        )
        assert inlined.stat().st_size > linked.stat().st_size

        # The bundled library mentions the CDN host in its own default config,
        # so the question is whether the page LOADS from it -- a script src --
        # not whether the string appears anywhere in the file.
        assert 'src="https://cdn.plot.ly' in linked.read_text(encoding="utf-8")
        assert 'src="https://cdn.plot.ly' not in inlined.read_text(encoding="utf-8")

    def test_dark_theme_styles_the_page(self, tmp_path: Path):
        _, _, figures = self.artifacts()
        path = build_html_report(
            tmp_path / "report.html",
            figures=[retention_figure(rates(frame()), theme="dark")],
            theme="dark",
            inline_plotlyjs=False,
        )
        html = path.read_text(encoding="utf-8")
        assert DARK_PALETTE.plane in html
        assert "color-scheme: dark" in html
