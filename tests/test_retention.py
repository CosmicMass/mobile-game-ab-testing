"""
Unit tests for mobile_game_ab_testing.retention

Run this file only:
    python -m pytest tests/test_retention.py -v
"""

import math

import pandas as pd
import pytest

from mobile_game_ab_testing.retention import (
    playtime_by_group,
    retention_by_group,
    retention_rate,
    retention_table,
    wilson_interval,
)


def frame() -> pd.DataFrame:
    """Ten users, with retention rates chosen to be exact fractions.

    gate_30: 3 of 5 retained on D1, 1 of 5 on D7.
    gate_40: 2 of 5 retained on D1, 0 of 5 on D7.
    """
    return pd.DataFrame(
        {
            "userid": range(10),
            "version": ["gate_30"] * 5 + ["gate_40"] * 5,
            "sum_gamerounds": [1, 2, 3, 4, 100, 5, 6, 7, 8, 200],
            "retention_1": [True, True, True, False, False, True, True, False, False, False],
            "retention_7": [True, False, False, False, False, False, False, False, False, False],
        }
    )


class TestWilsonInterval:
    """Pinned against published Wilson score interval values.

    The normal-approximation interval is what these numbers exist to rule
    out: at 0 successes it returns the degenerate (0, 0), claiming certainty
    from ten observations.
    """

    def test_half_of_ten(self):
        low, high = wilson_interval(5, 10)
        assert low == pytest.approx(0.2366, abs=1e-4)
        assert high == pytest.approx(0.7634, abs=1e-4)

    def test_zero_successes_still_has_a_real_upper_bound(self):
        low, high = wilson_interval(0, 10)
        assert low == 0.0
        assert high == pytest.approx(0.2775, abs=1e-4)

    def test_all_successes_still_has_a_real_lower_bound(self):
        low, high = wilson_interval(10, 10)
        assert low == pytest.approx(0.7225, abs=1e-4)
        assert high == 1.0

    def test_symmetric_about_one_half(self):
        low_a, high_a = wilson_interval(3, 10)
        low_b, high_b = wilson_interval(7, 10)
        assert low_a == pytest.approx(1 - high_b, abs=1e-12)
        assert high_a == pytest.approx(1 - low_b, abs=1e-12)

    def test_interval_contains_the_point_estimate(self):
        for successes in range(0, 11):
            low, high = wilson_interval(successes, 10)
            assert low <= successes / 10 <= high

    def test_interval_narrows_as_n_grows(self):
        widths = [
            wilson_interval(n // 2, n)[1] - wilson_interval(n // 2, n)[0]
            for n in (100, 1_000, 10_000)
        ]
        assert widths[0] > widths[1] > widths[2]

    def test_stays_inside_the_unit_interval(self):
        for n in (5, 50, 500):
            for successes in (0, n):
                low, high = wilson_interval(successes, n)
                assert 0.0 <= low <= high <= 1.0

    def test_higher_confidence_is_wider(self):
        narrow = wilson_interval(50, 100, confidence=0.80)
        wide = wilson_interval(50, 100, confidence=0.99)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_rejects_zero_n(self):
        with pytest.raises(ValueError, match="positive"):
            wilson_interval(0, 0)

    def test_rejects_successes_above_n(self):
        with pytest.raises(ValueError, match="within"):
            wilson_interval(11, 10)

    def test_rejects_negative_successes(self):
        with pytest.raises(ValueError, match="within"):
            wilson_interval(-1, 10)

    def test_rejects_confidence_outside_zero_one(self):
        with pytest.raises(ValueError, match="confidence"):
            wilson_interval(5, 10, confidence=1.0)


class TestRetentionRate:

    def test_overall_rate(self):
        result = retention_rate(frame(), "retention_1")
        assert result.group == "overall"
        assert result.n_users == 10
        assert result.retained == 5
        assert result.rate == 0.5

    def test_single_group_rate(self):
        result = retention_rate(frame(), "retention_1", group="gate_30")
        assert result.n_users == 5
        assert result.retained == 3
        assert result.rate == pytest.approx(0.6)

    def test_zero_retention_group(self):
        result = retention_rate(frame(), "retention_7", group="gate_40")
        assert result.retained == 0
        assert result.rate == 0.0
        assert result.ci_high > 0.0

    def test_carries_its_metric_and_confidence(self):
        result = retention_rate(frame(), "retention_7", confidence=0.99)
        assert result.metric == "retention_7"
        assert result.confidence == 0.99

    def test_nulls_leave_the_denominator(self):
        # A missing outcome is not a failed one -- counting it as False would
        # silently deflate the rate.
        df = frame()
        df["retention_1"] = pd.array(
            [True, True, None, False, False, True, True, False, False, False],
            dtype="boolean",
        )
        result = retention_rate(df, "retention_1")
        assert result.n_users == 9
        assert result.retained == 4

    def test_unknown_column_raises(self):
        with pytest.raises(KeyError, match="retention_30"):
            retention_rate(frame(), "retention_30")

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError, match="gate_99"):
            retention_rate(frame(), "retention_1", group="gate_99")


class TestRetentionByGroup:

    def test_one_row_per_group_in_first_seen_order(self):
        rates = retention_by_group(frame(), "retention_1")
        assert [r.group for r in rates] == ["gate_30", "gate_40"]

    def test_rates_are_per_group(self):
        rates = {r.group: r.rate for r in retention_by_group(frame(), "retention_1")}
        assert rates["gate_30"] == pytest.approx(0.6)
        assert rates["gate_40"] == pytest.approx(0.4)

    def test_denominators_are_per_group(self):
        for rate in retention_by_group(frame(), "retention_7"):
            assert rate.n_users == 5


class TestRetentionTable:

    def test_columns(self):
        table = retention_table(frame())
        assert list(table.columns) == [
            "group",
            "metric",
            "n_users",
            "retained",
            "rate",
            "ci_low",
            "ci_high",
            "confidence",
        ]

    def test_row_count_with_overall(self):
        # 2 metrics x (1 overall + 2 groups)
        assert len(retention_table(frame())) == 6

    def test_row_count_without_overall(self):
        assert len(retention_table(frame(), include_overall=False)) == 4

    def test_metric_subset(self):
        table = retention_table(frame(), metrics=("retention_7",), include_overall=False)
        assert set(table["metric"]) == {"retention_7"}
        assert len(table) == 2

    def test_values_match_the_single_rate_calls(self):
        table = retention_table(frame(), include_overall=False)
        row = table[(table["group"] == "gate_30") & (table["metric"] == "retention_1")]
        assert float(row["rate"].iloc[0]) == pytest.approx(0.6)


class TestPlaytimeByGroup:

    def test_columns(self):
        table = playtime_by_group(frame())
        assert list(table.columns) == [
            "group",
            "n_users",
            "mean_rounds",
            "median_rounds",
            "p90_rounds",
            "max_rounds",
        ]

    def test_one_row_per_group(self):
        assert playtime_by_group(frame())["group"].tolist() == ["gate_30", "gate_40"]

    def test_counts_and_medians(self):
        table = playtime_by_group(frame()).set_index("group")
        assert int(table.loc["gate_30", "n_users"]) == 5
        # gate_30 rounds are 1, 2, 3, 4, 100 -> median 3, max 100
        assert float(table.loc["gate_30", "median_rounds"]) == 3.0
        assert int(table.loc["gate_30", "max_rounds"]) == 100

    def test_mean_is_dragged_by_the_tail_but_median_is_not(self):
        # The reason the median is reported alongside the mean: one whale
        # moves the mean by 20x the median here.
        table = playtime_by_group(frame()).set_index("group")
        mean = float(table.loc["gate_30", "mean_rounds"])
        median = float(table.loc["gate_30", "median_rounds"])
        assert mean == pytest.approx(22.0)
        assert median == 3.0
        assert mean > median * 5

    def test_p90_is_between_median_and_max(self):
        table = playtime_by_group(frame()).set_index("group")
        for group in ("gate_30", "gate_40"):
            median = float(table.loc[group, "median_rounds"])
            p90 = float(table.loc[group, "p90_rounds"])
            maximum = float(table.loc[group, "max_rounds"])
            assert median <= p90 <= maximum
            assert not math.isnan(p90)
