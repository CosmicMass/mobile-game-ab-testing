"""
Unit tests for mobile_game_ab_testing.ab_test

The statistical functions are checked three ways: against closed-form values
computed by hand, against each other (the sample-size and power formulas must
invert one another), and against simulation -- ``TestAgainstSimulation`` draws
thousands of experiments from a known distribution and checks that the test's
false-positive rate really is alpha and that its predicted power really is the
rate at which it detects a known effect. A statistics module that is merely
self-consistent can still be uniformly wrong; those two are the tests that
would catch it.

Run this file only:
    python -m pytest tests/test_ab_test.py -v
"""

import math

import numpy as np
import pandas as pd
import pytest

from mobile_game_ab_testing.ab_test import (
    GroupOutcome,
    TwoGroupOutcome,
    achieved_power,
    analyze_metric,
    chi_square_test,
    effect_size,
    outcome_from_frame,
    power_check,
    required_sample_size,
    run_ab_test,
    summarize,
    summary_frame,
)

SEED = 20260909


def outcome(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
) -> TwoGroupOutcome:
    """Builds a TwoGroupOutcome from four counts."""
    return TwoGroupOutcome(
        control=GroupOutcome("gate_30", control_successes, control_total),
        treatment=GroupOutcome("gate_40", treatment_successes, treatment_total),
    )


class TestGroupOutcome:

    def test_rate_and_failures(self):
        arm = GroupOutcome("gate_30", 200, 1000)
        assert arm.rate == 0.2
        assert arm.failures == 800

    def test_rejects_more_successes_than_users(self):
        with pytest.raises(ValueError, match="within"):
            GroupOutcome("gate_30", 11, 10)

    def test_rejects_empty_arm(self):
        with pytest.raises(ValueError, match="positive"):
            GroupOutcome("gate_30", 0, 0)

    def test_rejects_negative_successes(self):
        with pytest.raises(ValueError, match="within"):
            GroupOutcome("gate_30", -1, 10)

    def test_contingency_table_layout(self):
        assert outcome(200, 1000, 250, 1000).contingency_table == [
            [200, 800],
            [250, 750],
        ]


class TestEffectSize:
    """Closed-form values for a 20% -> 25% move."""

    def test_absolute(self):
        assert effect_size(outcome(200, 1000, 250, 1000)).absolute == pytest.approx(0.05)

    def test_relative(self):
        # 5 points on a 20-point base is a 25% lift, not a 5% one. The two
        # numbers argue very differently and both are reported for that reason.
        assert effect_size(outcome(200, 1000, 250, 1000)).relative == pytest.approx(0.25)

    def test_cohens_h(self):
        expected = 2 * math.asin(math.sqrt(0.25)) - 2 * math.asin(math.sqrt(0.20))
        result = effect_size(outcome(200, 1000, 250, 1000))
        assert result.cohens_h == pytest.approx(expected)
        assert result.cohens_h == pytest.approx(0.1199023, abs=1e-6)

    def test_sign_follows_treatment_minus_control(self):
        assert effect_size(outcome(250, 1000, 200, 1000)).absolute == pytest.approx(-0.05)
        assert effect_size(outcome(250, 1000, 200, 1000)).relative == pytest.approx(-0.2)

    def test_interval_brackets_the_estimate(self):
        result = effect_size(outcome(200, 1000, 250, 1000))
        assert result.absolute_ci_low < result.absolute < result.absolute_ci_high

    def test_interval_narrows_with_more_users(self):
        small = effect_size(outcome(200, 1000, 250, 1000))
        large = effect_size(outcome(20_000, 100_000, 25_000, 100_000))
        assert (large.absolute_ci_high - large.absolute_ci_low) < (
            small.absolute_ci_high - small.absolute_ci_low
        )

    def test_zero_control_rate_gives_nan_relative_not_a_crash(self):
        # A relative lift on a zero base is undefined, not infinite, and must
        # not take the whole readout down with it.
        result = effect_size(outcome(0, 1000, 50, 1000))
        assert math.isnan(result.relative)
        assert result.absolute == pytest.approx(0.05)

    def test_rejects_bad_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            effect_size(outcome(200, 1000, 250, 1000), confidence=1.5)


class TestChiSquare:

    def test_identical_arms_give_zero_statistic(self):
        result = chi_square_test(outcome(10, 20, 10, 20))
        assert result.statistic == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert not result.significant

    def test_known_table(self):
        # [[20, 10], [10, 20]]: all four expected counts are 15, so
        # chi2 = 4 * (5^2 / 15) = 6.6667 on 1 dof.
        result = chi_square_test(outcome(20, 30, 10, 30))
        assert result.statistic == pytest.approx(6.6666667, abs=1e-6)
        assert result.p_value == pytest.approx(0.0098233, abs=1e-6)
        assert result.dof == 1
        assert result.significant

    def test_equals_squared_two_proportion_z(self):
        """An uncorrected 2x2 chi-square IS the pooled two-proportion z-test.

        The module docstring claims this; this test is what makes the claim
        checkable rather than decorative.
        """
        control_successes, control_total = 8_502, 44_700
        treatment_successes, treatment_total = 8_279, 45_489
        result = chi_square_test(
            outcome(control_successes, control_total, treatment_successes, treatment_total)
        )

        p_control = control_successes / control_total
        p_treatment = treatment_successes / treatment_total
        pooled = (control_successes + treatment_successes) / (control_total + treatment_total)
        standard_error = math.sqrt(
            pooled * (1 - pooled) * (1 / control_total + 1 / treatment_total)
        )
        z = (p_treatment - p_control) / standard_error

        assert result.statistic == pytest.approx(z**2, rel=1e-10)

    def test_yates_correction_is_more_conservative(self):
        uncorrected = chi_square_test(outcome(20, 30, 10, 30), correction=False)
        corrected = chi_square_test(outcome(20, 30, 10, 30), correction=True)
        assert corrected.statistic < uncorrected.statistic
        assert corrected.p_value > uncorrected.p_value

    def test_large_expected_counts_are_valid(self):
        result = chi_square_test(outcome(200, 1000, 250, 1000))
        assert result.approximation_valid
        assert result.min_expected >= 5

    def test_small_expected_counts_are_flagged(self):
        # 2x2 with expected cells under 5: the chi-square approximation does
        # not hold and the p-value must not be presented as if it did.
        result = chi_square_test(outcome(1, 4, 3, 4))
        assert not result.approximation_valid
        assert result.min_expected < 5

    def test_alpha_controls_the_significance_flag(self):
        assert chi_square_test(outcome(20, 30, 10, 30), alpha=0.05).significant
        assert not chi_square_test(outcome(20, 30, 10, 30), alpha=0.001).significant

    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            chi_square_test(outcome(20, 30, 10, 30), alpha=0.0)


class TestRequiredSampleSize:

    def test_known_design(self):
        # 20% baseline, detect a 2pp absolute move, alpha 0.05 two-sided,
        # power 0.80. Published tables put this design in the 6,300-6,600
        # per-arm range depending on the pooled/unpooled variance convention;
        # this implementation uses pooled-null / unpooled-alternative.
        n = required_sample_size(0.20, 0.02)
        assert 6_300 <= n <= 6_600
        assert n == 6_510

    def test_bigger_effects_need_fewer_users(self):
        assert required_sample_size(0.20, 0.05) < required_sample_size(0.20, 0.02)

    def test_more_power_needs_more_users(self):
        assert required_sample_size(0.20, 0.02, power=0.95) > required_sample_size(
            0.20, 0.02, power=0.80
        )

    def test_stricter_alpha_needs_more_users(self):
        assert required_sample_size(0.20, 0.02, alpha=0.01) > required_sample_size(
            0.20, 0.02, alpha=0.05
        )

    def test_one_sided_needs_fewer_users(self):
        assert required_sample_size(0.20, 0.02, two_sided=False) < required_sample_size(
            0.20, 0.02, two_sided=True
        )

    def test_relative_effect_matches_the_equivalent_absolute_one(self):
        # 10% of a 0.20 baseline is 0.02 in absolute terms.
        assert required_sample_size(0.20, 0.10, effect_kind="relative") == (
            required_sample_size(0.20, 0.02)
        )

    def test_negative_effect_sizes_the_same_regression(self):
        # Sizing to detect a 2pp drop is not the same design as a 2pp rise
        # (the variance at the alternative differs), but both must be finite
        # and in the same ballpark.
        assert required_sample_size(0.20, -0.02) > 0

    def test_zero_effect_is_rejected(self):
        with pytest.raises(ValueError, match="zero effect"):
            required_sample_size(0.20, 0.0)

    def test_effect_off_the_end_of_the_scale_is_rejected(self):
        with pytest.raises(ValueError, match="outside"):
            required_sample_size(0.95, 0.10)

    def test_rejects_bad_baseline(self):
        with pytest.raises(ValueError, match="baseline_rate"):
            required_sample_size(0.0, 0.02)

    def test_rejects_bad_power(self):
        with pytest.raises(ValueError, match="power"):
            required_sample_size(0.20, 0.02, power=1.0)


class TestAchievedPower:

    def test_inverts_required_sample_size(self):
        """The two formulas must agree, or one of them is wrong.

        Feeding achieved_power the n that required_sample_size returns for a
        given effect has to recover the target power it was sized for.
        """
        for baseline in (0.05, 0.19, 0.45, 0.80):
            for effect in (0.01, 0.02, 0.05):
                for target in (0.80, 0.90, 0.95):
                    n = required_sample_size(baseline, effect, power=target)
                    recovered = achieved_power(baseline, effect, n)
                    assert recovered == pytest.approx(target, abs=0.002)

    def test_power_at_zero_effect_equals_alpha(self):
        # Not zero: the chance of rejecting a true null IS the false-positive
        # rate. A power function that returns 0 here has dropped the far tail.
        for alpha in (0.01, 0.05, 0.10):
            assert achieved_power(0.20, 0.0, 5_000, alpha=alpha) == pytest.approx(
                alpha, abs=1e-9
            )

    def test_power_rises_with_sample_size(self):
        powers = [achieved_power(0.20, 0.02, n) for n in (500, 2_000, 10_000, 50_000)]
        assert powers == sorted(powers)
        assert powers[-1] > 0.99

    def test_power_rises_with_effect_size(self):
        powers = [achieved_power(0.20, e, 2_000) for e in (0.005, 0.02, 0.05, 0.10)]
        assert powers == sorted(powers)

    def test_sign_of_the_effect_is_ignored(self):
        assert achieved_power(0.20, 0.02, 5_000) == pytest.approx(
            achieved_power(0.20, -0.02, 5_000)
        )

    def test_stays_within_zero_and_one(self):
        for n in (10, 1_000, 1_000_000):
            assert 0.0 <= achieved_power(0.20, 0.05, n) <= 1.0

    def test_rejects_non_positive_n(self):
        with pytest.raises(ValueError, match="n_per_group"):
            achieved_power(0.20, 0.02, 0)


class TestAgainstSimulation:
    """Draws experiments from a known distribution and checks the test's own
    promises hold empirically.

    These are the tests that would catch a formula that is self-consistent but
    systematically wrong -- neither the closed-form checks nor the
    round-trip check above can see that.
    """

    def test_false_positive_rate_matches_alpha(self):
        # 3,000 experiments where both arms are drawn from the SAME rate.
        # A test claiming alpha = 0.05 must call about 5% of them significant.
        rng = np.random.default_rng(SEED)
        trials, n, rate = 3_000, 2_000, 0.19
        control = rng.binomial(n, rate, size=trials)
        treatment = rng.binomial(n, rate, size=trials)

        false_positives = sum(
            chi_square_test(outcome(int(c), n, int(t), n)).significant
            for c, t in zip(control, treatment)
        )
        observed = false_positives / trials
        # 3 standard errors at 3,000 trials is about 0.012.
        assert 0.038 <= observed <= 0.062

    def test_empirical_power_matches_the_predicted_power(self):
        # Size a study for 80% power at a 2pp lift, then run 2,000 of them
        # against a real 2pp lift. About 80% should come back significant.
        rng = np.random.default_rng(SEED + 1)
        baseline, effect, trials = 0.19, 0.02, 2_000
        n = required_sample_size(baseline, effect, power=0.80)
        predicted = achieved_power(baseline, effect, n)

        control = rng.binomial(n, baseline, size=trials)
        treatment = rng.binomial(n, baseline + effect, size=trials)
        detected = sum(
            chi_square_test(outcome(int(c), n, int(t), n)).significant
            for c, t in zip(control, treatment)
        )
        observed = detected / trials

        assert predicted == pytest.approx(0.80, abs=0.01)
        # 3 standard errors at 2,000 trials is about 0.027.
        assert abs(observed - predicted) < 0.035

    def test_underpowered_study_detects_a_real_effect_only_rarely(self):
        # The claim the power check exists to make legible: a real effect in
        # an undersized study is missed most of the time, so "not significant"
        # from one of these is not evidence the effect is absent.
        rng = np.random.default_rng(SEED + 2)
        baseline, effect, n, trials = 0.19, 0.02, 500, 2_000
        predicted = achieved_power(baseline, effect, n)
        assert predicted < 0.20

        control = rng.binomial(n, baseline, size=trials)
        treatment = rng.binomial(n, baseline + effect, size=trials)
        detected = sum(
            chi_square_test(outcome(int(c), n, int(t), n)).significant
            for c, t in zip(control, treatment)
        )
        assert abs(detected / trials - predicted) < 0.04


class TestPowerCheck:

    def test_specified_effect_is_used_when_given(self):
        result = power_check(outcome(1_900, 10_000, 2_100, 10_000), minimum_detectable_effect=0.01)
        assert result.powered_for == "specified"
        assert result.effect == pytest.approx(0.01)

    def test_observed_effect_is_used_when_not_given(self):
        result = power_check(outcome(1_900, 10_000, 2_100, 10_000))
        assert result.powered_for == "observed"
        assert result.effect == pytest.approx(0.02)

    def test_relative_minimum_detectable_effect(self):
        result = power_check(
            outcome(1_900, 10_000, 2_100, 10_000),
            minimum_detectable_effect=0.10,
            effect_kind="relative",
        )
        assert result.effect == pytest.approx(0.019)

    def test_binding_arm_is_the_smaller_one(self):
        result = power_check(outcome(1_900, 10_000, 800, 4_000))
        assert result.n_per_group == 4_000

    def test_large_study_is_well_powered(self):
        result = power_check(
            outcome(19_000, 100_000, 21_000, 100_000), minimum_detectable_effect=0.02
        )
        assert result.well_powered
        assert result.achieved_power > 0.99

    def test_small_study_is_not_well_powered(self):
        result = power_check(outcome(95, 500, 105, 500), minimum_detectable_effect=0.02)
        assert not result.well_powered

    def test_zero_observed_effect_has_no_required_sample_size(self):
        # No finite sample resolves a zero difference, so the field is None
        # rather than a made-up number.
        result = power_check(outcome(1_900, 10_000, 1_900, 10_000))
        assert result.required_n_per_group is None
        assert result.achieved_power == pytest.approx(0.05, abs=1e-9)


class TestRunAbTestVerdicts:
    """Each of the four significance-by-power quadrants must be named, not
    collapsed into "significant" / "not significant"."""

    def test_significant_and_well_powered(self):
        result = run_ab_test(
            outcome(19_000, 100_000, 21_000, 100_000),
            metric="retention_7",
            minimum_detectable_effect=0.02,
        )
        assert result.chi_square.significant and result.power.well_powered
        assert result.conclusive
        assert "real difference" in result.verdict

    def test_significant_but_underpowered(self):
        result = run_ab_test(
            outcome(950, 5_000, 1_100, 5_000),
            metric="retention_7",
            minimum_detectable_effect=0.002,
        )
        assert result.chi_square.significant and not result.power.well_powered
        assert not result.conclusive
        assert "underpowered" in result.verdict
        assert "smaller than measured" in result.verdict

    def test_not_significant_but_well_powered_is_evidence_of_absence(self):
        result = run_ab_test(
            outcome(19_000, 100_000, 19_000, 100_000),
            metric="retention_7",
            minimum_detectable_effect=0.02,
        )
        assert not result.chi_square.significant and result.power.well_powered
        assert result.conclusive
        assert "evidence of absence" in result.verdict

    def test_not_significant_and_underpowered_is_inconclusive(self):
        result = run_ab_test(
            outcome(95, 500, 100, 500),
            metric="retention_7",
            minimum_detectable_effect=0.01,
        )
        assert not result.chi_square.significant and not result.power.well_powered
        assert not result.conclusive
        assert "inconclusive" in result.verdict
        assert "Do not read this as 'no effect'" in result.verdict

    def test_invalid_approximation_short_circuits_the_verdict(self):
        result = run_ab_test(outcome(1, 4, 3, 4), metric="retention_7")
        assert not result.chi_square.approximation_valid
        assert not result.conclusive
        assert "not interpretable" in result.verdict

    def test_verdict_names_the_direction(self):
        worse = run_ab_test(outcome(21_000, 100_000, 19_000, 100_000), metric="retention_7")
        better = run_ab_test(outcome(19_000, 100_000, 21_000, 100_000), metric="retention_7")
        assert "lower" in worse.verdict
        assert "higher" in better.verdict

    def test_verdict_carries_both_effect_scales(self):
        result = run_ab_test(
            outcome(19_000, 100_000, 21_000, 100_000), metric="retention_7"
        )
        assert "pp" in result.verdict
        assert "relative" in result.verdict


class TestOutcomeFromFrame:

    def build_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "userid": range(10),
                "version": ["gate_30"] * 5 + ["gate_40"] * 5,
                "retention_7": [True, True, True, False, False, True, False, False, False, False],
            }
        )

    def test_counts_per_arm(self):
        result = outcome_from_frame(self.build_frame(), "retention_7")
        assert result.control.label == "gate_30"
        assert result.control.successes == 3
        assert result.control.total == 5
        assert result.treatment.successes == 1
        assert result.treatment.total == 5

    def test_nulls_leave_the_denominator(self):
        df = self.build_frame()
        df["retention_7"] = pd.array(
            [True, True, None, False, False, True, False, False, False, False],
            dtype="boolean",
        )
        result = outcome_from_frame(df, "retention_7")
        assert result.control.total == 4
        assert result.control.successes == 2

    def test_custom_arm_labels(self):
        df = self.build_frame()
        result = outcome_from_frame(df, "retention_7", control="gate_40", treatment="gate_30")
        assert result.control.label == "gate_40"
        assert result.control.successes == 1

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="retention_1"):
            outcome_from_frame(self.build_frame(), "retention_1")

    def test_absent_group_raises(self):
        with pytest.raises(ValueError, match="gate_99"):
            outcome_from_frame(self.build_frame(), "retention_7", treatment="gate_99")


class TestAnalyzeMetricAndReporting:

    def build_frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(SEED)
        n = 5_000
        return pd.DataFrame(
            {
                "userid": range(2 * n),
                "version": ["gate_30"] * n + ["gate_40"] * n,
                "retention_1": np.concatenate(
                    [rng.random(n) < 0.45, rng.random(n) < 0.44]
                ),
                "retention_7": np.concatenate(
                    [rng.random(n) < 0.19, rng.random(n) < 0.16]
                ),
            }
        )

    def test_end_to_end_from_a_frame(self):
        result = analyze_metric(self.build_frame(), "retention_7", minimum_detectable_effect=0.01)
        assert result.metric == "retention_7"
        assert result.outcome.control.total == 5_000
        assert result.chi_square.significant

    def test_agrees_with_the_explicit_two_step_path(self):
        df = self.build_frame()
        direct = run_ab_test(outcome_from_frame(df, "retention_7"), metric="retention_7")
        wrapped = analyze_metric(df, "retention_7")
        assert wrapped.chi_square.statistic == pytest.approx(direct.chi_square.statistic)
        assert wrapped.effect.absolute == pytest.approx(direct.effect.absolute)

    def test_summarize_reports_every_section(self):
        text = summarize(analyze_metric(self.build_frame(), "retention_7"))
        for expected in ("effect size", "significance", "power", "absolute", "p-value"):
            assert expected in text

    def test_summarize_handles_an_undefined_relative_effect(self):
        # A control arm that retained nobody: the relative lift is undefined
        # and the power maths has no variance to work with, but the effect
        # size and the chi-square are both perfectly well defined, so the
        # readout must degrade rather than raise.
        result = run_ab_test(outcome(0, 1_000, 50, 1_000), metric="retention_7")
        assert math.isnan(result.power.achieved_power)
        assert not result.conclusive
        assert "undefined" in result.verdict

        text = summarize(result)
        assert "n/a" in text

    def test_summary_frame_columns(self):
        df = self.build_frame()
        results = [analyze_metric(df, "retention_1"), analyze_metric(df, "retention_7")]
        table = summary_frame(results)
        assert len(table) == 2
        for column in ("metric", "p_value", "absolute_pp", "achieved_power", "conclusive"):
            assert column in table.columns

    def test_summary_frame_values_match_the_results(self):
        df = self.build_frame()
        result = analyze_metric(df, "retention_7")
        row = summary_frame([result]).iloc[0]
        assert float(row["p_value"]) == pytest.approx(result.chi_square.p_value)
        assert float(row["absolute_pp"]) == pytest.approx(result.effect.absolute * 100)
