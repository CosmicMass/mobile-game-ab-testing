"""Deciding whether an A/B difference in retention is real -- and whether
there was ever enough data to answer that.

``p < 0.05`` on its own is not a readout. Two failure modes it cannot see:

* **Underpowered and significant.** A study too small for the effect it
  reports does not just risk a false positive; conditional on reaching
  significance it systematically *overstates* the effect, because only the
  luckier draws cleared the bar. Shipping on the reported lift then reliably
  disappoints. (Type M -- magnitude -- error.)
* **Underpowered and not significant.** Read as "no difference," when the
  study never had the resolution to see a difference that would have mattered
  commercially. Absence of evidence, sold as evidence of absence.

So every test here returns three things together, and refuses to be read as
one: the **effect size** (absolute *and* relative -- 0.8 percentage points is
also a 4% relative drop, and the two argue differently in a product
meeting), the **significance test**, and a **power check** answering "what
effect could a sample this size actually detect?" :func:`run_ab_test`
composes them and states which of the four quadrants the result falls in.

The significance test is a chi-square on the 2x2 contingency table. With
``correction=False`` (the default here) an uncorrected 2x2 chi-square is
algebraically identical to the pooled two-proportion z-test -- chi2 = z^2 --
so nothing is given up by using the contingency-table form, and the same
function generalizes if a third arm is added later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence, cast

import pandas as pd
from scipy.stats import chi2_contingency, norm

from .loader import GROUP_COLUMN, KNOWN_GROUPS

# Below this, the chi-square approximation to the sampling distribution stops
# being trustworthy and an exact test (Fisher) is the right tool instead. The
# conventional threshold, applied to the smallest EXPECTED (not observed)
# cell count.
MIN_EXPECTED_CELL_COUNT = 5.0

EffectKind = Literal["absolute", "relative"]
PoweredFor = Literal["observed", "specified"]


@dataclass(frozen=True)
class GroupOutcome:
    """One experiment arm's binary outcome.

    Attributes:
        label: The group's name (e.g. ``"gate_30"``).
        successes: Users for whom the metric was true (retained).
        total: Users in the arm.
    """

    label: str
    successes: int
    total: int

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError(f"{self.label}: total must be positive, got {self.total}")
        if not 0 <= self.successes <= self.total:
            raise ValueError(
                f"{self.label}: successes ({self.successes}) must be within [0, {self.total}]"
            )

    @property
    def failures(self) -> int:
        """Users for whom the metric was false."""
        return self.total - self.successes

    @property
    def rate(self) -> float:
        """``successes / total``."""
        return self.successes / self.total


@dataclass(frozen=True)
class TwoGroupOutcome:
    """The two arms being compared.

    ``control`` is the incumbent (the variant already shipped); ``treatment``
    is the change under test. The direction matters: every effect below is
    signed as treatment minus control, so a negative number means the change
    made things worse.
    """

    control: GroupOutcome
    treatment: GroupOutcome

    @property
    def contingency_table(self) -> List[List[int]]:
        """The 2x2 table as ``[[c_success, c_failure], [t_success, t_failure]]``."""
        return [
            [self.control.successes, self.control.failures],
            [self.treatment.successes, self.treatment.failures],
        ]


@dataclass(frozen=True)
class EffectSize:
    """How big the difference is, on three scales that answer different
    questions.

    Attributes:
        control_rate: Incumbent rate.
        treatment_rate: Variant rate.
        absolute: ``treatment_rate - control_rate``, in rate units. Multiply
            by 100 for percentage points.
        relative: ``absolute / control_rate`` -- the lift a stakeholder
            usually means by "x% better". ``nan`` if the control rate is 0.
        cohens_h: The arcsine-transformed standardized difference, comparable
            across metrics with very different base rates (0.2 small, 0.5
            medium, 0.8 large by Cohen's rule of thumb).
        absolute_ci_low: Lower bound on ``absolute``.
        absolute_ci_high: Upper bound on ``absolute``.
        confidence: Nominal coverage of the interval on ``absolute``.
    """

    control_rate: float
    treatment_rate: float
    absolute: float
    relative: float
    cohens_h: float
    absolute_ci_low: float
    absolute_ci_high: float
    confidence: float


@dataclass(frozen=True)
class ChiSquareResult:
    """Outcome of the chi-square test of independence on the 2x2 table.

    Attributes:
        statistic: The chi-square statistic. With ``yates_correction`` off
            and a 2x2 table this equals the squared pooled two-proportion
            z-statistic.
        p_value: Two-sided p-value.
        dof: Degrees of freedom (1 for a 2x2 table).
        min_expected: Smallest expected cell count.
        approximation_valid: Whether ``min_expected`` clears
            :data:`MIN_EXPECTED_CELL_COUNT`. When false, the p-value is not
            dependable and an exact test should be used instead.
        yates_correction: Whether Yates' continuity correction was applied.
        alpha: Significance threshold used.
        significant: ``p_value < alpha``.
    """

    statistic: float
    p_value: float
    dof: int
    min_expected: float
    approximation_valid: bool
    yates_correction: bool
    alpha: float
    significant: bool


@dataclass(frozen=True)
class PowerResult:
    """Whether the sample was large enough to answer the question asked.

    Attributes:
        baseline_rate: The control rate the calculation is anchored on.
        effect: The absolute effect the check is powered for.
        powered_for: ``"specified"`` when ``effect`` came from a
            pre-registered minimum detectable effect -- the defensible
            reading. ``"observed"`` when it was taken from the data, which is
            post-hoc and only descriptive (see :func:`power_check`).
        n_per_group: The binding arm size, i.e. the smaller of the two.
        required_n_per_group: Users *per arm* needed to reach
            ``target_power`` for ``effect``. ``None`` when ``effect`` is zero,
            which no finite sample can resolve.
        achieved_power: Probability this design would detect ``effect``. At
            ``effect == 0`` this correctly equals ``alpha``.
        alpha: Significance threshold.
        target_power: The power level being held to (conventionally 0.80).
        two_sided: Whether the test is two-sided.
        well_powered: ``achieved_power >= target_power``.
    """

    baseline_rate: float
    effect: float
    powered_for: PoweredFor
    n_per_group: int
    required_n_per_group: Optional[int]
    achieved_power: float
    alpha: float
    target_power: float
    two_sided: bool
    well_powered: bool


@dataclass(frozen=True)
class ABTestResult:
    """Effect, significance and power as one inseparable readout.

    Attributes:
        metric: The column tested.
        outcome: The two arms' raw counts.
        effect: The measured difference on three scales.
        chi_square: The significance test.
        power: The sample-size adequacy check.
        conclusive: True only when the significance test is *both* valid and
            adequately powered -- i.e. when its verdict, either way, can be
            acted on.
        verdict: One-sentence bottom line naming which quadrant this is.
    """

    metric: str
    outcome: TwoGroupOutcome
    effect: EffectSize
    chi_square: ChiSquareResult
    power: PowerResult
    conclusive: bool
    verdict: str


def _z(probability: float) -> float:
    """Standard-normal quantile, as a plain float."""
    return float(norm.ppf(probability))


def _critical_z(alpha: float, two_sided: bool) -> float:
    """The critical value a test statistic must clear at ``alpha``."""
    return _z(1 - alpha / 2) if two_sided else _z(1 - alpha)


def outcome_from_frame(
    df: pd.DataFrame,
    metric: str,
    *,
    control: str = KNOWN_GROUPS[0],
    treatment: str = KNOWN_GROUPS[1],
    group_column: str = GROUP_COLUMN,
) -> TwoGroupOutcome:
    """Collapses a per-user frame into the two arms' success counts.

    Nulls in ``metric`` are dropped from that arm's denominator rather than
    counted as failures -- a missing outcome is not a negative one.

    Args:
        df: A loaded per-user frame.
        metric: Boolean column to measure (e.g. ``"retention_7"``).
        control: Group label for the incumbent arm.
        treatment: Group label for the arm under test.
        group_column: The A/B group column.

    Returns:
        The two arms' counts.

    Raises:
        KeyError: If ``metric`` or ``group_column`` is not a column.
        ValueError: If either group label matches no rows.
    """
    for column in (metric, group_column):
        if column not in df.columns:
            raise KeyError(f"no column {column!r} in frame")

    arms: List[GroupOutcome] = []
    for label in (control, treatment):
        values = df.loc[df[group_column] == label, metric].dropna()
        if values.empty:
            raise ValueError(f"no rows with {group_column} == {label!r}")
        arms.append(
            GroupOutcome(label=label, successes=int(values.sum()), total=int(values.size))
        )
    return TwoGroupOutcome(control=arms[0], treatment=arms[1])


def effect_size(outcome: TwoGroupOutcome, *, confidence: float = 0.95) -> EffectSize:
    """Measures the difference between the two arms on three scales.

    The interval on the absolute difference uses the unpooled (Wald) standard
    error, which is the standard choice and is well behaved at the sample
    sizes an A/B test on retention usually has. For very small arms, or rates
    near 0 or 1, a Newcombe hybrid-score interval would hold its coverage
    better.

    Args:
        outcome: The two arms' counts.
        confidence: Coverage for the interval on the absolute difference.

    Returns:
        The populated :class:`EffectSize`.

    Raises:
        ValueError: If ``confidence`` is not in ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    p_control = outcome.control.rate
    p_treatment = outcome.treatment.rate
    absolute = p_treatment - p_control

    relative = absolute / p_control if p_control > 0 else float("nan")
    cohens_h = 2 * math.asin(math.sqrt(p_treatment)) - 2 * math.asin(math.sqrt(p_control))

    standard_error = math.sqrt(
        p_control * (1 - p_control) / outcome.control.total
        + p_treatment * (1 - p_treatment) / outcome.treatment.total
    )
    half_width = _critical_z(1 - confidence, two_sided=True) * standard_error

    return EffectSize(
        control_rate=p_control,
        treatment_rate=p_treatment,
        absolute=absolute,
        relative=relative,
        cohens_h=cohens_h,
        absolute_ci_low=absolute - half_width,
        absolute_ci_high=absolute + half_width,
        confidence=confidence,
    )


def chi_square_test(
    outcome: TwoGroupOutcome,
    *,
    alpha: float = 0.05,
    correction: bool = False,
) -> ChiSquareResult:
    """Chi-square test of independence on the 2x2 outcome table.

    Args:
        outcome: The two arms' counts.
        alpha: Significance threshold.
        correction: Apply Yates' continuity correction. Off by default:
            at A/B-test sample sizes it is needlessly conservative, and
            leaving it off preserves the exact equivalence with the pooled
            two-proportion z-test (``statistic == z**2``). Turn it on for
            small tables.

    Returns:
        The populated :class:`ChiSquareResult`. Check
        ``approximation_valid`` before trusting ``p_value``.

    Raises:
        ValueError: If ``alpha`` is not in ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    # SciPy ships no type information for chi2_contingency, so an unannotated
    # unpack degrades every one of these to `object` and the float() calls
    # below stop type-checking. The cast states the shape SciPy documents
    # rather than silencing the checker on the results themselves.
    statistic, p_value, dof, expected = cast(
        "tuple[float, float, int, Any]",
        chi2_contingency(outcome.contingency_table, correction=correction),
    )
    min_expected = float(min(float(cell) for row in expected for cell in row))

    return ChiSquareResult(
        statistic=float(statistic),
        p_value=float(p_value),
        dof=int(dof),
        min_expected=min_expected,
        approximation_valid=min_expected >= MIN_EXPECTED_CELL_COUNT,
        yates_correction=correction,
        alpha=alpha,
        significant=float(p_value) < alpha,
    )


def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    effect_kind: EffectKind = "absolute",
) -> int:
    """Users needed *per arm* to detect a given effect.

    Uses the standard two-proportion formula, with the null variance pooled
    at the average of the two rates and the alternative variance unpooled:

        n = (z_alpha * sqrt(2 * p_bar * q_bar)
             + z_power * sqrt(p1*q1 + p2*q2))^2 / (p2 - p1)^2

    This is the calculation that should be run *before* an experiment
    launches, against the smallest effect worth shipping for. Run afterwards
    against the effect that was observed, it answers a different and much
    weaker question -- see :func:`power_check`.

    Args:
        baseline_rate: The control arm's expected rate, in ``(0, 1)``.
        minimum_detectable_effect: The smallest difference worth detecting.
            Read as an absolute change in rate, or as a fraction of
            ``baseline_rate`` when ``effect_kind`` is ``"relative"``. May be
            negative to size for a regression.
        alpha: Significance threshold.
        power: Target probability of detecting the effect.
        two_sided: Whether the eventual test is two-sided.
        effect_kind: How to read ``minimum_detectable_effect``.

    Returns:
        Users per arm, rounded up.

    Raises:
        ValueError: If any rate or probability is out of range, if the effect
            is zero, or if the implied treatment rate falls outside ``(0, 1)``.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if minimum_detectable_effect == 0:
        raise ValueError("a zero effect cannot be detected by any finite sample")

    absolute_effect = (
        minimum_detectable_effect
        if effect_kind == "absolute"
        else baseline_rate * minimum_detectable_effect
    )
    treatment_rate = baseline_rate + absolute_effect
    if not 0.0 < treatment_rate < 1.0:
        raise ValueError(
            f"the effect implies a treatment rate of {treatment_rate:.4f}, outside (0, 1)"
        )

    pooled = (baseline_rate + treatment_rate) / 2
    z_alpha = _critical_z(alpha, two_sided)
    z_power = _z(power)

    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power
        * math.sqrt(
            baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate)
        )
    ) ** 2
    return math.ceil(numerator / absolute_effect**2)


def achieved_power(
    baseline_rate: float,
    effect: float,
    n_per_group: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Probability that a design of this size would detect ``effect``.

    The inverse of :func:`required_sample_size`, using the same variance
    assumptions, so the two are mutually consistent: feeding this function
    the n that :func:`required_sample_size` returns for a given effect
    recovers the target power.

    At ``effect == 0`` the formula correctly returns ``alpha`` -- the
    probability of rejecting a true null is exactly the false-positive rate,
    not zero.

    Args:
        baseline_rate: The control arm's rate, in ``(0, 1)``.
        effect: Absolute difference to detect; sign is ignored.
        n_per_group: Users per arm.
        alpha: Significance threshold.
        two_sided: Whether the test is two-sided.

    Returns:
        Power, in ``[0, 1]``.

    Raises:
        ValueError: If a rate or probability is out of range, ``n_per_group``
            is not positive, or the implied treatment rate leaves ``(0, 1)``.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")

    treatment_rate = baseline_rate + abs(effect)
    if not 0.0 < treatment_rate <= 1.0:
        raise ValueError(
            f"the effect implies a treatment rate of {treatment_rate:.4f}, outside (0, 1]"
        )

    delta = abs(effect)
    pooled = (baseline_rate + treatment_rate) / 2
    se_null = math.sqrt(2 * pooled * (1 - pooled) / n_per_group)
    se_alternative = math.sqrt(
        (
            baseline_rate * (1 - baseline_rate)
            + treatment_rate * (1 - treatment_rate)
        )
        / n_per_group
    )
    if se_alternative == 0:
        return 1.0

    z_alpha = _critical_z(alpha, two_sided)
    upper = float(norm.cdf((delta - z_alpha * se_null) / se_alternative))
    if not two_sided:
        return upper
    # The far tail: rejecting in the direction opposite the true effect.
    # Negligible for a large delta, but it is what makes power equal alpha
    # exactly at delta = 0 rather than alpha/2.
    lower = float(norm.cdf((-delta - z_alpha * se_null) / se_alternative))
    return upper + lower


def power_check(
    outcome: TwoGroupOutcome,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    two_sided: bool = True,
    minimum_detectable_effect: Optional[float] = None,
    effect_kind: EffectKind = "absolute",
) -> PowerResult:
    """Asks whether this experiment had the sample size to answer its question.

    Two ways to run it, and the difference is not cosmetic:

    * Pass ``minimum_detectable_effect`` -- ideally the number fixed *before*
      the experiment, the smallest change that would actually justify
      shipping. The result then answers a real question: could this
      experiment have seen an effect that size?
    * Omit it, and the check is run against the effect observed in the data.
      This is post-hoc power. It is a descriptive diagnostic only, and it
      cannot be used to argue that a non-significant result "would have been
      significant with more users" -- for a non-significant result it is
      mathematically bound to come out low, so it adds no information beyond
      the p-value. It is reported because seeing "this design had 34% power
      for the effect it measured" makes the fragility legible in a way a
      p-value does not.

    Args:
        outcome: The two arms' counts.
        alpha: Significance threshold.
        target_power: Power level to hold the design to.
        two_sided: Whether the test is two-sided.
        minimum_detectable_effect: The pre-specified effect to power for. If
            ``None``, the observed absolute difference is used instead.
        effect_kind: How to read ``minimum_detectable_effect``.

    Returns:
        The populated :class:`PowerResult`.
    """
    baseline_rate = outcome.control.rate
    if minimum_detectable_effect is None:
        effect = abs(outcome.treatment.rate - baseline_rate)
        powered_for: PoweredFor = "observed"
    else:
        effect = abs(
            minimum_detectable_effect
            if effect_kind == "absolute"
            else baseline_rate * minimum_detectable_effect
        )
        powered_for = "specified"

    n_per_group = min(outcome.control.total, outcome.treatment.total)

    # A control arm that retained nobody (or everybody) has zero variance, so
    # the normal-approximation power math is undefined rather than merely
    # extreme. Reporting nan and well_powered=False keeps the readout honest;
    # letting required_sample_size raise here would take down a result whose
    # effect size and chi-square are both perfectly computable.
    if not 0.0 < baseline_rate < 1.0:
        return PowerResult(
            baseline_rate=baseline_rate,
            effect=effect,
            powered_for=powered_for,
            n_per_group=n_per_group,
            required_n_per_group=None,
            achieved_power=float("nan"),
            alpha=alpha,
            target_power=target_power,
            two_sided=two_sided,
            well_powered=False,
        )

    required: Optional[int] = None
    if effect != 0:
        required = required_sample_size(
            baseline_rate,
            effect,
            alpha=alpha,
            power=target_power,
            two_sided=two_sided,
            effect_kind="absolute",
        )

    power = achieved_power(
        baseline_rate, effect, n_per_group, alpha=alpha, two_sided=two_sided
    )

    return PowerResult(
        baseline_rate=baseline_rate,
        effect=effect,
        powered_for=powered_for,
        n_per_group=n_per_group,
        required_n_per_group=required,
        achieved_power=power,
        alpha=alpha,
        target_power=target_power,
        two_sided=two_sided,
        well_powered=power >= target_power,
    )


def _verdict(
    metric: str,
    effect: EffectSize,
    chi_square: ChiSquareResult,
    power: PowerResult,
) -> str:
    """Names which of the four significance-by-power quadrants a result is in.

    Args:
        metric: The column tested.
        effect: The measured difference.
        chi_square: The significance test.
        power: The sample-size adequacy check.

    Returns:
        A one-sentence bottom line.
    """
    if not chi_square.approximation_valid:
        return (
            f"{metric}: not interpretable -- the smallest expected cell count is "
            f"{chi_square.min_expected:.1f}, below {MIN_EXPECTED_CELL_COUNT:.0f}, so the "
            "chi-square approximation does not hold. Use an exact test."
        )

    if math.isnan(power.achieved_power):
        return (
            f"{metric}: the control arm's rate is {power.baseline_rate:.0%}, which leaves "
            "the power calculation undefined, so there is no way to say whether this "
            f"sample was adequate. The measured difference is "
            f"{effect.absolute * 100:+.2f} pp at p = {chi_square.p_value:.4g}."
        )

    direction = "higher" if effect.absolute > 0 else "lower"
    size = (
        f"{abs(effect.absolute) * 100:.2f} pp {direction} "
        f"({abs(effect.relative) * 100:.1f}% relative)"
    )
    powered_note = (
        f"powered for a {power.effect * 100:.2f} pp effect"
        if power.powered_for == "specified"
        else f"post-hoc power for the observed effect {power.achieved_power * 100:.0f}%"
    )

    if chi_square.significant and power.well_powered:
        return (
            f"{metric}: real difference -- treatment is {size}, p = "
            f"{chi_square.p_value:.4g}, and the sample was adequate ({powered_note}). "
            "Safe to act on."
        )
    if chi_square.significant and not power.well_powered:
        return (
            f"{metric}: significant but underpowered -- treatment is {size}, p = "
            f"{chi_square.p_value:.4g}, yet {powered_note}. The direction is probably "
            "right; expect the true effect to be smaller than measured, so do not plan "
            "against this magnitude."
        )
    if not chi_square.significant and power.well_powered:
        required = power.required_n_per_group
        detectable = (
            f"an effect of {power.effect * 100:.2f} pp would have been detected "
            f"({required:,} users per arm needed, {power.n_per_group:,} present)"
            if required is not None
            else "the design was adequate"
        )
        return (
            f"{metric}: no difference worth acting on -- p = {chi_square.p_value:.4g}, and "
            f"{detectable}. This is evidence of absence, not merely absent evidence."
        )
    required = power.required_n_per_group
    shortfall = (
        f"detecting a {power.effect * 100:.2f} pp effect needs {required:,} users per arm "
        f"and this test had {power.n_per_group:,}"
        if required is not None
        else "the two arms are exactly equal, which no sample size can resolve"
    )
    return (
        f"{metric}: inconclusive -- p = {chi_square.p_value:.4g} does not clear "
        f"{chi_square.alpha}, but the test was underpowered ({shortfall}). Do not read "
        "this as 'no effect'."
    )


def run_ab_test(
    outcome: TwoGroupOutcome,
    *,
    metric: str = "metric",
    alpha: float = 0.05,
    target_power: float = 0.80,
    two_sided: bool = True,
    confidence: float = 0.95,
    minimum_detectable_effect: Optional[float] = None,
    effect_kind: EffectKind = "absolute",
    correction: bool = False,
) -> ABTestResult:
    """Runs effect size, chi-square and the power check as one readout.

    Args:
        outcome: The two arms' counts.
        metric: Name of the column being tested, for the verdict text.
        alpha: Significance threshold.
        target_power: Power level to hold the design to.
        two_sided: Whether the test is two-sided.
        confidence: Coverage for the interval on the absolute difference.
        minimum_detectable_effect: Pre-specified effect to power for; if
            ``None``, the observed effect is used (post-hoc -- see
            :func:`power_check`).
        effect_kind: How to read ``minimum_detectable_effect``.
        correction: Apply Yates' continuity correction to the chi-square.

    Returns:
        The composed :class:`ABTestResult`.
    """
    effect = effect_size(outcome, confidence=confidence)
    chi_square = chi_square_test(outcome, alpha=alpha, correction=correction)
    power = power_check(
        outcome,
        alpha=alpha,
        target_power=target_power,
        two_sided=two_sided,
        minimum_detectable_effect=minimum_detectable_effect,
        effect_kind=effect_kind,
    )
    return ABTestResult(
        metric=metric,
        outcome=outcome,
        effect=effect,
        chi_square=chi_square,
        power=power,
        conclusive=chi_square.approximation_valid and power.well_powered,
        verdict=_verdict(metric, effect, chi_square, power),
    )


def analyze_metric(
    df: pd.DataFrame,
    metric: str,
    *,
    control: str = KNOWN_GROUPS[0],
    treatment: str = KNOWN_GROUPS[1],
    group_column: str = GROUP_COLUMN,
    alpha: float = 0.05,
    target_power: float = 0.80,
    two_sided: bool = True,
    confidence: float = 0.95,
    minimum_detectable_effect: Optional[float] = None,
    effect_kind: EffectKind = "absolute",
    correction: bool = False,
) -> ABTestResult:
    """Convenience wrapper: per-user frame in, full readout out.

    The keyword arguments are spelled out rather than forwarded as
    ``**kwargs`` so that a typo in a caller's keyword is a type error here
    instead of a silently ignored argument that changes the statistics.

    Args:
        df: A loaded per-user frame.
        metric: Boolean column to test.
        control: Group label for the incumbent arm.
        treatment: Group label for the arm under test.
        group_column: The A/B group column.
        alpha: Significance threshold.
        target_power: Power level to hold the design to.
        two_sided: Whether the test is two-sided.
        confidence: Coverage for the interval on the absolute difference.
        minimum_detectable_effect: Pre-specified effect to power for; if
            ``None``, the observed effect is used (post-hoc -- see
            :func:`power_check`).
        effect_kind: How to read ``minimum_detectable_effect``.
        correction: Apply Yates' continuity correction to the chi-square.

    Returns:
        The composed :class:`ABTestResult`.
    """
    outcome = outcome_from_frame(
        df, metric, control=control, treatment=treatment, group_column=group_column
    )
    return run_ab_test(
        outcome,
        metric=metric,
        alpha=alpha,
        target_power=target_power,
        two_sided=two_sided,
        confidence=confidence,
        minimum_detectable_effect=minimum_detectable_effect,
        effect_kind=effect_kind,
        correction=correction,
    )


def summarize(result: ABTestResult) -> str:
    """Formats a result as a multi-line analyst's readout.

    Args:
        result: A completed test.

    Returns:
        A printable block covering the counts, all three effect scales, the
        significance test, the power check, and the verdict.
    """
    outcome = result.outcome
    effect = result.effect
    chi_square = result.chi_square
    power = result.power

    relative = (
        f"{effect.relative * 100:+.2f}%" if not math.isnan(effect.relative) else "n/a"
    )
    required = (
        f"{power.required_n_per_group:,}"
        if power.required_n_per_group is not None
        else "not estimable (zero effect)"
    )

    lines: List[str] = [
        f"metric: {result.metric}",
        f"  {outcome.control.label:<12} {outcome.control.successes:>7,} / "
        f"{outcome.control.total:>7,} = {outcome.control.rate * 100:6.2f}%   (control)",
        f"  {outcome.treatment.label:<12} {outcome.treatment.successes:>7,} / "
        f"{outcome.treatment.total:>7,} = {outcome.treatment.rate * 100:6.2f}%   (treatment)",
        "",
        "  effect size",
        f"    absolute        {effect.absolute * 100:+.3f} pp  "
        f"[{effect.absolute_ci_low * 100:+.3f}, {effect.absolute_ci_high * 100:+.3f}] "
        f"at {effect.confidence * 100:.0f}%",
        f"    relative        {relative}",
        f"    Cohen's h       {effect.cohens_h:+.4f}",
        "",
        "  significance (chi-square, "
        f"{'Yates-corrected' if chi_square.yates_correction else 'uncorrected'})",
        f"    chi2            {chi_square.statistic:.4f}  (dof {chi_square.dof})",
        f"    p-value         {chi_square.p_value:.4g}  "
        f"({'significant' if chi_square.significant else 'not significant'} at "
        f"alpha = {chi_square.alpha})",
        f"    min expected    {chi_square.min_expected:,.1f}  "
        f"({'approximation holds' if chi_square.approximation_valid else 'TOO SMALL'})",
        "",
        f"  power (for a {power.effect * 100:.3f} pp effect, {power.powered_for})",
        f"    achieved        {power.achieved_power * 100:.1f}%  "
        f"(target {power.target_power * 100:.0f}%)",
        f"    n per arm       {power.n_per_group:,} present, {required} required",
        f"    verdict         {'adequately powered' if power.well_powered else 'UNDERPOWERED'}",
        "",
        f"  => {result.verdict}",
    ]
    return "\n".join(lines)


def summary_frame(results: Sequence[ABTestResult]) -> pd.DataFrame:
    """Flattens several results into one row-per-metric table.

    Args:
        results: Completed tests, typically one per retention metric.

    Returns:
        A frame with the counts, rates, effect sizes, p-value and power
        columns side by side for comparison.
    """
    return pd.DataFrame(
        [
            {
                "metric": r.metric,
                "control": r.outcome.control.label,
                "control_rate": r.outcome.control.rate,
                "treatment": r.outcome.treatment.label,
                "treatment_rate": r.outcome.treatment.rate,
                "absolute_pp": r.effect.absolute * 100,
                "absolute_ci_low_pp": r.effect.absolute_ci_low * 100,
                "absolute_ci_high_pp": r.effect.absolute_ci_high * 100,
                "relative_pct": r.effect.relative * 100,
                "cohens_h": r.effect.cohens_h,
                "chi2": r.chi_square.statistic,
                "p_value": r.chi_square.p_value,
                "significant": r.chi_square.significant,
                "n_per_arm": r.power.n_per_group,
                "required_n_per_arm": r.power.required_n_per_group,
                "achieved_power": r.power.achieved_power,
                "well_powered": r.power.well_powered,
                "conclusive": r.conclusive,
            }
            for r in results
        ]
    )
