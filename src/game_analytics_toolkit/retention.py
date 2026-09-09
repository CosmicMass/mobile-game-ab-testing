"""D1 / D7 retention rates per experiment group.

A retention rate is a proportion estimated from a sample, so a bare
percentage ("gate_40 D7 retention is 18.2%") is only half a number. Every
rate here comes back with a Wilson score interval attached. Wilson rather
than the textbook ``p +/- z * sqrt(p(1-p)/n)`` because the normal
approximation misbehaves exactly where retention analysis often operates --
rates well below 50%, and small per-cell counts once you slice by group and
by day -- where it can hand back a lower bound below zero. Wilson stays
inside [0, 1] and keeps its coverage in those regions.

The functions take the boolean retention columns produced by
``loader.load_cookie_cats``; ``True`` means the player came back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd
from scipy.stats import norm

from .loader import GROUP_COLUMN, RETENTION_COLUMNS, ROUNDS_COLUMN


@dataclass(frozen=True)
class RetentionRate:
    """One retention rate with its confidence interval.

    Attributes:
        group: The experiment group this rate is for, or ``"overall"`` for
            the whole frame.
        metric: The retention column this rate measures (e.g.
            ``"retention_7"``).
        n_users: Number of users in the denominator.
        retained: Number of users who returned (the numerator).
        rate: ``retained / n_users``.
        ci_low: Lower bound of the Wilson interval.
        ci_high: Upper bound of the Wilson interval.
        confidence: Nominal coverage of ``[ci_low, ci_high]`` (e.g. 0.95).
    """

    group: str
    metric: str
    n_users: int
    retained: int
    rate: float
    ci_low: float
    ci_high: float
    confidence: float


def wilson_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successes (here: users retained).
        n: Number of trials (users in the group). Must be positive.
        confidence: Desired coverage, strictly between 0 and 1.

    Returns:
        ``(low, high)``, both clamped to ``[0, 1]`` and guaranteed to
        bracket ``successes / n``.

    Raises:
        ValueError: If ``n <= 0``, ``successes`` is outside ``[0, n]``, or
            ``confidence`` is not in ``(0, 1)``.
    """
    if n <= 0:
        raise ValueError("n must be positive to estimate a rate")
    if not 0 <= successes <= n:
        raise ValueError(f"successes ({successes}) must be within [0, {n}]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    z = float(norm.ppf(1 - (1 - confidence) / 2))
    phat = successes / n
    denominator = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denominator
    half_width = (z / denominator) * math.sqrt(
        phat * (1 - phat) / n + z**2 / (4 * n**2)
    )

    # The Wilson interval contains phat by construction, but at successes == 0
    # or successes == n the closed form lands a few ulp the wrong side of the
    # boundary (1.0 comes back as 0.9999999999999999). Pulling the bounds onto
    # phat corrects only that floating-point error -- in exact arithmetic
    # neither min/max ever binds -- and it keeps the guarantee a caller drawing
    # an error bar depends on, that low <= phat <= high.
    low = max(0.0, min(centre - half_width, phat))
    high = min(1.0, max(centre + half_width, phat))
    return (low, high)


def _count_retained(series: pd.Series) -> tuple[int, int]:
    """Returns ``(retained, n_users)`` for a boolean retention column,
    ignoring nulls in the denominator."""
    present = series.dropna()
    return int(present.sum()), int(present.size)


def retention_rate(
    df: pd.DataFrame,
    metric: str,
    *,
    group: Optional[str] = None,
    group_column: str = GROUP_COLUMN,
    confidence: float = 0.95,
) -> RetentionRate:
    """Retention rate for one metric, over the whole frame or one group.

    Args:
        df: A loaded frame.
        metric: The boolean retention column to measure.
        group: A value of ``group_column`` to restrict to, or ``None`` for
            the whole frame.
        group_column: The A/B group column.
        confidence: Coverage for the Wilson interval.

    Returns:
        The populated :class:`RetentionRate`.

    Raises:
        KeyError: If ``metric`` is not a column.
        ValueError: If ``group`` is given but matches no rows.
    """
    if metric not in df.columns:
        raise KeyError(f"no column {metric!r} in frame")

    if group is None:
        subset = df
        label = "overall"
    else:
        subset = df[df[group_column] == group]
        label = group
        if len(subset) == 0:
            raise ValueError(f"no rows with {group_column} == {group!r}")

    retained, n_users = _count_retained(subset[metric])
    rate = retained / n_users if n_users else float("nan")
    low, high = wilson_interval(retained, n_users, confidence)
    return RetentionRate(
        group=label,
        metric=metric,
        n_users=n_users,
        retained=retained,
        rate=rate,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
    )


def retention_by_group(
    df: pd.DataFrame,
    metric: str,
    *,
    group_column: str = GROUP_COLUMN,
    confidence: float = 0.95,
) -> List[RetentionRate]:
    """One :class:`RetentionRate` per group, in the order groups first
    appear in ``df``.

    Args:
        df: A loaded frame.
        metric: The boolean retention column to measure.
        group_column: The A/B group column.
        confidence: Coverage for the Wilson intervals.

    Returns:
        A list of per-group rates.
    """
    groups = list(dict.fromkeys(df[group_column].tolist()))
    return [
        retention_rate(
            df,
            metric,
            group=str(group),
            group_column=group_column,
            confidence=confidence,
        )
        for group in groups
    ]


def retention_table(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str] = RETENTION_COLUMNS,
    group_column: str = GROUP_COLUMN,
    confidence: float = 0.95,
    include_overall: bool = True,
) -> pd.DataFrame:
    """Tidy long table of every rate, one row per (group, metric).

    Args:
        df: A loaded frame.
        metrics: Retention columns to include.
        group_column: The A/B group column.
        confidence: Coverage for the Wilson intervals.
        include_overall: Prepend a whole-frame row per metric.

    Returns:
        A frame with columns ``group, metric, n_users, retained, rate,
        ci_low, ci_high, confidence``.
    """
    rows: List[RetentionRate] = []
    for metric in metrics:
        if include_overall:
            rows.append(
                retention_rate(
                    df, metric, group=None, group_column=group_column, confidence=confidence
                )
            )
        rows.extend(
            retention_by_group(
                df, metric, group_column=group_column, confidence=confidence
            )
        )
    return pd.DataFrame([vars(r) for r in rows])


def playtime_by_group(
    df: pd.DataFrame,
    *,
    rounds_column: str = ROUNDS_COLUMN,
    group_column: str = GROUP_COLUMN,
) -> pd.DataFrame:
    """Engagement context for the retention numbers: how much each group
    actually played.

    Retention and playtime move together, so a difference in retention is
    easier to read next to the play-count distribution. The mean is included
    but the median and 90th percentile matter more here -- ``sum_gamerounds``
    is extremely right-skewed (one user in the published data logged ~49,000
    rounds), so the mean is pulled around by a handful of players.

    Args:
        df: A loaded frame.
        rounds_column: The play-count column.
        group_column: The A/B group column.

    Returns:
        A frame with columns ``group, n_users, mean_rounds, median_rounds,
        p90_rounds, max_rounds``.
    """
    grouped = df.groupby(group_column, sort=False)[rounds_column]
    table = pd.DataFrame(
        {
            "n_users": grouped.size().astype("int64"),
            "mean_rounds": grouped.mean(),
            "median_rounds": grouped.median(),
            "p90_rounds": grouped.quantile(0.9),
            "max_rounds": grouped.max().astype("int64"),
        }
    )
    return table.reset_index(names="group")
