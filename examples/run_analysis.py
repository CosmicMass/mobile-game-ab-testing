"""End-to-end Cookie Cats analysis: load, measure retention, test, report.

Run it directly:

    python examples/run_analysis.py

The question being answered is the one the experiment was run for: Cookie Cats
shows a "gate" -- a forced wait the player can pay or wait out -- and the test
moved it from level 30 to level 40. Does holding players back later keep more
of them playing?

If ``data/cookie_cats.csv`` is present the real dataset is used. If it is not,
the script generates a synthetic stand-in with the same shape and says so
loudly at the top and bottom of its output, so a reader never mistakes one for
the other. See ``data/README.md`` for the download.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from game_analytics_toolkit.ab_test import (
    ABTestResult,
    analyze_metric,
    summarize,
    summary_frame,
)
from game_analytics_toolkit.loader import (
    GROUP_COLUMN,
    KNOWN_GROUPS,
    RETENTION_COLUMNS,
    group_sizes,
    load_cookie_cats,
)
from game_analytics_toolkit.report import (
    build_html_report,
    effect_figure,
    playtime_figure,
    retention_figure,
    retention_summary_frame,
)
from game_analytics_toolkit.retention import (
    RetentionRate,
    playtime_by_group,
    retention_by_group,
    retention_rate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cookie_cats.csv"
OUTPUT_PATH = PROJECT_ROOT / "examples" / "output" / "report.html"

# The smallest move worth shipping for, fixed BEFORE looking at the result.
# A percentage point of D7 retention is a large commercial effect in a mobile
# game; powering the check against it is what turns "not significant" into a
# statement with content. Declaring it up front rather than reading it off the
# data is the difference between a power check and a rationalization.
MINIMUM_DETECTABLE_EFFECT = 0.01

# Marginals of the published dataset, used only to shape the synthetic
# stand-in so its output is representative when the CSV is absent.
SYNTHETIC_PROFILE = {
    "gate_30": {"n": 44_700, "retention_1": 0.4482, "retention_7": 0.1902},
    "gate_40": {"n": 45_489, "retention_1": 0.4423, "retention_7": 0.1820},
}

# Lognormal parameters chosen to land on the published column's median (16)
# and mean (about 52) at once: median = exp(mu), mean = exp(mu + sigma^2 / 2).
# A geometric draw was tried first and rejected -- it matched the median but
# topped out around 250 rounds, so the synthetic stand-in lost exactly the
# feature that makes this column interesting, and made the play-count chart's
# tail handling look like an over-reaction to nothing.
ROUNDS_LOG_MEAN = 2.77
ROUNDS_LOG_SIGMA = 1.54


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def synthetic_frame(seed: int = 20260909) -> pd.DataFrame:
    """Builds a stand-in dataset with the published data's shape.

    Retention is drawn independently per user at each group's published rate,
    and play counts from a lognormal, which reproduces the real column's
    extreme right skew: most players never get near the gate, while a few log
    thousands of rounds.

    This is NOT the Cookie Cats data. It exists so the pipeline is runnable
    before the CSV is downloaded.

    Args:
        seed: Seed for the generator, so runs are reproducible.

    Returns:
        A frame with the same columns and dtypes as the real export.
    """
    rng = np.random.default_rng(seed)
    versions: List[str] = []
    retention_1: List[np.ndarray] = []
    retention_7: List[np.ndarray] = []
    rounds: List[np.ndarray] = []

    for group in KNOWN_GROUPS:
        profile = SYNTHETIC_PROFILE[group]
        size = int(profile["n"])
        versions.extend([group] * size)
        retention_1.append(rng.random(size) < profile["retention_1"])
        retention_7.append(rng.random(size) < profile["retention_7"])
        rounds.append(
            rng.lognormal(ROUNDS_LOG_MEAN, ROUNDS_LOG_SIGMA, size).astype("int64")
        )

    return pd.DataFrame(
        {
            "userid": np.arange(len(versions)),
            "version": versions,
            "sum_gamerounds": np.concatenate(rounds),
            "retention_1": np.concatenate(retention_1),
            "retention_7": np.concatenate(retention_7),
        }
    )


def load_data() -> tuple[pd.DataFrame, bool]:
    """Loads the real CSV if present, otherwise a synthetic stand-in.

    Returns:
        ``(frame, is_real)`` -- the data, and whether it came from the
        downloaded file.
    """
    if DATA_PATH.exists():
        return load_cookie_cats(str(DATA_PATH)), True
    return synthetic_frame(), False


def main() -> None:
    """Runs the whole analysis and writes the HTML report."""
    df, is_real = load_data()

    section("STEP 0 -- the data")
    if is_real:
        print(f"  loaded {DATA_PATH}")
        provenance = "Kaggle 'Mobile Games: A/B Testing - Cookie Cats' (public dataset)"
    else:
        print(f"  !! {DATA_PATH} not found -- using SYNTHETIC data")
        print("  !! Every number below is generated, not measured. See data/README.md.")
        provenance = "SYNTHETIC stand-in data -- the Cookie Cats CSV was not present"
    print(f"  {len(df):,} users  |  groups: {group_sizes(df)}")

    # ------------------------------------------------------------------
    # retention: the rates themselves, each with the interval that says how
    # precisely they are known.
    # ------------------------------------------------------------------
    section("STEP 1 -- retention: D1 and D7 rates per group")
    # Printed from the RetentionRate objects rather than from
    # retention_table's frame: the dataclass fields are real floats, where a
    # frame row hands back an opaque scalar that has to be re-converted before
    # it can be arithmetic. per_group is kept separately for the figures,
    # which must not show an aggregate bar beside the arms it is made of.
    displayed: List[RetentionRate] = []
    per_group: List[RetentionRate] = []
    for metric in RETENTION_COLUMNS:
        displayed.append(retention_rate(df, metric))
        metric_rates = retention_by_group(df, metric)
        displayed.extend(metric_rates)
        per_group.extend(metric_rates)

    for rate in displayed:
        print(
            f"  {rate.group:<8} {rate.metric:<12} "
            f"{rate.rate * 100:6.2f}%   "
            f"[{rate.ci_low * 100:5.2f}%, {rate.ci_high * 100:5.2f}%]   "
            f"n = {rate.n_users:,}"
        )

    section("STEP 2 -- engagement: how much each group actually played")
    playtime = playtime_by_group(df)
    for row in playtime.itertuples(index=False):
        print(
            f"  {str(row.group):<8} median {row.median_rounds:6.1f}   "
            f"mean {row.mean_rounds:7.1f}   p90 {row.p90_rounds:7.1f}   "
            f"max {row.max_rounds:,}"
        )
    print("  (the mean sits well above the median: play counts are heavily right-skewed)")

    # ------------------------------------------------------------------
    # ab_test: effect size, significance and power together. The MDE is
    # passed in so the power check answers a question fixed in advance.
    # ------------------------------------------------------------------
    section("STEP 3 -- A/B test: is the difference real, and could we have seen it?")
    print(
        f"  control = {KNOWN_GROUPS[0]}, treatment = {KNOWN_GROUPS[1]}, "
        f"minimum detectable effect = {MINIMUM_DETECTABLE_EFFECT:.0%} absolute\n"
    )

    results: List[ABTestResult] = []
    for metric in RETENTION_COLUMNS:
        result = analyze_metric(
            df,
            metric,
            control=KNOWN_GROUPS[0],
            treatment=KNOWN_GROUPS[1],
            group_column=GROUP_COLUMN,
            minimum_detectable_effect=MINIMUM_DETECTABLE_EFFECT,
        )
        results.append(result)
        print(summarize(result))
        print()

    section("STEP 4 -- the answer, in one table")
    compact = summary_frame(results)[
        ["metric", "control_rate", "treatment_rate", "absolute_pp", "p_value", "conclusive"]
    ]
    print(compact.to_string(index=False))

    # ------------------------------------------------------------------
    # report: the same numbers as figures, written as one HTML page.
    # ------------------------------------------------------------------
    section("STEP 5 -- report")
    path = build_html_report(
        OUTPUT_PATH,
        figures=[
            retention_figure(per_group),
            effect_figure(results),
            playtime_figure(df),
        ],
        tables=[("Retention by group", retention_summary_frame(per_group))],
        verdicts=[r.verdict for r in results],
        subtitle=f"Source: {provenance}",
    )
    print(f"  wrote {path}")
    print("  (overwritten on every run, not tracked by git)")

    section("Done")
    if is_real:
        print("  Numbers above are from the public Kaggle dataset.")
    else:
        print("  !! Reminder: this run used SYNTHETIC data, not the real dataset.")
        print("  !! Download the CSV to data/cookie_cats.csv and re-run for real results.")


if __name__ == "__main__":
    main()
