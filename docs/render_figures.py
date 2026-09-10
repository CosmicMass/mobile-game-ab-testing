"""Regenerate the figures embedded in the README, from the real dataset.

    pip install -e ".[figures]"
    python docs/render_figures.py

Reads ``data/cookie_cats.csv`` (see ``data/README.md`` for the download) and
writes PNGs next to this file. The images checked into the repo are produced
by exactly this script and nothing else, so they cannot quietly drift from
what ``report.py`` actually draws -- re-run it after changing a figure.
"""

from __future__ import annotations

from pathlib import Path

from mobile_game_ab_testing.ab_test import analyze_metric
from mobile_game_ab_testing.loader import RETENTION_COLUMNS, load_cookie_cats
from mobile_game_ab_testing.report import (
    effect_figure,
    playtime_figure,
    retention_figure,
)
from mobile_game_ab_testing.retention import retention_by_group

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "cookie_cats.csv"

# 2x scale so the PNGs stay sharp on a high-DPI screen at README width.
IMAGE_WIDTH = 900
IMAGE_SCALE = 2


def main() -> None:
    """Renders the three README figures to ``docs/*.png``."""
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found -- see data/README.md for the download")

    df = load_cookie_cats(str(DATA_PATH))
    results = [
        analyze_metric(df, metric, minimum_detectable_effect=0.01)
        for metric in RETENTION_COLUMNS
    ]
    rates = [
        rate
        for metric in RETENTION_COLUMNS
        for rate in retention_by_group(df, metric)
    ]

    figures = {
        "retention.png": (retention_figure(rates), 440),
        "effect.png": (effect_figure(results), 320),
        "playtime.png": (playtime_figure(df), 340),
    }
    for name, (figure, height) in figures.items():
        out_path = HERE / name
        figure.write_image(
            str(out_path), width=IMAGE_WIDTH, height=height, scale=IMAGE_SCALE
        )
        print(f"wrote {out_path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
