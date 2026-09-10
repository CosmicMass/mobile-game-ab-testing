# Mobile Game A/B Testing

[![CI](https://github.com/CosmicMass/mobile-game-ab-testing/actions/workflows/ci.yml/badge.svg)](https://github.com/CosmicMass/mobile-game-ab-testing/actions/workflows/ci.yml)

A worked retention and A/B-test analysis of a real mobile-game experiment,
from the raw export to a decision you can defend — with a tested, reusable
statistics core (`ab_test.py`, `retention.py`).

## About this project

**This is a portfolio project built on a public dataset — not client or
employer work.** The data is the Kaggle
[Mobile Games: A/B Testing](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing)
set (Cookie Cats, 90,189 players, published by Aurelia Sui in 2019), which
anyone can download and re-run. The code — schema validation, interval
estimation, the sample-size / power / chi-square / effect-size pipeline, the
reporting — is my own, written to the same engineering bar as its sibling
[data-reconciliation-toolkit](https://github.com/CosmicMass/data-reconciliation-toolkit),
which *is* extracted from production work. This one is a single analysis done
properly, not a library: `loader.py` is specific to this dataset, while
`ab_test.py` and `retention.py` are written to be lifted into other work
unchanged. The point of the repo is to show retention and
experiment-analysis technique end to end on data you can check me on.

## The question

Cookie Cats puts a **gate** in the player's way — a forced wait the player can
sit out or pay to skip. Gates drive in-app purchases, but they also give
players an enforced break, which is supposed to make the game last longer.
The experiment moved the gate from **level 30** to **level 40**.

So: does making players go further before the first gate keep more of them
playing? The metrics are 1-day and 7-day retention.

## Why this is more than `p < 0.05`

A p-value on its own cannot see either of the two ways an experiment readout
goes wrong, and both are common:

- **Underpowered and significant.** A study too small for the effect it
  reports doesn't just risk a false positive — conditional on reaching
  significance it *systematically overstates* the effect, because only the
  luckier draws cleared the bar. Ship on the reported lift and it reliably
  disappoints.
- **Underpowered and not significant.** Read as "no difference," when the
  study never had the resolution to see a difference that would have mattered
  commercially. Absence of evidence sold as evidence of absence.

So `run_ab_test` returns the effect size, the significance test and a power
check as one object, and names which of the four quadrants the result falls
in:

|  | **well powered** | **underpowered** |
|---|---|---|
| **significant** | real difference, safe to act on | direction probably right, magnitude inflated |
| **not significant** | evidence of *absence* | inconclusive — not "no effect" |

Effect size is reported on three scales, because they argue differently: the
0.8 percentage-point drop in D7 retention below is also a 4.3% relative drop,
and Cohen's *h* is what lets you compare it to a change in a metric with a
totally different base rate.

## Architecture

```
src/mobile_game_ab_testing/
├── loader.py      # CSV loading, schema + dtype validation, refuse-to-load
├── retention.py   # D1/D7 rates per group, with Wilson intervals
├── ab_test.py     # sample size & power, chi-square, effect size, verdict
└── report.py      # summary tables + Plotly figures + one-page HTML report
```

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+. Core dependencies: `pandas`, `scipy`, `plotly`. The
`dev` extra adds `pytest`, `pyright`, `pandas-stubs` and `numpy`.

Then put the dataset in place — see [data/README.md](data/README.md):

```
data/cookie_cats.csv
```

## Quick start

### loader — refuse to analyze a file that isn't what it claims to be

Every downstream function takes the column names, the two group labels and
the boolean dtype of the retention columns on trust. A retention column read
as the strings `"True"`/`"False"` doesn't crash anything — every value is
truthy, so retention silently reads as 100%. The loader's job is to make that
impossible.

```python
from mobile_game_ab_testing.loader import load_cookie_cats

df = load_cookie_cats("data/cookie_cats.csv")   # or SchemaError listing every problem at once
```

### retention — a rate is not a number, it's a number with a width

```python
from mobile_game_ab_testing.retention import retention_by_group, wilson_interval

for rate in retention_by_group(df, "retention_7"):
    print(rate.group, rate.rate, rate.ci_low, rate.ci_high)

wilson_interval(0, 10)   # -> (0.0, 0.2775)  -- not the degenerate (0, 0) the normal approximation gives
```

### ab_test — is it real, and could we even have seen it?

```python
from mobile_game_ab_testing.ab_test import analyze_metric, summarize, required_sample_size

# How big does the experiment need to be, before running it?
required_sample_size(baseline_rate=0.19, minimum_detectable_effect=0.01)   # -> 24,641 per arm

result = analyze_metric(df, "retention_7", minimum_detectable_effect=0.01)
print(summarize(result))
print(result.verdict)
```

### report — the numbers as figures

```python
from mobile_game_ab_testing.report import retention_figure, effect_figure, build_html_report

build_html_report(
    "examples/output/report.html",
    figures=[retention_figure(rates), effect_figure([result])],
    verdicts=[result.verdict],
)
```

Every rate is drawn with its interval, and the effect chart is anchored on a
zero line — so an effect that isn't distinguishable from nothing *looks* like
one, without needing a caption to say so.

### Seeing it end to end

`examples/run_analysis.py` runs the whole pipeline and writes the HTML report.
If `data/cookie_cats.csv` is missing it generates a synthetic stand-in with the
published data's shape and says so loudly, so the script always runs and its
output is never mistaken for the real thing.

```bash
python examples/run_analysis.py
```

## Findings

The conclusion, on the full 90,189-player dataset: **moving the gate from
level 30 to level 40 did not improve retention, and 7-day retention got
measurably worse.**

| metric | gate_30 (control) | gate_40 (treatment) | absolute Δ | relative Δ | χ² p-value | powered? |
|---|---|---|---|---|---|---|
| **D1 retention** | 44.82% | 44.23% | −0.59 pp | −1.3% | 0.074 — **not significant** | yes (85%) |
| **D7 retention** | 19.02% | 18.20% | −0.82 pp | −4.3% | 0.0016 — **significant** | yes (96%) |

![Retention rate by group for D1 and D7, gate_30 versus gate_40, each bar
labelled and capped with its 95% Wilson interval](docs/retention.png)

- The **D7 drop is real**: the 95% interval on the difference is
  [−1.33, −0.31] pp — entirely below zero — and with ~45,000 players per arm
  the test had 96% power to catch a 1 pp move.
- The **D1 non-result is informative, not a shrug**: that same sample was
  powered (85%) to detect a 1 pp change and didn't, so this is evidence the
  gate move did not help D1, not merely absence of evidence.

![Treatment-minus-control effect for D1 and D7 with 95% confidence intervals,
against a zero reference line; the D1 interval touches zero, the D7 interval
sits entirely below it](docs/effect.png)

- `sum_gamerounds` carries the dataset's famous outlier — one gate_30 player
  logged 49,854 rounds against a median of ~16 — which is why `report.py`
  summarises play counts with the median and drops the top 1% from the
  distribution plot rather than letting one row set the axis.

![Box plot of rounds played per group with the top 1% of users excluded from
the view; the two groups' distributions are almost identical](docs/playtime.png)

The product read: the gate at level 30 is doing useful work; pushing it to 40
costs 7-day retention. Keep it where it is.

> Every number in this section is printed by `examples/run_analysis.py`, and
> every figure is written by `docs/render_figures.py` (`pip install -e
> ".[figures]"` first) — both run against the Kaggle CSV, so nothing here is
> hand-drawn or hand-typed. Re-run them to reproduce.

## Development

```bash
python -m pytest tests/ -v      # 160 tests
python -m pyright               # 0 errors
```

Both run in [CI](.github/workflows/ci.yml) on every push — the test job
across Python 3.10 through 3.13, so the `requires-python` floor is a claim
the badge actually backs.

The statistical functions are checked three ways: against closed-form values
computed by hand, against each other (the sample-size and power formulas must
invert one another — feed `achieved_power` the *n* that `required_sample_size`
returns and it has to recover the target power), and against **simulation** —
`TestAgainstSimulation` draws thousands of experiments from a known
distribution and checks that the false-positive rate really is α and that the
predicted power really is the rate at which a known effect gets detected. A
statistics module that is only self-consistent can still be uniformly wrong;
those last ones are what would catch it.

## License

MIT.
