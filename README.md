# Game Analytics Toolkit

Retention and A/B-test analysis for mobile game player data, from a raw
export to a decision you can defend.

## About this project

**This is a portfolio project built on a public dataset — not client or
employer work.** The data is the Kaggle
[Mobile Games: A/B Testing](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing)
set (Cookie Cats, 90,189 players, published by Aurelia Sui in 2019), which
anyone can download and re-run. The code — schema validation, interval
estimation, the sample-size / power / chi-square / effect-size pipeline, the
reporting — is my own, written to the same engineering bar as its sibling
[data-reconciliation-toolkit](https://github.com/CosmicMass/data-reconciliation-toolkit),
which *is* extracted from production work. The point of this repo is to show
retention and experiment-analysis technique end to end on data you can check
me on.

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

Effect size is reported on three scales, because they argue differently: a
0.8 percentage-point drop in D7 retention is also a 4% relative drop, and
Cohen's *h* is what lets you compare it to a change in a metric with a
totally different base rate.

## Architecture

```
src/game_analytics_toolkit/
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
from game_analytics_toolkit.loader import load_cookie_cats

df = load_cookie_cats("data/cookie_cats.csv")   # or SchemaError listing every problem at once
```

### retention — a rate is not a number, it's a number with a width

```python
from game_analytics_toolkit.retention import retention_by_group, wilson_interval

for rate in retention_by_group(df, "retention_7"):
    print(rate.group, rate.rate, rate.ci_low, rate.ci_high)

wilson_interval(0, 10)   # -> (0.0, 0.2775)  -- not the degenerate (0, 0) the normal approximation gives
```

### ab_test — is it real, and could we even have seen it?

```python
from game_analytics_toolkit.ab_test import analyze_metric, summarize, required_sample_size

# How big does the experiment need to be, before running it?
required_sample_size(baseline_rate=0.19, minimum_detectable_effect=0.01)   # -> 24,709 per arm

result = analyze_metric(df, "retention_7", minimum_detectable_effect=0.01)
print(summarize(result))
print(result.verdict)
```

### report — the numbers as figures

```python
from game_analytics_toolkit.report import retention_figure, effect_figure, build_html_report

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

Run against the Kaggle download, the analysis reaches this conclusion:
**moving the gate from level 30 to level 40 did not improve retention, and
7-day retention got measurably worse.**

- **D1 retention** is about **44.8%** (gate_30) vs **44.2%** (gate_40) — a
  difference of roughly half a percentage point that does not clear
  significance at α = 0.05.
- **D7 retention** is about **19.0%** (gate_30) vs **18.2%** (gate_40) — a
  drop of roughly 0.8 percentage points (about 4% relative) that *is*
  significant.
- With ~45,000 players per arm, the experiment is comfortably powered to
  detect a 1 percentage-point move, which is what makes the D1 non-result
  informative rather than merely inconclusive: it is evidence the gate move
  did not help, not a shrug.

The product read: the gate at level 30 is doing useful work, and pushing it
to 40 costs retention. Keep it where it is.

> Every figure above is printed by `examples/run_analysis.py`. Run it against
> your own download to reproduce them — that is the point of building this on
> public data.

## Development

```bash
python -m pytest tests/ -v      # 160 tests
python -m pyright               # 0 errors
```

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
