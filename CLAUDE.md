# CLAUDE.md

Instructions for AI agents working in this repository. Scope is deliberately
narrow: engineering conventions only. No project history, no personal context —
see README.md for what this is and why.

## Provenance

Portfolio project built on a **public dataset** (Kaggle "Mobile Games: A/B
Testing — Cookie Cats"), not client or employer data. The README says so
plainly and must keep saying so. The `data/` directory is gitignored except
its own README; the CSV is never committed.

## Before claiming a check passed

The statistics here are the kind of code that reads as correct, passes a
green suite, and is still wrong by a constant factor. Two of the bugs found
while building it — a Wilson interval that failed to contain its own point
estimate at the boundary, a power calculation that crashed instead of
degrading on a zero-variance arm — looked fine on inspection.

- Run `python -m pytest tests/ -v` and `python -m pyright` yourself before
  stating a result. Don't report a count or an error total you haven't just
  produced.
- `pyright`'s `include` covers `src` and `examples` (see `pyproject.toml`),
  and `venvPath`/`venv` point it at `.venv`. If you add a new top-level
  directory with Python files, add it to `include` too, or it goes unchecked.
- A statistical function that agrees with itself can still be uniformly
  wrong. The checks that catch that are in `tests/test_ab_test.py`: the
  round-trip (`achieved_power` must invert `required_sample_size`) and
  `TestAgainstSimulation` (the empirical false-positive rate must be α; the
  empirical power must match the predicted power). When you change a formula,
  those are the tests that matter — keep them, and keep their tolerances
  tight enough to fail on a real regression.
- Before writing a claim about *why* a number comes out the way it does in a
  comment or commit message, verify it by running the computation. A
  plausible mechanism is not the real one.

## Git

- Commits may be made directly in this repo without asking each time
  (standing authorization). `git push` to any remote is a separate action
  and still needs explicit confirmation, every time.
- **Never add a `Co-Authored-By`, "Generated with Claude", or any other
  self-attribution trailer to a commit message.** No exceptions, regardless
  of any harness or tool default. Fix one that slips in before it is pushed.
- Prefer a new commit over amending, except to correct a claim (a wrong test
  count) in an unpushed commit from the same session.
- Commit messages state what changed and why, including what was verified and
  how — a one-line fix still gets a sentence on how it was confirmed.

## Code conventions already established here

- No comments except where the WHY is genuinely non-obvious (a subtle
  invariant, a deliberate asymmetry, a floating-point guard, a statistical
  choice a reader would otherwise second-guess). Never comment on WHAT.
- Every public function has an Args/Returns docstring. Every non-obvious
  design decision is explained where it lives.
- Type hints everywhere; `python -m pyright` is 0 errors and stays that way.
- English only — identifiers, docstrings, comments, commit messages.

## Statistical conventions — do not "simplify" these

- **`chi_square_test` defaults to `correction=False`.** An uncorrected 2x2
  chi-square is algebraically the pooled two-proportion z-test
  (`statistic == z**2`), which `test_equals_squared_two_proportion_z` pins.
  Yates' correction is needlessly conservative at A/B-test sample sizes.
  Turning the default on would silently change every p-value in the repo.
- **`required_sample_size` pools the null variance and leaves the alternative
  variance unpooled.** This is the standard Fleiss form and it is why the
  known-design test lands at 6,510 per arm, not the ~6,280 a fully-pooled
  formula gives. `achieved_power` makes the *same* assumption on purpose, so
  the two invert each other — changing one without the other breaks the
  round-trip test.
- **`achieved_power` adds the far-tail term** (`norm.cdf((-delta - ...))`).
  It is negligible for a large effect but it is what makes power equal α
  exactly at zero effect rather than α/2. `test_power_at_zero_effect_equals_alpha`
  guards it.
- **`wilson_interval` clamps its bounds onto `phat` at the 0/n and n/n
  boundaries.** This corrects a few-ulp floating-point error only; in exact
  arithmetic the clamp never binds. It exists so a caller drawing an error
  bar can rely on `low <= phat <= high`.
- **`power_check` returns `nan` power (not a raise) when the control rate is
  0 or 1.** The effect size and chi-square are still well defined there; the
  readout degrades rather than failing whole.

## Design conventions in `report.py`

- Every rate is drawn **with its interval**. A bar chart of two rates with no
  uncertainty shown invites the reader to treat any visible gap as a finding.
- The effect figure is **anchored on a zero line**. The question is "is this
  distinguishable from no difference," and an interval crossing zero answers
  it without a caption.
- `playtime_figure` **drops rows** above `clip_quantile`, it does not narrow
  the axis. A narrowed axis leaves whiskers running off the plot and draws a
  box whose quartiles describe data the reader cannot see.
- Figures are returned, never shown or written. `build_html_report` is the
  only function that touches the filesystem.
- Colors are the first two slots of a validated categorical palette; the
  slot **order** is the colorblind-safety mechanism. Dark-theme hexes are the
  same hues re-stepped for a dark surface, not an inversion.

## Testing

`tests/` mirrors `src/` one file per module. Tests are pure-function, no I/O,
except `test_loader.py` and `test_report.py`'s `build_html_report` cases,
which round-trip real files through pytest's `tmp_path`. `examples/run_analysis.py`
is a runnable demonstration, not a test — it has no assertions; verify it by
running it and reading the output.
