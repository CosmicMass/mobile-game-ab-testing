"""mobile_game_ab_testing: retention and A/B-test analysis for mobile game
player data.

Four stages, mirroring how an experiment analysis actually runs:

    loader     -- read the raw export, validate its schema and types, and
                  refuse to hand on a frame that is not what it claims to be
    retention  -- D1 / D7 retention rates per experiment group, each with a
                  Wilson confidence interval rather than a bare point estimate
    ab_test    -- is the difference real, and is there enough data to trust
                  the answer: minimum sample size / power, chi-square, and
                  effect size (absolute and relative)
    report     -- the same numbers as tidy summary tables and Plotly figures

Built on a public dataset (Kaggle "Mobile Games: A/B Testing - Cookie Cats"),
not proprietary data -- see README.md for the full disclosure.
"""

__version__ = "0.1.0"
