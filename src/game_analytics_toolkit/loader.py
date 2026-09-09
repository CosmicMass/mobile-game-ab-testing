"""Reading the Cookie Cats A/B-test export, and refusing to pass on a frame
that is not what downstream code assumes it is.

The dataset is small and public, so it is tempting to skip validation and
``pd.read_csv`` straight into the analysis. The reason not to: every function
in ``retention`` and ``ab_test`` takes the column names, the two group
labels, and the boolean dtype of the retention columns on trust. A silently
wrong frame -- a retention column read as the strings ``"True"``/``"False"``,
a third ``version`` value from a re-export, a duplicated ``userid`` from a
bad concat -- does not crash. It produces a plausible-looking number that is
quietly wrong, which is the one outcome an experiment readout must never
have. So the loader's job is to fail loudly *here*, with every problem
listed at once, or return a frame the rest of the toolkit can rely on.

The column contract is specific to this dataset by design; ``validate_frame``
is reusable on any equivalently shaped export by passing your own column
names.
"""

from __future__ import annotations

from typing import List, Sequence

import pandas as pd

USER_COLUMN = "userid"
GROUP_COLUMN = "version"
ROUNDS_COLUMN = "sum_gamerounds"
RETENTION_1_COLUMN = "retention_1"
RETENTION_7_COLUMN = "retention_7"

RETENTION_COLUMNS: tuple[str, str] = (RETENTION_1_COLUMN, RETENTION_7_COLUMN)
REQUIRED_COLUMNS: tuple[str, ...] = (
    USER_COLUMN,
    GROUP_COLUMN,
    ROUNDS_COLUMN,
    RETENTION_1_COLUMN,
    RETENTION_7_COLUMN,
)

# The two A/B arms in the published dataset: the in-game gate (a forced wait
# / in-app-purchase prompt) sits at level 30 for one group and level 40 for
# the other.
KNOWN_GROUPS: tuple[str, str] = ("gate_30", "gate_40")

# Case-folded string tokens accepted as booleans, for exports that have been
# round-tripped through R, Excel, or a CSV writer that stringified the column.
_TRUE_TOKENS = frozenset({"true", "t", "yes", "y", "1"})
_FALSE_TOKENS = frozenset({"false", "f", "no", "n", "0"})


class SchemaError(ValueError):
    """Raised by ``load_cookie_cats`` when the frame fails its column
    contract. The message lists every problem found, not just the first."""


def _coerce_bool_column(series: pd.Series) -> pd.Series:
    """Best-effort conversion of a retention column to real booleans.

    A column already of ``bool`` dtype is returned unchanged. Otherwise each
    value is matched (case-insensitively, whitespace-trimmed) against a small
    vocabulary of true/false tokens; anything unrecognized becomes ``pd.NA``
    so ``validate_frame`` can report exactly which values did not convert
    rather than this function guessing.

    Args:
        series: The raw retention column.

    Returns:
        A boolean (or nullable-boolean, if any value failed to convert)
        Series aligned to the input index.
    """
    if series.dtype == bool:
        return series

    def convert(value: object) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in _TRUE_TOKENS:
                return True
            if token in _FALSE_TOKENS:
                return False
        return pd.NA

    converted = series.map(convert)
    if converted.isna().any():
        return converted.astype("boolean")
    return converted.astype(bool)


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the dataset's expected dtypes without discarding bad values.

    ``userid`` and ``sum_gamerounds`` are pushed through ``pd.to_numeric``
    with ``errors="coerce"`` (a non-numeric value becomes ``NaN`` for
    ``validate_frame`` to catch, rather than aborting the load); the
    retention columns go through :func:`_coerce_bool_column`. Columns not in
    the contract are left exactly as they are.

    Args:
        df: A frame straight from ``pd.read_csv``.

    Returns:
        A new frame with the contract columns coerced. The input is not
        mutated.
    """
    out = df.copy()
    for column in (USER_COLUMN, ROUNDS_COLUMN):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in RETENTION_COLUMNS:
        if column in out.columns:
            out[column] = _coerce_bool_column(out[column])
    if GROUP_COLUMN in out.columns:
        out[GROUP_COLUMN] = out[GROUP_COLUMN].astype("string").str.strip()
    return out


def validate_frame(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
    group_column: str = GROUP_COLUMN,
    known_groups: Sequence[str] = KNOWN_GROUPS,
    retention_columns: Sequence[str] = RETENTION_COLUMNS,
    user_column: str = USER_COLUMN,
    rounds_column: str = ROUNDS_COLUMN,
    strict_columns: bool = True,
) -> List[str]:
    """Checks a frame against the Cookie Cats column contract.

    Every check runs and every failure is collected, so a caller sees the
    full picture in one pass instead of fixing one problem, re-running, and
    hitting the next.

    Args:
        df: The frame to check (typically already through
            :func:`coerce_dtypes`).
        required_columns: Columns that must be present.
        group_column: The A/B group column.
        known_groups: The only values ``group_column`` may contain when
            ``strict_columns`` is set. At least two distinct groups must be
            present regardless, or there is nothing to compare.
        retention_columns: Columns that must be boolean-valued.
        user_column: The per-user key; must be present, non-null and unique.
        rounds_column: The play-count column; must be a non-negative integer.
        strict_columns: When true, columns outside ``required_columns`` and
            group values outside ``known_groups`` are reported as problems --
            the usual signal that the wrong file was loaded.

    Returns:
        Human-readable problem descriptions. An empty list means the frame
        satisfies the contract.
    """
    problems: List[str] = []

    if len(df) == 0:
        problems.append("frame has no rows")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        problems.append(f"missing required column(s): {missing}")

    if strict_columns:
        unexpected = [c for c in df.columns if c not in required_columns]
        if unexpected:
            problems.append(
                f"unexpected column(s) not in the Cookie Cats contract: {unexpected}"
            )

    if user_column in df.columns:
        n_null = int(df[user_column].isna().sum())
        if n_null:
            problems.append(f"{user_column}: {n_null} null value(s)")
        n_dupe = int(df[user_column].duplicated().sum())
        if n_dupe:
            problems.append(
                f"{user_column}: {n_dupe} duplicate value(s) -- rows are meant to be one per user"
            )

    if rounds_column in df.columns:
        rounds = pd.to_numeric(df[rounds_column], errors="coerce")
        n_bad = int(rounds.isna().sum())
        if n_bad:
            problems.append(f"{rounds_column}: {n_bad} non-numeric or null value(s)")
        n_negative = int((rounds < 0).sum())
        if n_negative:
            problems.append(f"{rounds_column}: {n_negative} negative value(s)")
        non_integer = rounds.dropna()
        if not non_integer.empty and not (non_integer % 1 == 0).all():
            problems.append(f"{rounds_column}: contains non-integer value(s)")

    for column in retention_columns:
        if column not in df.columns:
            continue
        series = df[column]
        if series.dtype == bool:
            continue
        non_bool = series.dropna().map(lambda v: not isinstance(v, bool))
        if series.isna().any() or bool(non_bool.any()):
            offenders = sorted(
                {repr(v) for v in series[series.map(lambda v: not isinstance(v, bool))].unique()}
            )
            problems.append(
                f"{column}: not boolean -- unconvertible value(s): {offenders[:10]}"
            )

    if group_column in df.columns:
        present = [g for g in df[group_column].dropna().unique()]
        if len(present) < 2:
            problems.append(
                f"{group_column}: needs at least 2 groups to compare, found {present}"
            )
        if strict_columns:
            unknown = sorted(str(g) for g in present if g not in known_groups)
            if unknown:
                problems.append(
                    f"{group_column}: value(s) outside {list(known_groups)}: {unknown}"
                )

    return problems


def load_cookie_cats(
    path: str,
    *,
    strict_columns: bool = True,
) -> pd.DataFrame:
    """Loads and validates the Cookie Cats CSV.

    Reads ``path``, applies the dataset's dtypes via :func:`coerce_dtypes`,
    then runs :func:`validate_frame`. If anything fails the contract, raises
    with every problem listed -- the frame is never returned in a
    known-broken state.

    Args:
        path: Path to the ``cookie_cats.csv`` download.
        strict_columns: Passed through to :func:`validate_frame`. Leave on
            unless you have deliberately trimmed or extended the file.

    Returns:
        The validated frame: ``userid`` and ``sum_gamerounds`` integer,
        ``retention_1`` / ``retention_7`` boolean, ``version`` string.

    Raises:
        SchemaError: If the frame fails its column contract.
    """
    raw = pd.read_csv(path)
    df = coerce_dtypes(raw)
    problems = validate_frame(df, strict_columns=strict_columns)
    if problems:
        listed = "\n".join(f"  - {p}" for p in problems)
        raise SchemaError(f"{path} does not match the Cookie Cats contract:\n{listed}")

    df[USER_COLUMN] = df[USER_COLUMN].astype("int64")
    df[ROUNDS_COLUMN] = df[ROUNDS_COLUMN].astype("int64")
    for column in RETENTION_COLUMNS:
        df[column] = df[column].astype(bool)
    df[GROUP_COLUMN] = df[GROUP_COLUMN].astype("string")
    return df


def group_sizes(df: pd.DataFrame, *, group_column: str = GROUP_COLUMN) -> dict[str, int]:
    """Row count per experiment group.

    Args:
        df: A loaded frame.
        group_column: The A/B group column.

    Returns:
        Mapping of group label to number of users, ordered as the groups
        first appear.
    """
    counts = df[group_column].value_counts(sort=False)
    return {str(label): int(count) for label, count in counts.items()}
