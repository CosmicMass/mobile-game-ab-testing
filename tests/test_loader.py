"""
Unit tests for mobile_game_ab_testing.loader

Run this file only:
    python -m pytest tests/test_loader.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

from mobile_game_ab_testing.loader import (
    SchemaError,
    coerce_dtypes,
    group_sizes,
    load_cookie_cats,
    validate_frame,
)


def clean_rows() -> dict[str, list[object]]:
    """A minimal frame that satisfies the contract, as plain lists."""
    return {
        "userid": [1, 2, 3, 4],
        "version": ["gate_30", "gate_30", "gate_40", "gate_40"],
        "sum_gamerounds": [10, 0, 250, 7],
        "retention_1": [True, False, True, False],
        "retention_7": [False, False, True, False],
    }


def write_csv(tmp_path: Path, frame: pd.DataFrame, name: str = "cookie_cats.csv") -> str:
    """Writes a frame to a CSV under tmp_path and returns the path."""
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


class TestLoadCookieCatsHappyPath:

    def test_loads_clean_file(self, tmp_path: Path):
        path = write_csv(tmp_path, pd.DataFrame(clean_rows()))
        df = load_cookie_cats(path)
        assert len(df) == 4
        assert list(df.columns) == [
            "userid",
            "version",
            "sum_gamerounds",
            "retention_1",
            "retention_7",
        ]

    def test_dtypes_are_normalized(self, tmp_path: Path):
        path = write_csv(tmp_path, pd.DataFrame(clean_rows()))
        df = load_cookie_cats(path)
        assert df["userid"].dtype == "int64"
        assert df["sum_gamerounds"].dtype == "int64"
        assert df["retention_1"].dtype == bool
        assert df["retention_7"].dtype == bool

    def test_values_survive_the_round_trip(self, tmp_path: Path):
        path = write_csv(tmp_path, pd.DataFrame(clean_rows()))
        df = load_cookie_cats(path)
        assert df["retention_1"].tolist() == [True, False, True, False]
        assert df["sum_gamerounds"].tolist() == [10, 0, 250, 7]

    def test_zero_gamerounds_is_valid(self, tmp_path: Path):
        # A user who installed and never played is a real, meaningful row --
        # it must not be confused with a missing value and rejected.
        rows = clean_rows()
        rows["sum_gamerounds"] = [0, 0, 0, 0]
        path = write_csv(tmp_path, pd.DataFrame(rows))
        assert len(load_cookie_cats(path)) == 4


class TestBooleanCoercion:
    """Retention columns must end up as real booleans however the file spells them.

    A retention column read as the strings "True"/"False" is the failure this
    guards: every value is truthy, so ``sum()`` silently returns the row count
    and every retention rate comes out as 100%.
    """

    def test_uppercase_string_booleans(self, tmp_path: Path):
        rows = clean_rows()
        rows["retention_1"] = ["TRUE", "FALSE", "TRUE", "FALSE"]
        path = write_csv(tmp_path, pd.DataFrame(rows))
        df = load_cookie_cats(path)
        assert df["retention_1"].dtype == bool
        assert df["retention_1"].tolist() == [True, False, True, False]

    def test_integer_booleans(self, tmp_path: Path):
        rows = clean_rows()
        rows["retention_7"] = [1, 0, 1, 0]
        path = write_csv(tmp_path, pd.DataFrame(rows))
        df = load_cookie_cats(path)
        assert df["retention_7"].dtype == bool
        assert df["retention_7"].tolist() == [True, False, True, False]

    def test_yes_no_strings(self, tmp_path: Path):
        rows = clean_rows()
        rows["retention_1"] = ["yes", "no", "y", "n"]
        path = write_csv(tmp_path, pd.DataFrame(rows))
        assert load_cookie_cats(path)["retention_1"].tolist() == [True, False, True, False]

    def test_string_booleans_would_otherwise_be_all_truthy(self, tmp_path: Path):
        # The bug being prevented, stated as a test: without coercion the
        # retention rate of this file reads as 100%, not 50%.
        rows = clean_rows()
        rows["retention_1"] = ["True", "False", "True", "False"]
        raw = pd.read_csv(write_csv(tmp_path, pd.DataFrame(rows)), dtype={"retention_1": str})
        assert raw["retention_1"].astype(bool).sum() == 4  # every string is truthy

        df = load_cookie_cats(write_csv(tmp_path, pd.DataFrame(rows)))
        assert int(df["retention_1"].sum()) == 2

    def test_unconvertible_value_is_reported_not_guessed(self, tmp_path: Path):
        rows = clean_rows()
        rows["retention_1"] = ["True", "False", "maybe", "False"]
        path = write_csv(tmp_path, pd.DataFrame(rows))
        with pytest.raises(SchemaError, match="retention_1"):
            load_cookie_cats(path)


class TestValidateFrameCatchesBadData:

    def test_missing_column(self):
        df = pd.DataFrame(clean_rows()).drop(columns=["retention_7"])
        problems = validate_frame(coerce_dtypes(df))
        assert any("retention_7" in p and "missing" in p for p in problems)

    def test_duplicate_userid(self):
        rows = clean_rows()
        rows["userid"] = [1, 1, 3, 4]
        problems = validate_frame(coerce_dtypes(pd.DataFrame(rows)))
        assert any("duplicate" in p for p in problems)

    def test_negative_gamerounds(self):
        rows = clean_rows()
        rows["sum_gamerounds"] = [10, -5, 250, 7]
        problems = validate_frame(coerce_dtypes(pd.DataFrame(rows)))
        assert any("negative" in p for p in problems)

    def test_non_numeric_gamerounds(self):
        rows = clean_rows()
        rows["sum_gamerounds"] = [10, "lots", 250, 7]
        problems = validate_frame(coerce_dtypes(pd.DataFrame(rows)))
        assert any("non-numeric" in p for p in problems)

    def test_unknown_group_value(self):
        rows = clean_rows()
        rows["version"] = ["gate_30", "gate_30", "gate_40", "gate_50"]
        problems = validate_frame(coerce_dtypes(pd.DataFrame(rows)))
        assert any("gate_50" in p for p in problems)

    def test_single_group_cannot_be_compared(self):
        rows = clean_rows()
        rows["version"] = ["gate_30"] * 4
        problems = validate_frame(coerce_dtypes(pd.DataFrame(rows)))
        assert any("at least 2 groups" in p for p in problems)

    def test_empty_frame(self):
        empty = pd.DataFrame({column: [] for column in clean_rows()})
        problems = validate_frame(coerce_dtypes(empty))
        assert any("no rows" in p for p in problems)

    def test_unexpected_column_is_flagged_when_strict(self):
        df = pd.DataFrame(clean_rows())
        df["campaign"] = "spring_sale"
        problems = validate_frame(coerce_dtypes(df), strict_columns=True)
        assert any("campaign" in p for p in problems)

    def test_unexpected_column_is_allowed_when_not_strict(self):
        df = pd.DataFrame(clean_rows())
        df["campaign"] = "spring_sale"
        assert validate_frame(coerce_dtypes(df), strict_columns=False) == []

    def test_unknown_group_is_allowed_when_not_strict(self):
        rows = clean_rows()
        rows["version"] = ["gate_30", "gate_30", "gate_40", "gate_50"]
        problems = validate_frame(coerce_dtypes(pd.DataFrame(rows)), strict_columns=False)
        assert problems == []

    def test_clean_frame_has_no_problems(self):
        assert validate_frame(coerce_dtypes(pd.DataFrame(clean_rows()))) == []

    def test_nullable_boolean_column_does_not_crash_the_check(self):
        # A retention column that half-converts leaves pd.NA behind. The
        # offender-listing path indexes the Series with a mask built from it,
        # and an NA inside a boolean mask raises in pandas -- so this asserts
        # the check REPORTS rather than blowing up.
        rows = clean_rows()
        rows["retention_7"] = [True, None, "nonsense", False]
        problems = validate_frame(coerce_dtypes(pd.DataFrame(rows)))
        assert any("retention_7" in p for p in problems)


class TestSchemaErrorMessage:

    def test_lists_every_problem_at_once(self, tmp_path: Path):
        # One pass should surface all of it, not stop at the first failure.
        rows = clean_rows()
        rows["userid"] = [1, 1, 3, 4]
        rows["sum_gamerounds"] = [10, -5, 250, 7]
        rows["version"] = ["gate_30", "gate_30", "gate_40", "gate_50"]
        path = write_csv(tmp_path, pd.DataFrame(rows))

        with pytest.raises(SchemaError) as excinfo:
            load_cookie_cats(path)

        message = str(excinfo.value)
        assert "duplicate" in message
        assert "negative" in message
        assert "gate_50" in message

    def test_names_the_offending_file(self, tmp_path: Path):
        rows = clean_rows()
        rows["userid"] = [1, 1, 3, 4]
        path = write_csv(tmp_path, pd.DataFrame(rows), name="broken.csv")
        with pytest.raises(SchemaError, match="broken.csv"):
            load_cookie_cats(path)


class TestCoerceDtypes:

    def test_does_not_mutate_the_input(self):
        original = pd.DataFrame(clean_rows())
        before = original["retention_1"].tolist()
        coerce_dtypes(original)
        assert original["retention_1"].tolist() == before

    def test_leaves_unknown_columns_alone(self):
        df = pd.DataFrame(clean_rows())
        df["campaign"] = ["a", "b", "c", "d"]
        assert coerce_dtypes(df)["campaign"].tolist() == ["a", "b", "c", "d"]

    def test_strips_whitespace_from_group_labels(self):
        rows = clean_rows()
        rows["version"] = [" gate_30", "gate_30 ", "gate_40", "gate_40"]
        assert validate_frame(coerce_dtypes(pd.DataFrame(rows))) == []


class TestGroupSizes:

    def test_counts_per_group(self):
        assert group_sizes(pd.DataFrame(clean_rows())) == {"gate_30": 2, "gate_40": 2}

    def test_uneven_groups(self):
        rows = clean_rows()
        rows["version"] = ["gate_30", "gate_30", "gate_30", "gate_40"]
        assert group_sizes(pd.DataFrame(rows)) == {"gate_30": 3, "gate_40": 1}
