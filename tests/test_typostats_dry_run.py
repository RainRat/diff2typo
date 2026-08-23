import os
import sys
import tempfile
import pytest
from unittest.mock import patch
import typostats


def test_dry_run_stdout_and_no_file_creation(capsys):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_in:
        tmp_in.write(b"teh -> the\nwrod -> word\n")
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path + ".out"

    try:
        test_args = [
            "typostats.py",
            tmp_in_path,
            "--output", tmp_out_path,
            "--dry-run",
            "-t",
        ]
        with patch.object(sys, "argv", test_args):
            typostats.main()

        captured = capsys.readouterr()
        assert "--- TYPOSTATS DRY RUN ---" in captured.err
        assert "Input Sources:" in captured.err
        assert tmp_out_path in captured.err
        assert "Sample Preview (First 5 replacements):" in captured.err
        assert "eh -> he (Count: 1)" in captured.err or "or -> ro" in captured.err

        # Output file should NOT be created in dry run
        assert not os.path.exists(tmp_out_path)
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)


def test_dry_run_empty_matches(capsys):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_in:
        tmp_in.write(b"hello -> world\n")
        tmp_in_path = tmp_in.name

    try:
        test_args = [
            "typostats.py",
            tmp_in_path,
            "--dry-run",
            "--min", "10",
        ]
        with patch.object(sys, "argv", test_args):
            typostats.main()

        captured = capsys.readouterr()
        assert "--- TYPOSTATS DRY RUN ---" in captured.err
        assert "(No replacements found matching criteria)" in captured.err
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)


def test_dry_run_sorting_options(capsys):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_in:
        tmp_in.write(b"teh -> the\nwrod -> word\n")
        tmp_in_path = tmp_in.name

    try:
        for sort_opt in ["typo", "correct", "count"]:
            test_args = [
                "typostats.py",
                tmp_in_path,
                "--dry-run",
                "-s", sort_opt,
                "-a",
            ]
            with patch.object(sys, "argv", test_args):
                typostats.main()

            captured = capsys.readouterr()
            assert "--- TYPOSTATS DRY RUN ---" in captured.err
            assert f"Sort By: {sort_opt}" in captured.err
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
