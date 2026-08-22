import os
import sys
import pytest
from unittest.mock import patch
import diff2typo


def test_diff2typo_dry_run_basic(tmp_path, caplog):
    """Verify that --dry-run prints execution summary and sample preview without creating the output file."""
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(
        "diff --git a/file.py b/file.py\n"
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "-teh house\n"
        "+the house\n"
    )
    output_file = tmp_path / "output_typos.txt"

    test_args = [
        "diff2typo.py",
        str(diff_file),
        "-o",
        str(output_file),
        "--dry-run",
    ]

    with patch.object(sys, "argv", test_args):
        with caplog.at_level("INFO"):
            diff2typo.main()

    assert not output_file.exists(), "Output file must not be written in dry-run mode."
    assert "--- DIFF2TYPO DRY RUN ---" in caplog.text
    assert f"Output Target: {output_file}" in caplog.text
    assert "Mode: typos" in caplog.text
    assert "Sample Typo Extraction Preview:" in caplog.text
    assert "teh -> the" in caplog.text
    assert "Dry run complete. No files were written." in caplog.text


def test_diff2typo_dry_run_both_mode(tmp_path, caplog):
    """Verify --dry-run with --mode both displays both typos and corrections preview without writing output."""
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(
        "diff --git a/file.py b/file.py\n"
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "-teh house\n"
        "+the house\n"
    )
    output_file = tmp_path / "output_both.txt"

    test_args = [
        "diff2typo.py",
        str(diff_file),
        "-o",
        str(output_file),
        "-M",
        "both",
        "--dry-run",
    ]

    with patch.object(sys, "argv", test_args):
        with caplog.at_level("INFO"):
            diff2typo.main()

    assert not output_file.exists()
    assert "--- DIFF2TYPO DRY RUN ---" in caplog.text
    assert "Mode: both" in caplog.text
    assert "Sample Typo Extraction Preview:" in caplog.text
    assert "Typos (" in caplog.text
    assert "Corrections (" in caplog.text
    assert "Dry run complete. No files were written." in caplog.text


def test_diff2typo_dry_run_corrections_mode(tmp_path, caplog):
    """Verify --dry-run with --mode corrections runs cleanly without writing output files."""
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(
        "diff --git a/file.py b/file.py\n"
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "-teh house\n"
        "+the house\n"
    )
    output_file = tmp_path / "output_corrections.txt"

    test_args = [
        "diff2typo.py",
        str(diff_file),
        "-o",
        str(output_file),
        "-M",
        "corrections",
        "--dry-run",
    ]

    with patch.object(sys, "argv", test_args):
        with caplog.at_level("INFO"):
            diff2typo.main()

    assert not output_file.exists()
    assert "--- DIFF2TYPO DRY RUN ---" in caplog.text
    assert "Mode: corrections" in caplog.text
    assert "Dry run complete. No files were written." in caplog.text
