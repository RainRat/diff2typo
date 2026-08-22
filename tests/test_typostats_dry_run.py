import os
import subprocess
import tempfile
import pytest

def test_typostats_dry_run_basic(tmp_path):
    input_file = tmp_path / "typos.txt"
    input_file.write_text("teh -> the\nwrod -> word\n", encoding="utf-8")
    output_file = tmp_path / "report.txt"

    res = subprocess.run(
        ["python", "typostats.py", str(input_file), "-o", str(output_file), "--dry-run"],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0
    assert "--- TYPOSTATS DRY RUN ---" in res.stderr
    assert "Input Source:" in res.stderr
    assert "Sample Typo Pattern Preview:" in res.stderr
    assert "Dry run complete. No output file written." in res.stderr
    assert not output_file.exists()


def test_typostats_dry_run_preview_content(tmp_path):
    input_file = tmp_path / "typos.txt"
    input_file.write_text("teh -> the\nwrod -> word\n", encoding="utf-8")

    res = subprocess.run(
        ["python", "typostats.py", str(input_file), "--dry-run", "-t"],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0
    assert "--- TYPOSTATS DRY RUN ---" in res.stderr
    assert "eh -> he" in res.stderr or "ro -> or" in res.stderr


def test_typostats_dry_run_empty_input(tmp_path):
    input_file = tmp_path / "empty.txt"
    input_file.write_text("", encoding="utf-8")

    res = subprocess.run(
        ["python", "typostats.py", str(input_file), "--dry-run"],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0
    assert "--- TYPOSTATS DRY RUN ---" in res.stderr
    assert "(No typo patterns found in input)" in res.stderr
