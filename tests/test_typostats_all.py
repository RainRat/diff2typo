import os
import subprocess
import sys
import pytest

def test_typostats_all_flag(tmp_path):
    typos_file = tmp_path / "typos.txt"
    typos_file.write_text("teh -> the\nrecieve -> receive\nm -> rn\nph -> f\nor -> o\na -> aa\n", encoding="utf-8")

    # Run typostats with --all flag
    # Using sys.executable to ensure we use the same python environment
    cmd = [sys.executable, "typostats.py", str(typos_file), "-a"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0

    # Check that enabled features are listed in stderr
    assert "Enabled features:" in result.stderr
    assert "keyboard, transposition, 1-to-2, 2-to-1, deletions/insertions" in result.stderr
    # Check that transposition summary is in stderr
    assert "Transpositions:" in result.stderr

    # Check stdout for some expected output
    # Since we have teh->the and recieve->receive, we expect transposition
    # he | eh and ei | ie
    assert "he" in result.stdout
    assert "eh" in result.stdout
    assert "ei" in result.stdout
    assert "ie" in result.stdout

    # Check for 1-to-2 (m -> rn)
    assert "rn" in result.stdout
    assert "m" in result.stdout

    # Check for 2-to-1 (ph -> f)
    assert "f" in result.stdout
    assert "ph" in result.stdout

    # Check for deletions/insertions (or -> o, a -> aa)
    assert " o " in result.stdout
    assert " or " in result.stdout
    assert " aa " in result.stdout
    assert " a " in result.stdout

def test_typostats_limit_alias(tmp_path):
    typos_file = tmp_path / "typos.txt"
    typos_file.write_text("a -> b\nc -> d\ne -> f\n", encoding="utf-8")

    cmd = [sys.executable, "typostats.py", str(typos_file), "-L", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0
    # Count lines in stdout that look like report rows (contain '|')
    rows = [line for line in result.stdout.splitlines() if "│" in line and "CORRECT" not in line and "─" not in line]
    assert len(rows) == 1
