import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo

def test_format_typos_single_word_structured():
    # Covers line 521: formatted.append(filter_to_letters(typo))
    typos = ["Error123"]
    assert diff2typo.format_typos(typos, "csv") == ["error"]
    assert diff2typo.format_typos(typos, "table") == ["error"]
    assert diff2typo.format_typos(typos, "list") == ["error"]
    assert diff2typo.format_typos(typos, "arrow") == ["Error123"]

def test_main_summary_extra_metrics(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    diff_file = tmp_path / "diff.txt"
    # Create multiple occurrences for min-count
    diff_file.write_text("--- a/f\n+++ b/f\n@@\n-teh\n+the\n-teh\n+the\n-eror\n+error\n")

    output_file = tmp_path / "output.txt"

    # We need to mock typos tool run or it might fail if not installed
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="")

        monkeypatch.setattr(sys, "argv", [
            "diff2typo.py",
            "--input", str(diff_file),
            "--output", str(output_file),
            "--min-count", "2",
            "--limit", "1"
        ])

        # We need to mock _should_enable_color to avoid ANSI codes in comparison if needed,
        # but here we just check if the text is present in stderr.
        with patch("diff2typo._should_enable_color", return_value=False):
            # We also need to mock time.perf_counter to have predictable duration if we cared,
            # but we just want to see the extra metrics.
            diff2typo.main()

    captured = capsys.readouterr()
    # Check if extra metrics are in stderr (where summary is written)
    assert "Min occurrences (--min-count):" in captured.err
    assert "2" in captured.err
    assert "Output limit (--limit):" in captured.err
    assert "1" in captured.err

def test_main_sort_by_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    diff_file = tmp_path / "diff.txt"
    # teh (3), eror (2), misspell (1)
    content = (
        "-teh\n+the\n" * 3 +
        "-eror\n+error\n" * 2 +
        "-mispel\n+misspell\n" * 1
    )
    diff_file.write_text(f"--- a/f\n+++ b/f\n@@\n{content}")

    output_file = tmp_path / "output.txt"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="")

        monkeypatch.setattr(sys, "argv", [
            "diff2typo.py",
            "--input", str(diff_file),
            "--output", str(output_file),
            "--sort", "count",
            "--quiet"
        ])

        diff2typo.main()

    results = output_file.read_text().strip().splitlines()
    # Order should be by count: teh, then eror, then mispel
    assert results == ["teh -> the", "eror -> error", "mispel -> misspell"]

def test_main_sort_by_alpha(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    diff_file = tmp_path / "diff.txt"
    # zzz (2), aaa (1)
    content = (
        "-zzz\n+zed\n" * 2 +
        "-aaa\n+abc\n" * 1
    )
    diff_file.write_text(f"--- a/f\n+++ b/f\n@@\n{content}")

    output_file = tmp_path / "output.txt"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="")

        # Default sort is alpha
        monkeypatch.setattr(sys, "argv", [
            "diff2typo.py",
            "--input", str(diff_file),
            "--output", str(output_file),
            "--quiet"
        ])

        diff2typo.main()

    results = output_file.read_text().strip().splitlines()
    assert results == ["aaa -> abc", "zzz -> zed"]
