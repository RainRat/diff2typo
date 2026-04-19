import sys
from pathlib import Path
from unittest.mock import patch


# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_write_diff_report_plain(tmp_path, capsys):
    # Test diff report without color
    input_file = "test.txt"
    original = ["Line 1", "Line 2"]
    modified = ["Line 1 modified", "Line 2"]

    with patch("multitool.YELLOW", ""): # Disable color
        multitool._write_diff_report(input_file, original, modified, sys.stdout)

    captured = capsys.readouterr()
    assert "--- a/test.txt" in captured.out
    assert "+++ b/test.txt" in captured.out
    assert "-Line 1" in captured.out
    assert "+Line 1 modified" in captured.out
    # Ensure no ANSI codes
    assert "\033[" not in captured.out

def test_write_diff_report_color(tmp_path, capsys):
    # Test diff report with color
    input_file = "test.txt"
    original = ["Line 1", "Line 2"]
    modified = ["Line 1 modified", "Line 2"]

    # Mock colors and FORCE_COLOR
    with patch("multitool.YELLOW", "\033[1;33m"), \
         patch("multitool.GREEN", "\033[1;32m"), \
         patch("multitool.RED", "\033[1;31m"), \
         patch("multitool.BLUE", "\033[1;34m"), \
         patch("multitool.RESET", "\033[0m"), \
         patch.dict("os.environ", {"FORCE_COLOR": "1"}):
        multitool._write_diff_report(input_file, original, modified, sys.stdout)

    captured = capsys.readouterr()
    # Check for colorized lines
    # Red for removed line
    assert "\033[1;31m-Line 1\033[0m" in captured.out
    # Green for added line
    assert "\033[1;32m+Line 1 modified\033[0m" in captured.out
    # Blue for @@ line
    assert "\033[1;34m@@" in captured.out

def test_write_diff_report_to_file(tmp_path):
    # Test diff report writing to a file
    input_file = "test.txt"
    original = ["Line 1"]
    modified = ["Line 1 modified"]
    output_path = tmp_path / "diff.patch"

    with patch("multitool.YELLOW", ""), open(output_path, "w") as f:
        multitool._write_diff_report(input_file, original, modified, f)

    content = output_path.read_text()
    assert "--- a/test.txt" in content
    assert "-Line 1" in content
    assert "+Line 1 modified" in content

def test_scrub_mode_diff_integration(tmp_path, capsys):
    # Test that scrub_mode correctly calls _write_diff_report
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh typo")

    # Run scrub_mode with diff=True and ad-hoc mapping
    multitool.scrub_mode(
        input_files=[str(input_file)],
        mapping_file=None,
        output_file="-",
        min_length=1,
        max_length=100,
        process_output=True,
        diff=True,
        quiet=True,
        ad_hoc=["teh:the"]
    )

    captured = capsys.readouterr()
    assert "--- a/" + str(input_file) in captured.out
    assert "-teh typo" in captured.out
    assert "+the typo" in captured.out

def test_standardize_mode_diff_integration(tmp_path, capsys):
    # Test that standardize_mode correctly calls _write_diff_report
    file1 = tmp_path / "file1.txt"
    file1.write_text("Database database database") # database is most frequent

    # We need at least 2 files or 2 occurrences to have a dominant word.
    # Actually standardize_mode builds a mapping from all input files.

    with patch("multitool.YELLOW", ""):
        multitool.standardize_mode(
            input_files=[str(file1)],
            output_file="-",
            min_length=1,
            max_length=100,
            process_output=True,
            diff=True,
            quiet=True
        )

    captured = capsys.readouterr()
    assert "--- a/" + str(file1) in captured.out
    assert "-Database database database" in captured.out
    assert "+database database database" in captured.out
