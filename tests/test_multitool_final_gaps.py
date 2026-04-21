import sys
import logging
import io
import subprocess
from pathlib import Path
from unittest.mock import patch
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_ngrams_clean_empty_word(tmp_path):
    """Cover line 1022: continue in _extract_ngram_items when word is empty after cleaning."""
    input_file = tmp_path / "input.txt"
    # "!!!" will be empty after filter_to_letters
    input_file.write_text("hello !!! world")
    output_file = tmp_path / "output.txt"

    multitool.ngrams_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        n=2,
        clean_items=True,
        quiet=True
    )

    content = output_file.read_text().strip()
    # Should be "hello world" if !!! was skipped
    assert content == "hello world"

def test_count_mode_arrow_stderr_coverage(tmp_path):
    """Cover lines 1467-1468 and 1489-1492: stderr metadata in count_mode arrow format."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple banana apple")

    # We need output_file='-', output_format='arrow', and quiet=False
    # Also need to mock sys.stderr.write to verify it was called
    with patch("sys.stderr", new=io.StringIO()) as mock_stderr:
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            # Force color and TTY for coverage of the branches
            with patch("os.environ", {"FORCE_COLOR": "1"}):
                # Ensure headers are considered colorable by setting TTY mock
                mock_stdout.isatty = lambda: True
                multitool.count_mode(
                    input_files=[str(input_file)],
                    output_file='-',
                    min_length=1,
                    max_length=100,
                    process_output=False,
                    output_format='arrow',
                    quiet=False
                )

                stderr_output = mock_stderr.getvalue()
                assert "ANALYSIS SUMMARY" in stderr_output
                assert "Word" in stderr_output # Header

def test_highlight_mode_limit(tmp_path):
    """Cover line 3091: limit in highlight_mode."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("line1\nline2\nline3")
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("line")
    output_file = tmp_path / "output.txt"

    multitool.highlight_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        limit=1,
        quiet=True
    )

    content = output_file.read_text().splitlines()
    assert len(content) == 1

def test_minimal_formatter_colorized_branch():
    """Cover line 3599: colorized branch in MinimalFormatter."""
    formatter = multitool.MinimalFormatter()
    record = logging.LogRecord(
        name="test", level=logging.ERROR, pathname="test.py",
        lineno=10, msg="Error message", args=None, exc_info=None
    )

    # Force YELLOW/RED/RESET to be colored to simulate color-capable environment
    with patch.dict("os.environ", {"FORCE_COLOR": "1"}), \
         patch("multitool.YELLOW", "\033[1;33m"), \
         patch("multitool.RED", "\033[1;31m"), \
         patch("multitool.RESET", "\033[0m"):
        # Patch LEVEL_COLORS directly as it's already initialized
        with patch.dict(formatter.LEVEL_COLORS, {logging.ERROR: "\033[1;31m"}):
            formatted = formatter.format(record)
            assert "\033[1;31mERROR\033[0m: Error message" in formatted

def test_main_entry_point():
    """Cover line 4862: the __main__ block by running as a subprocess."""
    # We just run --help to see if it works
    result = subprocess.run(
        [sys.executable, "multitool.py", "--help"],
        capture_output=True,
        text=True,
        check=True
    )
    assert "A multipurpose tool for cleaning, getting, and analyzing text files." in result.stdout

def test_count_mode_arrow_to_file_header_coverage(tmp_path):
    """Cover lines 1471-1472: header colors when outputting to a file in count_mode arrow."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple banana apple")
    output_file = tmp_path / "output.txt"

    # output_file is NOT '-', so headers go to the file
    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='arrow',
        quiet=True
    )

    content = output_file.read_text()
    assert "ANALYSIS SUMMARY" in content
    assert "Word" in content
