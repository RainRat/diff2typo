import sys
import logging
import io
import subprocess
import runpy
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
                assert "ITEM" in stderr_output # Header

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

    # Force sys.stderr.isatty to True
    with patch("sys.stderr.isatty", return_value=True):
        # Patch LEVEL_COLORS directly as it's already initialized
        with patch.dict(formatter.LEVEL_COLORS, {logging.ERROR: "\033[31m"}):
            with patch("multitool.RESET", "\033[0m"):
                formatted = formatter.format(record)
                assert "\033[31mERROR\033[0m: Error message" in formatted

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
    assert "ITEM" in content

def test_cycles_mode_visited_node(tmp_path):
    """Cover line 2021: if node in visited: return in cycles_mode."""
    # a -> b, c -> b
    # Starting at a visits a, b.
    # Starting at c visits c, then hits b which is already visited.
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> b\nc -> b")
    output_file = tmp_path / "output.txt"

    multitool.cycles_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True,
        clean_items=True
    )
    # No cycles expected
    assert output_file.read_text().strip() == ""

def test_map_mode_empty_line_skipping(tmp_path):
    """Cover line 3091: if not line_content: continue in map_mode."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple\n\nbanana")
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh -> the")
    output_file = tmp_path / "output.txt"

    multitool.map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        quiet=True
    )

    content = output_file.read_text().strip().splitlines()
    assert content == ["apple", "banana"]

def test_multitool_main_invocation():
    """Cover line 5156: the __main__ block using runpy."""
    with patch("sys.argv", ["multitool.py", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("multitool", run_name="__main__")
        assert excinfo.value.code == 0
