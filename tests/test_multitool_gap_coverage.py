import sys
import logging
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_count_mode_pairs_md_table(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh -> the\nteh -> the")
    output_file = tmp_path / "output.md"

    multitool.count_mode(
        [str(input_file)], str(output_file), 1, 100, False,
        pairs=True, output_format='md-table'
    )

    content = output_file.read_text()
    assert "| Typo | Correction | Count |" in content
    assert "| teh | the | 2 |" in content

def test_count_mode_pairs_empty_after_cleaning(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("!!! -> !!!\nteh -> the") # !!! will be empty after filter_to_letters
    output_file = tmp_path / "output.txt"

    multitool.count_mode(
        [str(input_file)], str(output_file), 1, 100, False,
        pairs=True, clean_items=True
    )

    content = output_file.read_text()
    assert "teh -> the: 1" in content
    assert "!!! -> !!!" not in content

def test_similarity_mode_empty_after_cleaning(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("!!! -> !!!\ncat -> bat")
    output_file = tmp_path / "output.txt"

    multitool.similarity_mode(
        [str(input_file)], str(output_file), 1, 100, False,
        clean_items=True
    )

    content = output_file.read_text()
    assert "cat -> bat" in content
    assert "!!!" not in content

def test_casing_mode_empty_norm(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("!!! Hello hello")
    output_file = tmp_path / "output.txt"

    multitool.casing_mode(
        [str(input_file)], str(output_file), 1, 100, False,
        clean_items=True
    )

    content = output_file.read_text()
    assert "hello" in content
    assert "!!!" not in content

def test_pairs_mode_empty_after_cleaning(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("!!! -> !!!\nteh -> the")
    output_file = tmp_path / "output.txt"

    multitool.pairs_mode(
        [str(input_file)], str(output_file), 1, 100, False,
        clean_items=True
    )

    content = output_file.read_text()
    assert "teh -> the" in content
    assert "!!!" not in content

def test_swap_mode_empty_after_cleaning(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("!!! -> !!!\nteh -> the")
    output_file = tmp_path / "output.txt"

    multitool.swap_mode(
        [str(input_file)], str(output_file), 1, 100, False,
        clean_items=True
    )

    content = output_file.read_text()
    assert "the -> teh" in content
    assert "!!!" not in content

def test_swap_mode_process_output(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("b -> a\nb -> a\nc -> a")
    output_file = tmp_path / "output.txt"

    multitool.swap_mode(
        [str(input_file)], str(output_file), 1, 100, True,
        clean_items=True
    )

    content = output_file.read_text().splitlines()
    assert content == ["a -> b", "a -> c"]

def test_sample_mode_no_count_no_percent(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("line1\nline2\nline3")
    output_file = tmp_path / "output.txt"

    # Calling with both None should use k = total_valid_items (line 2189)
    multitool.sample_mode(
        [str(input_file)], str(output_file), 1, 100, False,
        sample_count=None, sample_percent=None
    )

    content = output_file.read_text().splitlines()
    assert len(content) == 3

def test_minimal_formatter_colorized():
    formatter = multitool.MinimalFormatter()
    record = logging.LogRecord(
        name="test", level=logging.WARNING, pathname="test.py",
        lineno=10, msg="Warning message", args=None, exc_info=None
    )

    # We need to ensure multitool constants are set for color
    with patch("sys.stderr.isatty", return_value=True):
        with patch("sys.stdout.isatty", return_value=True):
            with patch.dict("os.environ", {}, clear=True):
                # Re-evaluate color constants or just check if they are expected in the output
                # The formatter uses self.LEVEL_COLORS which uses global color constants
                formatted = formatter.format(record)
                # If YELLOW is "" then it won't be in formatted
                # Let's check what YELLOW is
                if multitool.YELLOW:
                    assert multitool.YELLOW in formatted
                else:
                    assert "WARNING: Warning message" in formatted

def test_mode_help_action_error():
    argparse.ArgumentParser()
    action = multitool.ModeHelpAction(option_strings=["--mode-help"], dest="mode_help")

    mock_parser = MagicMock()
    # parser.error should raise SystemExit, as the real argparse.ArgumentParser.error does.
    mock_parser.error.side_effect = SystemExit(1)

    with pytest.raises(SystemExit):
        action(mock_parser, MagicMock(), "nonexistent_mode")

    mock_parser.error.assert_called_with("Unknown mode: nonexistent_mode")
