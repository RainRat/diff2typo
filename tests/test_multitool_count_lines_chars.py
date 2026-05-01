import sys
from pathlib import Path
from unittest.mock import patch
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_count_mode_lines(tmp_path):
    """Cover _extract_line_items and line-related labels/headers."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("line one\nline two\nline one")
    output_file = tmp_path / "output.txt"

    # Test arrow format for headers and labels
    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='arrow',
        lines=True,
        quiet=False,
        clean_items=False
    )

    content = output_file.read_text()
    assert "Line" in content # Header
    assert "line one" in content
    assert "Total lines encountered" in content

    # Test non-arrow format for item_label
    output_file_2 = tmp_path / "output2.txt"
    with patch("multitool.print_processing_stats") as mock_stats:
        multitool.count_mode(
            input_files=[str(input_file)],
            output_file=str(output_file_2),
            min_length=1,
            max_length=100,
            process_output=False,
            output_format='line',
            lines=True,
            quiet=True,
            clean_items=False
        )
        mock_stats.assert_called()
        args, kwargs = mock_stats.call_args
        assert kwargs['item_label'] == "line"

def test_count_mode_chars(tmp_path):
    """Cover _extract_char_items and character-related labels/headers."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("aabbc")
    output_file = tmp_path / "output.txt"

    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='arrow',
        chars=True,
        quiet=False
    )

    content = output_file.read_text()
    assert "Character" in content # Header
    assert "a" in content
    assert "Total characters encountered" in content

    # Test non-arrow format for item_label
    output_file_2 = tmp_path / "output2.txt"
    with patch("multitool.print_processing_stats") as mock_stats:
        multitool.count_mode(
            input_files=[str(input_file)],
            output_file=str(output_file_2),
            min_length=1,
            max_length=100,
            process_output=False,
            output_format='line',
            chars=True,
            quiet=True
        )
        mock_stats.assert_called()
        args, kwargs = mock_stats.call_args
        assert kwargs['item_label'] == "character"

def test_count_mode_chars_min_length_adjustment(tmp_path):
    """Cover line 1493: adjustment of min_length for chars."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("abc")
    output_file = tmp_path / "output.txt"

    # min_length defaults to 3 in _build_parser, but we pass it explicitly here
    # Line 1491: if chars and min_length == 3: min_length = 1
    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        output_format='line',
        chars=True,
        quiet=True
    )

    content = output_file.read_text()
    # If min_length was NOT adjusted, 'a', 'b', 'c' would be filtered out (length 1 < 3)
    assert "a: 1" in content
    assert "b: 1" in content
    assert "c: 1" in content
