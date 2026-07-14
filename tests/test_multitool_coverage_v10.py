import os
import sys
import io
import argparse
from unittest.mock import patch, MagicMock
import pytest

# Ensure the repository root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import multitool

# Helper to mock smart_open_output and prevent closing the StringIO object
class MockSmartOpen:
    def __init__(self, stream):
        self.stream = stream
    def __enter__(self):
        return self.stream
    def __exit__(self, *args):
        pass

def test_format_size_edge_cases():
    """Cover multitool.py lines 117, 119, 123, 124, 125-126."""
    # Negative size (117)
    assert multitool._format_size(-1) == "-1"

    # Zero size (119)
    assert multitool._format_size(0) == "0 B"

    # Integer value (123)
    assert multitool._format_size(1024) == "1 KB"

    # Non-integer value (124)
    # 1024 + 512 = 1536 bytes = 1.5 KB
    assert multitool._format_size(1536) == "1.5 KB"

    # Exabytes range (125-126)
    # 1 PB = 1024^5
    # 2^60 = 1024^6 = 1 EB
    eb_size = 2**60
    assert "EB" in multitool._format_size(eb_size)
    assert multitool._format_size(eb_size * 1024) == "1,024.0 EB"

def test_unzip_mode_coverage(tmp_path):
    """Cover multitool.py lines 5639-5643."""
    input_file = tmp_path / "pairs.txt"
    input_file.write_text("a -> b\nc -> d\n")

    output_left = io.StringIO()
    output_right = io.StringIO()

    # Extract left side (default)
    with patch('multitool.smart_open_output', return_value=MockSmartOpen(output_left)):
        multitool.unzip_mode(
            input_files=[str(input_file)],
            output_file='-',
            min_length=1,
            max_length=100,
            process_output=False,
            right_side=False
        )
    assert "a" in output_left.getvalue()
    assert "b" not in output_left.getvalue()

    # Extract right side
    with patch('multitool.smart_open_output', return_value=MockSmartOpen(output_right)):
        multitool.unzip_mode(
            input_files=[str(input_file)],
            output_file='-',
            min_length=1,
            max_length=100,
            process_output=False,
            right_side=True
        )
    assert "b" in output_right.getvalue()
    assert "a" not in output_right.getvalue()

def test_count_paragraphs_label(tmp_path, capsys):
    """Cover multitool.py line 3531."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("Para 1\n\nPara 2\n")

    # We need output_format != 'arrow' to hit 3531
    multitool.count_mode(
        input_files=[str(input_file)],
        output_file='-',
        min_length=1,
        max_length=100,
        process_output=False,
        paragraphs=True,
        output_format='line',
        quiet=False
    )

    captured = capsys.readouterr()
    # Check for "paragraph" in the processing stats
    assert "paragraph" in captured.err

def test_help_tuple_metavar(capsys):
    """Cover multitool.py line 8045."""
    # We want to test the part of show_mode_help that handles tuple metavars.

    test_mode = "test_mode"
    with patch.dict(multitool.MODE_DETAILS, {test_mode: {'summary': 's', 'description': 'd', 'example': 'e'}}):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='mode')
        mode_parser = subparsers.add_parser(test_mode)
        # Create a group that matches the filtering logic in show_mode_help
        group = mode_parser.add_argument_group("TEST OPTIONS")
        group.add_argument('--test', metavar=('A', 'B'), nargs=2, help="test help")

        with pytest.raises(SystemExit):
            multitool.show_mode_help(test_mode, parser)

        captured = capsys.readouterr()
        assert "A B" in captured.err

def test_main_min_length_defaults(tmp_path):
    """Cover multitool.py lines 9821, 9827."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("Some text here.")

    # Paragraphs mode (9821)
    with patch('sys.argv', ['multitool.py', 'paragraphs', str(input_file)]), \
         patch('multitool.paragraphs_mode') as mock_mode:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_mode.call_args
        assert kwargs["min_length"] == 20

    # Count mode with paragraphs (9827)
    with patch('sys.argv', ['multitool.py', 'count', str(input_file), '--paragraphs']), \
         patch('multitool.count_mode') as mock_mode:
        try:
            multitool.main()
        except SystemExit:
            pass
        _, kwargs = mock_mode.call_args
        assert kwargs["min_length"] == 20
