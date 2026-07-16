import os
import sys
import logging
import io
import argparse
import re
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

def test_search_mode_arrow_summary(tmp_path, capsys):
    """Cover multitool.py lines 5633-5646."""
    input_file = tmp_path / "test.txt"
    input_file.write_text("Hello world\nThis is a test\n")

    multitool.search_mode(
        input_files=[str(input_file)],
        query="test",
        output_file='-',
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='arrow',
        quiet=True
    )

    captured = capsys.readouterr()
    assert "SEARCH ANALYSIS SUMMARY" in captured.out
    assert "Matches found" in captured.out

def test_replace_mode_smart_case_regex_backref(tmp_path):
    """Cover multitool.py lines 6518-6521."""
    input_file = tmp_path / "test.txt"
    input_file.write_text("Hello World")

    multitool.replace_mode(
        input_files=[str(input_file)],
        old_text=r"(Hello)",
        new_text=r"Hi \1",
        output_file=str(input_file),
        use_regex=True,
        smart_case=True,
        quiet=True
    )

    # replace_mode adds a newline at the end when writing output
    assert input_file.read_text().strip() == "Hi Hello World"

def test_brokenlinks_suggestion_oserror(tmp_path, capsys):
    """Cover multitool.py lines 2874-2875."""
    md_file = tmp_path / "test.md"
    md_file.write_text("[link](nonexistent.md)")

    with patch('os.listdir', side_effect=OSError("mock error")):
        multitool.brokenlinks_mode(
            input_files=[str(md_file)],
            output_file='-',
            quiet=True
        )

def test_truncate_text_empty():
    """Cover multitool.py line 190."""
    assert multitool._truncate_text("") == ""
    assert multitool._truncate_text(None) == ""

def test_best_suggestion_tie_breaking():
    """Cover multitool.py lines 301-302."""
    # Tie in distance, same length -> alphabetical
    assert multitool._get_best_suggestion("apple", ["apples", "applee"], max_dist=1) == "applee"
    # Tie in distance, different length -> shorter wins
    assert multitool._get_best_suggestion("apple", ["appleS", "aple"], max_dist=1) == "aple"

def test_write_structured_data_toml_fallback_full(tmp_path):
    """Cover multitool.py lines 881-882."""
    data = {"key": "value"}

    mock_out = MagicMock()
    # Trigger line 881-882 by making seek fail
    mock_out.seek.side_effect = Exception("seek fail")

    with patch('multitool._TOML_AVAILABLE', True), \
         patch('toml.dump', side_effect=Exception("mock dump error")), \
         patch('multitool.smart_open_output', return_value=MockSmartOpen(mock_out)):
        multitool._write_structured_data(data, "dummy.toml", output_format='toml')

    mock_out.seek.assert_called_with(0)
    # It should still try to write JSON
    assert mock_out.write.called

def test_convert_mode_limit(tmp_path):
    """Cover multitool.py line 2361."""
    input_file = tmp_path / "test.json"
    input_file.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}')

    output = io.StringIO()
    with patch('multitool.smart_open_output', return_value=MockSmartOpen(output)):
        multitool.convert_mode(
            input_files=[str(input_file)],
            output_file="-",
            output_format='json',
            limit=2
        )

    import json
    data = json.loads(output.getvalue())
    assert len(data) == 2

def test_extensions_mode_exception_logging(tmp_path):
    """Cover multitool.py lines 3872-3873."""
    f = tmp_path / "test.txt"
    f.write_text("hello")

    # Need to patch os.path.getsize within multitool
    with patch('os.path.isfile', return_value=True), \
         patch('os.path.getsize', side_effect=Exception("size error")):
        with patch('logging.warning') as mock_warn:
            multitool.extensions_mode(
                input_files=[str(f)],
                output_file='-',
                quiet=False
            )
            mock_warn.assert_any_call("Failed to get info for '%s': size error" % str(f))

def test_extensions_mode_empty_arrow(capsys):
    """Cover multitool.py line 3894."""
    multitool.extensions_mode(
        input_files=[],
        output_file='-',
        output_format='arrow'
    )
    captured = capsys.readouterr()
    assert captured.out == ""

def test_extensions_mode_skip_non_file(tmp_path):
    """Cover multitool.py line 3860."""
    dir_path = tmp_path / "subdir"
    dir_path.mkdir()

    multitool.extensions_mode(
        input_files=[str(dir_path)],
        output_file='-',
        output_format='arrow'
    )
    # Just ensuring it skips the directory and doesn't crash

def test_help_flag_truncation_robust(capsys):
    """Cover multitool.py line 8174."""
    details = {
        'summary': 'S', 'description': 'D', 'example': 'E',
        'flags': 'x' * 100
    }
    with patch.dict(multitool.MODE_DETAILS, {'arrow': details}):
        summary = multitool.get_mode_summary_text()
        assert "..." in summary
