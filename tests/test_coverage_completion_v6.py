import os
import sys
import pytest
import logging
import json
import xml.etree.ElementTree as ET
from unittest.mock import patch, MagicMock
import importlib

# Add current directory to path so we can import the scripts
sys.path.append(os.getcwd())

import multitool
import cmdrunner

def test_write_structured_data_xml_sanitization(tmp_path):
    """Covers multitool.py line 794: XML root tag sanitization for invalid first char."""
    output_file = tmp_path / "test.xml"
    data = {"key": "value"}
    # Root tag starting with a digit is invalid in XML
    multitool._write_structured_data(data, str(output_file), "xml", root_tag="123root")

    content = output_file.read_text()
    assert "<_123root>" in content

    # Empty root tag
    multitool._write_structured_data(data, str(output_file), "xml", root_tag="")
    content = output_file.read_text()
    assert "<_>" in content

def test_convert_mode_limit(tmp_path):
    """Covers multitool.py line 1907: convert mode with limit."""
    # Use multiple files to ensure len(all_results) > 1
    f1 = tmp_path / "f1.json"
    f1.write_text('{"a": 1}')
    f2 = tmp_path / "f2.json"
    f2.write_text('{"a": 2}')
    f3 = tmp_path / "f3.json"
    f3.write_text('{"a": 3}')

    output_file = tmp_path / "output.json"
    multitool.convert_mode([str(f1), str(f2), str(f3)], str(output_file), limit=2, quiet=True)

    with open(output_file) as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 2
    assert data == [{"a": 1}, {"a": 2}]

def test_format_search_line_no_parts():
    """Covers multitool.py line 3842: _format_search_line when no prefixes are selected."""
    content = "test line"
    # Use color=True but both show_filename and line_numbers False
    result = multitool._format_search_line("file.txt", 0, content, True, False, False, True)
    assert result == content

def test_help_mode_truncation():
    """Covers multitool.py lines 6312 and 6317: help summary and flags truncation."""
    long_summary = "A" * 100
    long_flags = "B" * 100

    # We need to temporarily modify MODE_DETAILS to include a very long entry
    original_details = multitool.MODE_DETAILS
    # Use a category that exists in get_mode_summary_text
    test_details = {
        "arrow": {
            "summary": long_summary,
            "flags": long_flags,
            "description": "test",
            "example": "test"
        }
    }

    try:
        multitool.MODE_DETAILS = test_details
        help_text = multitool.get_mode_summary_text()

        # Check for ellipsis in the output
        assert "..." in help_text
        # Summary width is 33. "A"*100 should be truncated.
        # Flags width is 55. "B"*100 should be truncated.
        assert "A" * 30 + "..." in help_text
        assert "B" * 52 + "..." in help_text

    finally:
        multitool.MODE_DETAILS = original_details

def test_main_replace_positional(tmp_path, monkeypatch, caplog):
    """Covers multitool.py lines 8067-8077, 8089-8090: replace mode positional args and error handling."""
    f = tmp_path / "test.txt"
    f.write_text("hello world")

    # Scenario: 3+ positional args (OLD NEW FILE)
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'replace', 'hello', 'hi', str(f), '-I'])
    with patch('sys.exit'):
        multitool.main()
    assert f.read_text().strip() == "hi world"

    # Scenario: 2 positional args (OLD NEW) reading from stdin
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'replace', 'apple', 'orange', '--quiet'])
    multitool._STDIN_CACHE = None
    mock_stdin = MagicMock()
    mock_stdin.buffer.read.return_value = b"apple pie"
    with patch('sys.stdin', mock_stdin):
        with patch('sys.stdout', new_callable=MagicMock) as mock_stdout:
            with patch('sys.exit'):
                multitool.main()
                combined_output = "".join(call.args[0] for call in mock_stdout.write.call_args_list)
                assert "orange pie" in combined_output

    # Scenario: Missing args (8089-8090)
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'replace', 'only_one'])
    with caplog.at_level(logging.ERROR):
        with patch('sys.exit') as mock_exit:
            multitool.main()
            mock_exit.assert_called_with(1)
            assert "Replace mode requires both OLD and NEW text" in caplog.text

def test_main_search_positional(tmp_path, monkeypatch, caplog):
    """Covers multitool.py lines 8060-8065, 8092-8093: search mode positional args and error handling."""
    f = tmp_path / "test.txt"
    f.write_text("find me\nignore me")

    # Scenario: 2+ positional args (QUERY FILE)
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'search', 'find', str(f), '--quiet'])
    with patch('sys.stdout', new_callable=MagicMock) as mock_stdout:
        with patch('sys.exit'):
            multitool.main()
            combined_output = "".join(call.args[0] for call in mock_stdout.write.call_args_list)
            assert "find me" in combined_output

    # Scenario: 1 positional arg (QUERY) reading from stdin
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'search', 'apple', '--quiet'])
    multitool._STDIN_CACHE = None
    mock_stdin = MagicMock()
    mock_stdin.buffer.read.return_value = b"apple pie\nbanana bread"
    with patch('sys.stdin', mock_stdin):
        with patch('sys.stdout', new_callable=MagicMock) as mock_stdout:
            with patch('sys.exit'):
                multitool.main()
                combined_output = "".join(call.args[0] for call in mock_stdout.write.call_args_list)
                assert "apple pie" in combined_output

    # Scenario: Missing query (8092-8093)
    # We need to bypass _normalize_mode_args or give it empty input
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'search'])
    with caplog.at_level(logging.ERROR):
        with patch('sys.exit') as mock_exit:
            multitool.main()
            mock_exit.assert_called_with(1)
            assert "Search mode requires a search query" in caplog.text

def test_cmdrunner_tqdm_fallback():
    """Covers cmdrunner.py lines 15-26: tqdm fallback when tqdm is not installed."""
    with patch.dict(sys.modules, {'tqdm': None}):
        importlib.reload(cmdrunner)
        assert cmdrunner.tqdm.__name__ == "tqdm"
        it = [1, 2, 3]
        pbar = cmdrunner.tqdm(it)
        assert list(pbar) == it
        pbar.update(1)
        pbar.set_description("test")
        pbar.set_postfix(a=1)
        pbar.close()
        with pbar:
            pass
    importlib.reload(cmdrunner)
