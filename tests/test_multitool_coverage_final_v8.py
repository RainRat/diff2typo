import os
import sys
import logging
import io
import json
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

def test_orphans_mode_coverage_gaps(tmp_path, capsys):
    """Cover multitool.py lines 2760, 2787, 2789, 2827, 2831-2863."""
    md_file = tmp_path / "test.md"
    # To hit 2787 and 2789 we need _extract_markdown_links_detailed to yield ref: and broken-ref:
    # This happens when reference-style links are used.
    md_file.write_text("[text][label1]\n[text][label2]\n\n[label1]: http://ex.com")

    # We also need an unreferenced file for the Item/Reason table
    other_file = tmp_path / "other.txt"
    other_file.write_text("content")

    # We need to run it with arrow format and limit
    with patch('sys.stdout.isatty', return_value=True), \
         patch('multitool._should_enable_color', return_value=True):
        multitool.orphans_mode(
            input_files=[str(md_file), str(other_file), "-"], # "-" hits line 2760
            output_file='-',
            output_format='arrow',
            limit=2 # line 2827
        )

    captured = capsys.readouterr()
    assert "ORPHANS ANALYSIS" in captured.out
    assert "Item" in captured.out
    assert "Reason" in captured.out

def test_scan_mode_coverage_gaps(tmp_path, capsys):
    """Cover multitool.py lines 6756, 6824-6837."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple banana\ncherry")

    # multi-pattern match on same line (line 6756)
    with patch('sys.stdout.isatty', return_value=True), \
         patch('multitool._should_enable_color', return_value=True):
        multitool.scan_mode(
            input_files=[str(input_file)],
            mapping_file=None,
            ad_hoc=["apple", "banana"], # Correct arg name
            output_file='-',
            output_format='arrow',
            min_length=1,
            max_length=100,
            process_output=False
        )

    captured = capsys.readouterr()
    assert "SCAN ANALYSIS SUMMARY" in captured.out
    assert "Matches found" in captured.out

def test_fileinfo_mode_coverage_gaps(tmp_path, capsys):
    """Cover multitool.py lines 3536, 3538, 3554-3555, 3563, 3611."""
    dir_path = tmp_path / "subdir"
    dir_path.mkdir()
    f1 = tmp_path / "test.txt"
    f1.write_text("hello world")

    # 1. Stdin (line 3536) and directory (line 3538)
    multitool.fileinfo_mode(
        input_files=["-", str(dir_path), str(f1)],
        output_file='-',
        output_format='arrow',
        quiet=False # line 3611 logging
    )

    # 2. Empty results (line 3563)
    output = io.StringIO()
    with patch('multitool.smart_open_output', return_value=MockSmartOpen(output)):
        multitool.fileinfo_mode(
            input_files=[],
            output_file='-',
            output_format='arrow'
        )
    assert output.getvalue() == ""

    # 3. Exception handling (lines 3554-3555)
    f_fail = tmp_path / "fail.txt"
    f_fail.write_text("content")
    with patch('os.path.getsize', side_effect=Exception("mock error")):
        with patch('logging.warning') as mock_warn:
            multitool.fileinfo_mode(
                input_files=[str(f_fail)],
                output_file='-',
                output_format='arrow'
            )
            found = False
            for call in mock_warn.call_args_list:
                if "mock error" in str(call[0][0]):
                    found = True
            assert found

def test_sentences_mode_coverage_gaps(tmp_path):
    """Cover multitool.py lines 1673-1682, 2121, 3194, 3335, 3361, 3441, 9530, 9534."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("This is sentence one. This is sentence two. Short.")

    # main() sentences mode
    with patch('sys.argv', ['multitool.py', 'sentences', str(input_file)]):
        multitool.main()

    # main() count sentences (line format to hit 3441)
    with patch('sys.argv', ['multitool.py', 'count', str(input_file), '--sentences', '--raw']):
        multitool.main()

    # main() count sentences (arrow format to hit 3335 and 3361)
    with patch('sys.argv', ['multitool.py', 'count', str(input_file), '--sentences']), \
         patch('sys.stdout.isatty', return_value=True):
        multitool.main()

    # Direct call to _extract_sentence_items (lines 1673-1682)
    sentences = list(multitool._extract_sentence_items(str(input_file)))
    assert "This is sentence one." in sentences

def test_format_search_line_variations(capsys):
    """Cover multitool.py lines 4905-4926, including 4915."""
    line = "some content"
    # hit 4915 by having no parts (show_filename=False, line_numbers=False)
    result_no_parts = multitool._format_search_line(
        filename="file.txt",
        line_idx=0,
        line_content=line,
        is_match=True,
        show_filename=False,
        line_numbers=False,
        use_color=True
    )
    assert result_no_parts == line

    # hit color parts (4905-4918)
    result_color = multitool._format_search_line(
        filename="file.txt",
        line_idx=0,
        line_content=line,
        is_match=True,
        show_filename=True,
        line_numbers=True,
        use_color=True
    )
    assert "file.txt" in result_color

    # hit no color (4920-4926)
    result_no_color = multitool._format_search_line(
        filename="file.txt",
        line_idx=1,
        line_content=line,
        is_match=False,
        show_filename=True,
        line_numbers=True,
        use_color=False
    )
    assert "file.txt-2-" in result_no_color.replace(" ", "")

def test_anomalies_variations(tmp_path):
    """Cover multitool.py lines 4563, 4567, 4569, 4571, 4573."""
    input_file = tmp_path / "input.txt"
    # 'a' and 'be' hit 4563 skip.
    input_file.write_text("a be HEllow gIT w0rd pyTHon")

    anomalies = list(multitool._extract_anomalies(str(input_file)))
    types = [a[1] for a in anomalies]
    assert "[Shift]" in types
    assert "[Caps]" in types
    assert "[Num]" in types
    assert "[Bumpy]" in types
