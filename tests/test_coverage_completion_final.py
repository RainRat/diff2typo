import os
import sys
import shutil
import logging
import io
import re
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path

# Ensure the repository root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import multitool
import diff2typo

def test_resolve_full_mapping_complex_text(tmp_path):
    """Cover multitool.py lines 3309, 3315-3316: comments, empty lines, and : separator."""
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("# Comment\n\nword1: correction1\nword2 -> correction2\nword3")

    mapping = multitool._resolve_full_mapping(str(mapping_file), None, False)
    assert mapping == {
        "word1": "correction1",
        "word2": "correction2",
        "word3": ""
    }

def test_resolve_full_mapping_adhoc_no_colon():
    """Cover multitool.py line 3328: ad-hoc pairs without colon."""
    mapping = multitool._resolve_full_mapping(None, ad_hoc_pairs=["word1", "word2:corr"], clean_items=False)
    assert mapping == {
        "word1": "",
        "word2": "corr"
    }

def test_scrub_mode_no_changes(tmp_path, caplog):
    """Cover multitool.py line 3680: no changes needed in scrub mode with --in-place."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("no typos here")

    with caplog.at_level(logging.INFO):
        multitool.scrub_mode(
            input_files=[str(input_file)],
            mapping_file=None,
            ad_hoc=["typo:correction"],
            output_file='-',
            min_length=1,
            max_length=100,
            process_output=False,
            in_place=".bak", # Non-None for the branch
            quiet=False, # Must be False for logging
            clean_items=False
        )
        assert f"No changes needed for '{input_file}'." in caplog.text

def test_scrub_mode_backup_fail(tmp_path, caplog):
    """Cover multitool.py lines 3661-3663: backup failure in scrub mode."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("typo")

    # Mock _scrub_line to return a replacement
    with patch("multitool._scrub_line", return_value=("correction", 1)):
        with patch("shutil.copy2", side_effect=Exception("backup error")):
            with caplog.at_level(logging.ERROR):
                with pytest.raises(SystemExit) as excinfo:
                    multitool.scrub_mode(
                        input_files=[str(input_file)],
                        mapping_file=None,
                        ad_hoc=["typo:correction"],
                        output_file='-',
                        min_length=1,
                        max_length=100,
                        process_output=False,
                        in_place=".bak", # Non-empty string triggers backup
                        quiet=True,
                        clean_items=False
                    )
                assert excinfo.value.code == 1
                assert "Failed to create backup" in caplog.text

def test_standardize_mode_no_changes(tmp_path, caplog):
    """Cover multitool.py line 3622: no changes needed in standardize mode with --in-place."""
    input_file = tmp_path / "input.txt"

    # We need to find inconsistencies overall so mapping is not empty,
    # but one file should already be standardized.
    file1 = tmp_path / "file1.txt"
    file1.write_text("Apple Apple Apple apple") # apple (1) vs Apple (3) -> Apple wins

    file2 = tmp_path / "file2.txt"
    file2.write_text("Apple Apple") # This file already uses the winner

    with caplog.at_level(logging.INFO):
        multitool.standardize_mode(
            input_files=[str(file1), str(file2)],
            output_file='-',
            min_length=1,
            max_length=100,
            process_output=False,
            in_place=".bak",
            quiet=False
        )
        # file2 should trigger the "No changes needed" log
        assert f"No changes needed for '{file2}'." in caplog.text

def test_standardize_mode_backup_fail(tmp_path, caplog):
    """Cover multitool.py lines 3560-3562: backup failure in standardize mode."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple Apple")

    with patch("shutil.copy2", side_effect=Exception("backup error")):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit) as excinfo:
                multitool.standardize_mode(
                    input_files=[str(input_file)],
                    output_file='-',
                    min_length=1,
                    max_length=100,
                    process_output=False,
                    in_place=".bak",
                    quiet=True
                )
            assert excinfo.value.code == 1
            assert "Failed to create backup" in caplog.text

def test_rename_mode_non_existent_path(caplog):
    """Cover multitool.py line 3741: non-existent path in rename mode."""
    with caplog.at_level(logging.INFO):
        multitool.rename_mode(
            input_files=["non_existent_path"],
            mapping_file=None,
            ad_hoc=["a:b"],
            output_file='-',
            min_length=1,
            max_length=100,
            process_output=False,
            quiet=True
        )
        # Should just continue, no error logged specifically for skipping
        assert "non_existent_path" not in caplog.text

def test_rename_mode_fail(tmp_path, caplog):
    """Cover multitool.py lines 3759-3761: rename failure."""
    file_path = tmp_path / "typo_file"
    file_path.write_text("content")

    with patch("os.rename", side_effect=Exception("rename error")):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit) as excinfo:
                multitool.rename_mode(
                    input_files=[str(file_path)],
                    mapping_file=None,
                    ad_hoc=["typo:correction"],
                    output_file='-',
                    min_length=1,
                    max_length=100,
                    process_output=False,
                    in_place=True,
                    quiet=True,
                    clean_items=False
                )
            assert excinfo.value.code == 1
            assert "Failed to rename" in caplog.text

def test_rename_mode_dry_run_logging(tmp_path, caplog):
    """Cover multitool.py line 3767: dry run total renames logging."""
    file_path = tmp_path / "typo_file"
    file_path.write_text("content")

    with caplog.at_level(logging.WARNING):
        multitool.rename_mode(
            input_files=[str(file_path)],
            mapping_file=None,
            ad_hoc=["typo:correction"],
            output_file='-',
            min_length=1,
            max_length=100,
            process_output=False,
            in_place=True,
            dry_run=True,
            quiet=True,
            clean_items=False
        )
        assert "Total renames that would be made: 1" in caplog.text

def test_mode_help_action_usage_formatting(capsys):
    """Cover multitool.py line 4628: USAGE formatting for modes with positional labels."""
    # We trigger the ModeHelpAction by calling the parser
    with patch("sys.argv", ["multitool.py", "--mode-help", "search"]):
        with pytest.raises(SystemExit):
            multitool.main()

    captured = capsys.readouterr()
    output = captured.err + captured.out
    # Strip ANSI codes for comparison
    clean_output = re.sub(r'\x1b\[[0-9;]*m', '', output)
    assert "USAGE:       python multitool.py search QUERY [FILES...] [FLAGS]" in clean_output

def test_main_search_fallback(tmp_path):
    """Cover multitool.py lines 5673-5674: main() search mode fallbacks."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("query_string")

    # 2+ input paths: first is query, rest are files
    with patch("sys.argv", ["multitool.py", "search", "query_string", str(input_file)]):
        with patch("multitool.search_mode") as mock_search:
            multitool.main()
            mock_search.assert_called_once()
            args = mock_search.call_args[1]
            assert args['query'] == "query_string"
            assert args['input_files'] == [str(input_file)]

def test_main_search_missing_query(caplog):
    """Cover multitool.py lines 5690-5691: search mode missing query error."""
    with patch("sys.argv", ["multitool.py", "search"]):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit) as excinfo:
                multitool.main()
            assert excinfo.value.code == 1
            assert "Search mode requires a search query" in caplog.text

def test_diff2typo_compare_identical_words():
    """Cover diff2typo.py line 218: skipping identical words in aligned replacement."""
    # Use a mock to force a 'replace' block containing identical words
    with patch("difflib.SequenceMatcher.get_opcodes") as mock_opcodes:
        mock_opcodes.return_value = [('replace', 0, 2, 0, 2)]
        result = diff2typo._compare_word_lists(["same", "typo"], ["same", "fix"], min_length=1)
        assert "typo -> fix" in result
        assert len(result) == 1

def test_resolve_full_mapping_non_existent_file(caplog):
    """Cover multitool.py line 265: _read_file_lines_robust error handling."""
    with caplog.at_level(logging.ERROR):
        # We need to call something that uses _read_file_lines_robust via _resolve_full_mapping
        # but with a file that doesn't exist.
        # Actually _read_file_lines_robust is called directly by _resolve_full_mapping
        with pytest.raises(SystemExit) as excinfo:
            multitool._resolve_full_mapping("non_existent_file.txt", None, False)
        assert excinfo.value.code == 1
        assert "Input file 'non_existent_file.txt' not found." in caplog.text
