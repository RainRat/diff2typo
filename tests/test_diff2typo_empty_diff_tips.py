import sys
import logging
from unittest.mock import MagicMock, patch
import pytest
import diff2typo

def test_empty_diff_with_git_flag(caplog):
    """Test that empty diff with explicit -g/--git logs unstaged/staged tips."""
    args = MagicMock()
    args.git = ""
    args.git_log = None
    args.input_files = []
    args.input_files_flag = []
    args.dictionary_file = "words.csv"
    args.allowed_file = "allowed.csv"
    args.output_file = "-"
    args.output_format = "arrow"
    args.quiet = False
    args.mode = "typos"
    args.min_length = 2
    args.max_dist = None
    args.min_count = 1
    args.limit = None
    args.sort = "alpha"

    with patch("argparse.ArgumentParser.parse_args", return_value=args), \
         patch("sys.stdin.isatty", return_value=True), \
         patch("diff2typo._read_git_diff", return_value=""), \
         patch("diff2typo.smart_open_output"), \
         caplog.at_level(logging.INFO):

        diff2typo.main()

    log_texts = [record.message for record in caplog.records]
    assert any("The input diff is empty (no changes detected)." in msg for msg in log_texts)
    assert any("staged changes with '-g --cached', or previous commits with '-l HEAD~5'" in msg for msg in log_texts)

def test_empty_diff_with_git_log_flag(caplog):
    """Test that empty diff with explicit -l/--git-log logs commit range tips."""
    args = MagicMock()
    args.git = None
    args.git_log = "HEAD~5"
    args.input_files = []
    args.input_files_flag = []
    args.dictionary_file = "words.csv"
    args.allowed_file = "allowed.csv"
    args.output_file = "-"
    args.output_format = "arrow"
    args.quiet = False
    args.mode = "typos"
    args.min_length = 2
    args.max_dist = None
    args.min_count = 1
    args.limit = None
    args.sort = "alpha"

    with patch("argparse.ArgumentParser.parse_args", return_value=args), \
         patch("sys.stdin.isatty", return_value=True), \
         patch("diff2typo._read_git_log", return_value=""), \
         patch("diff2typo.smart_open_output"), \
         caplog.at_level(logging.INFO):

        diff2typo.main()

    log_texts = [record.message for record in caplog.records]
    assert any("The input diff is empty (no changes detected)." in msg for msg in log_texts)
    assert any("Try checking a different commit range, e.g. '-l HEAD~5'" in msg for msg in log_texts)

def test_empty_diff_with_direct_files(caplog):
    """Test that empty diff with files logs valid diff tips."""
    args = MagicMock()
    args.git = None
    args.git_log = None
    args.input_files = ["some_empty_file.diff"]
    args.input_files_flag = []
    args.dictionary_file = "words.csv"
    args.allowed_file = "allowed.csv"
    args.output_file = "-"
    args.output_format = "arrow"
    args.quiet = False
    args.mode = "typos"
    args.min_length = 2
    args.max_dist = None
    args.min_count = 1
    args.limit = None
    args.sort = "alpha"

    with patch("argparse.ArgumentParser.parse_args", return_value=args), \
         patch("sys.stdin.isatty", return_value=True), \
         patch("diff2typo._read_diff_sources", return_value=""), \
         patch("diff2typo.smart_open_output"), \
         caplog.at_level(logging.INFO):

        diff2typo.main()

    log_texts = [record.message for record in caplog.records]
    assert any("The input diff is empty (no changes detected)." in msg for msg in log_texts)
    assert any("Ensure the specified input files or piped input contain a valid Git diff or patch" in msg for msg in log_texts)
