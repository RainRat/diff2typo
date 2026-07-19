import sys
import subprocess
from unittest.mock import MagicMock, patch
import pytest
import diff2typo

def test_main_with_interactive_stdin_inside_git_worktree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    args = MagicMock()
    args.git = None
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

    git_rev_parse_mock = MagicMock(returncode=0, stdout="true\n")

    with patch("argparse.ArgumentParser.parse_args", return_value=args), \
         patch("sys.stdin.isatty", return_value=True), \
         patch("subprocess.run", return_value=git_rev_parse_mock), \
         patch("diff2typo._read_git_diff", return_value="some_diff_text") as mock_read_git_diff, \
         patch("diff2typo.find_typos", return_value=[]), \
         patch("diff2typo.read_words_mapping", return_value={}), \
         patch("diff2typo.read_allowed_words", return_value=set()), \
         patch("diff2typo.smart_open_output"):

        diff2typo.main()
        mock_read_git_diff.assert_called_once_with(None)

def test_main_with_interactive_stdin_outside_git_worktree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    args = MagicMock()
    args.git = None
    args.input_files = []
    args.input_files_flag = []
    args.quiet = False

    git_rev_parse_mock = MagicMock(returncode=1, stdout="false\n")

    with patch("argparse.ArgumentParser.parse_args", return_value=args), \
         patch("sys.stdin.isatty", return_value=True), \
         patch("subprocess.run", return_value=git_rev_parse_mock):

        with pytest.raises(SystemExit) as exc_info:
            diff2typo.main()
        assert exc_info.value.code == 1

def test_main_with_interactive_stdin_git_command_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    args = MagicMock()
    args.git = None
    args.input_files = []
    args.input_files_flag = []
    args.quiet = False

    with patch("argparse.ArgumentParser.parse_args", return_value=args), \
         patch("sys.stdin.isatty", return_value=True), \
         patch("subprocess.run", side_effect=FileNotFoundError):

        with pytest.raises(SystemExit) as exc_info:
            diff2typo.main()
        assert exc_info.value.code == 1

def test_main_with_non_interactive_stdin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    args = MagicMock()
    args.git = None
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
         patch("sys.stdin.isatty", return_value=False), \
         patch("diff2typo._read_diff_sources", return_value="some_diff_text") as mock_read_diff_sources, \
         patch("diff2typo.find_typos", return_value=[]), \
         patch("diff2typo.read_words_mapping", return_value={}), \
         patch("diff2typo.read_allowed_words", return_value=set()), \
         patch("diff2typo.smart_open_output"):

        diff2typo.main()
        mock_read_diff_sources.assert_called_once_with([])
