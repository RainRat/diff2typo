import sys
import subprocess
from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo


def test_read_git_log_success():
    mock_run_result = MagicMock(returncode=0, stdout="mock git log diff content", stderr="")
    with patch("subprocess.run", return_value=mock_run_result) as mock_run:
        result = diff2typo._read_git_log("HEAD~5 --oneline")
        assert result == "mock git log diff content"
        mock_run.assert_called_once_with(
            ["git", "log", "-p", "HEAD~5", "--oneline"],
            capture_output=True,
            text=True,
            check=True
        )


def test_read_git_log_success_no_args():
    mock_run_result = MagicMock(returncode=0, stdout="mock git log diff content empty args", stderr="")
    with patch("subprocess.run", return_value=mock_run_result) as mock_run:
        result = diff2typo._read_git_log(None)
        assert result == "mock git log diff content empty args"
        mock_run.assert_called_once_with(
            ["git", "log", "-p"],
            capture_output=True,
            text=True,
            check=True
        )


def test_read_git_log_called_process_error():
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["git"], stderr="error")):
        with pytest.raises(SystemExit) as exc_info:
            diff2typo._read_git_log("HEAD~5")
        assert exc_info.value.code == 1


def test_read_git_log_file_not_found_error():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(SystemExit) as exc_info:
            diff2typo._read_git_log("HEAD~5")
        assert exc_info.value.code == 1


def test_main_with_git_log_option(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    args = MagicMock()
    args.git = None
    args.git_log = "HEAD~3"
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
         patch("diff2typo._read_git_log", return_value="some_diff_text") as mock_read_git_log, \
         patch("diff2typo.find_typos", return_value=[]), \
         patch("diff2typo.read_words_mapping", return_value={}), \
         patch("diff2typo.read_allowed_words", return_value=set()), \
         patch("diff2typo.smart_open_output"):

        diff2typo.main()
        mock_read_git_log.assert_called_once_with("HEAD~3")


def test_cli_parser_accepts_git_log():
    # Verify that the parser correctly maps -l and --git-log
    test_args = ["-l", "HEAD~2"]
    with patch("sys.argv", ["diff2typo.py"] + test_args):
        # We need to temporarily suppress sys.exit during parse_args if anything went wrong
        parser = diff2typo.main.__globals__.get('argparse').ArgumentParser()
        # Since main doesn't return the parsed args, we can instantiate the parser manually to test parse_args
        # But wait, we can just test main's parsing via a mock call.
        # Let's mock a simple run to ensure parsing doesn't crash on these flags.
        with patch("diff2typo._read_git_log", return_value="some_diff_text"), \
             patch("diff2typo.find_typos", return_value=[]), \
             patch("diff2typo.read_words_mapping", return_value={}), \
             patch("diff2typo.read_allowed_words", return_value=set()), \
             patch("diff2typo.smart_open_output"):
            diff2typo.main()
