import sys
import subprocess
from pathlib import Path
from unittest import mock
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo


def test_read_git_log_success():
    mock_result = mock.Mock()
    mock_result.stdout = "commit 1234\nAuthor: Test\nDate: Test\n\n    Fix typo\n\ndiff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n-teh\n+the\n"

    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
        res = diff2typo._read_git_log("HEAD~5")
        mock_run.assert_called_once_with(
            ["git", "log", "-p", "HEAD~5"],
            capture_output=True,
            text=True,
            check=True
        )
        assert "teh" in res
        assert "the" in res


def test_read_git_log_no_args():
    mock_result = mock.Mock()
    mock_result.stdout = "git log output"

    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
        res = diff2typo._read_git_log(None)
        mock_run.assert_called_once_with(
            ["git", "log", "-p"],
            capture_output=True,
            text=True,
            check=True
        )
        assert res == "git log output"


def test_read_git_log_called_process_error():
    with mock.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "git", stderr="error message")):
        with pytest.raises(SystemExit) as exc_info:
            diff2typo._read_git_log("HEAD~5")
        assert exc_info.value.code == 1


def test_read_git_log_file_not_found_error():
    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(SystemExit) as exc_info:
            diff2typo._read_git_log("HEAD~5")
        assert exc_info.value.code == 1


def test_main_with_git_log_arg(tmp_path, monkeypatch):
    words_file = tmp_path / "words.csv"
    words_file.write_text("the\n")
    allowed_file = tmp_path / "allowed.csv"
    allowed_file.write_text("\n")
    output_file = tmp_path / "output.txt"

    test_args = [
        "diff2typo.py",
        "-l",
        "HEAD~5",
        "--output",
        str(output_file),
        "--dictionary",
        str(words_file),
        "--allowed",
        str(allowed_file),
        "--typos-path",
        "invalid_typos_path",  # to skip typos tool filtering
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    mock_git_log = "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n-teh\n+the\n"

    with mock.patch("diff2typo._read_git_log", return_value=mock_git_log) as mock_read:
        diff2typo.main()
        mock_read.assert_called_once_with("HEAD~5")

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "teh -> the" in content
