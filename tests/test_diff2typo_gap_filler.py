import subprocess
import sys
import runpy
from unittest.mock import MagicMock, patch
import pytest
import diff2typo

def test_read_git_diff_success():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="mocked git diff", returncode=0)
        result = diff2typo._read_git_diff("--stat")
        assert result == "mocked git diff"
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert "git" in args[0]
        assert "diff" in args[0]
        assert "--stat" in args[0]

def test_read_git_diff_error(caplog):
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, ["git", "diff"], stderr="git error")
        with pytest.raises(SystemExit) as excinfo:
            diff2typo._read_git_diff(None)
        assert excinfo.value.code == 1
        assert "Git command failed: git error" in caplog.text

def test_read_git_diff_not_found(caplog):
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(SystemExit) as excinfo:
            diff2typo._read_git_diff(None)
        assert excinfo.value.code == 1
        assert "Git executable not found." in caplog.text

def test_compare_word_lists_identical_words_in_replace():
    with patch("difflib.SequenceMatcher.get_opcodes") as mock_opcodes:
        mock_opcodes.return_value = [('replace', 0, 2, 0, 2)]
        before = ["identical", "teh"]
        after = ["identical", "the"]
        result = diff2typo._compare_word_lists(before, after, min_length=2)
        assert "teh -> the" in result
        assert len(result) == 1

def test_main_git_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("diff2typo._read_git_diff") as mock_git_diff:
        mock_git_diff.return_value = "--- a/file\n+++ b/file\n-teh\n+the\n"

        monkeypatch.setattr(sys, "argv", ["diff2typo.py", "--git", "HEAD", "--quiet", "--output", "out.txt"])

        (tmp_path / "words.csv").write_text("word,word")
        (tmp_path / "allowed.csv").write_text("word")

        diff2typo.main()

        mock_git_diff.assert_called_once_with("HEAD")
        assert (tmp_path / "out.txt").exists()
        assert "teh -> the" in (tmp_path / "out.txt").read_text()

def test_diff2typo_main_invocation():
    with patch("sys.argv", ["diff2typo.py", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("diff2typo", run_name="__main__")
        assert excinfo.value.code == 0
