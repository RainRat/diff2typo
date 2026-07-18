import sys
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo


def test_read_diff_sources_non_interactive(monkeypatch):
    """
    Test when no input files are specified but stdin is not interactive (e.g. piped input).
    It should read from stdin instead of checking Git or exiting.
    """
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(diff2typo, "_read_stdin_text", lambda: "piped stdin diff")

    result = diff2typo._read_diff_sources([])
    assert result == "piped stdin diff"


def test_read_diff_sources_interactive_inside_git(monkeypatch):
    """
    Test when no input files are specified, stdin is interactive, and we are inside a Git worktree.
    It should run `git diff` automatically.
    """
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def mock_run(args, **kwargs):
        if args == ["git", "rev-parse", "--is-inside-work-tree"]:
            return MagicMock(returncode=0, stdout="true\n")
        return MagicMock(returncode=1)

    monkeypatch.setattr(diff2typo.subprocess, "run", mock_run)
    monkeypatch.setattr(diff2typo, "_read_git_diff", lambda args: "git diff output")

    result = diff2typo._read_diff_sources([])
    assert result == "git diff output"


def test_read_diff_sources_interactive_outside_git(monkeypatch):
    """
    Test when no input files are specified, stdin is interactive, but we are NOT inside a Git worktree.
    It should print a helpful error message and exit with 1.
    """
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def mock_run(args, **kwargs):
        if args == ["git", "rev-parse", "--is-inside-work-tree"]:
            return MagicMock(returncode=0, stdout="false\n")
        return MagicMock(returncode=1)

    monkeypatch.setattr(diff2typo.subprocess, "run", mock_run)

    with pytest.raises(SystemExit) as excinfo:
        diff2typo._read_diff_sources([])
    assert excinfo.value.code == 1


def test_read_diff_sources_interactive_git_filenotfound(monkeypatch):
    """
    Test when no input files are specified, stdin is interactive, but git command itself is not found/available.
    It should print a helpful error message and exit with 1.
    """
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def mock_run(args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(diff2typo.subprocess, "run", mock_run)

    with pytest.raises(SystemExit) as excinfo:
        diff2typo._read_diff_sources([])
    assert excinfo.value.code == 1
