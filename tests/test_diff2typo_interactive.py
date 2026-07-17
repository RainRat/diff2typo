import sys
import logging
import pytest

import diff2typo


def test_is_git_repository_true(monkeypatch):
    """Test that _is_git_repository returns True when git rev-parse is successful."""
    class MockResult:
        returncode = 0

    def mock_run(*args, **kwargs):
        return MockResult()

    monkeypatch.setattr(diff2typo.subprocess, "run", mock_run)
    assert diff2typo._is_git_repository() is True


def test_is_git_repository_false(monkeypatch):
    """Test that _is_git_repository returns False when git rev-parse fails."""
    class MockResult:
        returncode = 128

    def mock_run(*args, **kwargs):
        return MockResult()

    monkeypatch.setattr(diff2typo.subprocess, "run", mock_run)
    assert diff2typo._is_git_repository() is False


def test_is_git_repository_file_not_found(monkeypatch):
    """Test that _is_git_repository returns False when git executable is missing."""
    def mock_run(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(diff2typo.subprocess, "run", mock_run)
    assert diff2typo._is_git_repository() is False


def test_main_interactive_stdin_no_git_repo(monkeypatch, caplog):
    """Test that diff2typo exits gracefully with a clear error when run in a terminal outside a Git repository."""
    monkeypatch.setattr(sys, "argv", ["diff2typo.py", "--quiet"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(diff2typo, "_is_git_repository", lambda: False)

    # We mock out read_words_mapping and read_allowed_words just in case
    monkeypatch.setattr(diff2typo, "read_words_mapping", lambda *args, **kwargs: {})
    monkeypatch.setattr(diff2typo, "read_allowed_words", lambda *args, **kwargs: set())

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            diff2typo.main()

    assert excinfo.value.code == 1
    assert any(
        "no Git repository was detected" in message
        for message in caplog.messages
    )


def test_main_interactive_stdin_with_git_repo(monkeypatch, tmp_path):
    """Test that diff2typo automatically triggers _read_git_diff when run in an interactive terminal inside a Git repository."""
    output_file = tmp_path / "output.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diff2typo.py",
            "--quiet",
            "--output",
            str(output_file),
        ],
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(diff2typo, "_is_git_repository", lambda: True)

    git_diff_called = []
    def mock_read_git_diff(args_str):
        git_diff_called.append(args_str)
        return "--- a/file\n+++ b/file\n@@\n-teh\n+the\n"

    monkeypatch.setattr(diff2typo, "_read_git_diff", mock_read_git_diff)
    monkeypatch.setattr(diff2typo, "read_words_mapping", lambda *args, **kwargs: {})
    monkeypatch.setattr(diff2typo, "read_allowed_words", lambda *args, **kwargs: set())

    try:
        diff2typo.main()
    except SystemExit:
        pass

    assert len(git_diff_called) == 1
    assert git_diff_called[0] == ""
    # Should have analyzed and found teh -> the
    assert output_file.read_text().strip().splitlines() == ["teh -> the"]
