import sys
import io
from pathlib import Path
import pytest
import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_stdin_default_behavior(monkeypatch, capsys):
    """Test that multitool defaults to reading from stdin when no input file is provided."""

    # Prepare input data
    input_data = "typo,correction\n"
    stdin_file = io.StringIO(input_data)

    # Mock sys.stdin
    monkeypatch.setattr(sys, 'stdin', stdin_file)

    # Mock sys.argv to run 'csv' mode without input arguments
    # We add --quiet to avoid log output clutter, although capsys captures it.
    monkeypatch.setattr(sys, 'argv', ['multitool.py', '--quiet', 'csv'])

    # Run main
    multitool.main()

    # Check output
    captured = capsys.readouterr()
    output = captured.out

    # With 'csv' mode and default settings (not first column only),
    # it should extract the second column "correction"
    assert "correction" in output
    assert "typo" not in output

def test_explicit_stdin_flag(monkeypatch, capsys):
    """Test that providing '-' explicitly still works."""

    input_data = "typo,correction\n"
    stdin_file = io.StringIO(input_data)
    monkeypatch.setattr(sys, 'stdin', stdin_file)

    # Explicitly pass '-' as input file
    monkeypatch.setattr(sys, 'argv', ['multitool.py', '--quiet', 'csv', '-'])

    multitool.main()

    captured = capsys.readouterr()
    output = captured.out

    assert "correction" in output
