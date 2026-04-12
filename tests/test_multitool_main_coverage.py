
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_main_no_args(capsys):
    """Cover lines 5660-5661: main() with no arguments prints summary and exits."""
    with patch("sys.argv", ["multitool.py"]):
        with pytest.raises(SystemExit) as excinfo:
            multitool.main()
        assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "Available Modes:" in captured.out

def test_normalize_mode_args_legacy(capsys):
    """Cover lines 5651-5655: legacy --mode flag normalization."""
    # Use _normalize_mode_args directly for cleaner coverage
    parser = multitool._build_parser()

    # Case 1: --mode arrow dummy.txt -> arrow dummy.txt (positional_mode is None)
    argv = ["--mode", "arrow", "dummy.txt"]
    normalized = multitool._normalize_mode_args(argv, parser)
    assert normalized == ["arrow", "dummy.txt"]

    # Case 2: arrow --mode arrow dummy.txt -> arrow dummy.txt (positional_mode is 'arrow')
    argv = ["arrow", "--mode", "arrow", "dummy.txt"]
    normalized = multitool._normalize_mode_args(argv, parser)
    assert normalized == ["arrow", "dummy.txt"]

def test_normalize_mode_args_conflict(capsys):
    """Cover lines 5647-5649: --mode flag conflict with positional mode."""
    parser = multitool._build_parser()
    argv = ["arrow", "--mode", "csv", "dummy.txt"]
    with pytest.raises(SystemExit):
        # parser.error calls sys.exit
        multitool._normalize_mode_args(argv, parser)

    captured = capsys.readouterr()
    assert "conflicts with positional mode" in captured.err

def test_normalize_mode_args_multiple(capsys):
    """Cover line 5637: multiple --mode flags error."""
    parser = multitool._build_parser()
    argv = ["--mode", "arrow", "--mode", "csv"]
    with pytest.raises(SystemExit):
        multitool._normalize_mode_args(argv, parser)

    captured = capsys.readouterr()
    assert "Only one --mode flag may be provided" in captured.err

def test_normalize_mode_args_missing_value(capsys):
    """Cover line 5640: --mode flag without value."""
    parser = multitool._build_parser()
    argv = ["--mode"]
    with pytest.raises(SystemExit):
        multitool._normalize_mode_args(argv, parser)

    captured = capsys.readouterr()
    assert "--mode requires a value" in captured.err
