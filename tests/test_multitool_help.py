import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_mode_help_all_implicit(monkeypatch, capsys):
    """Test 'multitool.py --mode-help' displays help for all modes."""
    monkeypatch.setattr(sys, 'argv', ['multitool.py', '--mode-help'])

    # Expect SystemExit because parser.exit() is called
    with pytest.raises(SystemExit):
        multitool.main()

    captured = capsys.readouterr()
    output = captured.err + captured.out

    # Check for presence of summary table header and content
    assert "Available Modes:" in output
    assert "arrow" in output
    assert "csv" in output
    # In table view, we print "Summary: ..." as just the text column
    assert "Extract the left side of '->' arrows." in output

def test_mode_help_all_explicit(monkeypatch, capsys):
    """Test 'multitool.py --mode-help all' displays help for all modes."""
    monkeypatch.setattr(sys, 'argv', ['multitool.py', '--mode-help', 'all'])

    with pytest.raises(SystemExit):
        multitool.main()

    captured = capsys.readouterr()
    output = captured.err + captured.out

    assert "Available Modes:" in output
    assert "arrow" in output
    assert "csv" in output

def test_mode_help_specific(monkeypatch, capsys):
    """Test 'multitool.py --mode-help arrow' displays help only for arrow mode."""
    monkeypatch.setattr(sys, 'argv', ['multitool.py', '--mode-help', 'arrow'])

    with pytest.raises(SystemExit):
        multitool.main()

    captured = capsys.readouterr()
    output = captured.err + captured.out

    assert "Mode: arrow" in output
    assert "Summary: Extract the left side of '->' arrows." in output

    # Should not contain other modes
    assert "Mode: csv" not in output

def test_mode_help_invalid(monkeypatch, capsys):
    """Test 'multitool.py --mode-help invalid' raises error."""
    monkeypatch.setattr(sys, 'argv', ['multitool.py', '--mode-help', 'invalid_mode'])

    with pytest.raises(SystemExit) as excinfo:
        multitool.main()

    assert excinfo.value.code != 0

    captured = capsys.readouterr()
    output = captured.err + captured.out

    assert "invalid choice" in output or "argument --mode-help: invalid choice" in output
