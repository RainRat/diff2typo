"""
Unit tests for the -L / --limit CLI option in gentypos.py.
"""

import sys
import pytest
import gentypos


def test_gentypos_limit_cli(monkeypatch, capsys):
    """Test that -L / --limit restricts the number of outputted typos in CLI mode."""
    test_args = ['gentypos.py', 'hello', 'world', '-L', '2', '--no-filter', '-q']
    monkeypatch.setattr(sys, 'argv', test_args)

    gentypos.main()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split('\n') if line]
    assert len(lines) == 2


def test_gentypos_limit_config_and_dry_run(monkeypatch, capsys):
    """Test -L / --limit in dry-run mode."""
    test_args = ['gentypos.py', 'hello', '-L', '5', '--dry-run']
    monkeypatch.setattr(sys, 'argv', test_args)

    with pytest.raises(SystemExit) as exc_info:
        gentypos.main()

    assert exc_info.value.code == 0
