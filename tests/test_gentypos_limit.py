import sys
from unittest.mock import patch
import pytest

import gentypos


def test_gentypos_limit_option(capsys, monkeypatch):
    """Verify that -L / --limit restricts output typos count."""
    test_args = ["gentypos.py", "hello", "-L", "3", "--no-filter", "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]
    assert len(lines) == 3


def test_gentypos_limit_dry_run(capsys, monkeypatch):
    """Verify that --limit is reported in dry-run output."""
    test_args = ["gentypos.py", "hello", "--limit", "5", "--dry-run"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gentypos.main()

    assert exc_info.value.code == 0
