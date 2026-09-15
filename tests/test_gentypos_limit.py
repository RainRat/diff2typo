import sys
import logging
from unittest.mock import patch
import pytest
import gentypos


def test_gentypos_limit_flag(capsys):
    test_args = ["gentypos.py", "hello", "world", "-L", "3", "-N"]
    with patch.object(sys, "argv", test_args):
        gentypos.main()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line.strip()]
    assert len(lines) == 3


def test_gentypos_limit_dry_run(caplog):
    test_args = ["gentypos.py", "hello", "world", "--limit", "5", "--dry-run"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            with caplog.at_level(logging.INFO):
                gentypos.main()
        assert exc_info.value.code == 0

    log_text = "\n".join(record.message for record in caplog.records)
    assert "Limit: 5" in log_text
