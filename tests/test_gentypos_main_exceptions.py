import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos


def test_init_config_write_error(tmp_path, monkeypatch, caplog):
    """Verify that OSError during --init-config writing logs an error and exits."""
    monkeypatch.chdir(tmp_path)
    test_args = ["gentypos.py", "--init-config", "new_config.yaml"]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("builtins.open", side_effect=OSError("Permission denied")):
        with pytest.raises(SystemExit) as exc_info:
            gentypos.main()

    assert exc_info.value.code == 1
    assert "Error writing configuration template to 'new_config.yaml'" in caplog.text


def test_main_summary_extra_metrics_limit(capsys, monkeypatch):
    """Verify that --limit includes 'Output limit (--limit)' in stderr summary."""
    test_args = ["gentypos.py", "hello", "-L", "2", "--no-filter"]
    monkeypatch.setattr(sys, "argv", test_args)

    gentypos.main()

    captured = capsys.readouterr()
    assert "Output limit (--limit):" in captured.err
    assert "2" in captured.err
