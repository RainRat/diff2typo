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


def test_main_write_file_exception(tmp_path, monkeypatch, caplog):
    """Verify that OSError during output file writing logs an error and exits."""
    output_file = tmp_path / "output.txt"
    test_args = ["gentypos.py", "hello", "world", "-o", str(output_file), "-N"]

    orig_open = open

    def mock_open_func(file, mode="r", *args, **kwargs):
        if str(file) == str(output_file) and "w" in mode:
            raise OSError("Disk write permission denied")
        return orig_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_func)
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        gentypos.main()

    assert exc_info.value.code == 1
    assert "Error writing to" in caplog.text

