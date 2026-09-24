import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos


def test_main_write_file_exception(tmp_path, monkeypatch, caplog):
    output_file = tmp_path / "output.txt"
    test_args = ["gentypos.py", "hello", "world", "-o", str(output_file), "-N"]

    orig_open = open

    def mock_open_func(file, mode="r", *args, **kwargs):
        if str(file) == str(output_file) and "w" in mode:
            raise OSError("Disk write permission denied")
        return orig_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_func)

    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            gentypos.main()

    assert exc_info.value.code == 1
    assert "Error writing to" in caplog.text


def test_main_summary_extra_metrics_limit(tmp_path, monkeypatch, capsys):
    output_file = tmp_path / "output.txt"
    test_args = ["gentypos.py", "hello", "world", "-o", str(output_file), "-L", "1", "-N"]

    with patch.object(sys, "argv", test_args):
        gentypos.main()

    captured = capsys.readouterr()
    assert "Output limit (--limit)" in captured.err
