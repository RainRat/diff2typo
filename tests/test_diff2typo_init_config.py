import os
import sys
import logging
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo


def test_init_config_default_filename(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["diff2typo.py", "--init-config"])

    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit) as exc_info:
            diff2typo.main()

    assert exc_info.value.code == 0
    config_file = tmp_path / "diff2typo.yaml"
    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert "diff2typo.yaml - Configuration file for diff2typo.py" in content
    assert "output_file:" in content
    assert "mode:" in content
    assert "Initialized sample configuration file at 'diff2typo.yaml'." in caplog.text


def test_init_config_custom_filename(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    custom_path = "custom_diff2typo.yaml"
    monkeypatch.setattr("sys.argv", ["diff2typo.py", "--init-config", custom_path])

    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit) as exc_info:
            diff2typo.main()

    assert exc_info.value.code == 0
    config_file = tmp_path / custom_path
    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert "diff2typo.yaml - Configuration file for diff2typo.py" in content
    assert f"Initialized sample configuration file at '{custom_path}'." in caplog.text


def test_generate_config_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["diff2typo.py", "--generate-config"])

    with pytest.raises(SystemExit) as exc_info:
        diff2typo.main()

    assert exc_info.value.code == 0
    assert (tmp_path / "diff2typo.yaml").exists()


def test_init_config_existing_file_error(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    existing_file = tmp_path / "diff2typo.yaml"
    existing_file.write_text("existing content", encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["diff2typo.py", "--init-config"])

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc_info:
            diff2typo.main()

    assert exc_info.value.code == 1
    assert "already exists. Aborting to prevent overwriting." in caplog.text


def test_init_config_write_error(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)

    def mock_open(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr("builtins.open", mock_open)
    monkeypatch.setattr("sys.argv", ["diff2typo.py", "--init-config"])

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc_info:
            diff2typo.main()

    assert exc_info.value.code == 1
    assert "Error writing configuration template" in caplog.text
