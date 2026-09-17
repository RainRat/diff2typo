import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos


def test_init_config_default_filename(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    target_file = tmp_path / "gentypos.yaml"
    assert not target_file.exists()

    test_args = ["gentypos.py", "--init-config"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            gentypos.main()

    assert exc_info.value.code == 0
    assert target_file.exists()
    content = target_file.read_text(encoding="utf-8")
    assert "input_file" in content
    assert "typo_types:" in content
    assert "deletion: true" in content


def test_init_config_custom_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    custom_file = tmp_path / "custom_gentypos.yaml"
    assert not custom_file.exists()

    test_args = ["gentypos.py", "--init-config", str(custom_file)]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            gentypos.main()

    assert exc_info.value.code == 0
    assert custom_file.exists()
    content = custom_file.read_text(encoding="utf-8")
    assert "input_file" in content
    assert "typo_types:" in content


def test_generate_config_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    alias_file = tmp_path / "alias_config.yaml"
    assert not alias_file.exists()

    test_args = ["gentypos.py", "--generate-config", str(alias_file)]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            gentypos.main()

    assert exc_info.value.code == 0
    assert alias_file.exists()


def test_init_config_existing_file_error(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    existing_file = tmp_path / "gentypos.yaml"
    existing_file.write_text("existing_content: true\n", encoding="utf-8")

    test_args = ["gentypos.py", "--init-config"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            gentypos.main()

    assert exc_info.value.code == 1
    assert "already exists" in caplog.text
