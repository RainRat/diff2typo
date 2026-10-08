import sys
import logging
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool


def test_init_config_default(tmp_path, monkeypatch):
    """Test generating default multitool.yaml file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["multitool.py", "--init-config"])

    with pytest.raises(SystemExit) as excinfo:
        multitool.main()

    assert excinfo.value.code == 0

    target = tmp_path / "multitool.yaml"
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "# multitool.yaml - Configuration file for multitool.py" in content
    assert "process_output: false" in content


def test_init_config_custom_path(tmp_path, monkeypatch):
    """Test generating configuration template at a custom path."""
    custom_path = tmp_path / "custom_multitool.yaml"
    monkeypatch.setattr(sys, "argv", ["multitool.py", "--generate-config", str(custom_path)])

    with pytest.raises(SystemExit) as excinfo:
        multitool.main()

    assert excinfo.value.code == 0

    assert custom_path.exists()
    content = custom_path.read_text(encoding="utf-8")
    assert "# multitool.yaml - Configuration file for multitool.py" in content


def test_init_config_already_exists(tmp_path, monkeypatch, caplog):
    """Test aborting if configuration file already exists."""
    monkeypatch.chdir(tmp_path)
    existing_config = tmp_path / "multitool.yaml"
    existing_config.write_text("existing: content", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["multitool.py", "--init-config"])

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            multitool.main()

    assert excinfo.value.code == 1
    assert any("already exists. Aborting to prevent overwriting." in record.message for record in caplog.records)


def test_init_config_write_error(tmp_path, monkeypatch, caplog):
    """Test error handling when writing configuration template fails."""
    invalid_path = tmp_path / "nonexistent_dir" / "multitool.yaml"
    monkeypatch.setattr(sys, "argv", ["multitool.py", "--init-config", str(invalid_path)])

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            multitool.main()

    assert excinfo.value.code == 1
    assert any("Error writing configuration template to" in record.message for record in caplog.records)
