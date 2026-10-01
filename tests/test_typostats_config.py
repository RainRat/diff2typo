import os
import sys
import pytest
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import typostats


def test_typostats_config_custom_file(monkeypatch, tmp_path):
    """Test loading settings from an explicit YAML config file via -C / --config."""
    typos_file = tmp_path / "sample.txt"
    typos_file.write_text("teh -> the\nteh -> the\nwrong -> right\n", encoding="utf-8")

    out_file = tmp_path / "out.csv"

    cfg_file = tmp_path / "my_config.yaml"
    cfg_data = {
        "output": str(out_file),
        "format": "csv",
        "min_count": 2,
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["typostats.py", str(typos_file), "-C", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    typostats.main()

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8").strip()
    assert "typo,correction,count" in content
    assert "e,h,2" in content or "eh,he,2" in content or "h,e,2" in content


def test_typostats_config_auto_fallback(monkeypatch, tmp_path):
    """Test automatic loading of typostats.yaml when present in current directory."""
    typos_file = tmp_path / "sample.txt"
    typos_file.write_text("teh -> the\n", encoding="utf-8")

    out_file = tmp_path / "out.toml"

    cfg_data = {
        "output": str(out_file),
        "format": "toml",
    }
    cfg_file = tmp_path / "typostats.yaml"
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    test_args = ["typostats.py", str(typos_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    typostats.main()

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8").strip()
    assert '=' in content


def test_typostats_config_cli_override(monkeypatch, tmp_path):
    """Test that explicit CLI arguments override values in the configuration file."""
    typos_file = tmp_path / "sample.txt"
    typos_file.write_text("teh -> the\n", encoding="utf-8")

    out_cfg = tmp_path / "out_cfg.txt"
    out_cli = tmp_path / "out_cli.csv"

    cfg_file = tmp_path / "custom.yaml"
    cfg_data = {
        "output": str(out_cfg),
        "format": "arrow",
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["typostats.py", str(typos_file), "-C", str(cfg_file), "-o", str(out_cli), "-f", "csv"]
    monkeypatch.setattr(sys, "argv", test_args)

    typostats.main()

    assert not out_cfg.exists()
    assert out_cli.exists()
    content = out_cli.read_text(encoding="utf-8").strip()
    assert "typo,correction,count" in content


def test_typostats_config_missing_pyyaml(monkeypatch, tmp_path):
    """Test error handling when PyYAML is unavailable and config file is specified."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("quiet: true", encoding="utf-8")

    monkeypatch.setattr(typostats, "_YAML_AVAILABLE", False)
    test_args = ["typostats.py", "--config", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        typostats.main()
    assert exc_info.value.code == 1


def test_typostats_config_file_not_found(monkeypatch):
    """Test error handling when specified config file does not exist."""
    test_args = ["typostats.py", "--config", "non_existent_config.yaml"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        typostats.main()
    assert exc_info.value.code == 1


def test_typostats_config_invalid_yaml(monkeypatch, tmp_path):
    """Test error handling when specified config file contains invalid YAML."""
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("sort: [invalid yaml::", encoding="utf-8")

    test_args = ["typostats.py", "--config", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        typostats.main()
    assert exc_info.value.code == 1


def test_typostats_config_all_options(monkeypatch, tmp_path):
    """Test loading and merging all options from a YAML configuration file."""
    typos_file = tmp_path / "data.txt"
    typos_file.write_text("teh -> the\nrm -> m\n", encoding="utf-8")

    out_file = tmp_path / "out_all.json"
    cfg_file = tmp_path / "full_config.yaml"

    cfg_data = {
        "input_files": [str(typos_file)],
        "output_file": str(out_file),
        "output_format": "json",
        "quiet": True,
        "min": 1,
        "sort": "typo",
        "reverse": True,
        "limit": 5,
        "transposition": True,
        "keyboard": True,
        "allow_1to2": True,
        "allow_2to1": True,
        "include_deletions": True,
        "exclude": ["*.bak"],
        "include": ["*.txt"],
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["typostats.py", "-C", str(cfg_file)])
    typostats.main()

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "replacements" in content
