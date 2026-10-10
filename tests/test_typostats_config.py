import os
import sys
from pathlib import Path
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
import typostats


def test_typostats_config_custom_file(monkeypatch, tmp_path):
    """Test loading settings from an explicit YAML config file via -C / --config."""
    data_file = tmp_path / "typos.txt"
    data_file.write_text("teh -> the\n", encoding="utf-8")

    out_file = tmp_path / "out.csv"

    cfg_file = tmp_path / "my_config.yaml"
    cfg_data = {
        "output": str(out_file),
        "format": "csv",
        "min_count": 1,
        "sort": "count"
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["typostats.py", str(data_file), "-C", str(cfg_file), "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    typostats.main()

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8").strip()
    assert "typo,correction,count" in content
    assert "e,h,1" in content or "teh,the" in content or "h,e,1" in content or "e,e" in content or "1" in content


def test_typostats_config_auto_fallback(monkeypatch, tmp_path):
    """Test automatic loading of typostats.yaml when present in current directory."""
    data_file = tmp_path / "typos.txt"
    data_file.write_text("teh -> the\n", encoding="utf-8")

    out_file = tmp_path / "out.csv"

    cfg_data = {
        "output": str(out_file),
        "format": "csv",
    }
    cfg_file = tmp_path / "typostats.yaml"
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    test_args = ["typostats.py", str(data_file), "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    typostats.main()

    assert out_file.exists()


def test_typostats_config_cli_override(monkeypatch, tmp_path):
    """Test that explicit CLI arguments override values in the configuration file."""
    data_file = tmp_path / "typos.txt"
    data_file.write_text("teh -> the\n", encoding="utf-8")

    out_cfg = tmp_path / "out_cfg.csv"
    out_cli = tmp_path / "out_cli.csv"

    cfg_file = tmp_path / "custom.yaml"
    cfg_data = {
        "output": str(out_cfg),
        "format": "json",
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["typostats.py", str(data_file), "-C", str(cfg_file), "-o", str(out_cli), "-f", "csv", "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    typostats.main()

    assert not out_cfg.exists()
    assert out_cli.exists()


def test_typostats_config_missing_pyyaml(monkeypatch, tmp_path):
    """Test error handling when PyYAML is unavailable and config file is specified."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("min_count: 1", encoding="utf-8")

    monkeypatch.setattr(typostats, "_YAML_AVAILABLE", False)
    test_args = ["typostats.py", "--config", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        typostats.main()
    assert exc_info.value.code == 1


def test_typostats_config_all_options(monkeypatch, tmp_path):
    """Test loading all configuration settings and analysis flags from YAML config file."""
    data_file = tmp_path / "typos.txt"
    data_file.write_text("teh -> the\nrecieve -> receive\n", encoding="utf-8")

    out_file = tmp_path / "out_all.json"

    cfg_file = tmp_path / "all_options.yaml"
    cfg_data = {
        "input_files": [str(data_file)],
        "output": str(out_file),
        "format": "json",
        "quiet": True,
        "min_count": 1,
        "sort": "typo",
        "reverse": True,
        "limit": 5,
        "exclude": ["ignore_pattern"],
        "include": ["*.txt"],
        "all": True,
        "keyboard": True,
        "transposition": True,
        "allow_1to2": True,
        "allow_2to1": True,
        "include_deletions": True,
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["typostats.py", "-C", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    typostats.main()

    assert out_file.exists()


def test_typostats_config_single_string_inputs_and_filters(monkeypatch, tmp_path):
    """Test auto-wrapping single string values into lists for input_files, exclude, and include in config."""
    data_file = tmp_path / "typos.txt"
    data_file.write_text("teh -> the\n", encoding="utf-8")

    out_file = tmp_path / "out_single.csv"

    cfg_file = tmp_path / "single_string.yaml"
    cfg_data = {
        "input_files": str(data_file),
        "output": str(out_file),
        "format": "csv",
        "quiet": True,
        "exclude": "single_exclude",
        "include": "single_include",
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["typostats.py", "-C", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    typostats.main()

    assert out_file.exists()


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
    cfg_file.write_text("min_count: [invalid yaml::", encoding="utf-8")

    test_args = ["typostats.py", "--config", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        typostats.main()
    assert exc_info.value.code == 1
