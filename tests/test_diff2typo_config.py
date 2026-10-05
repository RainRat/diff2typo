import os
import sys
import tempfile
import pytest
import yaml
from unittest.mock import patch
import diff2typo


def test_diff2typo_config_custom_file(monkeypatch, tmp_path):
    """Test loading settings from an explicit YAML config file via -C / --config."""
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text("--- a/foo.txt\n+++ b/foo.txt\n@@ -1 +1 @@\n-teh\n+the\n", encoding="utf-8")

    out_file = tmp_path / "out.txt"

    cfg_file = tmp_path / "my_config.yaml"
    cfg_data = {
        "output_file": str(out_file),
        "output_format": "csv",
        "min_length": 3,
        "mode": "typos"
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["diff2typo.py", str(diff_file), "-C", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    diff2typo.main()

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8").strip()
    assert content == "teh,the"


def test_diff2typo_config_auto_fallback(monkeypatch, tmp_path):
    """Test automatic loading of diff2typo.yaml when present in current directory."""
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text("--- a/foo.txt\n+++ b/foo.txt\n@@ -1 +1 @@\n-teh\n+the\n", encoding="utf-8")

    out_file = tmp_path / "out.txt"

    cfg_data = {
        "output_file": str(out_file),
        "output_format": "table",
        "min_length": 3
    }
    cfg_file = tmp_path / "diff2typo.yaml"
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    test_args = ["diff2typo.py", str(diff_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    diff2typo.main()

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8").strip()
    assert content == 'teh = "the"'


def test_diff2typo_config_cli_override(monkeypatch, tmp_path):
    """Test that explicit CLI arguments override values in the configuration file."""
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text("--- a/foo.txt\n+++ b/foo.txt\n@@ -1 +1 @@\n-teh\n+the\n", encoding="utf-8")

    out_cfg = tmp_path / "out_cfg.txt"
    out_cli = tmp_path / "out_cli.txt"

    cfg_file = tmp_path / "custom.yaml"
    cfg_data = {
        "output_file": str(out_cfg),
        "output_format": "csv",
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["diff2typo.py", str(diff_file), "-C", str(cfg_file), "-o", str(out_cli), "-f", "table"]
    monkeypatch.setattr(sys, "argv", test_args)

    diff2typo.main()

    assert not out_cfg.exists()
    assert out_cli.exists()
    content = out_cli.read_text(encoding="utf-8").strip()
    assert content == 'teh = "the"'


def test_diff2typo_config_missing_pyyaml(monkeypatch, tmp_path):
    """Test error handling when PyYAML is unavailable and config file is specified."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("mode: typos", encoding="utf-8")

    monkeypatch.setattr(diff2typo, "_YAML_AVAILABLE", False)
    test_args = ["diff2typo.py", "--config", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        diff2typo.main()
    assert exc_info.value.code == 1


def test_diff2typo_config_all_options(monkeypatch, tmp_path):
    """Test parsing and applying all configuration file options in diff2typo."""
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text("--- a/foo.txt\n+++ b/foo.txt\n@@ -1 +1 @@\n-teh\n+the\n", encoding="utf-8")

    out_file = tmp_path / "out.txt"

    cfg_file = tmp_path / "full_config.yaml"
    cfg_data = {
        "input_files": str(diff_file),
        "output_file": str(out_file),
        "max_length": 10,
        "max_dist": 3,
        "min_count": 1,
        "sort": "freq",
        "reverse": True,
        "limit": 5,
        "dictionary_file": "dict.csv",
        "allowed_file": "allow.csv",
        "typos_tool_path": "/usr/local/bin/typos",
        "exclude": "vendor/*",
        "include": ["*.txt"],
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["diff2typo.py", "-C", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    diff2typo.main()

    assert out_file.exists()


def test_diff2typo_config_list_filters_and_git_options(monkeypatch, tmp_path):
    """Test list handling for exclude/include/input_files in config."""
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text("--- a/foo.txt\n+++ b/foo.txt\n@@ -1 +1 @@\n-teh\n+the\n", encoding="utf-8")

    out_file = tmp_path / "out.txt"

    cfg_file = tmp_path / "git_config.yaml"
    cfg_data = {
        "input_files": [str(diff_file)],
        "output_file": str(out_file),
        "exclude": ["vendor/*", "build/*"],
        "include": "*.txt",
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["diff2typo.py", "-C", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    diff2typo.main()

    assert out_file.exists()


def test_diff2typo_config_git_options(monkeypatch, tmp_path):
    """Test git and git_log YAML config options with mocked git execution."""
    out_file = tmp_path / "out.txt"
    cfg_file = tmp_path / "git_options.yaml"
    cfg_data = {
        "output_file": str(out_file),
        "git": "HEAD~1",
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["diff2typo.py", "-C", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    diff_content = "--- a/foo.txt\n+++ b/foo.txt\n@@ -1 +1 @@\n-teh\n+the\n"
    monkeypatch.setattr(diff2typo, "_run_git_subcommand", lambda cmd, spec: diff_content)

    diff2typo.main()

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8").strip()
    assert "teh -> the" in content

    # Test git_log option in config
    cfg_log_file = tmp_path / "git_log_options.yaml"
    cfg_log_data = {
        "output_file": str(out_file),
        "git_log": "main..feature",
    }
    cfg_log_file.write_text(yaml.dump(cfg_log_data), encoding="utf-8")

    test_log_args = ["diff2typo.py", "-C", str(cfg_log_file)]
    monkeypatch.setattr(sys, "argv", test_log_args)

    diff2typo.main()

    assert out_file.exists()


def test_diff2typo_config_file_not_found(monkeypatch):
    """Test error handling when specified config file does not exist."""
    test_args = ["diff2typo.py", "--config", "non_existent_config.yaml"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        diff2typo.main()
    assert exc_info.value.code == 1


def test_diff2typo_config_invalid_yaml(monkeypatch, tmp_path):
    """Test error handling when specified config file contains invalid YAML."""
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("mode: [invalid yaml::", encoding="utf-8")

    test_args = ["diff2typo.py", "--config", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        diff2typo.main()
    assert exc_info.value.code == 1
