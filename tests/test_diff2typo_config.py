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


def test_diff2typo_config_all_options(monkeypatch, tmp_path):
    """Test configuration merging for max_length, max_dist, min_count, sort, reverse, limit, dictionary_file, allowed_file, typos_tool_path."""
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text("--- a/foo.txt\n+++ b/foo.txt\n@@ -1 +1 @@\n-teh\n+the\n", encoding="utf-8")

    out_file = tmp_path / "out.txt"
    dict_file = tmp_path / "custom_words.csv"
    dict_file.write_text("", encoding="utf-8")
    allowed_file = tmp_path / "custom_allowed.csv"
    allowed_file.write_text("", encoding="utf-8")

    cfg_file = tmp_path / "all_opts.yaml"
    cfg_data = {
        "output_file": str(out_file),
        "max_length": 10,
        "max_dist": 2,
        "min_count": 1,
        "sort": "count",
        "reverse": True,
        "limit": 5,
        "dictionary_file": str(dict_file),
        "allowed_file": str(allowed_file),
        "typos_tool_path": "nonexistent_typos_bin",
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["diff2typo.py", str(diff_file), "-C", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    diff2typo.main()

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8").strip()
    assert content == "teh -> the"


def test_diff2typo_config_list_filters_and_git_options(monkeypatch, tmp_path):
    """Test configuration merging for exclude, include, git, git_log, and input_files in config."""
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text("diff --git a/foo.txt b/foo.txt\n--- a/foo.txt\n+++ b/foo.txt\n@@ -1 +1 @@\n-teh\n+the\n", encoding="utf-8")

    out_file = tmp_path / "out.txt"

    cfg_file = tmp_path / "filters_git.yaml"
    cfg_data = {
        "output_file": str(out_file),
        "exclude": ["*.json"],
        "include": ["*.txt"],
        "git": None,
        "git_log": None,
        "input_files": str(diff_file)
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    test_args = ["diff2typo.py", "-C", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    diff2typo.main()

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8").strip()
    assert content == "teh -> the"
