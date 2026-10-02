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


def test_diff2typo_config_git_options(monkeypatch, tmp_path):
    """Test that git and git_log settings in YAML config file are correctly applied when CLI flags are absent."""
    out_file = tmp_path / "out.txt"

    cfg_file = tmp_path / "git_cfg.yaml"
    cfg_data = {
        "git": "HEAD~1",
        "output_file": str(out_file)
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    test_args = ["diff2typo.py", "--config", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args)

    diff_content = "--- a/bar.txt\n+++ b/bar.txt\n@@ -1 +1 @@\n-comit\n+commit\n"
    with patch("diff2typo._run_git_subcommand", return_value=diff_content) as mock_git:
        diff2typo.main()
        mock_git.assert_called_once_with(["git", "diff"], "HEAD~1")

    assert out_file.exists()
    assert "comit -> commit" in out_file.read_text(encoding="utf-8")

    # Now test git_log config option
    cfg_file_log = tmp_path / "git_log_cfg.yaml"
    cfg_data_log = {
        "git_log": "HEAD~3",
        "output_file": str(out_file)
    }
    cfg_file_log.write_text(yaml.dump(cfg_data_log), encoding="utf-8")

    test_args_log = ["diff2typo.py", "--config", str(cfg_file_log)]
    monkeypatch.setattr(sys, "argv", test_args_log)

    with patch("diff2typo._run_git_subcommand", return_value=diff_content) as mock_git_log:
        diff2typo.main()
        mock_git_log.assert_called_once_with(["git", "log", "-p"], "HEAD~3")


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
