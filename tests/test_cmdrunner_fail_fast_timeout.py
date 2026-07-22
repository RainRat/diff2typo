import sys
import logging
import os
from pathlib import Path
from unittest.mock import patch
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
import cmdrunner


def test_load_config_fail_fast_validation(tmp_path):
    config_file = tmp_path / "config.yaml"

    # Valid configurations
    config_file.write_text(yaml.safe_dump({
        "main_folder": "/tmp",
        "command_to_run": "echo",
        "fail_fast": True
    }))
    assert cmdrunner.load_config(str(config_file))["fail_fast"] is True

    # Invalid configurations (must be a boolean)
    config_file.write_text(yaml.safe_dump({
        "main_folder": "/tmp",
        "command_to_run": "echo",
        "fail_fast": "yes"
    }))
    with pytest.raises(cmdrunner.ConfigError, match="must be a boolean"):
        cmdrunner.load_config(str(config_file))


def test_load_config_timeout_validation(tmp_path):
    config_file = tmp_path / "config.yaml"

    # Valid configurations
    config_file.write_text(yaml.safe_dump({
        "main_folder": "/tmp",
        "command_to_run": "echo",
        "timeout": 5.5
    }))
    assert cmdrunner.load_config(str(config_file))["timeout"] == 5.5

    # Invalid configurations (must be a number, not a string)
    config_file.write_text(yaml.safe_dump({
        "main_folder": "/tmp",
        "command_to_run": "echo",
        "timeout": "five"
    }))
    with pytest.raises(cmdrunner.ConfigError, match="must be a number"):
        cmdrunner.load_config(str(config_file))

    # Invalid configurations (bool should not pass as int/float in strict check)
    config_file.write_text(yaml.safe_dump({
        "main_folder": "/tmp",
        "command_to_run": "echo",
        "timeout": True
    }))
    with pytest.raises(cmdrunner.ConfigError, match="must be a number"):
        cmdrunner.load_config(str(config_file))


def test_run_command_fail_fast_stops_immediately(tmp_path, caplog):
    base_dir = tmp_path / "projects"
    base_dir.mkdir()

    # We create three projects: proj1, proj2, proj3
    # If fail_fast is True and we run a failing command on proj1, we should stop immediately and not run on proj2 or proj3.
    # Standard alphabetical ordering means they run as proj1, proj2, proj3.
    (base_dir / "proj1").mkdir()
    (base_dir / "proj2").mkdir()
    (base_dir / "proj3").mkdir()

    # Create a command that always fails (exits 1)
    command = "python3 -c \"import sys; sys.exit(1)\""

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            cmdrunner.run_command_in_folders(
                str(base_dir),
                command,
                fail_fast=True
            )
        assert excinfo.value.code == 1

    # Check logs to confirm we processed the first project
    assert any("The command failed in 'proj1'" in msg for msg in caplog.messages)
    # And did NOT process the other projects
    assert not any("The command failed in 'proj2'" in msg for msg in caplog.messages)
    assert not any("The command failed in 'proj3'" in msg for msg in caplog.messages)


def test_run_command_no_fail_fast_continues(tmp_path, caplog):
    base_dir = tmp_path / "projects"
    base_dir.mkdir()

    (base_dir / "proj1").mkdir()
    (base_dir / "proj2").mkdir()

    command = "python3 -c \"import sys; sys.exit(1)\""

    with caplog.at_level(logging.ERROR):
        cmdrunner.run_command_in_folders(
            str(base_dir),
            command,
            fail_fast=False
        )

    # Both projects should be processed even though the command fails
    assert any("The command failed in 'proj1'" in msg for msg in caplog.messages)
    assert any("The command failed in 'proj2'" in msg for msg in caplog.messages)


def test_run_command_timeout_stops_immediately(tmp_path, caplog):
    base_dir = tmp_path / "projects"
    base_dir.mkdir()

    (base_dir / "proj1").mkdir()
    (base_dir / "proj2").mkdir()

    # Sleep command that takes 5 seconds, but we enforce 0.1s timeout
    command = "python3 -c \"import time; time.sleep(5)\""

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            cmdrunner.run_command_in_folders(
                str(base_dir),
                command,
                fail_fast=True,
                timeout=0.1
            )
        assert excinfo.value.code == 1

    # Check logs for timeout error
    assert any("timed out after 0.1 seconds" in msg for msg in caplog.messages)
    # Ensure proj2 was not processed
    assert not any("The command in 'proj2' timed out" in msg for msg in caplog.messages)


def test_run_command_timeout_no_fail_fast_continues(tmp_path, caplog):
    base_dir = tmp_path / "projects"
    base_dir.mkdir()

    (base_dir / "proj1").mkdir()
    (base_dir / "proj2").mkdir()

    command = "python3 -c \"import time; time.sleep(5)\""

    with caplog.at_level(logging.ERROR):
        cmdrunner.run_command_in_folders(
            str(base_dir),
            command,
            fail_fast=False,
            timeout=0.1
        )

    # Both projects should run and timeout
    # We should have timeout messages for both
    timeout_msgs = [msg for msg in caplog.messages if "timed out after 0.1 seconds" in msg]
    assert len(timeout_msgs) >= 2


def test_main_cli_overrides_fail_fast_and_timeout(tmp_path, monkeypatch, caplog):
    base_dir = tmp_path / "projects"
    base_dir.mkdir()
    (base_dir / "proj1").mkdir()

    # Setup config file with fail_fast: False
    config_data = {
        "main_folder": str(base_dir),
        "command_to_run": "python3 -c \"import sys; sys.exit(1)\"",
        "fail_fast": False,
        "timeout": 10.0
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config_data))

    # Run cmdrunner via CLI with --fail-fast and --timeout CLI overrides
    monkeypatch.setattr(sys, "argv", [
        "cmdrunner.py",
        str(config_file),
        "--fail-fast",
        "--timeout", "0.5"
    ])

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            cmdrunner.main()
        assert excinfo.value.code == 1

    # Ensure it failed under the CLI override
    assert any("The command failed in 'proj1'" in msg for msg in caplog.messages)
