import sys
import logging
import os
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


sys.path.append(str(Path(__file__).resolve().parents[1]))
import cmdrunner
from cmdrunner import MinimalFormatter, ConfigError


def test_load_config_success(tmp_path):
    config_data = {
        'base_directory': '/tmp',
        'command_to_run': 'echo test',
        'excluded_folders': ['venv'],
    }
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.safe_dump(config_data))

    loaded = cmdrunner.load_config(str(config_file))
    assert loaded == config_data


def test_load_config_missing_file(tmp_path):
    missing_file = tmp_path / 'missing.yaml'
    with pytest.raises(FileNotFoundError):
        cmdrunner.load_config(str(missing_file))


def test_load_config_malformed_yaml(tmp_path):
    bad_file = tmp_path / 'bad.yaml'
    bad_file.write_text(": invalid: yaml: content")

    with pytest.raises(ConfigError, match="Error parsing YAML file"):
        cmdrunner.load_config(str(bad_file))


def test_load_config_empty_file(tmp_path):
    empty_file = tmp_path / 'empty.yaml'
    empty_file.write_text("")

    with pytest.raises(ConfigError, match="empty or malformed"):
        cmdrunner.load_config(str(empty_file))


@pytest.mark.parametrize(
    "config_data,expected_message",
    [
        ({'command_to_run': 'echo test'}, "main_folder"),
        ({'main_folder': '/tmp'}, "command_to_run"),
    ],
)
def test_load_config_missing_required_fields(tmp_path, config_data, expected_message):
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.safe_dump(config_data))

    with pytest.raises(ConfigError, match=expected_message):
        cmdrunner.load_config(str(config_file))


def test_load_config_invalid_types(tmp_path):
    config_file = tmp_path / 'config_invalid.yaml'
    config_file.write_text(
        yaml.safe_dump(
            {
                'main_folder': 456,
                'base_directory': 123,
                'command_to_run': ['echo', 'test'],
                'excluded_folders': 'not-a-list',
            }
        )
    )

    with pytest.raises(ConfigError) as exc_info:
        cmdrunner.load_config(str(config_file))

    message = str(exc_info.value)
    assert "'main_folder' must be a string" in message
    assert "'base_directory' must be a string" in message
    assert "'command_to_run' must be a string" in message
    assert "must be a list" in message


def test_load_config_invalid_main_folder(tmp_path):
    config_file = tmp_path / 'config_invalid_main.yaml'
    config_file.write_text(
        yaml.safe_dump(
            {
                'main_folder': 123,
                'command_to_run': 'echo test',
            }
        )
    )

    with pytest.raises(ConfigError) as exc_info:
        cmdrunner.load_config(str(config_file))

    assert "'main_folder' must be a string" in exc_info.value.args[0]


def test_run_command_in_folders_creates_files(tmp_path):
    base_dir = tmp_path / 'projects'
    base_dir.mkdir()

    included = ['proj1', 'proj2']
    excluded = 'skip'

    for name in included + [excluded]:
        (base_dir / name).mkdir()

    command = "python3 -c \"open('test_file.txt','w').write('test')\""

    cmdrunner.run_command_in_folders(str(base_dir), command, excluded_folders=[excluded])

    for name in included:
        test_file = base_dir / name / 'test_file.txt'
        assert test_file.exists()
        assert test_file.read_text() == 'test'

    assert not (base_dir / excluded / 'test_file.txt').exists()


def test_run_command_in_folders_subprocess_failure(tmp_path, caplog):
    base_dir = tmp_path / 'projects'
    base_dir.mkdir()
    failing = base_dir / 'failing'
    failing.mkdir()

    command = "python3 -c \"import sys; sys.exit(1)\""

    with caplog.at_level(logging.ERROR):
        cmdrunner.run_command_in_folders(str(base_dir), command)

    assert any("The command failed" in message for message in caplog.messages)


def test_run_command_in_folders_invalid_base_dir(tmp_path):
    missing_dir = tmp_path / 'missing'

    with pytest.raises(SystemExit):
        cmdrunner.run_command_in_folders(str(missing_dir), 'echo test')


def test_run_command_in_folders_dry_run(tmp_path, caplog):
    base_dir = tmp_path / 'projects'
    base_dir.mkdir()

    for name in ['proj1', 'proj2']:
        (base_dir / name).mkdir()

    command = "python3 -c \"open('dry_run.txt','w').write('dry')\""

    with caplog.at_level('INFO'):
        cmdrunner.run_command_in_folders(str(base_dir), command, dry_run=True)

    for name in ['proj1', 'proj2']:
        assert not (base_dir / name / 'dry_run.txt').exists()

    for message in caplog.messages:
        if "dry_run.txt" in message:
            break
    else:  # pragma: no cover - fallback in case log expectations change
        pytest.fail('Expected dry run log message not found')


def test_main_integration(tmp_path, monkeypatch):
    base_dir = tmp_path / 'projects'
    base_dir.mkdir()

    for name in ['proj1', 'proj2']:
        (base_dir / name).mkdir()

    command = "python3 -c \"open('integration.txt','w').write('run')\""
    config = {
        'base_directory': str(base_dir),
        'command_to_run': command,
    }

    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.safe_dump(config))

    monkeypatch.setattr(sys, 'argv', ['cmdrunner.py', str(config_file)])

    cmdrunner.main()

    for name in ['proj1', 'proj2']:
        result_file = base_dir / name / 'integration.txt'
        assert result_file.exists()
        assert result_file.read_text() == 'run'


def test_excluded_folders_integration(tmp_path, monkeypatch):
    base_dir = tmp_path / 'projects'
    base_dir.mkdir()

    included = base_dir / 'proj1'
    excluded = base_dir / 'skip'
    included.mkdir()
    excluded.mkdir()

    command = (
        "python3 -c \"from pathlib import Path; Path('integration_excluded.txt').write_text('ok')\""
    )
    config = {
        'base_directory': str(base_dir),
        'command_to_run': command,
        'excluded_folders': [excluded.name],
    }

    config_file = tmp_path / 'config_excluded.yaml'
    config_file.write_text(yaml.safe_dump(config))

    monkeypatch.setattr(sys, 'argv', ['cmdrunner.py', str(config_file)])

    cmdrunner.main()

    included_result = included / 'integration_excluded.txt'
    excluded_result = excluded / 'integration_excluded.txt'

    assert included_result.exists()
    assert included_result.read_text() == 'ok'
    assert not excluded_result.exists()


def test_main_integration_dry_run(tmp_path, monkeypatch, caplog):
    base_dir = tmp_path / 'projects'
    base_dir.mkdir()

    for name in ['proj1', 'proj2']:
        (base_dir / name).mkdir()

    command = "python3 -c \"open('integration_dry_run.txt','w').write('run')\""
    config = {
        'base_directory': str(base_dir),
        'command_to_run': command,
    }

    config_file = tmp_path / 'config_dry_run.yaml'
    config_file.write_text(yaml.safe_dump(config))

    monkeypatch.setattr(sys, 'argv', ['cmdrunner.py', str(config_file), '--dry-run'])

    with caplog.at_level('INFO'):
        cmdrunner.main()

    for name in ['proj1', 'proj2']:
        assert not (base_dir / name / 'integration_dry_run.txt').exists()

    for record in caplog.records:
        if "Dry run" in record.message:
            break
    else:  # pragma: no cover - fallback in case log expectations change
        pytest.fail('Expected dry run log message not found')


def test_main_quiet_mode_suppresses_output(tmp_path, monkeypatch, capsys):
    base_dir = tmp_path / 'projects'
    base_dir.mkdir()

    for name in ['proj1', 'proj2']:
        (base_dir / name).mkdir()

    command = "python3 -c \"open('integration_quiet.txt','w').write('silent')\""
    config = {
        'base_directory': str(base_dir),
        'command_to_run': command,
    }

    config_file = tmp_path / 'config_quiet.yaml'
    config_file.write_text(yaml.safe_dump(config))

    monkeypatch.setattr(sys, 'argv', ['cmdrunner.py', str(config_file), '--quiet'])

    cmdrunner.main()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    for name in ['proj1', 'proj2']:
        result_file = base_dir / name / 'integration_quiet.txt'
        assert result_file.exists()
        assert result_file.read_text() == 'silent'


def test_main_missing_config(monkeypatch, tmp_path):
    missing_config = tmp_path / 'missing.yaml'
    monkeypatch.setattr(sys, 'argv', ['cmdrunner.py', str(missing_config)])

    with pytest.raises(SystemExit):
        cmdrunner.main()


def test_main_missing_base_directory(tmp_path, monkeypatch):
    config = {'command_to_run': 'echo test'}
    config_file = tmp_path / 'config_missing_base.yaml'
    config_file.write_text(yaml.safe_dump(config))

    monkeypatch.setattr(sys, 'argv', ['cmdrunner.py', str(config_file)])

    with pytest.raises(SystemExit):
        cmdrunner.main()


def test_main_missing_command_to_run(tmp_path, monkeypatch):
    base_dir = tmp_path / 'projects'
    base_dir.mkdir()
    config = {'base_directory': str(base_dir)}
    config_file = tmp_path / 'config_missing_command.yaml'
    config_file.write_text(yaml.safe_dump(config))

    monkeypatch.setattr(sys, 'argv', ['cmdrunner.py', str(config_file)])

    with pytest.raises(SystemExit):
        cmdrunner.main()

# --- Consolidated from test_cmdrunner_extra.py and expanded for full coverage ---

def test_minimal_formatter_info():
    formatter = MinimalFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="info message",
        args=None,
        exc_info=None
    )
    assert formatter.format(record) == "info message"

def test_minimal_formatter_warning_no_tty():
    formatter = MinimalFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=10,
        msg="warning message",
        args=None,
        exc_info=None
    )
    # Mock sys.stderr.isatty() to return False
    with patch("sys.stderr.isatty", return_value=False):
        assert formatter.format(record) == "WARNING: warning message"

def test_minimal_formatter_error_with_tty():
    with patch("cmdrunner.RED", "\033[31m"), patch("cmdrunner.RESET", "\033[0m"):
        # Patch the class attribute
        with patch.dict(MinimalFormatter.LEVEL_COLORS, {logging.ERROR: "\033[31m"}):
            formatter = MinimalFormatter()
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="error message",
                args=None,
                exc_info=None
            )
            # Mock sys.stderr.isatty() to return True
            with patch("sys.stderr.isatty", return_value=True):
                formatted = formatter.format(record)
                assert "\033[31m" in formatted
                assert "ERROR" in formatted
                assert "error message" in formatted

def test_minimal_formatter_uncolored_level_with_tty():
    """Covers line 46: when a level has no color defined in LEVEL_COLORS."""
    formatter = MinimalFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="test.py",
        lineno=10,
        msg="debug message",
        args=None,
        exc_info=None
    )
    with patch("sys.stderr.isatty", return_value=True):
        assert formatter.format(record) == "DEBUG: debug message"

def test_minimal_formatter_no_levelname():
    """Covers line 44: when levelname is empty or None."""
    formatter = MinimalFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=10,
        msg="warning message",
        args=None,
        exc_info=None
    )
    record.levelname = ""
    with patch("sys.stderr.isatty", return_value=True):
        assert formatter.format(record) == ": warning message"

def test_load_config_no_yaml_available():
    with patch("cmdrunner._YAML_AVAILABLE", False):
        with patch("logging.error") as mock_log_error:
            with pytest.raises(SystemExit) as excinfo:
                cmdrunner.load_config("any.yaml")
            assert excinfo.value.code == 1
            mock_log_error.assert_called_with("PyYAML is not installed. Install via 'pip install PyYAML' to use cmdrunner.")

def test_color_initialization_no_tty():
    """Covers line 25: when stdout is not a tty, colors should be disabled."""
    with patch("sys.stdout.isatty", return_value=False), \
         patch.dict(os.environ, {}, clear=False):
        if "cmdrunner" in sys.modules:
            importlib.reload(cmdrunner)
        assert cmdrunner.RED == ""
        assert cmdrunner.GREEN == ""

def test_color_initialization_no_color_env():
    """Covers line 25: when NO_COLOR is set, colors should be disabled."""
    with patch("sys.stdout.isatty", return_value=True), \
         patch.dict(os.environ, {"NO_COLOR": "1"}):
        if "cmdrunner" in sys.modules:
            importlib.reload(cmdrunner)
        assert cmdrunner.RED == ""
        assert cmdrunner.GREEN == ""

def test_color_initialization_enabled():
    """Verifies colors are enabled when tty is present and NO_COLOR is NOT set."""
    with patch("sys.stdout.isatty", return_value=True), \
         patch.dict(os.environ, {}, clear=True):
        # We need to make sure we don't accidentally have NO_COLOR from the environment
        if "cmdrunner" in sys.modules:
            importlib.reload(cmdrunner)
        assert cmdrunner.RED != ""
        assert cmdrunner.GREEN != ""

def test_main_block():
    import runpy
    with patch("sys.argv", ["cmdrunner.py", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("cmdrunner", run_name="__main__")
        assert excinfo.value.code == 0
