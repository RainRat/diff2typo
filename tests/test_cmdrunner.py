import sys
import logging
from pathlib import Path

import pytest
import yaml


sys.path.append(str(Path(__file__).resolve().parents[1]))
import cmdrunner


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

    with pytest.raises(yaml.YAMLError):
        cmdrunner.load_config(str(bad_file))


def test_load_config_empty_file(tmp_path):
    empty_file = tmp_path / 'empty.yaml'
    empty_file.write_text("")

    with pytest.raises(cmdrunner.ConfigError, match="empty or malformed"):
        cmdrunner.load_config(str(empty_file))


@pytest.mark.parametrize(
    "config_data,expected_message",
    [
        ({'command_to_run': 'echo test'}, "base_directory"),
        ({'base_directory': '/tmp'}, "command_to_run"),
    ],
)
def test_load_config_missing_required_fields(tmp_path, config_data, expected_message):
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.safe_dump(config_data))

    with pytest.raises(cmdrunner.ConfigError, match=expected_message):
        cmdrunner.load_config(str(config_file))


def test_load_config_invalid_types(tmp_path):
    config_file = tmp_path / 'config_invalid.yaml'
    config_file.write_text(
        yaml.safe_dump(
            {
                'base_directory': 123,
                'command_to_run': ['echo', 'test'],
                'excluded_folders': 'not-a-list',
            }
        )
    )

    with pytest.raises(cmdrunner.ConfigError) as exc_info:
        cmdrunner.load_config(str(config_file))

    message = str(exc_info.value)
    assert "'base_directory' must be a string" in message
    assert "'command_to_run' must be a string" in message
    assert "must be a list" in message


def test_run_command_in_folders_creates_files(tmp_path):
    base_dir = tmp_path / 'projects'
    base_dir.mkdir()

    included = ['proj1', 'proj2']
    excluded = 'skip'

    for name in included + [excluded]:
        (base_dir / name).mkdir()

    command = "python -c \"open('test_file.txt','w').write('test')\""

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

    command = "python -c \"import sys; sys.exit(1)\""

    with caplog.at_level(logging.ERROR):
        cmdrunner.run_command_in_folders(str(base_dir), command)

    assert any("Command failed" in message for message in caplog.messages)


def test_run_command_in_folders_invalid_base_dir(tmp_path):
    missing_dir = tmp_path / 'missing'

    with pytest.raises(SystemExit):
        cmdrunner.run_command_in_folders(str(missing_dir), 'echo test')


def test_run_command_in_folders_dry_run(tmp_path, caplog):
    base_dir = tmp_path / 'projects'
    base_dir.mkdir()

    for name in ['proj1', 'proj2']:
        (base_dir / name).mkdir()

    command = "python -c \"open('dry_run.txt','w').write('dry')\""

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

    command = "python -c \"open('integration.txt','w').write('run')\""
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
        "python -c \"from pathlib import Path; Path('integration_excluded.txt').write_text('ok')\""
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

    command = "python -c \"open('integration_dry_run.txt','w').write('run')\""
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
