import json
import sys
import types
from pathlib import Path

import pytest

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without PyYAML
    yaml = types.ModuleType("yaml")

    class YAMLError(Exception):
        """Fallback YAML error used when PyYAML is unavailable."""

    def safe_dump(data):
        return json.dumps(data)

    def safe_load(stream):
        if hasattr(stream, 'read'):
            stream = stream.read()
        try:
            return json.loads(stream)
        except json.JSONDecodeError as exc:  # pragma: no cover - exercised via malformed test
            raise YAMLError(str(exc)) from exc

    yaml.safe_dump = safe_dump
    yaml.safe_load = safe_load
    yaml.YAMLError = YAMLError
    sys.modules['yaml'] = yaml


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
    with pytest.raises(SystemExit):
        cmdrunner.load_config(str(missing_file))


def test_load_config_malformed_yaml(tmp_path):
    bad_file = tmp_path / 'bad.yaml'
    bad_file.write_text(": invalid: yaml: content")

    with pytest.raises(SystemExit):
        cmdrunner.load_config(str(bad_file))


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
