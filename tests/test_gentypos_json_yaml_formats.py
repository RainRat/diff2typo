import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

@pytest.fixture
def empty_config_file(tmp_path):
    config = tmp_path / "test_config.yaml"
    config.write_text("{}", encoding="utf-8")
    return str(config)

def test_gentypos_json_format_stdout(capsys, empty_config_file):
    test_args = [
        "gentypos.py",
        "hello",
        "-c", empty_config_file,
        "--no-filter",
        "-f", "json"
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    captured = capsys.readouterr()
    stdout_text = captured.out

    # Parse and verify the JSON
    parsed = json.loads(stdout_text)
    assert isinstance(parsed, dict)
    assert len(parsed) > 0
    # Every key is a generated typo, and its value should be 'hello'
    for typo, correction in parsed.items():
        assert correction == "hello"

def test_gentypos_yaml_format_stdout(capsys, empty_config_file):
    test_args = [
        "gentypos.py",
        "hello",
        "-c", empty_config_file,
        "--no-filter",
        "-f", "yaml"
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    captured = capsys.readouterr()
    stdout_text = captured.out

    if gentypos._YAML_AVAILABLE:
        import yaml
        parsed = yaml.safe_load(stdout_text)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0
        for typo, correction in parsed.items():
            assert correction == "hello"
    else:
        # Fallback to JSON
        parsed = json.loads(stdout_text)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0

def test_gentypos_auto_detect_json_extension(tmp_path, empty_config_file):
    output_json = tmp_path / "typos.json"
    test_args = [
        "gentypos.py",
        "hello",
        "-c", empty_config_file,
        "--no-filter",
        "-o", str(output_json)
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    assert output_json.exists()
    content = output_json.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert isinstance(parsed, dict)
    assert len(parsed) > 0
    for typo, correction in parsed.items():
        assert correction == "hello"

def test_gentypos_auto_detect_yaml_extension(tmp_path, empty_config_file):
    output_yaml = tmp_path / "typos.yaml"
    test_args = [
        "gentypos.py",
        "hello",
        "-c", empty_config_file,
        "--no-filter",
        "-o", str(output_yaml)
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    assert output_yaml.exists()
    content = output_yaml.read_text(encoding="utf-8")

    if gentypos._YAML_AVAILABLE:
        import yaml
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0
        for typo, correction in parsed.items():
            assert correction == "hello"
    else:
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

def test_gentypos_yaml_fallback_when_yaml_unavailable(capsys):
    test_args = [
        "gentypos.py",
        "hello",
        "--no-filter",
        "-f", "yaml"
    ]
    # Patch parse_yaml_config to avoid loading YAML config, and _YAML_AVAILABLE to False to trigger fallback
    with patch("gentypos.parse_yaml_config", return_value={}), \
         patch("gentypos._YAML_AVAILABLE", False):
        with patch.object(sys, 'argv', test_args):
            gentypos.main()

    captured = capsys.readouterr()
    stdout_text = captured.out
    # Falling back to JSON
    parsed = json.loads(stdout_text)
    assert isinstance(parsed, dict)
    assert len(parsed) > 0
