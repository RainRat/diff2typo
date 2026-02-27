import json
import sys
from pathlib import Path
import pytest
import yaml

# Ensure the root directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_format_typos_json_yaml():
    mapping = {'teh': 'the'}
    assert gentypos.format_typos(mapping, 'json') == {'teh': 'the'}
    assert gentypos.format_typos(mapping, 'yaml') == {'teh': 'the'}

def test_json_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.json"

    # Run gentypos via main with JSON format
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'hello', '--format', 'json', '--no-filter', '--output', str(output_file)])

    # Mock tqdm
    monkeypatch.setattr(gentypos, 'tqdm', lambda iterable, *_, **__: iterable)

    gentypos.main()

    with open(output_file, 'r') as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert any('hello' in v or k in ('helo', 'hllo') for k, v in data.items())

def test_yaml_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.yaml"

    # Run gentypos via main with YAML format
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'hello', '--format', 'yaml', '--no-filter', '--output', str(output_file)])

    # Mock tqdm
    monkeypatch.setattr(gentypos, 'tqdm', lambda iterable, *_, **__: iterable)

    gentypos.main()

    with open(output_file, 'r') as f:
        content = f.read()
        data = yaml.safe_load(content)

    assert isinstance(data, dict)
    assert any('hello' in v or k in ('helo', 'hllo') for k, v in data.items())
    assert content.endswith('\n')

def test_json_header_suppression(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.json"

    config_data = {
        'input_file': None,
        'dictionary_file': None,
        'output_file': str(output_file),
        'output_format': 'json',
        'output_header': 'THIS SHOULD NOT APPEAR'
    }

    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        json.dump(config_data, f)

    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'cat', '--config', str(config_file)])
    monkeypatch.setattr(gentypos, 'tqdm', lambda iterable, *_, **__: iterable)

    gentypos.main()

    with open(output_file, 'r') as f:
        content = f.read()

    # Should be valid JSON, meaning no header
    data = json.loads(content)
    assert isinstance(data, dict)
    assert 'THIS SHOULD NOT APPEAR' not in content
