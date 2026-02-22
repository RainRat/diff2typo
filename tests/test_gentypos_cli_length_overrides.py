import sys
import logging
import pytest
from pathlib import Path
from unittest.mock import patch
import yaml

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(gentypos, "tqdm", lambda iterable, *_, **__: iterable)

def test_min_length_cli_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.txt"
    # 'cat' is length 3. If we set --min-length 4, it should produce nothing.
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'cat', '--min-length', '4', '--output', str(output_file), '--no-filter'])
    gentypos.main()
    content = output_file.read_text().strip()
    assert content == ""

def test_max_length_cli_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.txt"
    # 'cat' is length 3. If we set --max-length 2, it should produce nothing.
    # Note: 'cat' typos like 'at' (len 2) would normally be generated,
    # but filtering happens on the INPUT words first, and then on typos.
    # Actually, let's check gentypos logic.
    # It filters input words by length, and THEN generates typos and filters them too.
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', 'cat', '--max-length', '2', '--output', str(output_file), '--no-filter'])
    gentypos.main()
    content = output_file.read_text().strip()
    assert content == ""

def test_length_overrides_with_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple\n") # len 5
    output_file = tmp_path / "output.txt"

    config = {
        'input_file': str(input_file),
        'dictionary_file': None,
        'output_file': 'dummy.txt',
        'output_format': 'list',
        'word_length': {'min_length': 1, 'max_length': 10}
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))

    # Override config with CLI: min-length 6. 'apple' (5) should be filtered out.
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', '--config', str(config_file), '--min-length', '6', '--output', str(output_file)])
    gentypos.main()
    assert output_file.read_text().strip() == ""

def test_max_length_override_with_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple\n") # len 5
    output_file = tmp_path / "output.txt"

    config = {
        'input_file': str(input_file),
        'dictionary_file': None,
        'output_file': 'dummy.txt',
        'output_format': 'list',
        'word_length': {'min_length': 1, 'max_length': 10}
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))

    # Override config with CLI: max-length 4. 'apple' (5) should be filtered out.
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', '--config', str(config_file), '--max-length', '4', '--output', str(output_file)])
    gentypos.main()
    assert output_file.read_text().strip() == ""
