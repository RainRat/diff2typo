import json
import sys
import types
from pathlib import Path

import pytest

# Provide a minimal stub for the yaml module used by gentypos
sys.modules.setdefault('yaml', types.SimpleNamespace(safe_load=lambda stream: {}, YAMLError=Exception))

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_load_custom_substitutions():
    custom = {'A': ['B', 'C'], 'd': ['E']}
    assert gentypos.load_custom_substitutions(custom) == {'a': {'b', 'c'}, 'd': {'e'}}

def test_generate_typos_by_replacement():
    adjacent = {'c': {'x', 'v'}, 'a': {'s'}, 't': {'r', 'y'}}
    custom = {'c': {'k'}}
    typos = gentypos.generate_typos_by_replacement('cat', adjacent, custom)
    assert typos == {'xat', 'vat', 'cst', 'car', 'cay', 'kat'}

def test_generate_typos_by_deletion():
    typos = gentypos.generate_typos_by_deletion('word')
    assert typos == {'ord', 'wrd', 'wod'}

def test_generate_typos_by_transposition():
    typos = gentypos.generate_typos_by_transposition('word', distance=1)
    assert typos == {'owrd', 'wrod', 'wodr'}

def test_generate_typos_by_duplication():
    typos = gentypos.generate_typos_by_duplication('cat')
    assert typos == {'ccat', 'caat', 'catt'}

def test_generate_all_typos():
    adjacent = {'c': {'v'}, 'a': {'s'}, 't': {'y'}}
    custom = {'a': {'@'}}
    typo_types = {'deletion': True, 'transposition': True, 'replacement': True, 'duplication': True}
    result = gentypos.generate_all_typos('cat', adjacent, custom, typo_types, transposition_distance=1, use_adjacent=True, use_custom=True)
    assert result == {'at', 'ct', 'ca', 'act', 'cta', 'vat', 'cst', 'c@t', 'cay', 'ccat', 'caat', 'catt'}

def test_format_typos():
    mapping = {'teh': 'the'}
    assert gentypos.format_typos(mapping, 'arrow') == ['teh -> the']
    assert gentypos.format_typos(mapping, 'csv') == ['teh,the']
    assert gentypos.format_typos(mapping, 'table') == ['teh = "the"']
    assert gentypos.format_typos(mapping, 'list') == ['teh']


def test_load_file_missing(tmp_path):
    with pytest.raises(SystemExit):
        gentypos.load_file(str(tmp_path / 'missing.txt'))


def test_parse_yaml_config_missing(tmp_path):
    with pytest.raises(SystemExit):
        gentypos.parse_yaml_config(str(tmp_path / 'missing.yaml'))


def test_parse_yaml_config_malformed(tmp_path, monkeypatch):
    bad_file = tmp_path / 'bad.yaml'
    bad_file.write_text('::bad yaml::')

    def bad_loader(stream):
        raise gentypos.yaml.YAMLError('boom')

    monkeypatch.setattr(gentypos.yaml, 'safe_load', bad_loader)

    with pytest.raises(SystemExit):
        gentypos.parse_yaml_config(str(bad_file))


def test_validate_config_missing_field():
    config = {
        'dictionary_file': 'dict.txt',
        'output_file': 'out.txt',
        'output_format': 'list',
        'typo_types': {},
    }

    with pytest.raises(SystemExit):
        gentypos.validate_config(config)


def test_run_typo_generation_filters_existing_words(monkeypatch):
    monkeypatch.setattr(gentypos, 'tqdm', lambda iterable, *_, **__: iterable)

    settings = types.SimpleNamespace(
        min_length=1,
        max_length=None,
        typo_types={'deletion': False, 'transposition': False, 'replacement': True, 'duplication': False},
        transposition_distance=1,
        repeat_modifications=1,
        enable_adjacent_substitutions=True,
        enable_custom_substitutions=False,
    )

    adjacent_keys = {'c': {'b', 'x'}, 'a': set(), 't': set()}

    result = gentypos._run_typo_generation(['cat'], {'bat'}, settings, adjacent_keys, {})

    assert 'bat' not in result
    assert result == {'xat': 'cat'}


def test_main_integration_success(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gentypos, 'tqdm', lambda iterable, *_, **__: iterable)

    input_file = tmp_path / 'input.txt'
    input_file.write_text('cat\n')

    dictionary_file = tmp_path / 'dictionary.txt'
    dictionary_file.write_text('dog\n')

    output_file = tmp_path / 'output.txt'

    config_data = {
        'input_file': str(input_file),
        'dictionary_file': str(dictionary_file),
        'output_file': str(output_file),
        'output_format': 'list',
        'typo_types': {
            'duplication': True,
            'deletion': False,
            'transposition': False,
            'replacement': False,
        },
        'word_length': {'min_length': 3, 'max_length': None},
    }

    config_file = tmp_path / 'config.yaml'
    config_file.write_text(json.dumps(config_data))

    def fake_safe_load(stream):
        return json.load(stream)

    monkeypatch.setattr(gentypos.yaml, 'safe_load', fake_safe_load)
    monkeypatch.setattr(sys, 'argv', ['gentypos.py', '--config', str(config_file)])

    gentypos.main()

    assert output_file.read_text().splitlines() == ['caat', 'catt', 'ccat']


def test_merge_defaults_recursive_merge():
    config = {
        'section': {
            'custom': 10
        }
    }
    defaults = {
        'section': {
            'custom': 1,
            'default': 2
        },
        'other': 3
    }

    gentypos._merge_defaults(config, defaults)

    assert config['section']['custom'] == 10
    assert config['section']['default'] == 2
    assert config['other'] == 3


def test_merge_defaults_type_mismatch():
    config = {'section': 'scalar'}
    defaults = {'section': {'sub': 1}}

    with pytest.raises(SystemExit):
        gentypos._merge_defaults(config, defaults)
