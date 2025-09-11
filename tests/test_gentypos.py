import sys
import types
from pathlib import Path
import pytest

# Provide a minimal stub for the yaml module used by gentypos
sys.modules.setdefault('yaml', types.SimpleNamespace(safe_load=lambda stream: {}, YAMLError=Exception))

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_get_adjacent_keys():
    with_diag = gentypos.get_adjacent_keys(include_diagonals=True)
    without_diag = gentypos.get_adjacent_keys(include_diagonals=False)
    assert 'y' in with_diag['g']
    assert 'y' not in without_diag['g']
    assert 'h' in without_diag['g']

def test_load_custom_substitutions():
    custom = {'A': ['B', 'C'], 'd': ['E']}
    assert gentypos.load_custom_substitutions(custom) == {'a': {'b', 'c'}, 'd': {'e'}}

def test_generate_typos_by_replacement():
    adjacent = {'c': {'x', 'v'}, 'a': {'s'}, 't': {'r', 'y'}}
    custom = {'c': {'k'}}
    typos = gentypos.generate_typos_by_replacement('cat', adjacent, custom)
    assert typos == {'xat', 'vat', 'cst', 'car', 'cay', 'kat'}

def test_generate_variations():
    typos = gentypos.generate_variations('word', {'deletion': True, 'transposition': True})
    assert typos == {'ord', 'wrd', 'wod', 'owrd', 'wrod', 'wodr'}

def test_generate_typos_by_duplication():
    typos = gentypos.generate_typos_by_duplication('cat', {'duplication': True})
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
