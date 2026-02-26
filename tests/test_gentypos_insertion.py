import sys
from pathlib import Path
import pytest

# Ensure the root directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_generate_typos_by_insertion():
    # 'a' is adjacent to 's' and 'w' in most layouts, but let's define it explicitly
    adjacent = {'a': {'s', 'w'}, 't': {'y'}}
    word = 'at'
    # Possible insertions:
    # pos 0: sat, wat (adjacent to 'a')
    # pos 1: ast, awt (adjacent to 'a'), ayt (adjacent to 't')
    # pos 2: aty (adjacent to 't')

    typos = gentypos.generate_typos_by_insertion(word, adjacent, use_custom=False)
    expected = {'sat', 'wat', 'ast', 'awt', 'ayt', 'aty'}
    assert typos == expected

def test_insertion_enabled_in_all_typos():
    adjacent = {'a': {'s'}}
    typo_types = {'insertion': True}
    # generate_all_typos(word, adjacent_keys, custom_subs, typo_types, ...)
    result = gentypos.generate_all_typos('a', adjacent, {}, typo_types)
    # word 'a', adjacent 's'
    # pos 0: sa
    # pos 1: as
    assert 'sa' in result
    assert 'as' in result

def test_insertion_disabled_by_default():
    adjacent = {'a': {'s'}}
    # If typo_types is empty or doesn't have 'insertion': True
    typo_types = {'insertion': False}
    result = gentypos.generate_all_typos('a', adjacent, {}, typo_types)
    assert 'sa' not in result
    assert 'as' not in result

def test_insertion_with_custom_subs():
    custom = {'a': {'@'}}
    word = 'a'
    typos = gentypos.generate_typos_by_insertion(word, {}, custom_subs=custom, use_adjacent=False)
    # pos 0: @a
    # pos 1: a@
    assert typos == {'@a', 'a@'}
