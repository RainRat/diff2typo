import pytest
from typostats import is_one_letter_replacement
from gentypos import generate_typos_by_replacement

def test_typostats_2_to_1_replacement():
    # Basic 2 -> 1
    assert is_one_letter_replacement("f", "ph", allow_two_char=True) == [("ph", "f")]
    # Context 2 -> 1
    assert is_one_letter_replacement("elefant", "elephant", allow_two_char=True) == [("ph", "f")]
    # Should not detect without flag
    assert is_one_letter_replacement("f", "ph", allow_two_char=False) == []

def test_gentypos_multi_char_substitution():
    adjacent = {}
    # 2 -> 1 substitution
    custom = {'ph': {'f'}}
    typos = generate_typos_by_replacement('elephant', adjacent, custom, use_adjacent=False, use_custom=True)
    assert 'elefant' in typos

    # 2 -> 2 substitution
    custom = {'ph': {'ff'}}
    typos = generate_typos_by_replacement('elephant', adjacent, custom, use_adjacent=False, use_custom=True)
    assert 'eleffant' in typos

    # 1 -> 2 substitution (already worked, but verifying consistency)
    custom = {'f': {'ph'}}
    typos = generate_typos_by_replacement('fast', adjacent, custom, use_adjacent=False, use_custom=True)
    assert 'phast' in typos

def test_gentypos_multi_char_multiple_occurrences():
    adjacent = {}
    custom = {'an': {'x'}}
    typos = generate_typos_by_replacement('banana', adjacent, custom, use_adjacent=False, use_custom=True)
    # banana -> bxana (first 'an'), banxa (second 'an')
    assert 'bxana' in typos
    assert 'banxa' in typos

def test_gentypos_overlapping_substitutions():
    adjacent = {}
    # If we have overlapping keys, both should be applied (one at a time)
    custom = {'ab': {'1'}, 'bc': {'2'}}
    typos = generate_typos_by_replacement('abc', adjacent, custom, use_adjacent=False, use_custom=True)
    assert '1c' in typos # ab -> 1
    assert 'a2' in typos # bc -> 2
