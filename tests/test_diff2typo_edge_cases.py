import sys
from pathlib import Path
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo

def test_split_into_subwords_numeric_handling():
    """
    Test that split_into_subwords now preserves numbers that are part of alphanumeric words.
    """
    # word123 -> 'word', '123'
    assert diff2typo.split_into_subwords('word123') == ['word', '123']

    # 123word -> '123', 'word'
    assert diff2typo.split_into_subwords('123word') == ['123', 'word']

    # mid123dle -> 'mid', '123', 'dle'
    assert diff2typo.split_into_subwords('mid123dle') == ['mid', '123', 'dle']

def test_split_into_subwords_pure_numbers():
    """
    Test that pure numbers are preserved.
    """
    assert diff2typo.split_into_subwords('123') == ['123']

def test_split_into_subwords_acronyms():
    """
    Test splitting behavior for acronyms, specifically noting how IPv6 is split.
    """
    # IPv6 -> 'I', 'Pv', '6'
    assert diff2typo.split_into_subwords('IPv6') == ['I', 'Pv', '6']

    # HTMLParser -> 'HTML' matches [A-Z]+, 'Parser' matches [A-Z]?[a-z]+
    assert diff2typo.split_into_subwords('HTMLParser') == ['HTML', 'Parser']

def test_split_into_subwords_empty_string():
    """Test handling of empty string."""
    assert diff2typo.split_into_subwords('') == ['']

def test_find_typos_casing_mismatch_acronyms_ignored():
    """
    Verify that casing mismatches in acronyms that result in different splitting
    are ignored (no typo reported) because of length mismatch in comparison.
    """
    diff_text = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@\n"
        "-IPv6 support\n"
        "+Ipv6 support\n"
    )
    # IPv6 -> ['I', 'Pv', '6']
    # Ipv6 -> ['Ipv', '6']
    # Lengths 3 vs 2 -> comparison skipped
    assert diff2typo.find_typos(diff_text) == []

def test_find_typos_numeric_suffix_ignored():
    """
    Verify that changing only the numeric suffix of a word is ignored.
    """
    diff_text = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@\n"
        "-variable1\n"
        "+variable2\n"
    )
    # variable1 -> ['variable', '1']
    # variable2 -> ['variable', '2']
    # variable matches.
    # 1 != 2. Filtered out (not letters).
    # No typo.
    assert diff2typo.find_typos(diff_text) == []

def test_find_typos_numeric_context_preserved():
    """
    Verify that if letters change alongside a stable number, it IS detected.
    """
    diff_text = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@\n"
        "-var1\n"
        "+vrr1\n"
    )
    # var1 -> ['var', '1']
    # vrr1 -> ['vrr', '1']
    # var != vrr -> typo detected
    # 1 == 1 -> context preserved
    assert diff2typo.find_typos(diff_text) == ['var -> vrr']

def test_find_typos_underscore_splitting():
    """Test that underscores split words correctly."""
    diff_text = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@\n"
        "-snake_case\n"
        "+snake_kase\n"
    )
    # snake_case -> ['snake', 'case']
    # snake_kase -> ['snake', 'kase']
    # case -> kase typo
    assert diff2typo.find_typos(diff_text) == ['case -> kase']

def test_find_typos_misalignment_fixed():
    """
    Test that splitting including numbers prevents misalignment when a number is detached,
    allowing nearby typos to be found.
    """
    # word123 -> ['word', '123']
    # word 123 -> ['word', '123']
    # Alignment matches.
    diff_text = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@\n"
        "-word123 eror\n"
        "+word 123 error\n"
    )
    assert diff2typo.find_typos(diff_text) == ['eror -> error']
