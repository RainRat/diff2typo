import sys
from pathlib import Path
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo

def test_split_into_subwords_numeric_handling():
    """
    Test that split_into_subwords drops numbers that are part of alphanumeric words
    due to the regex used for CamelCase splitting.
    """
    # word123 -> 'word' matches [a-z]+, '123' is ignored
    assert diff2typo.split_into_subwords('word123') == ['word']

    # 123word -> 'word' matches [a-z]+, '123' is ignored
    assert diff2typo.split_into_subwords('123word') == ['word']

    # mid123dle -> 'mid' matches [a-z]+, 'dle' matches [a-z]+
    assert diff2typo.split_into_subwords('mid123dle') == ['mid', 'dle']

def test_split_into_subwords_pure_numbers():
    """
    Test that pure numbers are preserved because the regex finds no match,
    triggering the fallback to append the original part.
    """
    assert diff2typo.split_into_subwords('123') == ['123']

def test_split_into_subwords_acronyms():
    """
    Test splitting behavior for acronyms, specifically noting how IPv6 is split.
    """
    # IPv6 -> 'I' matches [A-Z]+ (followed by P), 'Pv' matches [A-Z]?[a-z]+.
    # '6' is ignored.
    assert diff2typo.split_into_subwords('IPv6') == ['I', 'Pv']

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
    # IPv6 -> ['I', 'Pv']
    # Ipv6 -> ['Ipv']
    # Lengths 2 vs 1 -> comparison skipped
    assert diff2typo.find_typos(diff_text) == []

def test_find_typos_numeric_suffix_ignored():
    """
    Verify that changing only the numeric suffix of a word is ignored
    because the numeric part is dropped during splitting.
    """
    diff_text = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@\n"
        "-variable1\n"
        "+variable2\n"
    )
    # variable1 -> ['variable']
    # variable2 -> ['variable']
    # Match -> no typo
    assert diff2typo.find_typos(diff_text) == []

def test_find_typos_numeric_change_detected_with_letter_change():
    """
    Verify that if letters change alongside numbers, it IS detected.
    """
    diff_text = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@\n"
        "-var1\n"
        "+vrr2\n"
    )
    # var1 -> ['var']
    # vrr2 -> ['vrr']
    # var != vrr -> typo detected
    # Output format is "before -> after" (lowercased)
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
