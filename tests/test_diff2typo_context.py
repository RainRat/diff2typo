import sys
from pathlib import Path
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import diff2typo

def test_compare_word_lists_left_context_mismatch():
    """
    Test that a potential typo is rejected if the word preceding it
    does not match the word preceding the correction.
    """
    before = ['previous', 'typo', 'next']
    after = ['changed', 'correction', 'next']

    # 'previous' != 'changed', so we assume the whole block changed
    # and we shouldn't map 'typo' -> 'correction'.
    # This hits the `index > 0 and before_words[index - 1] != after_words[index - 1]` check.
    assert diff2typo._compare_word_lists(before, after, min_length=2) == []

def test_compare_word_lists_at_start():
    """
    Test finding a typo at the very start of the word list (index 0).
    Verifies that the absence of a left neighbor doesn't cause errors
    or false negatives.
    """
    before = ['typo', 'next']
    after = ['correction', 'next']

    # Logic: index=0. left check skipped. right check: 'next' == 'next'. Match.
    assert diff2typo._compare_word_lists(before, after, min_length=2) == ['typo -> correction']

def test_compare_word_lists_at_end():
    """
    Test finding a typo at the very end of the word list.
    Verifies that the absence of a right neighbor doesn't cause errors
    or false negatives.
    """
    before = ['previous', 'typo']
    after = ['previous', 'correction']

    # Logic: index=1 (len-1). right check skipped. left check: 'previous' == 'previous'. Match.
    assert diff2typo._compare_word_lists(before, after, min_length=2) == ['typo -> correction']

def test_compare_word_lists_single_word():
    """
    Test finding a typo when the list contains only that single word.
    """
    before = ['typo']
    after = ['correction']

    # Logic: index=0. left skipped. right skipped. Match.
    assert diff2typo._compare_word_lists(before, after, min_length=2) == ['typo -> correction']

def test_compare_word_lists_min_length_filtering():
    """
    Test that candidates shorter than min_length are filtered out.
    """
    before = ['a', 'longword']
    after = ['b', 'longword']

    # 'a' and 'b' are length 1. min_length=2 should filter them.
    assert diff2typo._compare_word_lists(before, after, min_length=2) == []

    # If min_length=1, it should be found
    assert diff2typo._compare_word_lists(before, after, min_length=1) == ['a -> b']

def test_compare_word_lists_multiple_typos_with_context():
    """
    Test finding multiple typos in a sequence where context is preserved locally.
    """
    # before: word1 typo1 word2 typo2 word3
    # after:  word1 corr1 word2 corr2 word3
    # Using 'typ1' -> 'typ' (len 3) to satisfy min_length=2
    before = ['word1', 'typ1', 'word2', 'typ2', 'word3']
    after = ['word1', 'cor1', 'word2', 'cor2', 'word3']

    results = diff2typo._compare_word_lists(before, after, min_length=2)

    # filter_to_letters removes digits, so 'typ1' -> 'typ', 'cor1' -> 'cor'
    assert results == ['typ -> cor', 'typ -> cor']
