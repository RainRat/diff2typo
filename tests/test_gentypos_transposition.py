import sys
from pathlib import Path

import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import gentypos

def test_transposition_distance_two():
    # word: "1234"
    # distance=2
    # i=0: swap word[0] ('1') and word[2] ('3'). Middle: word[1:2] ('2').
    # Result: '3' + '2' + '1' + '4' = "3214"
    # i=1: swap word[1] ('2') and word[3] ('4'). Middle: word[2:3] ('3').
    # Result: '1' + '4' + '3' + '2' = "1432"
    typos = gentypos.generate_typos_by_transposition("1234", distance=2)
    assert typos == {"3214", "1432"}

def test_transposition_distance_equal_length():
    # word: "abc" (len=3), distance=3
    # Loop: range(len - distance) -> range(0). Empty.
    typos = gentypos.generate_typos_by_transposition("abc", distance=3)
    assert typos == set()

def test_transposition_distance_greater_than_length():
    typos = gentypos.generate_typos_by_transposition("abc", distance=5)
    assert typos == set()

def test_transposition_skips_identical_chars():
    # word: "aba"
    # distance=2
    # i=0: swap word[0] ('a') and word[2] ('a'). Identical -> skip.
    typos = gentypos.generate_typos_by_transposition("aba", distance=2)
    assert typos == set()

def test_transposition_distance_clamped_to_one():
    # distance=0 should be treated as 1
    # word: "ab" -> "ba"
    typos = gentypos.generate_typos_by_transposition("ab", distance=0)
    assert typos == {"ba"}

    # distance=-1 should be treated as 1
    typos = gentypos.generate_typos_by_transposition("ab", distance=-1)
    assert typos == {"ba"}

def test_transposition_complex_word():
    # word: "hello"
    # distance=2
    # i=0: swap 'h', 'l' (index 2). 'l' + 'e' + 'h' + 'l' + 'o' = "lehlo"
    # i=1: swap 'e', 'l' (index 3). 'h' + 'l' + 'l' + 'e' + 'o' = "hlleo"
    # i=2: swap 'l', 'o' (index 4). 'h' + 'e' + 'o' + 'l' + 'l' = "heoll"
    typos = gentypos.generate_typos_by_transposition("hello", distance=2)
    assert typos == {"lehlo", "hlleo", "heoll"}
