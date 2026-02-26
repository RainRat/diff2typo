import sys
from pathlib import Path
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from gentypos import generate_typos_by_replacement, load_custom_substitutions

def test_multi_char_substring_replacement_basic():
    custom_subs = {"ph": {"f"}}
    typos = generate_typos_by_replacement(
        "alphabet",
        adjacent_keys={},
        custom_subs=custom_subs,
        use_adjacent=False,
        use_custom=True
    )
    assert "alfabet" in typos
    assert len(typos) == 1

def test_multi_char_substring_replacement_multiple_occurrences():
    # "banana" has "na" at index 2 and 4.
    custom_subs = {"na": {"ba"}}
    typos = generate_typos_by_replacement(
        "banana",
        adjacent_keys={},
        custom_subs=custom_subs,
        use_adjacent=False,
        use_custom=True
    )
    # Each occurrence should be replaced individually
    assert "babana" in typos # index 2
    assert "banaba" in typos # index 4
    assert len(typos) == 2

def test_multi_char_substring_replacement_overlapping():
    # "aaaa" with "aa" -> "x"
    # Overlapping indices for "aa": 0, 1, 2
    custom_subs = {"aa": {"x"}}
    typos = generate_typos_by_replacement(
        "aaaa",
        adjacent_keys={},
        custom_subs=custom_subs,
        use_adjacent=False,
        use_custom=True
    )
    assert "xaa" in typos # index 0
    assert "axa" in typos # index 1
    assert "aax" in typos # index 2
    assert len(typos) == 3

def test_load_custom_substitutions_non_iterable_value():
    # This should trigger the TypeError block and handle the value as a single item
    custom_subs_raw = {"a": 123}
    subs = load_custom_substitutions(custom_subs_raw)
    assert subs["a"] == {"123"}

def test_multi_char_substring_replacement_multiple_values():
    custom_subs = {"ph": {"f", "v"}}
    typos = generate_typos_by_replacement(
        "phone",
        adjacent_keys={},
        custom_subs=custom_subs,
        use_adjacent=False,
        use_custom=True
    )
    assert "fone" in typos
    assert "vone" in typos
    assert len(typos) == 2
