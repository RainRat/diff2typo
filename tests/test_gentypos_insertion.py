import pytest
from gentypos import generate_typos_by_insertion, get_adjacent_keys, generate_all_typos
import subprocess
import sys

def test_generate_typos_by_insertion_basic():
    word = "a"
    adjacent_keys = {"a": {"s", "q", "z"}}
    # Insertion can happen before 'a' or after 'a'
    # Before 'a': sa, qa, za
    # After 'a': as, aq, az
    expected = {"sa", "qa", "za", "as", "aq", "az"}
    result = generate_typos_by_insertion(word, adjacent_keys, custom_subs={})
    assert result == expected

def test_generate_typos_by_insertion_with_custom_subs():
    word = "a"
    adjacent_keys = {}
    custom_subs = {"a": {"e", "i"}}
    # Before 'a': ea, ia
    # After 'a': ae, ai
    expected = {"ea", "ia", "ae", "ai"}
    result = generate_typos_by_insertion(word, adjacent_keys, custom_subs=custom_subs)
    assert result == expected

def test_generate_all_typos_includes_insertion():
    word = "a"
    adjacent_keys = {"a": {"s"}}
    custom_subs = {}
    # Only insertion enabled
    typo_types = {"deletion": False, "transposition": False, "replacement": False, "duplication": False, "insertion": True}
    result = generate_all_typos(word, adjacent_keys, custom_subs, typo_types)
    assert "sa" in result
    assert "as" in result
    assert "a" not in result # base word not a typo

def test_cli_include_insertions():
    # Run gentypos.py via subprocess to test CLI integration
    # We'll use a simple word and check if any expected insertion typo is in the output
    # 'w' is next to 'e' and 's' on QWERTY
    word = "w"
    result = subprocess.run(
        [sys.executable, "gentypos.py", word, "-i", "--no-filter", "--format", "list", "-m", "0", "-o", "-"],
        capture_output=True,
        text=True
    )
    output = result.stdout.splitlines()
    # 'ew' or 'we' should be in output if insertion is working
    assert any(t in output for t in ["ew", "we", "sw", "ws"])

def test_insertion_disabled_by_default():
    word = "w"
    result = subprocess.run(
        [sys.executable, "gentypos.py", word, "--no-filter", "--format", "list", "-m", "0", "-o", "-"],
        capture_output=True,
        text=True
    )
    output = result.stdout.splitlines()
    # By default, deletion, transposition, replacement, duplication are ON.
    # Insertion is OFF.
    # For a single letter "w":
    # Deletion: "" (but gentypos might skip empty)
    # Transposition: none
    # Replacement: "e", "s", "q" etc
    # Duplication: "ww"
    # Insertion (if it WERE on): "ew", "we" etc.
    assert "ww" in output
    assert not any(t in output for t in ["ew", "we", "sw", "ws"])
