import subprocess
import os
import pytest

def run_multitool(args, input_text=None):
    cmd = ["python", "multitool.py"] + args
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=input_text)
    return stdout, stderr, process.returncode

def test_repeated_basic():
    input_text = "the the quick quick brown fox"
    stdout, stderr, code = run_multitool(["repeated", "-f", "arrow"], input_text=input_text)
    assert code == 0
    assert "the the -> the" in stdout
    assert "quick quick -> quick" in stdout

def test_repeated_case_insensitive_default():
    input_text = "The the Quick quick"
    stdout, stderr, code = run_multitool(["repeated", "-f", "arrow"], input_text=input_text)
    assert code == 0
    # Output should be lowercase because clean_items=True by default
    assert "the the -> the" in stdout
    assert "quick quick -> quick" in stdout

def test_repeated_case_sensitive_raw():
    input_text = "The the Quick quick"
    stdout, stderr, code = run_multitool(["repeated", "-f", "arrow", "--raw"], input_text=input_text)
    assert code == 0
    # "The the" should NOT match because matching is case-sensitive with --raw
    assert "The the" not in stdout
    # "quick quick" should still match
    # Wait, in my implementation:
    # match_word = filter_to_letters(word) if clean_items else word
    # If clean_items is False (--raw), match_word is exact word.
    # So "The" != "the".
    # But "quick" == "quick" matches.
    # Actually, "Quick" and "quick" don't match.

    input_text_2 = "the the Quick Quick"
    stdout, stderr, code = run_multitool(["repeated", "-f", "arrow", "--raw"], input_text=input_text_2)
    assert "the the -> the" in stdout
    assert "Quick Quick -> Quick" in stdout

def test_repeated_smart_split():
    input_text = "doubled doubledWord"
    # Without --smart, "doubledWord" is one word.
    stdout, stderr, code = run_multitool(["repeated", "-f", "arrow"], input_text=input_text)
    assert "doubled doubled -> doubled" not in stdout

    # With --smart, "doubledWord" becomes ["doubled", "Word"]
    stdout, stderr, code = run_multitool(["repeated", "-f", "arrow", "--smart"], input_text=input_text)
    assert "doubled doubled -> doubled" in stdout

def test_repeated_min_length():
    input_text = "a a the the"
    # Default min-length is 3
    stdout, stderr, code = run_multitool(["repeated", "-f", "arrow"], input_text=input_text)
    assert "a a -> a" not in stdout
    assert "the the -> the" in stdout

    # Set min-length to 1
    stdout, stderr, code = run_multitool(["repeated", "-f", "arrow", "-m", "1"], input_text=input_text)
    assert "a a -> a" in stdout

def test_repeated_across_lines():
    input_text = "the\nthe"
    stdout, stderr, code = run_multitool(["repeated", "-f", "arrow"], input_text=input_text)
    assert "the the -> the" in stdout

def test_repeated_csv_output():
    input_text = "the the"
    stdout, stderr, code = run_multitool(["repeated", "-f", "csv"], input_text=input_text)
    assert "the the,the" in stdout
