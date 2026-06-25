import os
import pytest
from multitool import standardize_mode

def test_standardize_transitive_fuzzy(tmp_path):
    # Word A (freq 100), Word B (freq 10), Word C (freq 1)
    # dist(A,B)=1, dist(B,C)=1, dist(A,C)=2
    # threshold = 10.0, fuzzy=1
    content = (["wordaaaa "] * 100) + (["wordaaab "] * 10) + (["wordaabb "] * 1)
    test_file = tmp_path / "test.txt"
    test_file.write_text("".join(content))

    output_file = tmp_path / "output.txt"

    # Run with fuzzy=1
    standardize_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        quiet=True,
        clean_items=True,
        fuzzy=1,
        threshold=10.0
    )

    result = output_file.read_text()
    # "wordaabb" -> "wordaaab" -> "wordaaaa"
    assert "wordaaab" not in result
    assert "wordaabb" not in result
    assert result.count("wordaaaa") == 111

def test_standardize_casing_in_fuzzy_block(tmp_path):
    # fuzzy > 0, but no fuzzy match found for this word
    # Only casing variations: "Apple" (10), "apple" (1)
    content = (["Apple "] * 10) + (["apple "] * 1)
    test_file = tmp_path / "test.txt"
    test_file.write_text("".join(content))

    output_file = tmp_path / "output.txt"

    # Run with fuzzy=1
    standardize_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        quiet=True,
        clean_items=True,
        fuzzy=1
    )

    result = output_file.read_text()
    # "apple" should become "Apple"
    assert "apple " not in result
    assert result.count("Apple") == 11

def test_standardize_typo_filters(tmp_path):
    # Test --keyboard and --transposition filters in standardize mode
    # Keyboard typo: 'w' -> 'e' (adjacent)
    # Transposition: 'ab' -> 'ba'
    content = (["word "] * 100) + (["eord "] * 10) + (["wrod "] * 10)
    test_file = tmp_path / "test.txt"
    test_file.write_text("".join(content))

    output_file = tmp_path / "output.txt"

    # Run with --keyboard
    standardize_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        quiet=True,
        clean_items=True,
        fuzzy=1,
        keyboard=True
    )
    result = output_file.read_text()
    assert "eord" not in result
    assert "wrod" in result # Not a keyboard typo

    # Run with --transposition
    standardize_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        quiet=True,
        clean_items=True,
        fuzzy=1,
        transposition=True
    )
    result = output_file.read_text()
    assert "wrod" not in result
    assert "eord" in result # Not a transposition
