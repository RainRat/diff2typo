import pytest
from unittest.mock import patch, MagicMock
import multitool

def test_classify_typo_logic():
    """Test the core classification logic for various typo types."""
    adj_keys = multitool.get_adjacent_keys()

    # Transposition [T]
    assert multitool.classify_typo("teh", "the", adj_keys) == "[T]"
    # Deletion [D]
    assert multitool.classify_typo("helo", "hello", adj_keys) == "[D]"
    # Insertion [I]
    assert multitool.classify_typo("helloo", "hello", adj_keys) == "[I]"
    # Keyboard [K] (k and l are adjacent)
    assert multitool.classify_typo("helko", "hello", adj_keys) == "[K]"
    # Replacement [R] (a and o are not adjacent)
    assert multitool.classify_typo("hella", "hello", adj_keys) == "[R]"
    # Multi-character [M]
    assert multitool.classify_typo("abc", "def", adj_keys) == "[M]"

def test_classify_mode_basic():
    """Test that classify_mode correctly processes pairs and calls output functions."""
    input_pairs = [
        ("teh", "the"),
        ("helo", "hello"),
        ("helloo", "hello"),
        ("hella", "hello"),
        ("helko", "hello"),
        ("abc", "def"),
    ]

    with patch("multitool._extract_pairs", return_value=input_pairs), \
         patch("multitool._write_paired_output") as mock_write, \
         patch("multitool.print_processing_stats"):

        multitool.classify_mode(
            input_files=["dummy.txt"],
            output_file="out.txt",
            min_length=2,
            max_length=100,
            process_output=False
        )

        # Verify results passed to _write_paired_output
        args, kwargs = mock_write.call_args
        results = args[0]

        results_dict = dict(results)
        assert results_dict["teh"] == "the [T]"
        assert results_dict["helo"] == "hello [D]"
        assert results_dict["helloo"] == "hello [I]"
        assert results_dict["hella"] == "hello [R]"
        assert results_dict["helko"] == "hello [K]"
        assert results_dict["abc"] == "def [M]"

def test_classify_mode_show_dist():
    """Test that --show-dist adds distance information in classify_mode."""
    input_pairs = [("teh", "the")]

    with patch("multitool._extract_pairs", return_value=input_pairs), \
         patch("multitool._write_paired_output") as mock_write, \
         patch("multitool.print_processing_stats"):

        multitool.classify_mode(
            input_files=["dummy.txt"],
            output_file="out.txt",
            min_length=2,
            max_length=100,
            process_output=False,
            show_dist=True
        )

        args, _ = mock_write.call_args
        results = dict(args[0])
        assert results["teh"] == "the [T] (dist: 2)"

def test_get_adjacent_keys_coverage():
    """Test get_adjacent_keys with and without diagonals for coverage."""
    adj_with = multitool.get_adjacent_keys(include_diagonals=True)
    adj_without = multitool.get_adjacent_keys(include_diagonals=False)

    # 's' is surrounded by 'q', 'w', 'e', 'a', 'd', 'z', 'x', 'c'
    # Diagonals for 's' are 'q', 'e', 'z', 'c'
    # Non-diagonals for 's' are 'w', 'a', 'd', 'x'

    assert 'q' in adj_with['s']
    assert 'q' not in adj_without['s']
    assert 'w' in adj_without['s']

def test_classify_typo_edge_cases():
    """Test edge cases in classify_typo for full coverage."""
    adj_keys = multitool.get_adjacent_keys()

    # Empty strings (lines 163-164)
    assert multitool.classify_typo("", "the", adj_keys) == "[?]"
    assert multitool.classify_typo("teh", "", adj_keys) == "[?]"

    # Identical strings (reaches line 202)
    assert multitool.classify_typo("the", "the", adj_keys) == "[?]"

    # Replacement [R] that is NOT keyboard adjacent (line 196)
    # 'q' and 'p' are not adjacent
    assert multitool.classify_typo("q", "p", adj_keys) == "[R]"

    # Multi-character [M] (line 199-200)
    assert multitool.classify_typo("abcd", "axcy", adj_keys) == "[M]"

def test_classify_mode_filtering():
    """Test filtering and cleaning in classify_mode (lines 1420-1430)."""
    input_pairs = [
        ("t", "the"),    # Too short
        ("teh", "t"),    # Too short
        ("!!!", "the"),  # Empty after cleaning
        ("teh", "!!!"),  # Empty after cleaning
    ]

    with patch("multitool._extract_pairs", return_value=input_pairs), \
         patch("multitool._write_paired_output") as mock_write, \
         patch("multitool.print_processing_stats"):

        multitool.classify_mode(
            input_files=["dummy.txt"],
            output_file="out.txt",
            min_length=3,
            max_length=10,
            process_output=False,
            clean_items=True
        )

        args, _ = mock_write.call_args
        results = args[0]
        assert len(results) == 0

def test_classify_mode_no_clean():
    """Test classify_mode with clean_items=False (lines 1424-1426)."""
    input_pairs = [("TeH", "the")]

    with patch("multitool._extract_pairs", return_value=input_pairs), \
         patch("multitool._write_paired_output") as mock_write, \
         patch("multitool.print_processing_stats"):

        multitool.classify_mode(
            input_files=["dummy.txt"],
            output_file="out.txt",
            min_length=2,
            max_length=100,
            process_output=False,
            clean_items=False
        )

        args, _ = mock_write.call_args
        results = dict(args[0])
        # "TeH" contains uppercase, filter_to_letters would make it "teh"
        # Since clean_items=False, it should stay "TeH"
        assert "TeH" in results

def test_classify_mode_process_output():
    """Test classify_mode with process_output=True (line 1441)."""
    # Duplicate pairs
    input_pairs = [("teh", "the"), ("teh", "the")]

    with patch("multitool._extract_pairs", return_value=input_pairs), \
         patch("multitool._write_paired_output") as mock_write, \
         patch("multitool.print_processing_stats"):

        multitool.classify_mode(
            input_files=["dummy.txt"],
            output_file="out.txt",
            min_length=2,
            max_length=100,
            process_output=True
        )

        args, _ = mock_write.call_args
        results = args[0]
        # Should be deduplicated and sorted
        assert len(results) == 1
        assert results[0] == ("teh", "the [T]")
