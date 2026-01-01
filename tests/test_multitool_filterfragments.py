import sys
import time
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_filter_fragments_mode(tmp_path):
    list1 = tmp_path / "list1.txt"
    list1.write_text("apple\ncar\nplane\ncarpet\n")
    list2 = tmp_path / "list2.txt"
    list2.write_text("an applepie\ncarpeted floor\ncar\n")
    output_file = tmp_path / "output.txt"
    multitool.filter_fragments_mode([str(list1)], str(list2), str(output_file), 1, 10, True)
    assert output_file.read_text().splitlines() == ["plane"]

@pytest.mark.performance
def test_filter_fragments_mode_performance(tmp_path):
    # Create large temporary files for performance testing
    num_words = 1000

    list1_path = tmp_path / "perf_list1.txt"
    list2_path = tmp_path / "perf_list2.txt"
    output_path = tmp_path / "perf_output.txt"

    # Generate a list of unique words
    words1 = {f"word{i}" for i in range(num_words)}
    # Generate a larger set of words for the second list, ensuring some overlap
    words2 = {f"longerword{i}" for i in range(num_words * 2)}

    with open(list1_path, "w") as f:
        for word in words1:
            f.write(word + "\n")

    with open(list2_path, "w") as f:
        for word in words2:
            f.write(word + "\n")

    start_time = time.time()
    multitool.filter_fragments_mode([str(list1_path)], str(list2_path), str(output_path), 1, 20, True)
    end_time = time.time()

    duration = end_time - start_time
    print(f"filter_fragments_mode execution time: {duration:.4f} seconds")

    assert duration < 1.0

def test_filter_fragments_empty_input(tmp_path):
    """Test when the input file is empty."""
    list1 = tmp_path / "empty.txt"
    list1.write_text("")
    list2 = tmp_path / "list2.txt"
    list2.write_text("word\n")
    output_file = tmp_path / "output.txt"

    multitool.filter_fragments_mode([str(list1)], str(list2), str(output_file), 1, 10, True)

    assert output_file.read_text() == ""

def test_filter_fragments_empty_filter(tmp_path):
    """Test when the filter file is empty (should keep all inputs)."""
    # Note: 'word1' and 'word2' will be cleaned to 'word' and 'word' by filter_to_letters
    # and deduplicated if process_output=True.
    list1 = tmp_path / "list1.txt"
    list1.write_text("apple\nbanana\n")
    list2 = tmp_path / "empty.txt"
    list2.write_text("")
    output_file = tmp_path / "output.txt"

    multitool.filter_fragments_mode([str(list1)], str(list2), str(output_file), 1, 10, True)

    assert sorted(output_file.read_text().splitlines()) == ["apple", "banana"]

def test_filter_fragments_substring_behavior(tmp_path):
    """
    Verify strict substring directionality.
    Logic: Input word is removed if it appears INSIDE any word from File2.
    """
    list1 = tmp_path / "input.txt"
    list1.write_text("sub\nsuper\n")
    list2 = tmp_path / "filter.txt"
    list2.write_text("superman\n")

    output_file = tmp_path / "output.txt"
    multitool.filter_fragments_mode([str(list1)], str(list2), str(output_file), 1, 10, True)

    # 'super' should be removed (found in superman).
    # 'sub' should be kept.
    assert sorted(output_file.read_text().splitlines()) == ["sub"]

def test_filter_fragments_substring_reverse_case(tmp_path):
    """
    Verify that if Filter word is substring of Input word, Input is NOT removed.
    (Unless Input word contains another Filter word or itself).
    """
    list1 = tmp_path / "input.txt"
    list1.write_text("superman\n")
    list2 = tmp_path / "filter.txt"
    list2.write_text("super\n")

    # Is 'superman' a substring of 'super'? No.
    # So 'superman' should be kept.

    output_file = tmp_path / "output.txt"
    multitool.filter_fragments_mode([str(list1)], str(list2), str(output_file), 1, 20, True)

    assert output_file.read_text().splitlines() == ["superman"]
