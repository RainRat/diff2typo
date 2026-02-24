import pytest
import sys
from pathlib import Path
import io

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_levenshtein_distance():
    """Verify Levenshtein distance calculation for various scenarios."""
    assert multitool.levenshtein_distance("kitten", "sitting") == 3
    assert multitool.levenshtein_distance("flaw", "lawn") == 2
    assert multitool.levenshtein_distance("gumbo", "gambol") == 2
    assert multitool.levenshtein_distance("", "") == 0
    assert multitool.levenshtein_distance("abc", "") == 3
    assert multitool.levenshtein_distance("", "abc") == 3
    assert multitool.levenshtein_distance("abc", "abc") == 0
    assert multitool.levenshtein_distance("abc", "ab") == 1
    assert multitool.levenshtein_distance("ab", "abc") == 1
    # Test longer first string (branch coverage)
    # "longerstring" (12) vs "short" (5).
    # Distance is 10 (Standard Levenshtein)
    assert multitool.levenshtein_distance("longerstring", "short") == 10

def test_similarity_mode_basic(tmp_path):
    """Verify basic similarity mode filtering."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple -> apple\napple -> apples\napple -> apply\napple -> banana\n")
    output_file = tmp_path / "output.txt"

    # Default min_dist=0, so all should pass if length allows
    multitool.similarity_mode([str(input_file)], str(output_file), 1, 100, False, min_dist=0)

    content = output_file.read_text().splitlines()
    assert "apple -> apple" in content
    assert "apple -> apples" in content
    assert "apple -> apply" in content
    assert "apple -> banana" in content

def test_similarity_mode_min_dist(tmp_path):
    """Verify min_dist filter in similarity mode."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple -> apple\napple -> apples\napple -> apply\n")
    output_file = tmp_path / "output.txt"

    # distance apple/apple: 0
    # distance apple/apples: 1
    # distance apple/apply: 1
    multitool.similarity_mode([str(input_file)], str(output_file), 1, 100, False, min_dist=1)
    content = output_file.read_text().splitlines()
    assert "apple -> apple" not in content # distance 0
    assert "apple -> apples" in content # distance 1
    assert "apple -> apply" in content # distance 1

def test_similarity_mode_max_dist(tmp_path):
    """Verify max_dist filter in similarity mode."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple -> apple\napple -> apples\napple -> banana\n")
    output_file = tmp_path / "output.txt"

    multitool.similarity_mode([str(input_file)], str(output_file), 1, 100, False, max_dist=1)
    content = output_file.read_text().splitlines()
    assert "apple -> apple" in content # dist 0
    assert "apple -> apples" in content # dist 1
    assert "apple -> banana" not in content # dist > 1

def test_similarity_mode_show_dist(tmp_path):
    """Verify show_dist flag appends distance to output."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple -> apples\n")
    output_file = tmp_path / "output.txt"

    multitool.similarity_mode([str(input_file)], str(output_file), 1, 100, False, show_dist=True)
    content = output_file.read_text().strip()
    assert "apple -> apples (changes: 1)" == content

def test_similarity_mode_raw(tmp_path):
    """Verify clean_items=False (raw) disables character cleaning."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("Apple -> apples\n")
    output_file = tmp_path / "output.txt"

    # With clean_items=True (default), "Apple" -> "apple", distance is 1
    multitool.similarity_mode([str(input_file)], str(output_file), 1, 100, False, clean_items=True, max_dist=1)
    assert "apple -> apples" in output_file.read_text()

    # With clean_items=False, "Apple" vs "apples", distance is 2 (A->a, +s)
    multitool.similarity_mode([str(input_file)], str(output_file), 1, 100, False, clean_items=False, max_dist=1)
    # The output should NOT contain "Apple -> apples" because dist is 2 and max_dist is 1
    assert "Apple -> apples" not in output_file.read_text()

def test_similarity_mode_length_filtering(tmp_path):
    """Verify length filtering in similarity mode."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> ab\napple -> apples\n")
    output_file = tmp_path / "output.txt"

    multitool.similarity_mode([str(input_file)], str(output_file), 3, 100, False)
    content = output_file.read_text().splitlines()
    assert "a -> ab" not in content
    assert "apple -> apples" in content

def test_similarity_cli(tmp_path, monkeypatch, capsys):
    """Verify CLI integration for similarity mode."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh -> the\napple -> banana\n")
    output_file = tmp_path / "output.txt"

    # distance teh/the is 2.
    monkeypatch.setattr(sys, 'argv', [
        'multitool.py', 'similarity', str(input_file),
        '--output', str(output_file),
        '--max-dist', '2',
        '--show-dist',
        '--quiet'
    ])

    multitool.main()

    content = output_file.read_text()
    assert "teh -> the (changes: 2)" in content
    assert "apple -> banana" not in content

def test_similarity_mode_empty_input(tmp_path):
    """Verify similarity mode handles empty input correctly."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("")
    output_file = tmp_path / "output.txt"

    multitool.similarity_mode([str(input_file)], str(output_file), 1, 100, False)
    assert output_file.read_text().strip() == ""

def test_similarity_mode_process_output(tmp_path):
    """Verify deduplication when process_output=True."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh -> the\nteh -> the\n")
    output_file = tmp_path / "output.txt"

    # process_output=True should deduplicate
    multitool.similarity_mode([str(input_file)], str(output_file), 1, 100, True)
    content = output_file.read_text().strip().splitlines()
    assert len(content) == 1
    assert "teh -> the" in content[0]

def test_similarity_mode_invalid_cleaning(tmp_path):
    """Verify items that become empty after cleaning are skipped."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("123 -> 456\napple -> apples\n")
    output_file = tmp_path / "output.txt"

    multitool.similarity_mode([str(input_file)], str(output_file), 1, 100, False)
    content = output_file.read_text().splitlines()
    # "123" becomes empty string after filter_to_letters
    assert not any("123" in line for line in content)
    assert any("apple -> apples" in line for line in content)
