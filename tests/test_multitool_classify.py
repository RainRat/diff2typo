from unittest.mock import MagicMock
import sys
from pathlib import Path
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

@pytest.fixture(autouse=True)
def reset_stdin_cache():
    """Reset the global stdin cache before and after each test."""
    multitool._STDIN_CACHE = None
    multitool._STDIN_ENCODING = None
    yield
    multitool._STDIN_CACHE = None
    multitool._STDIN_ENCODING = None

def test_get_adjacent_keys_diagonals():
    adj = multitool.get_adjacent_keys(include_diagonals=True)
    # 's' is surrounded by 'q', 'w', 'e', 'a', 'd', 'z', 'x', 'c'
    assert 'w' in adj['s']
    assert 'q' in adj['s']
    assert 'x' in adj['s']
    assert len(adj['s']) == 8

def test_get_adjacent_keys_no_diagonals():
    adj = multitool.get_adjacent_keys(include_diagonals=False)
    # 's' has neighbors 'w' (up), 'a' (left), 'd' (right), 'x' (down)
    # wait, qwerty keyboard layout is offset.
    # multitool.py implementation uses a grid approach on the rows.
    # row 0: qwertyuiop
    # row 1: asdfghjkl
    # row 2: zxcvbnm
    # 's' is (1, 1). Neighbors: (0, 1)='w', (1, 0)='a', (1, 2)='d', (2, 1)='x'
    assert 'w' in adj['s']
    assert 'a' in adj['s']
    assert 'd' in adj['s']
    assert 'x' in adj['s']
    assert 'q' not in adj['s']
    assert 'e' not in adj['s']
    assert 'z' not in adj['s']
    assert 'c' not in adj['s']
    assert len(adj['s']) == 4

def test_classify_typo_logic():
    adj = multitool.get_adjacent_keys()

    # [T] Transposition
    assert multitool.classify_typo("teh", "the", adj) == "[T]"

    # [D] Deletion
    assert multitool.classify_typo("helo", "hello", adj) == "[D]"

    # [I] Insertion
    assert multitool.classify_typo("helloo", "hello", adj) == "[I]"

    # [K] Keyboard
    assert multitool.classify_typo("helko", "hello", adj) == "[K]"

    # [R] Replacement
    assert multitool.classify_typo("hella", "hello", adj) == "[R]"

    # [M] Multi-character
    assert multitool.classify_typo("abc", "def", adj) == "[M]"

    # Edge cases
    assert multitool.classify_typo("", "the", adj) == "[?]"
    assert multitool.classify_typo("the", "", adj) == "[?]"
    assert multitool.classify_typo("a", "b", adj) == "[R]"
    assert multitool.classify_typo("a", "q", adj) == "[K]" # 'a' (1, 0), 'q' (0, 0)

    # Coverage for line 202: return "[?]"
    # This is hard to reach because [M] covers anything with distance > 1,
    # and [T], [D], [I], [R], [K] cover distance 1.
    # Distance 0?
    assert multitool.classify_typo("the", "the", adj) == "[?]"

def test_classify_mode_basic(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh -> the\nhelo -> hello")
    output_file = tmp_path / "output.txt"

    multitool.classify_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=2,
        max_length=100,
        process_output=False,
        output_format='arrow'
    )

    content = output_file.read_text()
    assert "teh" in content and "the" in content and "[T]" in content
    assert "helo" in content and "hello" in content and "[D]" in content

def test_classify_mode_show_dist(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh -> the")
    output_file = tmp_path / "output.txt"

    multitool.classify_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=2,
        max_length=100,
        process_output=False,
        show_dist=True,
        output_format='arrow'
    )

    content = output_file.read_text()
    assert "teh" in content and "the" in content and "[T] [D:2]" in content

def test_classify_mode_process_output(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh -> the\nteh -> the")
    output_file = tmp_path / "output.txt"

    multitool.classify_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=2,
        max_length=100,
        process_output=True,
        output_format='arrow'
    )

    content = output_file.read_text()
    assert "[T]" in content
    assert content.count("teh") == 1

def test_classify_mode_raw(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("TeH -> the")
    output_file = tmp_path / "output.txt"

    # When clean_items=False, classify_typo is called with "TeH" and "the".
    # Since "T" != "t", it's not a simple transposition of lowercase letters if we don't normalize.
    # Actually, classify_typo uses .lower() for keyboard adjacency check but not for transposition.
    # "TeH" vs "the" -> T!=t, e==e, H!=e. Diffs at 0 and 2. Not adjacent diffs for [T].
    # Distance is 2 (T->t, H->e). So it becomes [M].
    multitool.classify_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=2,
        max_length=100,
        process_output=False,
        clean_items=False,
        output_format='arrow'
    )

    content = output_file.read_text()
    assert "TeH" in content and "the" in content and "[M]" in content

def test_classify_mode_empty_sides(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("!!! -> the\nteh -> !!!")
    output_file = tmp_path / "output.txt"

    multitool.classify_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=2,
        max_length=100,
        process_output=False,
        output_format='arrow'
    )

    content = output_file.read_text()
    # Both pairs should be skipped because one side becomes empty after filter_to_letters
    assert "the" not in content
    assert "teh" not in content

def test_classify_mode_formats(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh -> the")

    # JSON
    json_out = tmp_path / "output.json"
    multitool.classify_mode([str(input_file)], str(json_out), 2, 100, False, output_format='json')
    assert '"teh": "the [T]"' in json_out.read_text()

    # CSV
    csv_out = tmp_path / "output.csv"
    multitool.classify_mode([str(input_file)], str(csv_out), 2, 100, False, output_format='csv')
    assert "teh,the,[T]" in csv_out.read_text()

    # MD-Table
    md_out = tmp_path / "output.md"
    multitool.classify_mode([str(input_file)], str(md_out), 2, 100, False, output_format='md-table')
    assert "| teh | the | [T] |" in md_out.read_text()
