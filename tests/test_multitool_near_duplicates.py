import sys
from pathlib import Path
import json
import pytest

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_near_duplicates_logic(tmp_path):
    test_file = tmp_path / "test_words_nd.txt"
    test_file.write_text("apple\naple\nbanana\nbananna\ncherry\nberry\n")
    output_file = tmp_path / "output.txt"

    # Run near_duplicates with max-dist 1
    multitool.near_duplicates_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        max_dist=1
    )
    content = output_file.read_text()
    assert "aple -> apple" in content
    assert "banana -> bananna" in content
    assert "cherry -> berry" not in content

    # Run near_duplicates with max-dist 2
    multitool.near_duplicates_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        max_dist=2
    )
    content = output_file.read_text()
    assert "berry -> cherry" in content

    # Test show-dist
    multitool.near_duplicates_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        max_dist=1,
        show_dist=True
    )
    content = output_file.read_text()
    assert "aple -> apple (changes: 1)" in content

    # Test output format JSON
    multitool.near_duplicates_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        max_dist=1,
        output_format="json"
    )
    with open(output_file) as f:
        data = json.load(f)
    assert data["aple"] == "apple"

def test_near_duplicates_multiple_files(tmp_path):
    file1 = tmp_path / "file1.txt"
    file1.write_text("apple\n")
    file2 = tmp_path / "file2.txt"
    file2.write_text("aple\n")
    output_file = tmp_path / "output.txt"

    multitool.near_duplicates_mode(
        input_files=[str(file1), str(file2)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        max_dist=1
    )
    assert "aple -> apple" in output_file.read_text()

def test_near_duplicates_min_dist(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("apple\naple\napples\n")
    output_file = tmp_path / "output.txt"

    # distance aple/apple is 1
    # distance aple/apples is 2
    # distance apple/apples is 1

    # min_dist=2 should only find aple/apples
    multitool.near_duplicates_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        min_dist=2,
        max_dist=2
    )
    content = output_file.read_text().strip()
    assert content == "aple -> apples"

def test_near_duplicates_length_filtering(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("a\nab\napple\naple\n")
    output_file = tmp_path / "output.txt"

    # min_length=3 should skip 'a' and 'ab'
    multitool.near_duplicates_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        max_dist=1
    )
    content = output_file.read_text()
    assert "aple -> apple" in content
    assert "a -> ab" not in content

def test_near_duplicates_cleaning(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Apple\naple\n")
    output_file = tmp_path / "output.txt"

    # clean_items=True (default) should find them
    multitool.near_duplicates_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        max_dist=1,
        clean_items=True
    )
    assert "apple" in output_file.read_text()

    # clean_items=False should treat them as different (dist 2: A->a, +e? No, Apple vs aple is dist 2: A->a, p->p, p->l, l->e... wait)
    # A p p l e
    # a p l e
    # 1 0 1 1 = 3 changes?
    # Actually:
    # A -> a (1)
    # p -> p (0)
    # p -> l (1)
    # l -> e (1)
    # e -> "" (1)
    # Total 4.
    # If we use max_dist=1, they won't be found.

    multitool.near_duplicates_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        max_dist=1,
        clean_items=False
    )
    assert "Apple -> aple" not in output_file.read_text()

def test_near_duplicates_process_output_sorting(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("zebra\nzebraz\napple\naple\n")
    output_file = tmp_path / "output.txt"

    # process_output=True should sort results alphabetically
    multitool.near_duplicates_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        max_dist=1
    )
    lines = output_file.read_text().splitlines()
    assert "aple -> apple" in lines[0]
    assert "zebra -> zebraz" in lines[1]

def test_near_duplicates_no_process_output(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("zebra\nzebraz\napple\naple\n")
    output_file = tmp_path / "output.txt"

    multitool.near_duplicates_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        max_dist=1
    )
    lines = output_file.read_text().splitlines()
    assert "aple -> apple" in lines[0]
    assert "zebra -> zebraz" in lines[1]

def test_near_duplicates_optimization(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("a\n" + "b" * 10 + "\n")
    output_file = tmp_path / "output.txt"

    multitool.near_duplicates_mode(
        input_files=[str(test_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        max_dist=1
    )
    assert output_file.read_text().strip() == ""

def test_near_duplicates_cli(tmp_path, monkeypatch):
    test_file = tmp_path / "test_words.txt"
    test_file.write_text("apple\naple\n")
    output_file = tmp_path / "output.txt"

    monkeypatch.setattr(sys, 'argv', [
        'multitool.py', 'near_duplicates', str(test_file),
        '--output', str(output_file),
        '--max-dist', '1',
        '--quiet'
    ])
    multitool.main()
    assert "aple -> apple" in output_file.read_text()
