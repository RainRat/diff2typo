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

def test_fuzzymatch_logic(tmp_path):
    list1 = tmp_path / "list1.txt"
    list1.write_text("aple\nbananna\n")
    list2 = tmp_path / "list2.txt"
    list2.write_text("apple\nbanana\ncherry\n")
    output_file = tmp_path / "output.txt"

    # Run fuzzymatch with max-dist 1
    multitool.fuzzymatch_mode(
        input_files=[str(list1)],
        file2=str(list2),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        max_dist=1
    )
    content = output_file.read_text()
    assert "aple -> apple" in content
    assert "bananna -> banana" in content
    assert "cherry" not in content

def test_fuzzymatch_min_dist(tmp_path):
    list1 = tmp_path / "list1.txt"
    list1.write_text("apple\n")
    list2 = tmp_path / "list2.txt"
    list2.write_text("apple\naple\napples\n")
    output_file = tmp_path / "output.txt"

    # apple vs apple is dist 0
    # apple vs aple is dist 1
    # apple vs apples is dist 1

    # min_dist=1 should find aple and apples, but NOT apple
    multitool.fuzzymatch_mode(
        input_files=[str(list1)],
        file2=str(list2),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        min_dist=1,
        max_dist=1
    )
    lines = output_file.read_text().splitlines()
    assert "apple -> aple" in lines
    assert "apple -> apples" in lines
    assert "apple -> apple" not in lines

def test_fuzzymatch_show_dist(tmp_path):
    list1 = tmp_path / "list1.txt"
    list1.write_text("aple\n")
    list2 = tmp_path / "list2.txt"
    list2.write_text("apple\n")
    output_file = tmp_path / "output.txt"

    multitool.fuzzymatch_mode(
        input_files=[str(list1)],
        file2=str(list2),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        max_dist=1,
        show_dist=True
    )
    content = output_file.read_text()
    assert "aple -> apple (changes: 1)" in content

def test_fuzzymatch_formats(tmp_path):
    list1 = tmp_path / "list1.txt"
    list1.write_text("aple\n")
    list2 = tmp_path / "list2.txt"
    list2.write_text("apple\n")
    output_file = tmp_path / "output.json"

    multitool.fuzzymatch_mode(
        input_files=[str(list1)],
        file2=str(list2),
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

def test_fuzzymatch_cli(tmp_path, monkeypatch):
    list1 = tmp_path / "list1.txt"
    list1.write_text("aple\n")
    list2 = tmp_path / "list2.txt"
    list2.write_text("apple\n")
    output_file = tmp_path / "output.txt"

    # Test positional file2
    monkeypatch.setattr(sys, 'argv', [
        'multitool.py', 'fuzzymatch', str(list1), str(list2),
        '--output', str(output_file),
        '--quiet'
    ])
    multitool.main()
    assert "aple -> apple" in output_file.read_text()

    # Test --file2 flag
    monkeypatch.setattr(sys, 'argv', [
        'multitool.py', 'fuzzymatch', str(list1),
        '--file2', str(list2),
        '--output', str(output_file),
        '--quiet'
    ])
    multitool.main()
    assert "aple -> apple" in output_file.read_text()
