import json
import csv
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_count_mode_filtering(tmp_path):
    input_file = tmp_path / "input.txt"
    # words: a(4), b(3), c(2), d(1)
    input_file.write_text("a a a a b b b c c d")
    output_file = tmp_path / "output.txt"

    # Min count 3
    multitool.count_mode([str(input_file)], str(output_file), 1, 10, False, min_count=3)
    assert output_file.read_text().splitlines() == ["a: 4", "b: 3"]

    # Max count 2
    multitool.count_mode([str(input_file)], str(output_file), 1, 10, False, max_count=2)
    assert output_file.read_text().splitlines() == ["c: 2", "d: 1"]

    # Range 2-3
    multitool.count_mode([str(input_file)], str(output_file), 1, 10, False, min_count=2, max_count=3)
    assert output_file.read_text().splitlines() == ["b: 3", "c: 2"]

def test_count_mode_json_output(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple banana apple")
    output_file = tmp_path / "output.json"

    multitool.count_mode([str(input_file)], str(output_file), 1, 10, False, output_format='json')

    data = json.loads(output_file.read_text())
    assert data == [
        {"item": "apple", "count": 2},
        {"item": "banana", "count": 1}
    ]

def test_count_mode_csv_output(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple banana apple")
    output_file = tmp_path / "output.csv"

    multitool.count_mode([str(input_file)], str(output_file), 1, 10, False, output_format='csv')

    with open(output_file, 'r', newline='') as f:
        reader = list(csv.reader(f))
        assert reader == [["apple", "2"], ["banana", "1"]]

def test_count_mode_markdown_output(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple banana apple")
    output_file = tmp_path / "output.md"

    multitool.count_mode([str(input_file)], str(output_file), 1, 10, False, output_format='markdown')

    assert output_file.read_text().splitlines() == ["- apple: 2", "- banana: 1"]
