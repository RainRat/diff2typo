import sys
import json
import csv
import pytest
from pathlib import Path

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_table_mode(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text('apple = "red"\nbanana = "yellow"\ncherry = "fruit"\n')
    output_file = tmp_path / "output.txt"

    multitool.table_mode([str(input_file)], str(output_file), 1, 10, True)
    assert output_file.read_text().splitlines() == ["apple", "banana", "cherry"]

    output_file_right = tmp_path / "output_right.txt"
    multitool.table_mode([str(input_file)], str(output_file_right), 1, 10, True, right_side=True)
    assert output_file_right.read_text().splitlines() == ["fruit", "red", "yellow"]

    input_file_no_end_quote = tmp_path / "no_end.txt"
    input_file_no_end_quote.write_text('missing = "quote\n')
    multitool.table_mode([str(input_file_no_end_quote)], str(output_file), 1, 20, False, right_side=True)
    assert output_file.read_text().strip() == "quote"

def test_markdown_mode(tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text(
        "- item1\n"
        "* item2\n"
        "+ item3\n"
        "- typo: correction\n"
        "* apple -> red\n"
        "- \n"
        "no bullet\n"
    )
    output_file = tmp_path / "output.txt"

    multitool.markdown_mode([str(input_file)], str(output_file), 1, 100, True)
    results = sorted(output_file.read_text().splitlines())
    assert results == ["apple", "item", "typo"]

    output_file_right = tmp_path / "output_right.txt"
    multitool.markdown_mode([str(input_file)], str(output_file_right), 1, 100, True, right_side=True)
    results_right = sorted(output_file_right.read_text().splitlines())
    assert results_right == ["correction", "red"]

def test_md_table_mode(tmp_path):
    input_file = tmp_path / "table.md"
    input_file.write_text(
        "| Typo | Correction |\n"
        "| :--- | :--- |\n"
        "| teh | the |\n"
        "| recieve | receive |\n"
    )
    output_file = tmp_path / "output.txt"

    multitool.md_table_mode([str(input_file)], str(output_file), 1, 100, True)
    results = sorted(output_file.read_text().splitlines())
    assert results == ["recieve", "teh"]

    output_file_right = tmp_path / "output_right.txt"
    multitool.md_table_mode([str(input_file)], str(output_file_right), 1, 100, True, right_side=True)
    results_right = sorted(output_file_right.read_text().splitlines())
    assert results_right == ["receive", "the"]

    input_file_custom = tmp_path / "custom.md"
    input_file_custom.write_text(
        "| Item | Count |\n"
        "| --- | --- |\n"
        "| word | 1 |\n"
    )
    multitool.md_table_mode([str(input_file_custom)], str(output_file), 1, 100, True)
    assert output_file.read_text().strip() == "word"

def test_json_mode(tmp_path):
    input_file = tmp_path / "data.json"
    data = {
        "items": [
            {"name": "apple", "color": "red"},
            {"name": "banana", "color": "yellow"}
        ],
        "meta": {"source": "market"}
    }
    input_file.write_text(json.dumps(data))
    output_file = tmp_path / "output.txt"

    multitool.json_mode([str(input_file)], str(output_file), 1, 100, True, key="items.name")
    results = sorted(output_file.read_text().splitlines())
    assert results == ["apple", "banana"]

    multitool.json_mode([str(input_file)], str(output_file), 1, 100, True, key="meta.source")
    assert output_file.read_text().strip() == "market"

    input_root = tmp_path / "root.json"
    input_root.write_text(json.dumps(["one", "two"]))
    multitool.json_mode([str(input_root)], str(output_file), 1, 100, True, key="")
    assert sorted(output_file.read_text().splitlines()) == ["one", "two"]

    input_empty = tmp_path / "empty.json"
    input_empty.write_text("  ")
    multitool.json_mode([str(input_empty)], str(output_file), 1, 100, True, key="")
    assert output_file.read_text().strip() == ""

def test_yaml_mode(tmp_path):
    pytest.importorskip("yaml")

    input_file = tmp_path / "data.yaml"
    input_file.write_text(
        "items:\n"
        "  - name: apple\n"
        "    color: red\n"
        "  - name: banana\n"
        "    color: yellow\n"
        "meta:\n"
        "  source: market\n"
    )
    output_file = tmp_path / "output.txt"

    multitool.yaml_mode([str(input_file)], str(output_file), 1, 100, True, key="items.name")
    results = sorted(output_file.read_text().splitlines())
    assert results == ["apple", "banana"]

    multitool.yaml_mode([str(input_file)], str(output_file), 1, 100, True, key="meta.source")
    assert output_file.read_text().strip() == "market"

    input_root = tmp_path / "root.yaml"
    input_root.write_text("- one\n- two\n")
    multitool.yaml_mode([str(input_root)], str(output_file), 1, 100, True, key="")
    assert sorted(output_file.read_text().splitlines()) == ["one", "two"]

    input_multi = tmp_path / "multi.yaml"
    input_multi.write_text("a: 1\n---\nb: 2\n")
    multitool.yaml_mode([str(input_multi)], str(output_file), 1, 100, True, key="")
    assert sorted(output_file.read_text().splitlines()) == ["a", "b"]

def test_write_output_formats(tmp_path):
    items = ["apple", "banana"]
    output_file = tmp_path / "output"

    multitool.write_output(items, str(output_file), output_format='json')
    assert json.loads(output_file.read_text()) == items

    multitool.write_output(items, str(output_file), output_format='csv')
    with open(output_file, newline='') as f:
        reader = csv.reader(f)
        assert list(reader) == [["apple"], ["banana"]]

    multitool.write_output(items, str(output_file), output_format='markdown')
    assert output_file.read_text() == "- apple\n- banana\n"

    multitool.write_output(items, str(output_file), output_format='md-table')
    lines = output_file.read_text().splitlines()
    assert "| Item |" in lines[0]
    assert "| apple |" in lines[2]

    pytest.importorskip("yaml")
    multitool.write_output(items, str(output_file), output_format='yaml')
    import yaml
    assert yaml.safe_load(output_file.read_text()) == items

    multitool.write_output(items, str(output_file), output_format='line', limit=1)
    assert output_file.read_text() == "apple\n"

def test_write_paired_output_formats(tmp_path):
    pairs = [("teh", "the"), ("recieve", "receive")]
    output_file = tmp_path / "output"

    multitool._write_paired_output(pairs, str(output_file), 'json', 'Pairs')
    assert json.loads(output_file.read_text()) == dict(pairs)

    multitool._write_paired_output(pairs, str(output_file), 'csv', 'Pairs')
    with open(output_file, newline='') as f:
        reader = csv.reader(f)
        assert list(reader) == [["teh", "the"], ["recieve", "receive"]]

    multitool._write_paired_output(pairs, str(output_file), 'table', 'Pairs')
    assert 'teh = "the"' in output_file.read_text()

    multitool._write_paired_output(pairs, str(output_file), 'markdown', 'Pairs')
    assert "- teh: the\n" in output_file.read_text()

    multitool._write_paired_output(pairs, str(output_file), 'md-table', 'Pairs')
    lines = output_file.read_text().splitlines()
    assert "| Typo | Correction |" in lines[0]
    assert "| teh | the |" in lines[2]

    pytest.importorskip("yaml")
    multitool._write_paired_output(pairs, str(output_file), 'yaml', 'Pairs')
    import yaml
    assert yaml.safe_load(output_file.read_text()) == dict(pairs)

    multitool._write_paired_output(pairs, str(output_file), 'arrow', 'Pairs')
    assert "teh -> the\n" in output_file.read_text()

    multitool._write_paired_output(pairs, str(output_file), 'line', 'Pairs', limit=1)
    assert output_file.read_text() == "teh -> the\n"
