import sys
from pathlib import Path
import json
import yaml

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_write_paired_output_table(tmp_path):
    output_file = tmp_path / "output.txt"
    pairs = [("key1", "value1"), ("key2", "value2")]
    multitool._write_paired_output(pairs, str(output_file), "table", "Test")

    content = output_file.read_text().splitlines()
    assert 'key1 = "value1"' in content
    assert 'key2 = "value2"' in content

def test_write_paired_output_table_empty(tmp_path):
    output_file = tmp_path / "output_empty.txt"
    pairs = []
    multitool._write_paired_output(pairs, str(output_file), "table", "Test")
    assert output_file.read_text() == ""

def test_write_paired_output_markdown(tmp_path):
    output_file = tmp_path / "output.md"
    pairs = [("key1", "value1"), ("key2", "value2")]
    multitool._write_paired_output(pairs, str(output_file), "markdown", "Test")

    content = output_file.read_text().splitlines()
    assert "- key1: value1" in content
    assert "- key2: value2" in content

def test_write_paired_output_markdown_empty(tmp_path):
    output_file = tmp_path / "output_empty.md"
    pairs = []
    multitool._write_paired_output(pairs, str(output_file), "markdown", "Test")
    assert output_file.read_text() == ""

def test_write_paired_output_md_table(tmp_path):
    output_file = tmp_path / "output.md"
    pairs = [("key1", "value1"), ("key2", "value2")]
    multitool._write_paired_output(pairs, str(output_file), "md-table", "Map")

    content = output_file.read_text().splitlines()
    assert "| Typo | Correction |" in content
    assert "| :--- | :--- |" in content
    assert "| key1 | value1 |" in content
    assert "| key2 | value2 |" in content

def test_write_paired_output_md_table_empty(tmp_path):
    output_file = tmp_path / "output_empty.md"
    pairs = []
    multitool._write_paired_output(pairs, str(output_file), "md-table", "Test")

    content = output_file.read_text()
    assert content == ""

def test_write_paired_output_arrow_empty(tmp_path):
    output_file = tmp_path / "output_arrow_empty.txt"
    pairs = []
    multitool._write_paired_output(pairs, str(output_file), "arrow", "Test")

    content = output_file.read_text()
    assert content == ""

def test_write_paired_output_default_fallback(tmp_path):
    output_file = tmp_path / "output_fallback.txt"
    pairs = [("key1", "value1")]
    multitool._write_paired_output(pairs, str(output_file), "unknown", "Test")

    content = output_file.read_text().strip()
    assert content == "key1 -> value1"

def test_write_paired_output_json(tmp_path):
    output_file = tmp_path / "output.json"
    pairs = [("key1", "value1"), ("key2", "value2")]
    multitool._write_paired_output(pairs, str(output_file), "json", "Test")

    with open(output_file, 'r') as f:
        data = json.load(f)
    assert data == {"key1": "value1", "key2": "value2"}

def test_write_paired_output_yaml(tmp_path):
    output_file = tmp_path / "output.yaml"
    pairs = [("key1", "value1"), ("key2", "value2")]
    multitool._write_paired_output(pairs, str(output_file), "yaml", "Test")

    with open(output_file, 'r') as f:
        data = yaml.safe_load(f)
    assert data == {"key1": "value1", "key2": "value2"}

def test_write_paired_output_yaml_fallback(tmp_path, monkeypatch):
    output_file = tmp_path / "output_no_yaml.txt"
    pairs = [("key1", "value1")]

    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == 'yaml':
            raise ImportError
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        multitool._write_paired_output(pairs, str(output_file), "yaml", "Test")

    content = output_file.read_text().strip()
    assert content == "key1: value1"
