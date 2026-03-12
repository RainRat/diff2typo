import json
import yaml
from multitool import _extract_pairs

def test_extract_pairs_json_list_typo_correct(tmp_path):
    """Test JSON list of objects with typo/correct keys."""
    d = tmp_path / "test.json"
    data = [
        {"typo": "teht", "correct": "that"},
        {"typo": "teh", "correct": "the"}
    ]
    d.write_text(json.dumps(data))

    pairs = list(_extract_pairs([str(d)]))
    assert pairs == [("teht", "that"), ("teh", "the")]

def test_extract_pairs_json_dict_replacements(tmp_path):
    """Test JSON dict with 'replacements' key."""
    d = tmp_path / "test.json"
    data = {
        "replacements": [
            {"typo": "teht", "correct": "that"},
            {"typo": "teh", "correct": "the"}
        ]
    }
    d.write_text(json.dumps(data))

    pairs = list(_extract_pairs([str(d)]))
    assert pairs == [("teht", "that"), ("teh", "the")]

def test_extract_pairs_json_dict_generic(tmp_path):
    """Test JSON dict with generic keys."""
    d = tmp_path / "test.json"
    data = {
        "key1": "val1",
        "key2": "val2"
    }
    d.write_text(json.dumps(data))

    pairs = list(_extract_pairs([str(d)]))
    assert pairs == [("key1", "val1"), ("key2", "val2")]

def test_extract_pairs_yaml_list_typo_correct(tmp_path):
    """Test YAML list of objects with typo/correct keys."""
    d = tmp_path / "test.yaml"
    content = """
- typo: teht
  correct: that
- typo: teh
  correct: the
"""
    d.write_text(content)

    pairs = list(_extract_pairs([str(d)]))
    assert pairs == [("teht", "that"), ("teh", "the")]

def test_extract_pairs_yaml_list_generic(tmp_path):
    """Test YAML list of objects with generic keys."""
    d = tmp_path / "test.yaml"
    content = """
- key1: val1
- key2: val2
"""
    d.write_text(content)

    pairs = list(_extract_pairs([str(d)]))
    assert pairs == [("key1", "val1"), ("key2", "val2")]

def test_extract_pairs_yaml_multi_document(tmp_path):
    """Test multi-document YAML file."""
    d = tmp_path / "test.yaml"
    content = """
doc1_k: doc1_v
---
doc2_k: doc2_v
"""
    d.write_text(content)

    pairs = list(_extract_pairs([str(d)]))
    assert pairs == [("doc1_k", "doc1_v"), ("doc2_k", "doc2_v")]

def test_extract_pairs_yaml_error(tmp_path, caplog):
    """Test YAML parsing error handling."""
    d = tmp_path / "test.yaml"
    content = """
- : [ :
"""
    d.write_text(content)

    pairs = list(_extract_pairs([str(d)]))
    assert pairs == []
    assert f"Failed to parse YAML in '{d}'" in caplog.text

def test_extract_pairs_json_error(tmp_path, caplog):
    """Test JSON parsing error handling."""
    d = tmp_path / "test.json"
    content = "{ invalid json }"
    d.write_text(content)

    pairs = list(_extract_pairs([str(d)]))
    assert pairs == []
    assert f"Failed to parse JSON in '{d}'" in caplog.text

def test_extract_pairs_json_empty(tmp_path):
    """Test empty JSON file."""
    d = tmp_path / "test.json"
    d.write_text("")

    pairs = list(_extract_pairs([str(d)]))
    assert pairs == []

def test_extract_pairs_markdown_table_edge_cases(tmp_path):
    """Test Markdown table with empty edge parts and dividers."""
    d = tmp_path / "test.md"
    content = """
| typo | correction |
| :--- | :--- |
| teht | that |
|  | empty |
"""
    d.write_text(content)

    pairs = list(_extract_pairs([str(d)]))
    assert pairs == [("teht", "that"), ("", "empty")]

def test_extract_pairs_markdown_list_bullets(tmp_path):
    """Test Markdown list with different bullet types."""
    d = tmp_path / "test.md"
    content = """
- typo1 -> corr1
* typo2 -> corr2
+ typo3 -> corr3
"""
    d.write_text(content)

    pairs = list(_extract_pairs([str(d)]))
    assert pairs == [("typo1", "corr1"), ("typo2", "corr2"), ("typo3", "corr3")]
