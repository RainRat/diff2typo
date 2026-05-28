import json
from unittest.mock import patch
import multitool
from multitool import _yield_structured_docs

def test_yield_structured_docs_jsonl_fallback_with_errors(tmp_path):
    """
    Test that _yield_structured_docs correctly falls back to JSON Lines
    when the file is not a valid single JSON object, and that it
    gracefully skips malformed lines.
    """
    # Create a 'json' file that is NOT a valid single JSON object
    # but contains multiple JSON lines, including some malformed ones.
    jsonl_content = '{"name": "valid1"}\nnot a json\n{"name": "valid2"}\n'
    json_file = tmp_path / "test_fallback.json"
    json_file.write_text(jsonl_content)

    docs = list(_yield_structured_docs(str(json_file)))

    assert len(docs) == 2
    assert docs[0] == {"name": "valid1"}
    assert docs[1] == {"name": "valid2"}

def test_yield_structured_docs_jsonl_fallback_empty_lines(tmp_path):
    """
    Test that _yield_structured_docs skips empty lines during JSONL fallback.
    """
    jsonl_content = '{"a": 1}\n\n{"b": 2}\n'
    json_file = tmp_path / "test_empty.json"
    json_file.write_text(jsonl_content)

    docs = list(_yield_structured_docs(str(json_file)))

    assert len(docs) == 2
    assert docs[0] == {"a": 1}
    assert docs[1] == {"b": 2}

def test_yield_structured_docs_empty_file(tmp_path):
    """
    Test that _yield_structured_docs returns an empty iterator for empty files.
    """
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("")

    docs = list(_yield_structured_docs(str(empty_file)))
    assert len(docs) == 0

def test_yield_structured_docs_whitespace_file(tmp_path):
    """
    Test that _yield_structured_docs returns an empty iterator for whitespace-only files.
    """
    ws_file = tmp_path / "whitespace.json"
    ws_file.write_text("   \n   ")

    docs = list(_yield_structured_docs(str(ws_file)))
    assert len(docs) == 0

def test_yield_structured_docs_yaml_error(tmp_path):
    """
    Test that _yield_structured_docs gracefully handles YAML parsing errors.
    """
    yaml_file = tmp_path / "error.yaml"
    yaml_file.write_text("invalid: [unclosed list")

    docs = list(_yield_structured_docs(str(yaml_file)))
    assert len(docs) == 0

def test_yield_structured_docs_toml_error(tmp_path):
    """
    Test that _yield_structured_docs gracefully handles TOML parsing errors.
    """
    toml_file = tmp_path / "error.toml"
    toml_file.write_text("invalid = [unclosed list")

    docs = list(_yield_structured_docs(str(toml_file)))
    assert len(docs) == 0

def test_yield_structured_docs_fallback_success(tmp_path):
    """
    Test that _yield_structured_docs falls back to JSON/YAML for unknown extensions.
    """
    unknown_file = tmp_path / "data.txt"
    unknown_file.write_text('{"key": "value"}')

    docs = list(_yield_structured_docs(str(unknown_file)))
    assert len(docs) == 1
    assert docs[0] == {"key": "value"}

def test_yield_structured_docs_fallback_yaml_success(tmp_path):
    """
    Test that _yield_structured_docs falls back to YAML for unknown extensions if JSON fails.
    """
    unknown_file = tmp_path / "data.txt"
    unknown_file.write_text('key: value')

    docs = list(_yield_structured_docs(str(unknown_file)))
    assert len(docs) == 1
    assert docs[0] == {"key": "value"}

def test_yield_structured_docs_fallback_error(tmp_path):
    """
    Test that _yield_structured_docs gracefully handles errors in fallback for unknown extensions.
    """
    unknown_file = tmp_path / "data.txt"
    unknown_file.write_text('invalid: [unclosed list')

    docs = list(_yield_structured_docs(str(unknown_file)))
    assert len(docs) == 0

def test_yield_structured_docs_toml_available_fallback(tmp_path):
    """
    Test that _yield_structured_docs uses 'toml' library if 'tomllib' is not available.
    """
    toml_file = tmp_path / "test.toml"
    toml_file.write_text('key = "value"')

    with patch("multitool._TOMLLIB_AVAILABLE", False), \
         patch("multitool._TOML_AVAILABLE", True), \
         patch("toml.loads", return_value={"key": "value"}):
        docs = list(_yield_structured_docs(str(toml_file)))
        assert len(docs) == 1
        assert docs[0] == {"key": "value"}
