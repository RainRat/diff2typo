import multitool
import json
import os
import pytest

def test_toml_write_fallback_logic(tmp_path, caplog):
    output_file = str(tmp_path / "output.toml")
    data = {"key": "value"}

    # 1. Test missing toml package fallback
    multitool._TOML_AVAILABLE = False
    multitool._write_structured_data(data, output_file, "toml")

    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        content = f.read()
    assert json.loads(content) == data
    assert "TOML output requires the 'toml' package" in caplog.text
    caplog.clear()

    # 2. Test non-dict root fallback
    multitool._TOML_AVAILABLE = True
    multitool._write_structured_data(["not", "a", "dict"], output_file, "toml")
    assert "TOML output requires a dictionary root" in caplog.text
    with open(output_file, 'r') as f:
        content = f.read()
    assert json.loads(content) == ["not", "a", "dict"]
