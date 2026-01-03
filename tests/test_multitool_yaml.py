import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

# Replicate the disable_tqdm fixture
@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_yaml_mode_simple_key(tmp_path):
    input_file = tmp_path / "input.yaml"
    input_file.write_text("- name: alice\n- name: bob\n")
    output_file = tmp_path / "output.txt"

    multitool.yaml_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="name",
    )

    result = output_file.read_text().splitlines()
    assert sorted(result) == ["alice", "bob"]

def test_yaml_mode_nested_key(tmp_path):
    input_file = tmp_path / "input.yaml"
    input_file.write_text(
        "meta:\n"
        "  id: 1\n"
        "  user: alice\n"
        "---\n"
        "meta:\n"
        "  id: 2\n"
        "  user: bob\n"
    )
    output_file = tmp_path / "output.txt"

    multitool.yaml_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="meta.user",
    )

    result = output_file.read_text().splitlines()
    assert sorted(result) == ["alice", "bob"]

def test_yaml_mode_list_handling(tmp_path):
    input_file = tmp_path / "input.yaml"
    output_file = tmp_path / "output.txt"

    # Case: Root is object, key points to list of strings
    input_file.write_text("tags:\n  - alpha\n  - beta\n")

    multitool.yaml_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="tags",
    )
    result = output_file.read_text().splitlines()
    assert sorted(result) == ["alpha", "beta"]

def test_yaml_mode_missing_key(tmp_path):
    input_file = tmp_path / "input.yaml"
    input_file.write_text("- name: alice\n- age: 30\n")
    output_file = tmp_path / "output.txt"

    multitool.yaml_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="name",
    )

    result = output_file.read_text().splitlines()
    assert sorted(result) == ["alice"]

def test_yaml_mode_malformed_yaml(tmp_path, caplog):
    input_file = tmp_path / "input.yaml"
    input_file.write_text("key: value\n  invalid_indent: value")
    output_file = tmp_path / "output.txt"

    with caplog.at_level(logging.ERROR):
        multitool.yaml_mode(
            [str(input_file)],
            str(output_file),
            min_length=1,
            max_length=100,
            process_output=True,
            key="key",
        )

    assert "Failed to parse YAML" in caplog.text
    assert output_file.exists()
    assert output_file.read_text() == ""

def test_yaml_mode_deeply_nested_list(tmp_path):
    input_file = tmp_path / "input.yaml"
    input_file.write_text(
        "items:\n"
        "  - value: one\n"
        "  - value: two\n"
        "---\n"
        "items:\n"
        "  - value: three\n"
    )
    output_file = tmp_path / "output.txt"

    multitool.yaml_mode(
        [str(input_file)],
        str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        key="items.value",
    )

    result = output_file.read_text().splitlines()
    assert sorted(result) == ["one", "three", "two"]

def test_yaml_mode_missing_pyyaml(tmp_path, monkeypatch):
    # Mock import error for yaml
    import builtins
    real_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'yaml':
            raise ImportError("No module named 'yaml'")
        return real_import(name, globals, locals, fromlist, level)

    # We need to monkeypatch builtins.__import__
    # But since multitool.py likely imports it inside the function, we need to ensure it's not already in sys.modules
    # or reload it.

    # Actually, multitool.py has a local import inside _extract_yaml_items:
    # try:
    #     import yaml
    # except ImportError:
    #     ... sys.exit(1)

    # So we can just monkeypatch sys.modules to remove yaml if it's there,
    # and use the mock import.

    # Easier strategy: Patch sys.modules to raise error on access or just pretend it's not there
    # But python imports are cached.

    # Let's try to mock sys.exit to verify it exits
    mock_exit = MagicMock(side_effect=SystemExit)
    monkeypatch.setattr(sys, 'exit', mock_exit)

    # We need to simulate ImportError when `import yaml` is called.
    # Since `multitool` is already imported, and `_extract_yaml_items` does the import at runtime.

    with monkeypatch.context() as m:
        m.delitem(sys.modules, 'yaml', raising=False)
        m.setattr(builtins, '__import__', mock_import)

        input_file = tmp_path / "input.yaml"
        output_file = tmp_path / "output.txt"

        with pytest.raises(SystemExit):
            multitool.yaml_mode(
                [str(input_file)],
                str(output_file),
                1, 100, False, "key"
            )
