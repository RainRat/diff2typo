import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_markdown_mode_basic(tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text(
        "# Title\n"
        "- item1\n"
        "* item2\n"
        "+ item3\n"
        "Not an item\n"
        "  - nested item\n"
    )
    output_file = tmp_path / "output.txt"
    multitool.markdown_mode([str(input_file)], str(output_file), 1, 20, True)

    assert sorted(output_file.read_text().splitlines()) == ["item", "nesteditem"]

def test_markdown_mode_raw(tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text(
        "- Item 1\n"
        "* item_2\n"
    )
    output_file = tmp_path / "output.txt"
    multitool.markdown_mode([str(input_file)], str(output_file), 1, 20, True, clean_items=False)

    assert sorted(output_file.read_text().splitlines()) == ["Item 1", "item_2"]

def test_markdown_mode_separators(tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text(
        "- typo1: correction1\n"
        "- typo2 -> correction2\n"
        "- lone_word\n"
    )
    output_file = tmp_path / "output.txt"

    # Left side (default)
    multitool.markdown_mode([str(input_file)], str(output_file), 1, 20, True)
    assert sorted(output_file.read_text().splitlines()) == ["loneword", "typo"]

    # Right side
    multitool.markdown_mode([str(input_file)], str(output_file), 1, 20, True, right_side=True)
    assert sorted(output_file.read_text().splitlines()) == ["correction", "loneword"]

def test_markdown_mode_cli(monkeypatch, tmp_path):
    # Regression test for NameError in main()
    input_file = tmp_path / "input.md"
    input_file.write_text("- item1\n")
    output_file = tmp_path / "output.txt"

    monkeypatch.setattr(sys, 'argv', [
        'multitool.py', 'markdown', str(input_file), '--output', str(output_file), '--min-length', '1'
    ])

    multitool.main()
    assert output_file.read_text().strip() == "item"
