import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_extract_paragraph_items(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("This is the first paragraph.\nIt has multiple lines.\n\nThis is the second paragraph.\n\n\nThird paragraph here.\n")
    
    paragraphs = list(multitool._extract_paragraph_items(str(input_file)))
    assert paragraphs == [
        "This is the first paragraph. It has multiple lines.",
        "This is the second paragraph.",
        "Third paragraph here."
    ]

def test_paragraphs_mode(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("Paragraph one.\n\nParagraph two.\n")
    output_file = tmp_path / "output.txt"
    
    multitool.paragraphs_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=5,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True
    )
    
    content = output_file.read_text().splitlines()
    assert "paragraphone" in content
    assert "paragraphtwo" in content

def test_count_mode_paragraphs(tmp_path):
    input_file = tmp_path / "input.txt"
    # Duplicate paragraphs
    input_file.write_text("Hello world.\n\nHello world.\n\nDifferent one.\n")
    output_file = tmp_path / "output.txt"
    
    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        paragraphs=True,
        output_format='arrow',
        quiet=True,
        clean_items=False
    )
    
    lines = output_file.read_text().strip().splitlines()
    assert any("hello world" in l.lower() for l in lines)
    assert any("different one" in l.lower() for l in lines)
