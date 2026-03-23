import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_write_paired_output_arrow_rich_visual(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh -> the\naddr -> address")
    output_file = tmp_path / "output.txt"

    with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
        multitool.pairs_mode(
            [str(input_file)], str(output_file), 1, 100, False,
            output_format='arrow'
        )

    content = output_file.read_text()

    assert "Typo" in content
    assert "Correction" in content
    assert "───" in content
    assert "│" in content

    assert "teh" in content
    assert "the" in content
    assert "addr" in content
    assert "address" in content

    lines = content.strip().splitlines()
    data_lines = [line for line in lines if "│" in line and "Typo" not in line and "──" not in line]
    assert len(data_lines) == 2

    assert any("  addr │ address" in line for line in data_lines)
    assert any("  teh  │ the" in line for line in data_lines)

def test_write_paired_output_arrow_rich_visual_empty(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("")
    output_file = tmp_path / "output.txt"

    with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
        multitool.pairs_mode(
            [str(input_file)], str(output_file), 1, 100, False,
            output_format='arrow'
        )

    content = output_file.read_text()
    assert "Typo" in content
    assert "Correction" in content
