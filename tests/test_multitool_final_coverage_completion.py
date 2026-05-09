from unittest.mock import MagicMock
import sys
from pathlib import Path
import pytest
import runpy

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable=None, *_, **__: iterable if iterable is not None else MagicMock())

def test_cycles_mode_visited_branch(tmp_path):
    """Covers line 2021: if node in visited: return in cycles_mode.walk"""
    # Graph: a -> b, c -> b
    # Processing 'a' will visit 'a' and 'b'.
    # Processing 'c' will visit 'c' and then see 'b' is already in 'visited'.
    mapping_file = tmp_path / "cycles.txt"
    mapping_file.write_text("a -> b\nc -> b")
    output_file = tmp_path / "output.txt"

    multitool.cycles_mode(
        input_files=[str(mapping_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        quiet=True
    )
    # No cycles expected
    assert output_file.read_text().strip() == ""

def test_map_mode_empty_line(tmp_path):
    """Covers line 3091: if not line_content: continue in map_mode"""
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple\n\nbanana")
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("apple -> fruit")
    output_file = tmp_path / "output.txt"

    multitool.map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        quiet=True
    )

    content = output_file.read_text().splitlines()
    assert "fruit" in content
    assert "banana" in content
    assert len(content) == 2 # The empty line was skipped

def test_multitool_main_block(monkeypatch):
    """Covers line 5156: the if __name__ == "__main__": block using runpy"""
    # Mock sys.argv to run a simple command that exits quickly
    monkeypatch.setattr(sys, "argv", ["multitool.py", "--mode-help", "words"])

    # We expect SystemExit because ModeHelpAction calls parser.exit()
    with pytest.raises(SystemExit) as cm:
        runpy.run_path("multitool.py", run_name="__main__")

    # ModesHelpAction calls parser.exit() which defaults to status 0
    assert cm.value.code == 0
