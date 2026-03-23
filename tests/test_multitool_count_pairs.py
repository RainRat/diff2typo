import sys
import json
from pathlib import Path

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import multitool

def test_count_mode_pairs_basic(tmp_path):
    input_file = tmp_path / "typos.txt"
    # Mix of formats that _extract_pairs should handle
    input_file.write_text("""
teh -> the
teh -> the
recieve -> receive
teh -> the
recieve -> receive
    """.strip())

    output_file = tmp_path / "output.csv"

    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        output_format='csv',
        pairs=True,
        quiet=True
    )

    content = output_file.read_text().splitlines()
    assert content[0] == "typo,correction,count"
    # Result should be sorted by count
    assert "teh,the,3" in content[1]
    assert "recieve,receive,2" in content[2]

def test_count_mode_pairs_json(tmp_path):
    input_file = tmp_path / "typos.txt"
    input_file.write_text("teh -> the\nteh -> the")

    output_file = tmp_path / "output.json"

    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        output_format='json',
        pairs=True,
        quiet=True
    )

    with open(output_file) as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["typo"] == "teh"
    assert data[0]["correction"] == "the"
    assert data[0]["count"] == 2

def test_count_mode_pairs_arrow_visual(tmp_path):
    input_file = tmp_path / "typos.txt"
    input_file.write_text("teh -> the\nrecieve -> receive")

    output_file = tmp_path / "output.txt"

    # Mock isatty to False to avoid ANSI colors in tests, but check headers
    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        output_format='arrow',
        pairs=True,
        quiet=True
    )

    content = output_file.read_text()
    assert "TYPO -> CORRECTION" in content
    assert "teh -> the" in content
    assert "recieve -> receive" in content

def test_count_mode_pairs_filtering(tmp_path):
    input_file = tmp_path / "typos.txt"
    input_file.write_text("a -> b\nteh -> the\nteh -> the")

    output_file = tmp_path / "output.csv"

    # Filter with min_length=3
    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=False,
        output_format='csv',
        pairs=True,
        quiet=True
    )

    content = output_file.read_text()
    assert "a,b" not in content
    assert "teh,the,2" in content
