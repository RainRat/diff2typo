import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_zip_mode_basic(tmp_path):
    file1 = tmp_path / "typos.txt"
    file1.write_text("teh\nadn\n")
    file2 = tmp_path / "corrections.txt"
    file2.write_text("the\nand\n")
    output = tmp_path / "output.txt"

    multitool.zip_mode([str(file1)], str(file2), str(output), 1, 10, False)
    assert output.read_text().splitlines() == ["teh -> the", "adn -> and"]

def test_zip_mode_csv(tmp_path):
    file1 = tmp_path / "typos.txt"
    file1.write_text("teh\nadn\n")
    file2 = tmp_path / "corrections.txt"
    file2.write_text("the\nand\n")
    output = tmp_path / "output.csv"

    multitool.zip_mode([str(file1)], str(file2), str(output), 1, 10, False, output_format='csv')
    assert output.read_text().splitlines() == ["teh,the", "adn,and"]

def test_zip_mode_table(tmp_path):
    file1 = tmp_path / "typos.txt"
    file1.write_text("teh\nadn\n")
    file2 = tmp_path / "corrections.txt"
    file2.write_text("the\nand\n")
    output = tmp_path / "output.toml"

    multitool.zip_mode([str(file1)], str(file2), str(output), 1, 10, False, output_format='table')
    assert output.read_text().splitlines() == ['teh = "the"', 'adn = "and"']

def test_zip_mode_json(tmp_path):
    file1 = tmp_path / "typos.txt"
    file1.write_text("teh\nadn\n")
    file2 = tmp_path / "corrections.txt"
    file2.write_text("the\nand\n")
    output = tmp_path / "output.json"

    multitool.zip_mode([str(file1)], str(file2), str(output), 1, 10, False, output_format='json')
    data = json.loads(output.read_text())
    assert data == {"teh": "the", "adn": "and"}

def test_zip_mode_mismatched_lengths(tmp_path):
    file1 = tmp_path / "typos.txt"
    file1.write_text("teh\nadn\nexcess\n")
    file2 = tmp_path / "corrections.txt"
    file2.write_text("the\nand\n")
    output = tmp_path / "output.txt"

    multitool.zip_mode([str(file1)], str(file2), str(output), 1, 10, False)
    # Should stop at shortest
    assert output.read_text().splitlines() == ["teh -> the", "adn -> and"]

def test_zip_mode_filtering(tmp_path):
    file1 = tmp_path / "typos.txt"
    file1.write_text("teh\na\nlongtypo\n")
    file2 = tmp_path / "corrections.txt"
    file2.write_text("the\nb\nlongcorrection\n")
    output = tmp_path / "output.txt"

    # Filter with min_length=3, max_length=5
    multitool.zip_mode([str(file1)], str(file2), str(output), 3, 5, False)
    # Only "teh -> the" should remain
    assert output.read_text().splitlines() == ["teh -> the"]

def test_zip_mode_raw(tmp_path):
    file1 = tmp_path / "typos.txt"
    file1.write_text("TeH\n")
    file2 = tmp_path / "corrections.txt"
    file2.write_text("The\n")
    output = tmp_path / "output.txt"

    # Without raw, it should be lowercased
    multitool.zip_mode([str(file1)], str(file2), str(output), 1, 10, False, clean_items=True)
    assert output.read_text().splitlines() == ["teh -> the"]

    # With raw, it should preserve casing
    multitool.zip_mode([str(file1)], str(file2), str(output), 1, 10, False, clean_items=False)
    assert output.read_text().splitlines() == ["TeH -> The"]
