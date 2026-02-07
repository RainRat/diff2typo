import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

def test_table_mode_basic(tmp_path):
    input_file = tmp_path / "input.toml"
    input_file.write_text('teh = "the"\nadn = "and"\nignored line\n')
    output_file = tmp_path / "output.txt"

    multitool.table_mode([str(input_file)], str(output_file), 1, 10, True)
    assert output_file.read_text().splitlines() == ["adn", "teh"]

def test_table_mode_right(tmp_path):
    input_file = tmp_path / "input.toml"
    input_file.write_text('teh = "the"\nadn = "and"\n')
    output_file = tmp_path / "output.txt"

    multitool.table_mode([str(input_file)], str(output_file), 1, 10, True, right_side=True)
    assert output_file.read_text().splitlines() == ["and", "the"]

def test_table_mode_complex_values(tmp_path):
    input_file = tmp_path / "input.toml"
    # Use raw string to avoid python-level escape confusion
    input_file.write_text(r'complex-typo = "correction with spaces and \"quotes\""' + "\n")
    output_file = tmp_path / "output.txt"

    # Use raw to avoid cleaning complex values
    multitool.table_mode([str(input_file)], str(output_file), 1, 100, False, right_side=True, clean_items=False)

    # Line in file: complex-typo = "correction with spaces and \"quotes\""
    # parts[1]: correction with spaces and \"quotes\""
    # val_part: correction with spaces and \"quotes\""
    # val_part[:-1]: correction with spaces and \"quotes\"
    assert output_file.read_text().strip() == r'correction with spaces and \"quotes\"'
