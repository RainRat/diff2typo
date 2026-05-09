import multitool
from unittest.mock import patch
import io

def test_toml_mode_extraction(tmp_path):
    toml_content = """
[tool.poetry.dependencies]
python = "^3.10"
tqdm = "*"

[replacements]
teh = "the"
recieve = "receive"
"""
    toml_file = tmp_path / "test.toml"
    toml_file.write_text(toml_content)

    # Test extracting keys from a nested table
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'toml', str(toml_file), '-k', 'tool.poetry.dependencies', '-R']):
            multitool.main()
            output = fake_out.getvalue()
            assert "python" in output
            assert "tqdm" in output

    # Test extracting from a flat table
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'toml', str(toml_file), '-k', 'replacements', '-R']):
            multitool.main()
            output = fake_out.getvalue()
            assert "teh" in output
            assert "recieve" in output

def test_extract_pairs_from_toml(tmp_path):
    toml_content = """
[replacements]
teh = "the"
recieve = "receive"

[[replacements.list]]
typo = "andd"
correct = "and"
"""
    toml_file = tmp_path / "test_pairs.toml"
    toml_file.write_text(toml_content)

    # Use pairs mode to verify extraction logic in _extract_pairs
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'pairs', str(toml_file), '-f', 'csv', '-R']):
            multitool.main()
            output = fake_out.getvalue()
            assert "teh,the" in output
            assert "recieve,receive" in output
            assert "andd,and" in output

def test_toml_output_format(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("word1\nword2\n")

    # Test output-format=toml for simple list
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'line', str(input_file), '-f', 'toml', '-R']):
            multitool.main()
            output = fake_out.getvalue()
            assert 'word1 = ""' in output
            assert 'word2 = ""' in output

    # Test output-format=toml for pairs
    mapping_file = tmp_path / "map.csv"
    mapping_file.write_text("teh,the\n")
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'pairs', str(mapping_file), '-f', 'toml', '-R']):
            multitool.main()
            output = fake_out.getvalue()
            assert 'teh = "the"' in output
