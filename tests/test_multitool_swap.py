import pytest
import json
from pathlib import Path
from unittest.mock import patch
import sys
import io

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_swap_arrow(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh -> the\ntaht -> that\n")
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')

    content = output_file.read_text()
    assert "the -> teh" in content
    assert "that -> taht" in content

def test_swap_csv(tmp_path):
    input_file = tmp_path / "input.csv"
    input_file.write_text("teh,the\ntaht,that\n")
    output_file = tmp_path / "output.csv"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='csv')

    content = output_file.read_text().splitlines()
    assert "the,teh" in content
    assert "that,taht" in content

def test_swap_table(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text('teh = "the"\ntaht = "that"\n')
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='table')

    content = output_file.read_text()
    assert 'the = "teh"' in content
    assert 'that = "taht"' in content

def test_swap_raw(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("TeH -> The!\n")
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow', clean_items=False)

    content = output_file.read_text()
    assert "The! -> TeH" in content

def test_swap_clean(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("TeH -> The!\n")
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow', clean_items=True)

    content = output_file.read_text()
    assert "the -> teh" in content

def test_swap_json_input(tmp_path):
    d = {"teh": "the", "taht": "that"}
    input_file = tmp_path / "test.json"
    input_file.write_text(json.dumps(d))
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    content = output_file.read_text()
    assert "the -> teh" in content
    assert "that -> taht" in content

def test_swap_filtering(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("a -> apple\n")
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 3, 100, False, output_format='arrow')
    content = output_file.read_text()
    assert "apple -> a" not in content # 'a' is too short

def test_swap_json_replacements_list(tmp_path):
    d = {"replacements": [{"typo": "teh", "correct": "the"}, {"typo": "adn", "correct": "and"}]}
    input_file = tmp_path / "test.json"
    input_file.write_text(json.dumps(d))
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    content = output_file.read_text()
    assert "the -> teh" in content
    assert "and -> adn" in content

def test_swap_json_list_of_dicts(tmp_path):
    d = [{"typo": "teh", "correct": "the"}, {"typo": "adn", "correct": "and"}]
    input_file = tmp_path / "test.json"
    input_file.write_text(json.dumps(d))
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    content = output_file.read_text()
    assert "the -> teh" in content
    assert "and -> adn" in content

def test_swap_yaml_dict(tmp_path):
    pytest.importorskip("yaml")
    import yaml
    d = {"teh": "the", "adn": "and"}
    input_file = tmp_path / "test.yaml"
    input_file.write_text(yaml.dump(d))
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    content = output_file.read_text()
    assert "the -> teh" in content
    assert "and -> adn" in content

def test_swap_yaml_list_of_dicts(tmp_path):
    pytest.importorskip("yaml")
    import yaml
    d = [{"teh": "the"}, {"adn": "and"}]
    input_file = tmp_path / "test.yaml"
    input_file.write_text(yaml.dump(d))
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    content = output_file.read_text()
    assert "the -> teh" in content
    assert "and -> adn" in content

def test_swap_yaml_list_of_dicts_explicit(tmp_path):
    pytest.importorskip("yaml")
    import yaml
    d = [{"typo": "teh", "correct": "the"}]
    input_file = tmp_path / "test.yaml"
    input_file.write_text(yaml.dump(d))
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    content = output_file.read_text()
    assert "the -> teh" in content

def test_swap_markdown_style(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("- teh: the\n* adn: and\n+ taht: that\n")
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    content = output_file.read_text()
    assert "the -> teh" in content
    assert "and -> adn" in content
    assert "that -> taht" in content

def test_swap_csv_fallback(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh,the,extra\nadn,and\n")
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    content = output_file.read_text()
    assert "the -> teh" in content
    assert "and -> adn" in content

def test_swap_empty_lines_comments(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("\n# comment\n  \nteh -> the\n")
    output_file = tmp_path / "output.txt"

    multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    content = output_file.read_text().strip()
    assert content == "the -> teh"

def test_swap_json_invalid(tmp_path, caplog):
    input_file = tmp_path / "test.json"
    input_file.write_text("{invalid")
    output_file = tmp_path / "output.txt"

    with caplog.at_level("ERROR"):
        multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    assert "Failed to parse JSON" in caplog.text

def test_swap_yaml_invalid(tmp_path, caplog):
    pytest.importorskip("yaml")
    input_file = tmp_path / "test.yaml"
    input_file.write_text(":")
    output_file = tmp_path / "output.txt"

    with caplog.at_level("ERROR"):
        multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    assert "Failed to parse YAML" in caplog.text

def test_swap_yaml_missing_dependency(tmp_path, caplog, monkeypatch):
    monkeypatch.setitem(sys.modules, 'yaml', None)
    input_file = tmp_path / "test.yaml"
    input_file.write_text("a: b")
    output_file = tmp_path / "output.txt"

    with caplog.at_level("ERROR"):
        multitool.swap_mode([str(input_file)], str(output_file), 1, 100, False, output_format='arrow')
    assert "PyYAML not installed" in caplog.text

def test_swap_cli_integration(tmp_path, monkeypatch, capsys):
    """Verify CLI entry point and stdin handling for swap mode."""
    input_text = "teh -> the\n"
    monkeypatch.setattr(sys, 'stdin', io.StringIO(input_text))

    # Run with '-' (stdin) and arrow output format
    monkeypatch.setattr(sys, 'argv', ['multitool.py', '--quiet', 'swap', '-', '-f', 'arrow'])

    multitool.main()

    captured = capsys.readouterr()
    assert "the -> teh" in captured.out
