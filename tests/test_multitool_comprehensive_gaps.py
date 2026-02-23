import json
import sys
import pytest
import logging
from multitool import main
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    """Replace tqdm with identity to avoid progress output during tests."""
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

def test_stats_yaml_output(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple\nbanana\n")
    output_file = tmp_path / "stats.yaml"
    monkeypatch.setattr(sys, 'argv', ["multitool.py", "stats", str(input_file), "-o", str(output_file), "-f", "yaml"])
    main()
    with open(output_file, "r") as f:
        data = yaml.safe_load(f)
    assert data["items"]["total_encountered"] == 2

def test_stats_yaml_fallback(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple -> banana\n")
    output_file = tmp_path / "stats.yaml"
    monkeypatch.setattr(sys, 'argv', ["multitool.py", "stats", str(input_file), "-o", str(output_file), "-f", "yaml", "--pairs"])
    main()
    content = output_file.read_text()
    assert "items:" in content
    assert "pairs:" in content

def test_stats_markdown_table_pairs(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple -> banana\n")
    output_file = tmp_path / "stats.md"
    monkeypatch.setattr(sys, 'argv', ["multitool.py", "stats", str(input_file), "-o", str(output_file), "-f", "md-table", "--pairs"])
    main()
    content = output_file.read_text()
    assert "### PAIRED DATA STATISTICS" in content
    assert "| Total pairs extracted | 1 |" in content

def test_count_md_table(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple banana apple\n")
    output_file = tmp_path / "count.md"
    monkeypatch.setattr(sys, 'argv', ["multitool.py", "count", str(input_file), "-o", str(output_file), "-f", "md-table"])
    main()
    content = output_file.read_text()
    assert "| Item | Count |" in content
    assert "| apple | 2 |" in content

def test_map_positional_fallback(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh\n")
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text("teh -> the\n")
    output_file = tmp_path / "output.txt"
    # Positional: input, mapping
    monkeypatch.setattr(sys, 'argv', ["multitool.py", "map", str(input_file), str(mapping_file), "-o", str(output_file)])
    main()
    assert output_file.read_text().strip() == "the"

def test_zip_positional_fallback(tmp_path, monkeypatch):
    file1 = tmp_path / "file1.txt"
    file1.write_text("apple\n")
    file2 = tmp_path / "file2.txt"
    file2.write_text("banana\n")
    output_file = tmp_path / "output.txt"
    # Positional: file1, file2
    monkeypatch.setattr(sys, 'argv', ["multitool.py", "zip", str(file1), str(file2), "-o", str(output_file)])
    main()
    assert "apple -> banana" in output_file.read_text()

def test_filterfragments_positional_fallback(tmp_path, monkeypatch):
    file1 = tmp_path / "file1.txt"
    file1.write_text("app\nbanana\n")
    file2 = tmp_path / "file2.txt"
    file2.write_text("apple\n")
    output_file = tmp_path / "output.txt"
    monkeypatch.setattr(sys, 'argv', ["multitool.py", "filterfragments", str(file1), str(file2), "-o", str(output_file)])
    main()
    content = output_file.read_text().strip()
    assert content == "banana"

def test_set_operation_positional_fallback(tmp_path, monkeypatch):
    file1 = tmp_path / "file1.txt"
    file1.write_text("apple\nbanana\n")
    file2 = tmp_path / "file2.txt"
    file2.write_text("banana\ncherry\n")
    output_file = tmp_path / "output.txt"
    # Positional: file1, file2
    monkeypatch.setattr(sys, 'argv', ["multitool.py", "set_operation", str(file1), str(file2), "--operation", "intersection", "-o", str(output_file)])
    main()
    assert output_file.read_text().strip() == "banana"

def test_pairs_empty_skip(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text(" -> \napple -> banana\n")
    output_file = tmp_path / "output.txt"
    monkeypatch.setattr(sys, 'argv', ["multitool.py", "pairs", str(input_file), "-o", str(output_file)])
    main()
    assert output_file.read_text().strip() == "apple -> banana"

def test_swap_empty_skip_and_process(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text(" -> \napple -> banana\napple -> banana\n")
    output_file = tmp_path / "output.txt"
    monkeypatch.setattr(sys, 'argv', ["multitool.py", "swap", str(input_file), "-o", str(output_file), "-P"])
    main()
    assert output_file.read_text().strip() == "banana -> apple"

def test_sample_default_k(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple\nbanana\n")
    output_file = tmp_path / "output.txt"
    # Testing internal logic that is hard to reach via CLI due to required group
    multitool.sample_mode([str(input_file)], str(output_file), 1, 100, False)
    assert "apple" in output_file.read_text()
    assert "banana" in output_file.read_text()

def test_zip_missing_file2_error(monkeypatch, caplog):
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'zip', 'file1.txt'])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    assert "Zip mode requires a secondary file" in caplog.text

def test_map_missing_mapping_error(monkeypatch, caplog):
    monkeypatch.setattr(sys, 'argv', ['multitool.py', 'map', 'file1.txt'])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    assert "Map mode requires a mapping file" in caplog.text
