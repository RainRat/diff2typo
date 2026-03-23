from multitool import map_mode
import json

def test_map_mode_pairs(tmp_path):
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the\nwrd,word")

    input_file = tmp_path / "input.txt"
    input_file.write_text("teh\nhello\nwrd")

    output_file = tmp_path / "output.csv"

    map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='csv',
        pairs=True
    )

    content = output_file.read_text().strip().splitlines()
    # sorted(set(results)) in map_mode
    # Expected: hello,hello; teh,the; wrd,word
    assert "teh,the" in content
    assert "hello,hello" in content
    assert "wrd,word" in content
    assert len(content) == 3

def test_map_mode_smart_case(tmp_path):
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the")

    input_file = tmp_path / "input.txt"
    input_file.write_text("Teh\nTEH\nteh")

    output_file = tmp_path / "output.txt"

    map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='line',
        smart_case=True
    )

    content = output_file.read_text().strip().splitlines()
    assert content == ["The", "THE", "the"]

def test_map_mode_pairs_and_smart_case(tmp_path):
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the")

    input_file = tmp_path / "input.txt"
    input_file.write_text("Teh")

    output_file = tmp_path / "output.json"

    map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format='json',
        pairs=True,
        smart_case=True
    )

    with open(output_file) as f:
        data = json.load(f)

    assert data == {"Teh": "The"}

def test_map_mode_drop_missing_pairs(tmp_path):
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("teh,the")

    input_file = tmp_path / "input.txt"
    input_file.write_text("teh\nunknown")

    output_file = tmp_path / "output.csv"

    map_mode(
        input_files=[str(input_file)],
        mapping_file=str(mapping_file),
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='csv',
        pairs=True,
        drop_missing=True
    )

    content = output_file.read_text().strip().splitlines()
    assert content == ["teh,the"]
