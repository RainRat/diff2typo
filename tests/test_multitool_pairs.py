import json
import csv
import io
import sys
from multitool import main
import pytest

def test_pairs_json_to_csv(tmp_path, capsys):
    # Create input JSON file
    input_file = tmp_path / "input.json"
    data = [
        {"typo": "teh", "correct": "the"},
        {"typo": "receieve", "correct": "receive"}
    ]
    input_file.write_text(json.dumps(data))

    # Run multitool pairs mode
    sys.argv = [
        "multitool.py", "pairs", str(input_file),
        "--output-format", "csv",
        "--quiet"
    ]

    # We use capsys to capture stdout
    main()

    out, err = capsys.readouterr()

    # Verify CSV output
    # csv.reader expects an iterable of strings
    reader = csv.reader(io.StringIO(out))
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0] == ["teh", "the"]
    assert rows[1] == ["receieve", "receive"]

def test_pairs_cleaning_and_filtering(tmp_path, capsys):
    # Create input text file in arrow format
    input_file = tmp_path / "input.txt"
    input_file.write_text("Teh -> The\nshort -> s\nTooLongWordIndeed -> Correction")

    # Run multitool pairs mode with cleaning and length filter
    sys.argv = [
        "multitool.py", "pairs", str(input_file),
        "--min-length", "3",
        "--max-length", "10",
        "--output-format", "arrow",
        "--quiet"
    ]

    main()

    out, err = capsys.readouterr()

    # "Teh -> The" should become "teh -> the" (cleaned to lowercase)
    # "short -> s" should be filtered out (s is too short)
    # "TooLongWordIndeed -> Correction" should be filtered out (TooLongWordIndeed is too long)

    assert "teh -> the" in out
    assert "short -> s" not in out
    assert "tooLongWordIndeed" not in out.lower()

def test_pairs_deduplication(tmp_path, capsys):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh -> the\nteh -> the\nabc -> def")

    sys.argv = [
        "multitool.py", "pairs", str(input_file),
        "--process-output",
        "--output-format", "arrow",
        "--quiet"
    ]

    main()

    out, err = capsys.readouterr()
    lines = [line.strip() for line in out.strip().split('\n')]
    assert len(lines) == 2
    assert "teh -> the" in lines
    assert "abc -> def" in lines

def test_pairs_yaml_output(tmp_path, capsys):
    input_file = tmp_path / "input.txt"
    input_file.write_text("teh -> the\nabc -> def")

    sys.argv = [
        "multitool.py", "pairs", str(input_file),
        "--output-format", "yaml",
        "--quiet"
    ]

    main()

    out, err = capsys.readouterr()
    # If PyYAML is installed, it will be proper YAML.
    # Otherwise it will be the fallback format "key: value".
    # Both should contain "teh: the".
    assert "teh: the" in out
    assert "abc: def" in out

def test_single_item_yaml_output(tmp_path, capsys):
    input_file = tmp_path / "input.txt"
    input_file.write_text("the\nreceive")

    sys.argv = [
        "multitool.py", "line", str(input_file),
        "--output-format", "yaml",
        "--quiet"
    ]

    main()

    out, err = capsys.readouterr()
    # For single items, it should be a YAML list format.
    assert "- the" in out
    assert "- receive" in out
