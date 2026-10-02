import json
import pytest
import multitool

def test_count_mode_reverse_word_counts(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple banana apple cherry banana apple cherry cherry cherry\n", encoding="utf-8")
    output_file = tmp_path / "output.json"

    # Default order: descending frequency (cherry: 4, apple: 3, banana: 2)
    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format="json",
        reverse=False,
    )
    data_default = json.loads(output_file.read_text(encoding="utf-8"))
    items_default = [d["item"] for d in data_default]
    counts_default = [d["count"] for d in data_default]

    assert counts_default == [4, 3, 2]
    assert items_default == ["cherry", "apple", "banana"]

    # Reverse order: ascending frequency (banana: 2, apple: 3, cherry: 4)
    multitool.count_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        output_format="json",
        reverse=True,
    )
    data_reverse = json.loads(output_file.read_text(encoding="utf-8"))
    items_reverse = [d["item"] for d in data_reverse]
    counts_reverse = [d["count"] for d in data_reverse]

    assert counts_reverse == [2, 3, 4]
    assert items_reverse == ["banana", "apple", "cherry"]


def test_count_mode_reverse_cli(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("alpha beta alpha beta beta gamma gamma gamma gamma\n", encoding="utf-8")
    output_file = tmp_path / "output.csv"

    # Run CLI with -r / --reverse
    monkeypatch.setattr(
        "sys.argv",
        [
            "multitool.py",
            "count",
            str(input_file),
            "-o",
            str(output_file),
            "-f",
            "csv",
            "-m",
            "1",
            "-r",
            "-q",
        ],
    )
    multitool.main()

    lines = [line.strip() for line in output_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == ["alpha,2", "beta,3", "gamma,4"]
