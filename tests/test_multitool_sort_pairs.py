import json
import pytest
import multitool


def test_sort_pairs_alpha(tmp_path):
    input_file = tmp_path / "pairs.txt"
    output_file = tmp_path / "out.txt"

    input_file.write_text("zebra -> animal\napple -> fruit\nbanana -> yellow\n")

    multitool.sort_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        by="alpha",
        reverse=False,
        unique=False,
        pairs=True,
        output_format="line",
        quiet=True,
    )

    lines = [line.strip() for line in output_file.read_text().strip().split("\n") if line.strip()]
    assert lines == [
        "apple -> fruit",
        "banana -> yellow",
        "zebra -> animal",
    ]


def test_sort_pairs_length(tmp_path):
    input_file = tmp_path / "pairs.json"
    output_file = tmp_path / "out.json"

    data = {"elephant": "animal", "cat": "pet", "hippopotamus": "large"}
    input_file.write_text(json.dumps(data))

    multitool.sort_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        by="length",
        reverse=False,
        unique=False,
        pairs=True,
        output_format="json",
        quiet=True,
    )

    out_data = json.loads(output_file.read_text())
    keys = list(out_data.keys())
    assert keys == ["cat", "elephant", "hippopotamus"]


def test_sort_pairs_numeric(tmp_path):
    input_file = tmp_path / "pairs.txt"
    output_file = tmp_path / "out.txt"

    input_file.write_text("item100 -> c\nitem2 -> a\nitem10 -> b\n")

    multitool.sort_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        by="numeric",
        reverse=False,
        unique=False,
        pairs=True,
        output_format="line",
        quiet=True,
        clean_items=False,
    )

    lines = [line.strip() for line in output_file.read_text().strip().split("\n") if line.strip()]
    assert lines == [
        "item2 -> a",
        "item10 -> b",
        "item100 -> c",
    ]


def test_sort_pairs_reverse(tmp_path):
    input_file = tmp_path / "pairs.txt"
    output_file = tmp_path / "out.txt"

    input_file.write_text("apple -> fruit\nbanana -> yellow\nzebra -> animal\n")

    multitool.sort_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=False,
        by="alpha",
        reverse=True,
        unique=False,
        pairs=True,
        output_format="line",
        quiet=True,
    )

    lines = [line.strip() for line in output_file.read_text().strip().split("\n") if line.strip()]
    assert lines == [
        "zebra -> animal",
        "banana -> yellow",
        "apple -> fruit",
    ]


def test_sort_pairs_unique(tmp_path):
    input_file = tmp_path / "pairs.txt"
    output_file = tmp_path / "out.txt"

    input_file.write_text("banana -> yellow\napple -> fruit\nbanana -> yellow\napple -> fruit\n")

    multitool.sort_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        by="alpha",
        reverse=False,
        unique=True,
        pairs=True,
        output_format="line",
        quiet=True,
    )

    lines = [line.strip() for line in output_file.read_text().strip().split("\n") if line.strip()]
    assert lines == [
        "apple -> fruit",
        "banana -> yellow",
    ]


def test_sort_pairs_cli_integration(tmp_path, monkeypatch):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.csv"

    data = {"zebra": "stripes", "alpha": "first"}
    input_file.write_text(json.dumps(data))

    monkeypatch.setattr(
        "sys.argv",
        [
            "multitool.py",
            "sort",
            str(input_file),
            "-p",
            "-o",
            str(output_file),
            "-f",
            "csv",
            "-q",
        ],
    )

    multitool.main()

    content = output_file.read_text()
    assert "alpha" in content
    assert "zebra" in content
    lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
    assert len(lines) == 2  # 2 csv rows
    assert lines[0].startswith("alpha")
    assert lines[1].startswith("zebra")
