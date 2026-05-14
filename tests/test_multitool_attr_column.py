import json
import pytest
from multitool import classify_mode, similarity_mode, near_duplicates_mode, fuzzymatch_mode, discovery_mode

def test_classify_mode_attr_arrow(capsys, tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("teh -> the\n", encoding="utf-8")

    classify_mode(
        input_files=[str(input_file)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=False,
        show_dist=True,
        output_format="arrow",
        quiet=True
    )

    captured = capsys.readouterr()
    assert "Typo" in captured.out
    assert "Correction" in captured.out
    assert "Attr" in captured.out
    assert "teh" in captured.out
    assert "the" in captured.out
    assert "[T]" in captured.out
    assert "[D:2]" in captured.out

def test_classify_mode_attr_json(capsys, tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("teh -> the\n", encoding="utf-8")

    classify_mode(
        input_files=[str(input_file)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=False,
        show_dist=False,
        output_format="json",
        quiet=True
    )

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["teh"] == "the [T]"

def test_similarity_mode_attr_csv(capsys, tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("house -> horse\n", encoding="utf-8")

    similarity_mode(
        input_files=[str(input_file)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=False,
        show_dist=True,
        output_format="csv",
        quiet=True
    )

    captured = capsys.readouterr()
    # CSV output: typo,correction,attr
    assert "house,horse,[R] [D:1]" in captured.out

def test_near_duplicates_attr(capsys, tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("house\nhorse\n", encoding="utf-8")

    near_duplicates_mode(
        input_files=[str(input_file)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=False,
        min_dist=1,
        max_dist=1,
        show_dist=True,
        output_format="json",
        quiet=True
    )

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["horse"] == "house [R] [D:1]" or data["house"] == "horse [R] [D:1]"

def test_fuzzymatch_attr(capsys, tmp_path):
    file1 = tmp_path / "list1.txt"
    file1.write_text("teh\n", encoding="utf-8")
    file2 = tmp_path / "list2.txt"
    file2.write_text("the\n", encoding="utf-8")

    fuzzymatch_mode(
        input_files=[str(file1)],
        file2=str(file2),
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=False,
        min_dist=1,
        max_dist=2,
        show_dist=True,
        output_format="json",
        quiet=True
    )

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["teh"] == "the [T] [D:2]"

def test_discovery_mode_attr(capsys, tmp_path):
    input_file = tmp_path / "test.txt"
    input_file.write_text("the the the the the\nteh\n", encoding="utf-8")

    discovery_mode(
        input_files=[str(input_file)],
        output_file="-",
        min_length=1,
        max_length=1000,
        process_output=False,
        freq_min=2,
        rare_max=1,
        max_dist=2,
        show_dist=True,
        output_format="csv",
        quiet=True
    )

    captured = capsys.readouterr()
    assert "teh,the,[T] [D:2]" in captured.out
