import os
import multitool


def test_unzip_mode_two_files(tmp_path):
    """Test unzip_mode extracting both left and right sides simultaneously with file2."""
    input_file = tmp_path / "pairs.txt"
    input_file.write_text("hello -> world\nteh -> the\nfoo -> bar\n", encoding="utf-8")

    out_left = tmp_path / "left.txt"
    out_right = tmp_path / "right.txt"

    multitool.unzip_mode(
        input_files=[str(input_file)],
        output_file=str(out_left),
        min_length=1,
        max_length=100,
        process_output=False,
        file2=str(out_right),
    )

    left_content = out_left.read_text(encoding="utf-8").splitlines()
    right_content = out_right.read_text(encoding="utf-8").splitlines()

    assert left_content == ["hello", "teh", "foo"]
    assert right_content == ["world", "the", "bar"]


def test_unzip_mode_two_files_processed(tmp_path):
    """Test unzip_mode extracting both left and right sides with process_output (dedup/sort)."""
    input_file = tmp_path / "pairs.txt"
    input_file.write_text("b -> y\na -> z\nb -> y\n", encoding="utf-8")

    out_left = tmp_path / "left.txt"
    out_right = tmp_path / "right.txt"

    multitool.unzip_mode(
        input_files=[str(input_file)],
        output_file=str(out_left),
        min_length=1,
        max_length=100,
        process_output=True,
        file2=str(out_right),
    )

    left_content = out_left.read_text(encoding="utf-8").splitlines()
    right_content = out_right.read_text(encoding="utf-8").splitlines()

    assert left_content == ["a", "b"]
    assert right_content == ["y", "z"]


def test_unzip_cli_output2_alias(tmp_path, monkeypatch):
    """Test unzip mode via CLI with --output2 and --output-right aliases."""
    pairs_file = tmp_path / "data.csv"
    pairs_file.write_text("bad,good\nwrong,right\n", encoding="utf-8")

    left_file = tmp_path / "left_out.txt"
    right_file = tmp_path / "right_out.txt"

    monkeypatch.setattr(
        "sys.argv",
        [
            "multitool.py",
            "unzip",
            str(pairs_file),
            "-o",
            str(left_file),
            "--output2",
            str(right_file),
        ],
    )
    multitool.main()

    assert left_file.read_text(encoding="utf-8").splitlines() == ["bad", "wrong"]
    assert right_file.read_text(encoding="utf-8").splitlines() == ["good", "right"]
