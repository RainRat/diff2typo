
import os
import json
from multitool import fileinfo_mode

def test_fileinfo_mode_basic(tmp_path):
    f1 = tmp_path / "test1.txt"
    f1.write_text("hello world\nline 2")

    out_file = tmp_path / "out.json"

    fileinfo_mode(
        input_files=[str(f1)],
        output_file=str(out_file),
        output_format="json",
        quiet=True
    )

    with open(out_file) as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["file"] == str(f1)
    assert data[0]["lines"] == 2
    assert data[0]["words"] == 4
    assert data[0]["size"] > 0
    assert data[0]["encoding"] == "utf-8"

def test_fileinfo_mode_multiple(tmp_path):
    f1 = tmp_path / "test1.txt"
    f1.write_text("one")
    f2 = tmp_path / "test2.txt"
    f2.write_text("two words")

    out_file = tmp_path / "out.json"

    fileinfo_mode(
        input_files=[str(f1), str(f2)],
        output_file=str(out_file),
        output_format="json",
        quiet=True
    )

    with open(out_file) as f:
        data = json.load(f)

    assert len(data) == 2
    assert data[0]["words"] == 1
    assert data[1]["words"] == 2
