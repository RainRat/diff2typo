import pytest
import os
from multitool import main

def test_set_operation_symmetric_difference(tmp_path, capsys):
    file1 = tmp_path / "file1.txt"
    file1.write_text("apple\nbanana\ncherry\n")

    file2 = tmp_path / "file2.txt"
    file2.write_text("banana\ncherry\ndate\n")

    output = tmp_path / "output.txt"

    # Run symmetric_difference
    # apple is only in file1
    # banana is in both
    # cherry is in both
    # date is only in file2
    # Result should be apple, date

    import sys
    test_args = [
        "multitool.py", "set_operation", str(file1),
        "--file2", str(file2),
        "--operation", "symmetric_difference",
        "--output", str(output),
        "--raw"
    ]

    import multitool
    multitool._STDIN_CACHE = None

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(sys, "argv", test_args)
        main()

    content = output.read_text().splitlines()
    assert sorted(content) == ["apple", "date"]

def test_set_operation_symmetric_difference_no_overlap(tmp_path):
    file1 = tmp_path / "file1.txt"
    file1.write_text("apple\n")

    file2 = tmp_path / "file2.txt"
    file2.write_text("banana\n")

    output = tmp_path / "output.txt"

    import sys
    test_args = [
        "multitool.py", "set_operation", str(file1),
        "--file2", str(file2),
        "--operation", "symmetric_difference",
        "--output", str(output),
        "--raw"
    ]

    import multitool
    multitool._STDIN_CACHE = None

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(sys, "argv", test_args)
        main()

    content = output.read_text().splitlines()
    assert sorted(content) == ["apple", "banana"]

def test_set_operation_symmetric_difference_full_overlap(tmp_path):
    file1 = tmp_path / "file1.txt"
    file1.write_text("apple\nbanana\n")

    file2 = tmp_path / "file2.txt"
    file2.write_text("apple\nbanana\n")

    output = tmp_path / "output.txt"

    import sys
    test_args = [
        "multitool.py", "set_operation", str(file1),
        "--file2", str(file2),
        "--operation", "symmetric_difference",
        "--output", str(output),
        "--raw"
    ]

    import multitool
    multitool._STDIN_CACHE = None

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(sys, "argv", test_args)
        main()

    content = output.read_text().splitlines()
    assert content == []
