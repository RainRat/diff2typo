import pytest
from multitool import main
import sys
from io import StringIO

def test_sort_alpha(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("banana\napple\ncherry\n")

    output_file = tmp_path / "output.txt"

    args = ["multitool.py", "sort", str(input_file), "-o", str(output_file), "--by", "alpha"]
    monkeypatch.setattr(sys, 'argv', args)

    main()

    assert output_file.read_text().strip().split('\n') == ["apple", "banana", "cherry"]

def test_sort_length(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("banana\napple\ncherryberry\n")

    output_file = tmp_path / "output.txt"

    args = ["multitool.py", "sort", str(input_file), "-o", str(output_file), "--by", "length"]
    monkeypatch.setattr(sys, 'argv', args)

    main()

    assert output_file.read_text().strip().split('\n') == ["apple", "banana", "cherryberry"]

def test_sort_numeric(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("item10\nitem2\nitem1\n")

    output_file = tmp_path / "output.txt"

    # Use -R to prevent digits from being stripped during cleaning
    args = ["multitool.py", "sort", str(input_file), "-o", str(output_file), "--by", "numeric", "-R"]
    monkeypatch.setattr(sys, 'argv', args)

    main()

    assert output_file.read_text().strip().split('\n') == ["item1", "item2", "item10"]

def test_sort_reverse(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple\nbanana\ncherry\n")

    output_file = tmp_path / "output.txt"

    args = ["multitool.py", "sort", str(input_file), "-o", str(output_file), "--reverse"]
    monkeypatch.setattr(sys, 'argv', args)

    main()

    assert output_file.read_text().strip().split('\n') == ["cherry", "banana", "apple"]

def test_sort_unique(tmp_path, monkeypatch):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple\nbanana\napple\n")

    output_file = tmp_path / "output.txt"

    args = ["multitool.py", "sort", str(input_file), "-o", str(output_file), "-u"]
    monkeypatch.setattr(sys, 'argv', args)

    main()

    assert output_file.read_text().strip().split('\n') == ["apple", "banana"]
