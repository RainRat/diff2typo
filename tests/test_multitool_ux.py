import sys
from io import StringIO
import pytest
from multitool import main

def test_zip_positional(tmp_path, monkeypatch):
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("apple\nbanana")
    f2.write_text("cherry\ndate")

    output = StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", ["multitool.py", "zip", str(f1), str(f2)])

    main()

    result = output.getvalue()
    assert "apple -> cherry" in result
    assert "banana -> date" in result

def test_map_positional(tmp_path, monkeypatch):
    i = tmp_path / "i.txt"
    m = tmp_path / "m.csv"
    i.write_text("apple")
    m.write_text("apple,pear")

    output = StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", ["multitool.py", "map", str(i), str(m)])

    main()

    result = output.getvalue()
    assert "pear" in result

def test_set_operation_positional(tmp_path, monkeypatch):
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("apple\nbanana")
    f2.write_text("apple")

    output = StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", ["multitool.py", "set_operation", str(f1), str(f2), "--operation", "intersection"])

    main()

    result = output.getvalue()
    assert "apple" in result
    assert "banana" not in result

def test_filterfragments_positional(tmp_path, monkeypatch):
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("app\norange")
    f2.write_text("apple")

    output = StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", ["multitool.py", "filterfragments", str(f1), str(f2)])

    main()

    result = output.getvalue()
    assert "orange" in result
    assert "app" not in result

def test_zip_error_missing_second_file(tmp_path, monkeypatch):
    f1 = tmp_path / "f1.txt"
    f1.write_text("apple")

    monkeypatch.setattr(sys, "argv", ["multitool.py", "zip", str(f1)])

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
