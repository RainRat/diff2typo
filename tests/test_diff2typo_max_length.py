import sys
import pytest
from diff2typo import find_typos, main

def test_find_typos_max_length():
    diff_text = """--- a/file.txt
+++ b/file.txt
@@ -1,2 +1,2 @@
-teh short
+the supercalifragilisticexpialidocious
"""
    # Without max_length
    typos_all = find_typos(diff_text, min_length=2)
    assert "teh -> the" in typos_all or "short -> supercalifragilisticexpialidocious" in typos_all

    # With max_length = 5
    typos_filtered = find_typos(diff_text, min_length=2, max_length=5)
    assert "teh -> the" in typos_filtered
    assert not any("supercalifragilisticexpialidocious" in t for t in typos_filtered)

def test_diff2typo_cli_max_length(tmp_path, monkeypatch, capsys):
    diff_file = tmp_path / "test.diff"
    diff_file.write_text("""--- a/file.txt
+++ b/file.txt
@@ -1,2 +1,2 @@
-teh shortword
+the superlongwordthatshouldbefiltered
""")

    out_file = tmp_path / "output.txt"
    dict_file = tmp_path / "words.csv"
    dict_file.write_text("the\nsuperlongwordthatshouldbefiltered\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diff2typo.py",
            str(diff_file),
            "-o",
            str(out_file),
            "-d",
            str(dict_file),
            "--max-length",
            "5",
        ],
    )

    main()

    output = out_file.read_text()
    assert "teh -> the" in output
    assert "superlongwordthatshouldbefiltered" not in output
