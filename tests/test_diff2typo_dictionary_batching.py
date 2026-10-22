import pytest
import tempfile
import os
import diff2typo


def test_read_words_mapping_single_file(tmp_path):
    dict1 = tmp_path / "dict1.csv"
    dict1.write_text("word1,fix1\nword2,fix2\n")

    mapping = diff2typo.read_words_mapping(str(dict1))
    assert mapping == {"word1": {"fix1"}, "word2": {"fix2"}}


def test_read_words_mapping_multiple_files(tmp_path):
    dict1 = tmp_path / "dict1.csv"
    dict1.write_text("word1,fix1\nword2\n")

    dict2 = tmp_path / "dict2.csv"
    dict2.write_text("word1,fix1_alt\nword3,fix3\n")

    mapping = diff2typo.read_words_mapping([str(dict1), str(dict2)])
    assert mapping["word1"] == {"fix1", "fix1_alt"}
    assert mapping["word2"] == set()
    assert mapping["word3"] == {"fix3"}


def test_read_words_mapping_directory_and_glob(tmp_path):
    sub_dir = tmp_path / "dicts"
    sub_dir.mkdir()

    d1 = sub_dir / "dict_a.csv"
    d1.write_text("apple\nbanana\n")

    d2 = sub_dir / "dict_b.csv"
    d2.write_text("cherry\n")

    mapping = diff2typo.read_words_mapping(str(sub_dir))
    assert "apple" in mapping
    assert "banana" in mapping
    assert "cherry" in mapping

    glob_pattern = os.path.join(str(sub_dir), "*.csv")
    mapping_glob = diff2typo.read_words_mapping(glob_pattern)
    assert "apple" in mapping_glob
    assert "cherry" in mapping_glob


def test_cli_dictionary_batching(monkeypatch, tmp_path, caplog):
    d1 = tmp_path / "d1.csv"
    d1.write_text("teh,the\n")
    d2 = tmp_path / "d2.csv"
    d2.write_text("wrod,word\n")

    diff_file = tmp_path / "test.diff"
    diff_file.write_text("""--- a/file.txt
+++ b/file.txt
@@ -1,2 +1,2 @@
-This is teh house.
+This is the house.
-Another wrod here.
+Another word here.
""")

    test_args = [
        "diff2typo.py",
        str(diff_file),
        "-d", str(d1), str(d2),
        "--dry-run"
    ]
    monkeypatch.setattr("sys.argv", test_args)

    diff2typo.main()
