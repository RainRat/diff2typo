import diff2typo


def test_diff2typo_reverse_sort_alpha(tmp_path, monkeypatch, caplog):
    diff_file = tmp_path / "test.diff"
    diff_file.write_text(
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1,3 +1,3 @@\n"
        "-apple banana cherry\n"
        "+aple bannana chery\n"
    )

    out_file = tmp_path / "output.txt"
    test_args = [
        "diff2typo.py",
        str(diff_file),
        "-o",
        str(out_file),
        "-f",
        "list",
        "--sort",
        "alpha",
        "-r",
        "-d",
        "nonexistent_words.csv",
        "-a",
        "nonexistent_allowed.csv",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    diff2typo.main()

    lines = out_file.read_text().strip().split("\n")
    # Expected reverse alphabetical order of typos (before words): cherry, banana, apple
    assert lines == ["cherry", "banana", "apple"]


def test_diff2typo_reverse_sort_count(tmp_path, monkeypatch, caplog):
    diff_file = tmp_path / "test.diff"
    diff_file.write_text(
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1,5 +1,5 @@\n"
        "-apple apple apple banana banana cherry\n"
        "+aple aple aple bannana bannana chery\n"
    )

    out_file = tmp_path / "output.txt"
    test_args = [
        "diff2typo.py",
        str(diff_file),
        "-o",
        str(out_file),
        "-f",
        "list",
        "--sort",
        "count",
        "--reverse",
        "-d",
        "nonexistent_words.csv",
        "-a",
        "nonexistent_allowed.csv",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    diff2typo.main()

    lines = out_file.read_text().strip().split("\n")
    # Count normal order: apple (3), banana (2), cherry (1)
    # Reverse count order: cherry (1), banana (2), apple (3)
    assert lines == ["cherry", "banana", "apple"]


def test_diff2typo_reverse_dry_run(tmp_path, monkeypatch, caplog):
    diff_file = tmp_path / "test.diff"
    diff_file.write_text(
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-apple\n"
        "+aple\n"
    )

    test_args = [
        "diff2typo.py",
        str(diff_file),
        "-n",
        "-r",
        "-d",
        "nonexistent_words.csv",
        "-a",
        "nonexistent_allowed.csv",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with caplog.at_level("INFO"):
        diff2typo.main()

    assert "Reverse: True" in caplog.text
