import sys

from diff2typo import format_typos, main

def test_format_typos_markdown_pairs():
    typos = ["teh -> the", "fiel -> file"]
    result = format_typos(typos, "markdown")
    expected = [
        "| Typo | Correction |",
        "| :--- | :--- |",
        "| `teh` | `the` |",
        "| `fiel` | `file` |"
    ]
    assert result == expected

def test_format_typos_markdown_singles():
    typos = ["teh", "fiel"]
    result = format_typos(typos, "md")
    expected = [
        "- `teh`",
        "- `fiel`"
    ]
    assert result == expected

def test_format_typos_markdown_mixed():
    typos = ["teh -> the", "fiel"]
    result = format_typos(typos, "markdown")
    expected = [
        "| Typo | Correction |",
        "| :--- | :--- |",
        "| `teh` | `the` |",
        "",
        "- `fiel`"
    ]
    assert result == expected

def test_main_markdown_auto_detect_extension(monkeypatch, tmp_path):
    diff_content = (
        "diff --git a/test.txt b/test.txt\n"
        "--- a/test.txt\n"
        "+++ b/test.txt\n"
        "@@ -1 +1 @@\n"
        "-teh\n"
        "+the\n"
    )
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    output_file = tmp_path / "output.md"

    test_args = ["diff2typo.py", str(diff_file), "-o", str(output_file), "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "| Typo | Correction |" in content
    assert "| `teh` | `the` |" in content

def test_format_typos_markdown_empty():
    assert format_typos([], "markdown") == []
    assert format_typos([], "md") == []

def test_main_markdown_explicit_flag(monkeypatch, tmp_path):
    diff_content = (
        "diff --git a/test.txt b/test.txt\n"
        "--- a/test.txt\n"
        "+++ b/test.txt\n"
        "@@ -1 +1 @@\n"
        "-teh\n"
        "+the\n"
    )
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    output_file = tmp_path / "output.txt"

    test_args = ["diff2typo.py", str(diff_file), "-o", str(output_file), "-f", "md", "-q"]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "| Typo | Correction |" in content
    assert "| `teh` | `the` |" in content

def test_main_markdown_corrections_mode(monkeypatch, tmp_path):
    diff_content = (
        "diff --git a/test.txt b/test.txt\n"
        "--- a/test.txt\n"
        "+++ b/test.txt\n"
        "@@ -1 +1 @@\n"
        "-teh\n"
        "+the\n"
    )
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    dict_file = tmp_path / "words.csv"
    dict_file.write_text("teh\n", encoding="utf-8")

    output_file = tmp_path / "output.md"

    test_args = [
        "diff2typo.py",
        str(diff_file),
        "-o", str(output_file),
        "-M", "corrections",
        "-d", str(dict_file),
        "-q"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "| Typo | Correction |" in content
    assert "| `teh` | `the` |" in content

def test_main_markdown_audit_mode(monkeypatch, tmp_path):
    diff_content = (
        "diff --git a/test.txt b/test.txt\n"
        "--- a/test.txt\n"
        "+++ b/test.txt\n"
        "@@ -1 +1 @@\n"
        "-the\n"
        "+teh\n"
    )
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    dict_file = tmp_path / "words.csv"
    dict_file.write_text("the\n", encoding="utf-8")

    output_file = tmp_path / "output.md"

    test_args = [
        "diff2typo.py",
        str(diff_file),
        "-o", str(output_file),
        "-M", "audit",
        "-d", str(dict_file),
        "-q"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "| Typo | Correction |" in content
    assert "| `the` | `teh` |" in content

def test_main_markdown_both_mode(monkeypatch, tmp_path):
    diff_content = (
        "diff --git a/test.txt b/test.txt\n"
        "--- a/test.txt\n"
        "+++ b/test.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-fiel\n"
        "+file\n"
        "-teh\n"
        "+the\n"
    )
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    dict_file = tmp_path / "words.csv"
    dict_file.write_text("teh,thee\nfile\n", encoding="utf-8")

    output_file = tmp_path / "output.txt"

    test_args = [
        "diff2typo.py",
        str(diff_file),
        "-o", str(output_file),
        "-f", "markdown",
        "-M", "both",
        "-d", str(dict_file),
        "-q"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "### Typos" in content
    assert "### Corrections" in content
    assert "| Typo | Correction |" in content

def test_main_markdown_both_mode_partial_empty(monkeypatch, tmp_path):
    diff_content = (
        "diff --git a/test.txt b/test.txt\n"
        "--- a/test.txt\n"
        "+++ b/test.txt\n"
        "@@ -1 +1 @@\n"
        "-teh\n"
        "+the\n"
    )
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    # When "teh" is in large_dictionary (simple word without fixes in words.csv),
    # "teh" is excluded from typos_list but included in corrections_list (since it's a known typo in words_mapping).
    dict_file = tmp_path / "words.csv"
    dict_file.write_text("teh\n", encoding="utf-8")

    output_file_1 = tmp_path / "output1.md"
    test_args_1 = [
        "diff2typo.py",
        str(diff_file),
        "-o", str(output_file_1),
        "-M", "both",
        "-d", str(dict_file),
        "-q"
    ]
    monkeypatch.setattr(sys, "argv", test_args_1)
    main()

    content1 = output_file_1.read_text(encoding="utf-8")
    assert "### Typos" not in content1
    assert "### Corrections" in content1

    # When "teh" is not in large_dictionary/mapping, typos_list is populated, corrections_list is empty.
    dict_file.write_text("file\n", encoding="utf-8")
    output_file_2 = tmp_path / "output2.md"
    test_args_2 = [
        "diff2typo.py",
        str(diff_file),
        "-o", str(output_file_2),
        "-M", "both",
        "-d", str(dict_file),
        "-q"
    ]
    monkeypatch.setattr(sys, "argv", test_args_2)
    main()

    content2 = output_file_2.read_text(encoding="utf-8")
    assert "### Typos" in content2
    assert "### Corrections" not in content2
