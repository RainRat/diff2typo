import json
import sys
from multitool import main


def test_orphans_mode_files(tmp_path, monkeypatch, capsys):
    file1 = tmp_path / "file1.md"
    file2 = tmp_path / "file2.md"
    file3 = tmp_path / "file3.md"
    image = tmp_path / "image.png"
    unused = tmp_path / "unused.png"

    file1.write_text("[link](file2.md) ![alt](image.png)")
    file2.write_text("Hello")
    file3.write_text("World")
    image.write_text("image content")
    unused.write_text("unused content")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multitool.py",
            "orphans",
            str(file1),
            str(file2),
            str(file3),
            str(image),
            str(unused),
            "--output-format",
            "json",
        ],
    )
    main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert str(file3) in result
    assert result[str(file3)] == "Unreferenced file"
    assert str(unused) in result
    assert result[str(unused)] == "Unreferenced file"
    assert str(file1) in result


def test_orphans_mode_labels(tmp_path, monkeypatch, capsys):
    md_file = tmp_path / "file.md"
    md_file.write_text(
        "[text][label1]\n\n[label1]: http://example.com\n[label2]: http://unused.com"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["multitool.py", "orphans", str(md_file), "--output-format", "json"],
    )
    main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)

    orphan_label = f"{str(md_file)} (label: label2)"
    assert orphan_label in result
    assert result[orphan_label] == "Unused Markdown reference definition"
    assert f"{str(md_file)} (label: label1)" not in result


def test_orphans_mode_shortcut_labels(tmp_path, monkeypatch, capsys):
    md_file = tmp_path / "file.md"
    md_file.write_text(
        "Check [label1] for more info.\n\n[label1]: http://example.com\n[label2]: http://unused.com"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["multitool.py", "orphans", str(md_file), "--output-format", "json"],
    )
    main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert f"{str(md_file)} (label: label1)" not in result
    assert f"{str(md_file)} (label: label2)" in result


def test_orphans_mode_images_ref_style(tmp_path, monkeypatch, capsys):
    md_file = tmp_path / "file.md"
    img_file = tmp_path / "img.png"
    img_file.write_text("png")

    md_file.write_text("![alt][imglabel]\n\n[imglabel]: img.png")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multitool.py",
            "orphans",
            str(md_file),
            str(img_file),
            "--output-format",
            "json",
        ],
    )
    main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert str(img_file) not in result
    assert f"{str(md_file)} (label: imglabel)" not in result


def test_orphans_mode_arrow_format_and_limit(tmp_path, monkeypatch, capsys):
    f1 = tmp_path / "doc1.md"
    f2 = tmp_path / "doc2.md"
    f3 = tmp_path / "doc3.md"
    f1.write_text("[a]: http://unused1.com\n[b]: http://unused2.com")
    f2.write_text("hello")
    f3.write_text("world")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multitool.py",
            "orphans",
            str(f1),
            str(f2),
            str(f3),
            "-f",
            "arrow",
            "-L",
            "2",
        ],
    )
    main()

    captured = capsys.readouterr()
    assert "ORPHANS ANALYSIS" in captured.out
    assert "Item" in captured.out
    assert "Reason" in captured.out
    assert "Unreferenced file" in captured.out
