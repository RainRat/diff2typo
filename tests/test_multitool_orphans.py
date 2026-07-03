import os
import pytest
from multitool import main
import sys
from io import StringIO

def test_orphans_mode_files(tmp_path):
    # Setup:
    # file1.md references file2.md
    # file3.md is an orphan
    # image.png is referenced by file1.md
    # unused.png is an orphan

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

    # Run orphans mode on all files
    sys.argv = ["multitool.py", "orphans", str(file1), str(file2), str(file3), str(image), str(unused), "--output-format", "json"]

    output = StringIO()
    sys.stdout = output

    try:
        main()
    except SystemExit:
        pass
    finally:
        sys.stdout = sys.__stdout__

    import json
    result = json.loads(output.getvalue())

    # result is a dict because of _write_paired_output fallback or structured data
    # Actually Orphans mode uses _write_paired_output which for JSON returns a dict {item: reason}

    assert str(file3) in result
    assert result[str(file3)] == "Unreferenced file"
    assert str(unused) in result
    assert result[str(unused)] == "Unreferenced file"
    # file1 is also an orphan because it's not referenced by any other file
    assert str(file1) in result

def test_orphans_mode_labels(tmp_path):
    # Setup:
    # file.md has [label1]: url (used)
    # file.md has [label2]: url (unused)

    md_file = tmp_path / "file.md"
    md_file.write_text("[text][label1]\n\n[label1]: http://example.com\n[label2]: http://unused.com")

    sys.argv = ["multitool.py", "orphans", str(md_file), "--output-format", "json"]

    output = StringIO()
    sys.stdout = output

    try:
        main()
    except SystemExit:
        pass
    finally:
        sys.stdout = sys.__stdout__

    import json
    result = json.loads(output.getvalue())

    orphan_label = f"{str(md_file)} (label: label2)"
    assert orphan_label in result
    assert result[orphan_label] == "Unused Markdown reference definition"
    assert f"{str(md_file)} (label: label1)" not in result

def test_orphans_mode_shortcut_labels(tmp_path):
    # Setup:
    # file.md has [label1] (shortcut link, used)
    # file.md has [label2]: url (unused)

    md_file = tmp_path / "file.md"
    md_file.write_text("Check [label1] for more info.\n\n[label1]: http://example.com\n[label2]: http://unused.com")

    sys.argv = ["multitool.py", "orphans", str(md_file), "--output-format", "json"]

    output = StringIO()
    sys.stdout = output

    try:
        main()
    except SystemExit:
        pass
    finally:
        sys.stdout = sys.__stdout__

    import json
    result = json.loads(output.getvalue())

    assert f"{str(md_file)} (label: label1)" not in result
    assert f"{str(md_file)} (label: label2)" in result

def test_orphans_mode_images_ref_style(tmp_path):
    # Setup:
    # file.md has ![alt][imglabel] (used)
    # img.png is defined by [imglabel]: img.png

    md_file = tmp_path / "file.md"
    img_file = tmp_path / "img.png"
    img_file.write_text("png")

    md_file.write_text("![alt][imglabel]\n\n[imglabel]: img.png")

    sys.argv = ["multitool.py", "orphans", str(md_file), str(img_file), "--output-format", "json"]

    output = StringIO()
    sys.stdout = output

    try:
        main()
    except SystemExit:
        pass
    finally:
        sys.stdout = sys.__stdout__

    import json
    result = json.loads(output.getvalue())

    assert str(img_file) not in result
    assert f"{str(md_file)} (label: imglabel)" not in result
