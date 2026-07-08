import pytest
from multitool import main
import sys
import os

def test_links_mode_reference_style(tmp_path, capsys):
    # Create a markdown file with reference-style links
    md_file = tmp_path / "refs.md"
    md_file.write_text("""
[Link Text][label]
![Alt Text][img_label]

[label]: https://example.com/page
[img_label]: /assets/image.png
""")

    # Run links mode
    sys.argv = ["multitool.py", "links", str(md_file), "--raw", "--pairs"]
    main()

    captured = capsys.readouterr()
    assert "Link Text -> https://example.com/page" in captured.out
    assert "Alt Text -> /assets/image.png" in captured.out
