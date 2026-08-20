import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos


@pytest.fixture
def empty_config_file(tmp_path):
    config = tmp_path / "test_config.yaml"
    config.write_text("{}", encoding="utf-8")
    return str(config)


def test_gentypos_markdown_format_stdout(capsys, empty_config_file):
    test_args = [
        "gentypos.py",
        "hello",
        "-c", empty_config_file,
        "--no-filter",
        "-f", "markdown"
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().split('\n') if line.strip()]

    assert lines[0] == "| Typo | Correction |"
    assert lines[1] == "| :--- | :--- |"
    assert len(lines) > 2
    for row in lines[2:]:
        assert row.startswith("| `")
        assert row.endswith("` |")
        assert "`hello`" in row


def test_gentypos_md_format_stdout(capsys, empty_config_file):
    test_args = [
        "gentypos.py",
        "hello",
        "-c", empty_config_file,
        "--no-filter",
        "-f", "md"
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.strip().split('\n') if line.strip()]

    assert lines[0] == "| Typo | Correction |"
    assert lines[1] == "| :--- | :--- |"
    assert len(lines) > 2
    for row in lines[2:]:
        assert "`hello`" in row


def test_gentypos_auto_detect_md_extension(tmp_path, empty_config_file):
    output_md = tmp_path / "typos.md"
    test_args = [
        "gentypos.py",
        "hello",
        "-c", empty_config_file,
        "--no-filter",
        "-o", str(output_md)
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    assert output_md.exists()
    content = output_md.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.strip().split('\n') if line.strip()]

    assert lines[0] == "| Typo | Correction |"
    assert lines[1] == "| :--- | :--- |"
    assert len(lines) > 2


def test_gentypos_auto_detect_markdown_extension(tmp_path, empty_config_file):
    output_markdown = tmp_path / "typos.markdown"
    test_args = [
        "gentypos.py",
        "hello",
        "-c", empty_config_file,
        "--no-filter",
        "-o", str(output_markdown)
    ]
    with patch.object(sys, 'argv', test_args):
        gentypos.main()

    assert output_markdown.exists()
    content = output_markdown.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.strip().split('\n') if line.strip()]

    assert lines[0] == "| Typo | Correction |"
    assert lines[1] == "| :--- | :--- |"
    assert len(lines) > 2


def test_format_typos_markdown_direct():
    mapping = {"helo": "hello", "hlelo": "hello"}
    result = gentypos.format_typos(mapping, "markdown")

    assert result[0] == "| Typo | Correction |"
    assert result[1] == "| :--- | :--- |"
    assert "| `helo` | `hello` |" in result
    assert "| `hlelo` | `hello` |" in result
