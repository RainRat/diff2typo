import pytest
from multitool import main, _build_parser


def test_multitool_parser_dry_run_short_flag():
    parser = _build_parser()

    # scrub mode
    args_scrub = parser.parse_args(["scrub", "file.txt", "-a", "teh:the", "-n"])
    assert args_scrub.dry_run is True

    # rename mode
    args_rename = parser.parse_args(["rename", ".", "-a", "foo:bar", "-n"])
    assert args_rename.dry_run is True

    # replace mode
    args_replace = parser.parse_args(["replace", "foo", "bar", "file.txt", "-n"])
    assert args_replace.dry_run is True

    # standardize mode
    args_standardize = parser.parse_args(["standardize", ".", "-n"])
    assert args_standardize.dry_run is True


def test_multitool_scrub_short_dry_run_execution(tmp_path, monkeypatch, caplog):
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is teh test.", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["multitool.py", "scrub", str(test_file), "-a", "teh:the", "-I", "-n"]
    )

    with caplog.at_level("WARNING"):
        main()

    # File content should remain unchanged because of dry-run
    assert test_file.read_text(encoding="utf-8") == "This is teh test."
    assert "Would make" in caplog.text or "Dry Run" in caplog.text


def test_multitool_replace_short_dry_run_execution(tmp_path, monkeypatch, caplog):
    test_file = tmp_path / "replace_test.txt"
    test_file.write_text("Hello world", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["multitool.py", "replace", "world", "there", str(test_file), "-I", "-n"]
    )

    with caplog.at_level("WARNING"):
        main()

    # File content should remain unchanged because of dry-run
    assert test_file.read_text(encoding="utf-8") == "Hello world"
    assert "Would make" in caplog.text or "Dry Run" in caplog.text
