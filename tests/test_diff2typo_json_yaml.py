import json
import pytest
import diff2typo


def test_format_typos_json():
    typos = ["teh -> the", "wrod -> word"]
    res = diff2typo.format_typos(typos, "json")
    parsed = json.loads(res[0])
    assert parsed == {"teh": "the", "wrod": "word"}


def test_format_typos_yaml():
    typos = ["teh -> the", "wrod -> word"]
    res = diff2typo.format_typos(typos, "yaml")
    if diff2typo._YAML_AVAILABLE:
        import yaml
        parsed = yaml.safe_load(res[0])
        assert parsed == {"teh": "the", "wrod": "word"}
    else:
        parsed = json.loads(res[0])
        assert parsed == {"teh": "the", "wrod": "word"}


def test_format_typos_json_single_word():
    typos = ["typoonly"]
    res = diff2typo.format_typos(typos, "json")
    parsed = json.loads(res[0])
    assert parsed == {"typoonly": ""}


def test_format_typos_yaml_no_pyyaml(monkeypatch):
    monkeypatch.setattr(diff2typo, "_YAML_AVAILABLE", False)
    typos = ["teh -> the"]
    res = diff2typo.format_typos(typos, "yaml")
    parsed = json.loads(res[0])
    assert parsed == {"teh": "the"}


def test_main_json_and_yaml_modes(tmp_path, monkeypatch):
    diff_content = (
        "diff --git a/file.txt b/file.txt\n"
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-teh house\n"
        "+the house\n"
    )
    diff_file = tmp_path / "test.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    json_out = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "diff2typo.py",
            str(diff_file),
            "--output",
            str(json_out),
            "--mode",
            "typos",
            "-q",
        ],
    )
    diff2typo.main()

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data == {"teh": "the"}


def test_main_both_mode_json(tmp_path, monkeypatch):
    diff_content = (
        "diff --git a/file.txt b/file.txt\n"
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-teh house\n"
        "+the house\n"
    )
    diff_file = tmp_path / "test.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    json_out = tmp_path / "both.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "diff2typo.py",
            str(diff_file),
            "--output",
            str(json_out),
            "--mode",
            "both",
            "-q",
        ],
    )
    diff2typo.main()

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert "typos" in data
    assert "corrections" in data
    assert data["typos"] == {"teh": "the"}


def test_main_both_mode_yaml(tmp_path, monkeypatch):
    diff_content = (
        "diff --git a/file.txt b/file.txt\n"
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-teh house\n"
        "+the house\n"
    )
    diff_file = tmp_path / "test.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    yaml_out = tmp_path / "both.yaml"
    monkeypatch.setattr(
        "sys.argv",
        [
            "diff2typo.py",
            str(diff_file),
            "--output",
            str(yaml_out),
            "--mode",
            "both",
            "-q",
        ],
    )
    diff2typo.main()

    if diff2typo._YAML_AVAILABLE:
        import yaml
        data = yaml.safe_load(yaml_out.read_text(encoding="utf-8"))
    else:
        data = json.loads(yaml_out.read_text(encoding="utf-8"))

    assert "typos" in data
    assert "corrections" in data
    assert data["typos"] == {"teh": "the"}


def test_main_both_mode_yaml_no_pyyaml(tmp_path, monkeypatch):
    monkeypatch.setattr(diff2typo, "_YAML_AVAILABLE", False)
    diff_content = (
        "diff --git a/file.txt b/file.txt\n"
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-teh house\n"
        "+the house\n"
    )
    diff_file = tmp_path / "test.diff"
    diff_file.write_text(diff_content, encoding="utf-8")

    yaml_out = tmp_path / "both.yml"
    monkeypatch.setattr(
        "sys.argv",
        [
            "diff2typo.py",
            str(diff_file),
            "--output",
            str(yaml_out),
            "--mode",
            "both",
            "-q",
        ],
    )
    diff2typo.main()

    data = json.loads(yaml_out.read_text(encoding="utf-8"))
    assert "typos" in data
    assert "corrections" in data
    assert data["typos"] == {"teh": "the"}
