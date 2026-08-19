import json
import os
import tempfile
from unittest.mock import patch

import diff2typo


SAMPLE_DIFF = """diff --git a/file.txt b/file.txt
index 123..456 100644
--- a/file.txt
+++ b/file.txt
@@ -1,2 +1,2 @@
-This is teh first line.
+This is the first line.
-He had a recieve error.
+He had a receive error.
"""


def test_format_typos_json():
    typos = ["teh -> the", "recieve -> receive", "lonelyword"]
    res = diff2typo.format_typos(typos, "json")
    output = "\n".join(res)
    data = json.loads(output)
    assert isinstance(data, list)
    assert len(data) == 3
    assert data[0] == {"typo": "teh", "correction": "the"}
    assert data[1] == {"typo": "recieve", "correction": "receive"}
    assert data[2] == {"typo": "lonelyword", "correction": ""}


def test_format_typos_yaml():
    typos = ["teh -> the", "recieve -> receive"]
    res = diff2typo.format_typos(typos, "yaml")
    output = "\n".join(res)
    if diff2typo._YAML_AVAILABLE:
        import yaml
        data = yaml.safe_load(output)
        assert isinstance(data, list)
        assert data[0] == {"typo": "teh", "correction": "the"}
        assert data[1] == {"typo": "recieve", "correction": "receive"}


def test_format_typos_yaml_fallback(caplog):
    typos = ["teh -> the"]
    with patch("diff2typo._YAML_AVAILABLE", False):
        res = diff2typo.format_typos(typos, "yaml")
        output = "\n".join(res)
        data = json.loads(output)
        assert data[0] == {"typo": "teh", "correction": "the"}
        assert "PyYAML not installed" in caplog.text


def test_main_json_format_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_file = os.path.join(tmpdir, "test.diff")
        out_file = os.path.join(tmpdir, "output.json")
        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DIFF)

        test_args = ["diff2typo.py", diff_file, "-o", out_file, "-f", "json", "-q", "-d", "nonexistent.csv"]
        with patch("sys.argv", test_args):
            diff2typo.main()

        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        typos = {item["typo"]: item["correction"] for item in data}
        assert typos.get("teh") == "the"
        assert typos.get("recieve") == "receive"


def test_main_yaml_format_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_file = os.path.join(tmpdir, "test.diff")
        out_file = os.path.join(tmpdir, "output.yaml")
        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DIFF)

        test_args = ["diff2typo.py", diff_file, "-o", out_file, "-f", "yaml", "-q", "-d", "nonexistent.csv"]
        with patch("sys.argv", test_args):
            diff2typo.main()

        with open(out_file, "r", encoding="utf-8") as f:
            content = f.read()
        if diff2typo._YAML_AVAILABLE:
            import yaml
            data = yaml.safe_load(content)
            assert isinstance(data, list)
            typos = {item["typo"]: item["correction"] for item in data}
            assert typos.get("teh") == "the"


def test_main_auto_extension_detection_json_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_file = os.path.join(tmpdir, "test.diff")
        json_out = os.path.join(tmpdir, "out.json")
        yaml_out = os.path.join(tmpdir, "out.yaml")
        yml_out = os.path.join(tmpdir, "out.yml")
        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DIFF)

        test_args = ["diff2typo.py", diff_file, "-o", json_out, "-q", "-d", "nonexistent.csv"]
        with patch("sys.argv", test_args):
            diff2typo.main()

        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)

        test_args = ["diff2typo.py", diff_file, "-o", yml_out, "-q", "-d", "nonexistent.csv"]
        with patch("sys.argv", test_args):
            diff2typo.main()

        with open(yml_out, "r", encoding="utf-8") as f:
            content = f.read()
        assert content.strip()


def test_main_both_mode_json_and_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_file = os.path.join(tmpdir, "test.diff")
        out_json = os.path.join(tmpdir, "both.json")
        dict_file = os.path.join(tmpdir, "dict.csv")

        # Large dictionary contains 'recieve' as a known typo mapping to 'receive'
        with open(dict_file, "w", encoding="utf-8") as f:
            f.write("recieve,receive\n")

        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DIFF)

        test_args = [
            "diff2typo.py",
            diff_file,
            "-o",
            out_json,
            "-M",
            "both",
            "-f",
            "json",
            "-d",
            dict_file,
            "-q",
        ]
        with patch("sys.argv", test_args):
            diff2typo.main()

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "typos" in data
        assert "corrections" in data
        assert isinstance(data["typos"], list)
        assert isinstance(data["corrections"], list)


def test_main_both_mode_yaml_fallback(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_file = os.path.join(tmpdir, "test.diff")
        out_yaml = os.path.join(tmpdir, "both.yaml")
        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DIFF)

        test_args = [
            "diff2typo.py",
            diff_file,
            "-o",
            out_yaml,
            "-M",
            "both",
            "-f",
            "yaml",
            "-q",
            "-d",
            "nonexistent.csv",
        ]
        with patch("diff2typo._YAML_AVAILABLE", False), patch("sys.argv", test_args):
            diff2typo.main()

        with open(out_yaml, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "typos" in data
        assert "corrections" in data


def test_main_both_mode_yaml_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_file = os.path.join(tmpdir, "test.diff")
        out_yaml = os.path.join(tmpdir, "both.yaml")
        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DIFF)

        test_args = [
            "diff2typo.py",
            diff_file,
            "-o",
            out_yaml,
            "-M",
            "both",
            "-f",
            "yaml",
            "-q",
            "-d",
            "nonexistent.csv",
        ]
        with patch("diff2typo._YAML_AVAILABLE", True), patch("sys.argv", test_args):
            diff2typo.main()

        assert os.path.exists(out_yaml)
        with open(out_yaml, "r", encoding="utf-8") as f:
            content = f.read()
        assert "typos:" in content or "corrections:" in content


def test_main_both_mode_single_items_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_file = os.path.join(tmpdir, "test.diff")
        out_json = os.path.join(tmpdir, "both.json")
        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DIFF)

        test_args = [
            "diff2typo.py",
            diff_file,
            "-o",
            out_json,
            "-M",
            "both",
            "-f",
            "json",
            "-q",
            "-d",
            "nonexistent.csv",
        ]
        with patch("diff2typo.process_typos_mode", return_value=["standalone_typo"]), patch("diff2typo.process_corrections_mode", return_value=["standalone_corr"]), patch("sys.argv", test_args):
            diff2typo.main()

        assert os.path.exists(out_json)
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["typos"] == [{"typo": "standalone_typo", "correction": ""}]
        assert data["corrections"] == [{"typo": "standalone_corr", "correction": ""}]

