import json
import os
import subprocess
import pytest

def run_multitool(args, input_data=None):
    cmd = ["python", "multitool.py"] + args
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=input_data)
    return stdout, stderr, process.returncode

def test_diff_simple_items(tmp_path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    file1.write_text("apple\nbanana\ncherry\n")
    file2.write_text("apple\ndate\ncherry\n")

    stdout, stderr, code = run_multitool(["diff", str(file1), str(file2)])

    assert code == 0
    assert "- banana" in stdout
    assert "+ date" in stdout
    assert "apple" not in stdout # Should only show differences

def test_diff_pairs(tmp_path):
    file1 = tmp_path / "pairs1.txt"
    file2 = tmp_path / "pairs2.txt"

    file1.write_text("teh -> the\nwierd -> weird\n")
    file2.write_text("teh -> the\nwierd -> wired\nnew -> newer\n")

    # In pairs mode, wierd changes its correction, and new is added.
    stdout, stderr, code = run_multitool(["diff", str(file1), str(file2), "--pairs"])

    assert code == 0
    assert "+ new -> newer" in stdout
    assert "~ wierd: weird -> wired" in stdout
    assert "teh" not in stdout

def test_diff_json_output(tmp_path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    file1.write_text("apple\nbanana\n")
    file2.write_text("apple\ndate\n")

    stdout, stderr, code = run_multitool(["diff", str(file1), str(file2), "-f", "json"])

    assert code == 0
    data = json.loads(stdout)
    assert data["added"] == ["date"]
    assert data["removed"] == ["banana"]

def test_diff_pairs_json_output(tmp_path):
    file1 = tmp_path / "pairs1.txt"
    file2 = tmp_path / "pairs2.txt"

    file1.write_text("teh -> the\nwierd -> weird\n")
    file2.write_text("teh -> the\nwierd -> wired\nnew -> newer\n")

    stdout, stderr, code = run_multitool(["diff", str(file1), str(file2), "--pairs", "-f", "json"])

    assert code == 0
    data = json.loads(stdout)
    assert data["added"] == {"new": "newer"}
    assert data["removed"] == {}
    assert data["changed"] == {"wierd": "weird -> wired"}

def test_diff_no_file2_error():
    stdout, stderr, code = run_multitool(["diff", "file1.txt"])
    assert code != 0
    assert "requires a secondary file" in stderr
