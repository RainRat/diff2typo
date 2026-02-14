import subprocess
import pytest
import os
import json

def run_multitool(args, input_text=None):
    process = subprocess.Popen(
        ['python', 'multitool.py'] + args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=input_text)
    return stdout, stderr, process.returncode

def test_swap_arrow():
    input_text = "teh -> the\ntaht -> that\n"
    stdout, stderr, code = run_multitool(['swap', '-f', 'arrow'], input_text=input_text)
    assert code == 0
    assert "the -> teh" in stdout
    assert "that -> taht" in stdout

def test_swap_csv():
    input_text = "teh,the\ntaht,that\n"
    stdout, stderr, code = run_multitool(['swap', '-f', 'csv'], input_text=input_text)
    assert code == 0
    assert "the,teh" in stdout
    assert "that,taht" in stdout

def test_swap_table():
    input_text = 'teh = "the"\ntaht = "that"\n'
    stdout, stderr, code = run_multitool(['swap', '-f', 'table'], input_text=input_text)
    assert code == 0
    assert 'the = "teh"' in stdout
    assert 'that = "taht"' in stdout

def test_swap_raw():
    # Test that --raw preserves casing/punctuation
    input_text = "TeH -> The!\n"
    stdout, stderr, code = run_multitool(['swap', '--raw', '-f', 'arrow'], input_text=input_text)
    assert code == 0
    assert "The! -> TeH" in stdout

def test_swap_clean():
    # Test that default behavior cleans
    input_text = "TeH -> The!\n"
    stdout, stderr, code = run_multitool(['swap', '-f', 'arrow'], input_text=input_text)
    assert code == 0
    assert "the -> teh" in stdout

def test_swap_json_input(tmp_path):
    d = {"teh": "the", "taht": "that"}
    f = tmp_path / "test.json"
    f.write_text(json.dumps(d))

    stdout, stderr, code = run_multitool(['swap', str(f), '-f', 'arrow'])
    assert code == 0
    assert "the -> teh" in stdout
    assert "that -> taht" in stdout

def test_swap_filtering():
    # Test min-length filtering
    input_text = "a -> apple\n"
    stdout, stderr, code = run_multitool(['swap', '--min-length', '3'], input_text=input_text)
    assert code == 0
    assert "apple -> a" not in stdout # 'a' is too short
