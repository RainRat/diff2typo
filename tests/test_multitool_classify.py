import subprocess
import pytest

def test_classify_mode_basic():
    """Test that classify mode correctly labels various types of typos."""
    input_text = (
        "teh -> the\n"      # [T]
        "helo -> hello\n"    # [D]
        "helloo -> hello\n"  # [I]
        "hella -> hello\n"   # [R]
        "helko -> hello\n"   # [K]
        "abc -> def\n"       # [M]
    )
    process = subprocess.Popen(
        ['python', 'multitool.py', 'classify', '-m', '2'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=input_text)

    assert "teh -> the [T]" in stdout
    assert "helo -> hello [D]" in stdout
    assert "helloo -> hello [I]" in stdout
    assert "hella -> hello [R]" in stdout
    assert "helko -> hello [K]" in stdout
    assert "abc -> def [M]" in stdout

def test_classify_mode_show_dist():
    """Test that --show-dist adds distance information to the labels."""
    input_text = "teh -> the"
    process = subprocess.Popen(
        ['python', 'multitool.py', 'classify', '-m', '2', '--show-dist'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=input_text)

    assert "teh -> the [T] (dist: 2)" in stdout

def test_classify_mode_formats():
    """Test that classify mode works with different output formats."""
    input_text = "teh -> the"
    formats = ['json', 'csv', 'md-table']

    for fmt in formats:
        process = subprocess.Popen(
            ['python', 'multitool.py', 'classify', '-m', '2', '--format', fmt],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_text)

        if fmt == 'json':
            assert '"teh": "the [T]"' in stdout
        elif fmt == 'csv':
            assert 'teh,the [T]' in stdout
        elif fmt == 'md-table':
            assert '| teh | the [T] |' in stdout
