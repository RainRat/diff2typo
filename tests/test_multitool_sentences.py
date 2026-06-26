import pytest
import sys
import os
from io import StringIO
from unittest.mock import patch
from multitool import main

def test_sentences_mode_basic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("This is a sentence. This is another! And a third? Yes.")

    with patch('sys.stdout', new=StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'sentences', str(f), '--raw']):
            main()
            output = fake_out.getvalue().strip()

    sentences = output.split('\n')
    assert "This is a sentence." in sentences
    assert "This is another!" in sentences
    assert "And a third?" in sentences
    assert "Yes." in sentences
    assert len(sentences) == 4

def test_sentences_mode_multi_line(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("This is a\nmulti-line sentence. Next\none starts here.")

    with patch('sys.stdout', new=StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'sentences', str(f), '--raw']):
            main()
            output = fake_out.getvalue().strip()

    sentences = output.split('\n')
    assert "This is a multi-line sentence." in sentences
    assert "Next one starts here." in sentences
    assert len(sentences) == 2

def test_count_sentences(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world. Hello world. Goodbye world.")

    with patch('sys.stdout', new=StringIO()) as fake_out:
        # We use --format line to get simple output for verification
        with patch('sys.argv', ['multitool.py', 'count', str(f), '--sentences', '--format', 'line', '--raw']):
            main()
            output = fake_out.getvalue().strip()

    lines = output.split('\n')
    # Expected:
    # Hello world.: 2
    # Goodbye world.: 1
    assert "hello world.: 2" in output.lower()
    assert "goodbye world.: 1" in output.lower()

def test_sentences_mode_cleaning(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("This is a sentence. (With extra  spaces).")

    with patch('sys.stdout', new=StringIO()) as fake_out:
        with patch('sys.argv', ['multitool.py', 'sentences', str(f), '--raw']):
            main()
            output = fake_out.getvalue().strip()

    sentences = output.split('\n')
    assert "This is a sentence." in sentences
    assert "(With extra spaces)." in sentences
