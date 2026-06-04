import pytest
import os
import tempfile
from multitool import replace_mode

def test_replace_ignore_case():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("The quick brown fox jumps over the lazy dog.\n")
        f.flush()
        temp_name = f.name

    try:
        output_file = temp_name + ".out"
        replace_mode(
            input_files=[temp_name],
            old_text="the",
            new_text="that",
            output_file=output_file,
            ignore_case=True
        )

        with open(output_file, 'r') as f:
            content = f.read()
            # "The" and "the" should both be replaced by literal "that"
            assert "that quick brown fox jumps over that lazy dog." in content
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        if os.path.exists(output_file):
            os.remove(output_file)

def test_replace_regex_backref_smart_case():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("ERROR_code and error_CODE and Error_Code\n")
        f.flush()
        temp_name = f.name

    try:
        output_file = temp_name + ".out"
        replace_mode(
            input_files=[temp_name],
            old_text=r"([a-z]+)_([a-z]+)",
            new_text=r"\2_\1",
            output_file=output_file,
            use_regex=True,
            smart_case=True
        )

        with open(output_file, 'r') as f:
            content = f.read()
            # "ERROR_code" -> "Code_ERROR" (starts with capital)
            # "error_CODE" -> "code_error" (starts with lower)
            # "Error_Code" -> "Code_Error" (starts with capital)
            assert "Code_ERROR and code_error and Code_Error" in content
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        if os.path.exists(output_file):
            os.remove(output_file)

def test_replace_smart_case():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("THE quick brown fox jumps over The lazy dog.\n")
        f.flush()
        temp_name = f.name

    try:
        output_file = temp_name + ".out"
        replace_mode(
            input_files=[temp_name],
            old_text="the",
            new_text="that",
            output_file=output_file,
            smart_case=True
        )

        with open(output_file, 'r') as f:
            content = f.read()
            # "THE" -> "THAT", "The" -> "That"
            assert "THAT quick brown fox jumps over That lazy dog." in content
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        if os.path.exists(output_file):
            os.remove(output_file)

def test_replace_regex_smart_case():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("FIXme and fixME and FixMe\n")
        f.flush()
        temp_name = f.name

    try:
        output_file = temp_name + ".out"
        replace_mode(
            input_files=[temp_name],
            old_text=r"fixme",
            new_text="resolved",
            output_file=output_file,
            use_regex=True,
            smart_case=True
        )

        with open(output_file, 'r') as f:
            content = f.read()
            # smart case applies to regex matches too
            # "FIXme" (starts with capital) -> "Resolved"
            # "fixME" (starts with lower) -> "resolved"
            # "FixMe" (starts with capital) -> "Resolved"
            assert "Resolved and resolved and Resolved" in content
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        if os.path.exists(output_file):
            os.remove(output_file)
