import os
import shutil
import tempfile
import pytest
from multitool import main
import sys
from unittest.mock import patch

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

def test_rename_basic_file(temp_dir):
    # Setup files
    file_path = os.path.join(temp_dir, "teh_file.txt")
    with open(file_path, "w") as f:
        f.write("content")

    mapping_path = os.path.join(temp_dir, "mapping.csv")
    with open(mapping_path, "w") as f:
        f.write("teh,the\n")

    # Run rename mode (dry run)
    test_args = ["multitool.py", "rename", file_path, "--mapping", mapping_path, "--dry-run"]
    with patch.object(sys, 'argv', test_args):
        with patch('sys.stdout', new_callable=pytest.importorskip('io').StringIO) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            assert "teh_file.txt" in output and "the_file.txt" in output
            assert "Original" in output and "New Name" in output

    # Verify file not actually renamed
    assert os.path.exists(file_path)
    assert not os.path.exists(os.path.join(temp_dir, "the_file.txt"))

    # Run rename mode (in-place)
    test_args = ["multitool.py", "rename", file_path, "--mapping", mapping_path, "--in-place"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify file renamed
    assert not os.path.exists(file_path)
    assert os.path.exists(os.path.join(temp_dir, "the_file.txt"))

def test_rename_directory_and_nested_file(temp_dir):
    # Setup: subdir/teh_nested.txt
    subdir = os.path.join(temp_dir, "teh_dir")
    os.mkdir(subdir)
    nested_file = os.path.join(subdir, "teh_nested.txt")
    with open(nested_file, "w") as f:
        f.write("content")

    mapping_path = os.path.join(temp_dir, "mapping.csv")
    with open(mapping_path, "w") as f:
        f.write("teh,the\n")

    # Run rename mode (in-place)
    # We pass both the dir and the file
    test_args = ["multitool.py", "rename", subdir, nested_file, "--mapping", mapping_path, "--in-place"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify both renamed (bottom-up ensures nested_file is renamed before subdir)
    new_subdir = os.path.join(temp_dir, "the_dir")
    new_nested_file = os.path.join(new_subdir, "the_nested.txt")

    assert os.path.exists(new_subdir)
    assert os.path.exists(new_nested_file)
    assert not os.path.exists(subdir)
    assert not os.path.exists(nested_file)

def test_rename_smart_case(temp_dir):
    # Setup
    file1 = os.path.join(temp_dir, "TehFile.txt")
    file2 = os.path.join(temp_dir, "teh_file.txt")
    for f in [file1, file2]:
        with open(f, "w") as fh:
            fh.write("c")

    mapping_path = os.path.join(temp_dir, "mapping.csv")
    with open(mapping_path, "w") as f:
        f.write("teh,the\n")

    test_args = ["multitool.py", "rename", file1, file2, "--mapping", mapping_path, "--in-place", "--smart-case"]
    with patch.object(sys, 'argv', test_args):
        main()

    assert os.path.exists(os.path.join(temp_dir, "TheFile.txt"))
    assert os.path.exists(os.path.join(temp_dir, "the_file.txt"))
    assert not os.path.exists(file1)
    assert not os.path.exists(file2)
