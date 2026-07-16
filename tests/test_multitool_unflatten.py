
import json
import subprocess
import os
import traceback

def test_unflatten_basic():
    input_content = "user.name -> John\nuser.age -> 30\n"
    with open("test_input.txt", "w") as f:
        f.write(input_content)

    result = subprocess.run(
        ["python3", "multitool.py", "unflatten", "test_input.txt", "--output-format", "json", "--raw"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data == {"user": {"name": "John", "age": "30"}}
    os.remove("test_input.txt")

def test_unflatten_list():
    input_content = "items.0 -> apple\nitems.1 -> banana\n"
    with open("test_input.txt", "w") as f:
        f.write(input_content)

    result = subprocess.run(
        ["python3", "multitool.py", "unflatten", "test_input.txt", "--output-format", "json", "--raw"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data == {"items": ["apple", "banana"]}
    os.remove("test_input.txt")

def test_unflatten_key():
    input_content = "users.0.name -> Alice\nusers.1.name -> Bob\nother.data -> 123\n"
    with open("test_input.txt", "w") as f:
        f.write(input_content)

    result = subprocess.run(
        ["python3", "multitool.py", "unflatten", "test_input.txt", "-k", "users", "--output-format", "json", "--raw"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data == [{"name": "Alice"}, {"name": "Bob"}]
    os.remove("test_input.txt")

if __name__ == "__main__":
    try:
        test_unflatten_basic()
        print("Basic test passed!")
        test_unflatten_list()
        print("List test passed!")
        test_unflatten_key()
        print("Key test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()
        exit(1)

def test_unflatten_ambiguous_numeric_keys():
    # Test that non-sequential or zero-padded numeric keys do not cause a crash
    # and are correctly preserved as a dictionary.
    input_content = 'root.0 -> a\nroot.01 -> b\n'
    with open("test_input_ambiguous.txt", "w") as f:
        f.write(input_content)

    result = subprocess.run(
        ["python3", "multitool.py", "unflatten", "test_input_ambiguous.txt", "--output-format", "json", "--raw"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    # Since keys are '0' and '01', they don't form a simple [0, 1] sequence of string keys
    # and should be kept as a dictionary.
    assert data == {"root": {"0": "a", "01": "b"}}
    os.remove("test_input_ambiguous.txt")
