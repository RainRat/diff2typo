from multitool import discovery_mode
import json

def test_discovery_mode_basic(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello hello hello hello hello\nworld world world world world\nhelo\nworldd\n")

    output_file = tmp_path / "output.txt"

    discovery_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        rare_max=1,
        freq_min=5,
        min_dist=1,
        max_dist=1,
        output_format='line'
    )

    content = output_file.read_text().splitlines()
    assert "helo -> hello [D]" in content
    assert "worldd -> world [I]" in content
    assert len(content) == 2

def test_discovery_mode_json_format(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("apple apple apple apple apple\naple\n")

    output_file = tmp_path / "output.json"

    discovery_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        rare_max=1,
        freq_min=5,
        min_dist=1,
        max_dist=1,
        output_format='json'
    )

    with open(output_file) as f:
        data = json.load(f)
    assert data == {"aple": "apple [D]"}

def test_discovery_mode_thresholds(tmp_path):
    input_file = tmp_path / "input.txt"
    # apple appears 3 times, aple appears 2 times
    input_file.write_text("apple apple apple\naple aple\n")

    output_file = tmp_path / "output.txt"

    # If rare_max is 1, aple (count 2) should not be considered a typo
    discovery_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        rare_max=1,
        freq_min=3,
        min_dist=1,
        max_dist=1,
        output_format='line'
    )
    assert output_file.read_text() == ""

    # If rare_max is 2, aple should be considered a typo
    discovery_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        rare_max=2,
        freq_min=3,
        min_dist=1,
        max_dist=1,
        output_format='line'
    )
    assert "aple -> apple" in output_file.read_text()

def test_discovery_mode_distance(tmp_path):
    input_file = tmp_path / "input.txt"
    # correction: "distance"
    # typo 1: "distnce" (dist 1)
    # typo 2: "distnc" (dist 2)
    text = " ".join(["distance"] * 10) + "\ndistnce\ndistnc\n"
    input_file.write_text(text)

    output_file = tmp_path / "output.txt"

    # Max distance 1
    discovery_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        rare_max=1,
        freq_min=5,
        min_dist=1,
        max_dist=1,
        output_format='line'
    )
    content = output_file.read_text().splitlines()
    assert "distnce -> distance [D]" in content
    assert "distnc -> distance" not in content

    # Max distance 2
    discovery_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        rare_max=1,
        freq_min=5,
        min_dist=1,
        max_dist=2,
        output_format='line'
    )
    content = output_file.read_text().splitlines()
    assert "distnce -> distance [D]" in content
    assert "distnc -> distance [M]" in content

def test_discovery_mode_smart_splitting(tmp_path):
    input_file = tmp_path / "input.txt"
    # "Message" appears 10 times, "Mesage" appears once inside a CamelCase word
    text = " ".join(["Message"] * 10) + "\nUpdatedMesage\n"
    input_file.write_text(text)

    output_file = tmp_path / "output.txt"

    # Without smart splitting, "UpdatedMesage" is treated as one word and ignored (rare, but no similar freq word)
    discovery_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        rare_max=1,
        freq_min=5,
        min_dist=1,
        max_dist=1,
        output_format='line',
        smart=False
    )
    assert "mesage -> message" not in output_file.read_text()

    # With smart splitting, "UpdatedMesage" is split into "Updated" and "Mesage"
    # "Mesage" (rare) is matched against "Message" (freq)
    discovery_mode(
        input_files=[str(input_file)],
        output_file=str(output_file),
        min_length=3,
        max_length=100,
        process_output=True,
        rare_max=1,
        freq_min=5,
        min_dist=1,
        max_dist=1,
        output_format='line',
        smart=True
    )
    assert "mesage -> message" in output_file.read_text()
