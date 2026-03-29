from multitool import cycles_mode

def test_cycles_mode_simple(tmp_path):
    # a -> b, b -> a
    mapping_file = tmp_path / "cycles.txt"
    mapping_file.write_text("a -> b\nb -> a")

    output_file = tmp_path / "output.txt"

    cycles_mode(
        input_files=[str(mapping_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    content = output_file.read_text()
    assert "a -> b -> a" in content or "b -> a -> b" in content

def test_cycles_mode_conflict_cycle(tmp_path):
    # a -> b, a -> c, c -> a
    mapping_file = tmp_path / "conflict_cycle.txt"
    mapping_file.write_text("a -> b\na -> c\nc -> a")

    output_file = tmp_path / "output.txt"

    cycles_mode(
        input_files=[str(mapping_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    content = output_file.read_text()
    assert "a -> c -> a" in content or "c -> a -> c" in content

def test_cycles_mode_multi_hop(tmp_path):
    # a -> b, b -> c, c -> a
    mapping_file = tmp_path / "cycles_multi.txt"
    mapping_file.write_text("a -> b\nb -> c\nc -> a")

    output_file = tmp_path / "output.txt"

    cycles_mode(
        input_files=[str(mapping_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    content = output_file.read_text()
    assert "a -> b -> c -> a" in content or "b -> c -> a -> b" in content or "c -> a -> b -> c" in content

def test_cycles_mode_no_cycles(tmp_path):
    # a -> b, b -> c
    mapping_file = tmp_path / "no_cycles.txt"
    mapping_file.write_text("a -> b\nb -> c")

    output_file = tmp_path / "output.txt"

    cycles_mode(
        input_files=[str(mapping_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    content = output_file.read_text().strip()
    assert content == ""

def test_cycles_mode_self_loop(tmp_path):
    # a -> a
    mapping_file = tmp_path / "self_loop.txt"
    mapping_file.write_text("a -> a")

    output_file = tmp_path / "output.txt"

    cycles_mode(
        input_files=[str(mapping_file)],
        output_file=str(output_file),
        min_length=1,
        max_length=100,
        process_output=True,
        output_format='line',
        quiet=True,
        clean_items=True
    )

    content = output_file.read_text()
    assert "a -> a" in content
