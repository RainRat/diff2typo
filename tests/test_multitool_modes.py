
import sys
import json
import pytest
from pathlib import Path

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import multitool

@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    monkeypatch.setattr(multitool, "tqdm", lambda iterable, *_, **__: iterable)

# ZIP MODE
def test_zip_mode_basic(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.write_text("teh\n")
    f2 = tmp_path / "f2.txt"
    f2.write_text("the\n")
    out = tmp_path / "out.txt"
    multitool.zip_mode([str(f1)], str(f2), str(out), 1, 100, False)
    assert out.read_text().strip() == "teh -> the"

# SWAP MODE
def test_swap_mode_basic(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("the -> teh\n")
    out = tmp_path / "out.txt"
    multitool.swap_mode([str(f)], str(out), 1, 100, False)
    assert out.read_text().strip() == "teh -> the"

# PAIRS MODE
def test_pairs_mode_basic(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("teh,the\n")
    out = tmp_path / "out.txt"
    multitool.pairs_mode([str(f)], str(out), 1, 100, False)
    assert out.read_text().strip() == "teh -> the"

# CONFLICT MODE
def test_conflict_mode_basic(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("teh -> the\nteh -> tea\n")
    out = tmp_path / "out.txt"
    multitool.conflict_mode([str(f)], str(out), 1, 100, False)
    content = out.read_text()
    assert "teh" in content
    assert "the" in content
    assert "tea" in content

# SIMILARITY MODE
def test_similarity_mode_basic(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("cat -> bat\n")
    out = tmp_path / "out.txt"
    multitool.similarity_mode([str(f)], str(out), 1, 100, False, min_dist=1, max_dist=1)
    assert out.read_text().strip() == "cat -> bat"

# FUZZYMATCH MODE
def test_fuzzymatch_mode_basic(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.write_text("cat\n")
    f2 = tmp_path / "f2.txt"
    f2.write_text("bat\n")
    out = tmp_path / "out.txt"
    multitool.fuzzymatch_mode([str(f1)], str(f2), str(out), 1, 100, False, min_dist=1, max_dist=1)
    assert out.read_text().strip() == "cat -> bat"

# NEAR DUPLICATES MODE
def test_near_duplicates_mode_basic(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("cat\nbat\n")
    out = tmp_path / "out.txt"
    multitool.near_duplicates_mode([str(f)], str(out), 1, 100, False, min_dist=1, max_dist=1)
    content = out.read_text().strip()
    assert content == "cat -> bat" or content == "bat -> cat"
