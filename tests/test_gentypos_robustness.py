import sys
import logging
import pytest
import yaml
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import gentypos

def test_load_file_none():
    """Test that load_file returns an empty set when path is None (Line 377)."""
    assert gentypos.load_file(None) == set()

def test_run_typo_generation_length_filtering():
    """Test that _run_typo_generation respects min_length and max_length (Lines 614, 616)."""
    word_list = ["a", "cat", "hippopotamus"]
    all_words = set()
    # min_length=3, max_length=5. "a" is too short, "hippopotamus" is too long.
    settings = SimpleNamespace(
        min_length=3,
        max_length=5,
        repeat_modifications=1,
        typo_types={'deletion': True},
        enable_adjacent_substitutions=False,
        enable_custom_substitutions=False,
        transposition_distance=1
    )

    # We pass empty adjacent_keys and custom_subs since they are disabled in settings
    typos = gentypos._run_typo_generation(word_list, all_words, settings, {}, {}, quiet=True)

    # "cat" -> "at", "ct", "ca"
    assert "at" in typos
    assert "ct" in typos
    assert "ca" in typos
    # Ensure "a" (from "a") and typos from "hippopotamus" are not there
    assert len(typos) == 3

def test_setup_generation_tools_merge_non_list(tmp_path):
    """Test merging logic when existing custom substitution is not a list (Line 556)."""
    subs_file = tmp_path / "extra_subs.yaml"
    subs_file.write_text(yaml.dump({"a": ["i"]}))

    settings = SimpleNamespace(
        custom_substitutions_config={"a": "e"}, # String, not a list
        substitutions_file=str(subs_file),
        enable_custom_substitutions=True,
        enable_adjacent_substitutions=False
    )

    adj, custom = gentypos._setup_generation_tools(settings)
    assert custom["a"] == {"e", "i"}

def test_setup_generation_tools_merge_existing_none(tmp_path):
    """Test merging logic when existing custom substitution is None."""
    subs_file = tmp_path / "extra_subs.yaml"
    subs_file.write_text(yaml.dump({"a": ["i"]}))

    settings = SimpleNamespace(
        custom_substitutions_config={"a": None},
        substitutions_file=str(subs_file),
        enable_custom_substitutions=True,
        enable_adjacent_substitutions=False
    )

    adj, custom = gentypos._setup_generation_tools(settings)
    assert custom["a"] == {"i"}

def test_setup_generation_tools_custom_disabled():
    """Test disabling custom substitutions (Lines 561-562)."""
    settings = SimpleNamespace(
        custom_substitutions_config={"a": ["e"]},
        enable_custom_substitutions=False,
        enable_adjacent_substitutions=False
    )
    adj, custom = gentypos._setup_generation_tools(settings)
    assert custom == {}

def test_load_custom_substitutions_none_safe():
    """
    Test that load_custom_substitutions safely handles None values.
    """
    custom_subs = {"a": None, "b": ["e", None]}
    # Should skip None value for "a" and skip None inside list for "b"
    result = gentypos.load_custom_substitutions(custom_subs)
    assert "a" not in result
    assert result["b"] == {"e"}

def test_load_custom_substitutions_non_iterable_safe():
    """
    Test that load_custom_substitutions safely handles non-iterable values.
    """
    custom_subs = {"a": 123} # Not iterable
    result = gentypos.load_custom_substitutions(custom_subs)
    assert result["a"] == {"123"}

def test_load_custom_substitutions_string_safe():
    """
    Test that load_custom_substitutions handles a single string as iterable (Line 197).
    """
    custom_subs = {"a": "e"} # String
    result = gentypos.load_custom_substitutions(custom_subs)
    assert result["a"] == {"e"}
