import sys
import importlib
from unittest.mock import patch

import pytest

import diff2typo
import gentypos
import multitool
import typostats


def test_diff2typo_tqdm_fallback_activation():
    with patch.dict(sys.modules, {"tqdm": None}):
        importlib.reload(diff2typo)
        assert hasattr(diff2typo, "tqdm")
        assert diff2typo.tqdm.__module__ == "diff2typo"

        fallback_class = diff2typo.tqdm
        instance = fallback_class([1, 2, 3])
        assert list(instance) == [1, 2, 3]

        instance_none = fallback_class(None)
        assert list(instance_none) == []

        instance.update(5)
        instance.set_description("Description")
        instance.set_postfix(key="value")
        instance.close()

        with instance as pbar:
            pass

    importlib.reload(diff2typo)


def test_gentypos_tqdm_fallback_activation():
    with patch.dict(sys.modules, {"tqdm": None}):
        importlib.reload(gentypos)
        assert hasattr(gentypos, "tqdm")
        assert gentypos.tqdm.__module__ == "gentypos"

        fallback_class = gentypos.tqdm
        instance = fallback_class([4, 5, 6])
        assert list(instance) == [4, 5, 6]

        instance_none = fallback_class(None)
        assert list(instance_none) == []

        instance.update(5)
        instance.set_description("Description")
        instance.set_postfix(key="value")
        instance.close()

        with instance as pbar:
            pass

    importlib.reload(gentypos)


def test_multitool_tqdm_fallback_activation():
    with patch.dict(sys.modules, {"tqdm": None}):
        importlib.reload(multitool)
        assert hasattr(multitool, "tqdm")
        assert multitool.tqdm.__module__ == "multitool"

        fallback_class = multitool.tqdm
        instance = fallback_class([7, 8, 9])
        assert list(instance) == [7, 8, 9]

        instance_none = fallback_class(None)
        assert list(instance_none) == []

        instance.update(5)
        instance.set_description("Description")
        instance.set_postfix(key="value")
        instance.close()

        with instance as pbar:
            pass

    importlib.reload(multitool)


def test_typostats_tqdm_fallback_activation(tmp_path):
    with patch.dict(sys.modules, {"tqdm": None}):
        importlib.reload(typostats)
        assert typostats._TQDM_AVAILABLE is False
        assert typostats.tqdm is None

        test_file = tmp_path / "typos.txt"
        test_file.write_text("teh -> the\n")
        pairs = list(typostats._extract_pairs([str(test_file)], quiet=False))
        assert pairs == [("teh", "the")]

    importlib.reload(typostats)


def test_diff2typo_yaml_fallback_activation():
    with patch.dict(sys.modules, {"yaml": None}):
        importlib.reload(diff2typo)
        assert diff2typo._YAML_AVAILABLE is False

    importlib.reload(diff2typo)
