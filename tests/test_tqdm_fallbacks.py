import sys
import importlib
from unittest.mock import patch

import pytest

import diff2typo
import gentypos


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
