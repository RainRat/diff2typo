import sys
import importlib
from unittest.mock import patch
import multitool

def test_multitool_tqdm_fallback_activation():
    orig_dict = sys.modules["multitool"].__dict__.copy()
    try:
        with patch.dict(sys.modules, {"tqdm": None}):
            importlib.reload(multitool)
            assert hasattr(multitool, "tqdm")

            fallback_class = multitool.tqdm
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
    finally:
        sys.modules["multitool"].__dict__.clear()
        sys.modules["multitool"].__dict__.update(orig_dict)


def test_multitool_tomllib_fallback_activation():
    orig_dict = sys.modules["multitool"].__dict__.copy()
    try:
        with patch.dict(sys.modules, {"tomllib": None}):
            importlib.reload(multitool)
            assert not multitool._TOMLLIB_AVAILABLE
    finally:
        sys.modules["multitool"].__dict__.clear()
        sys.modules["multitool"].__dict__.update(orig_dict)
