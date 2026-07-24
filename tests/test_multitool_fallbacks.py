import sys
import importlib.util
from unittest.mock import patch

def test_multitool_tqdm_and_tomllib_fallbacks_isolated():
    with patch.dict(sys.modules, {"tqdm": None, "tomllib": None}):
        spec = importlib.util.spec_from_file_location("multitool_fallback_test", "multitool.py")
        multitool_fallback = importlib.util.module_from_spec(spec)
        sys.modules["multitool_fallback_test"] = multitool_fallback
        spec.loader.exec_module(multitool_fallback)

        assert hasattr(multitool_fallback, "tqdm")
        assert multitool_fallback.tqdm.__module__ == "multitool_fallback_test"

        fb = multitool_fallback.tqdm([1, 2])
        assert list(fb) == [1, 2]

        fb_none = multitool_fallback.tqdm(None)
        assert list(fb_none) == []

        with fb as pbar:
            pass

        fb.update(1)
        fb.close()
        fb.set_description("desc")
        fb.set_postfix(x=1)

        assert not multitool_fallback._TOMLLIB_AVAILABLE

        sys.modules.pop("multitool_fallback_test", None)
