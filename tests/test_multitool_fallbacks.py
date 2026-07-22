import sys
import importlib.util
from unittest.mock import patch

def test_multitool_fallbacks_tqdm_and_tomllib():
    with patch.dict(sys.modules, {"tqdm": None, "tomllib": None}):
        spec = importlib.util.spec_from_file_location("multitool_fallback_test", "multitool.py")
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert module is not None

        spec.loader.exec_module(module)

        assert hasattr(module, "tqdm")
        fallback_class = module.tqdm

        instance = fallback_class([1, 2, 3])
        assert list(instance) == [1, 2, 3]

        instance_none = fallback_class(None)
        assert list(instance_none) == []

        instance.update(5)
        instance.set_description("Test description")
        instance.set_postfix(test_key="test_value")
        instance.close()

        with instance as pbar:
            pass

        assert hasattr(module, "_TOMLLIB_AVAILABLE")
        assert module._TOMLLIB_AVAILABLE is False
