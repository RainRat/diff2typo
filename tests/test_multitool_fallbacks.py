import sys
import importlib.util
from unittest.mock import patch


def test_multitool_fallbacks_via_isolated_load():
    # Load multitool.py under a different module name to avoid touching the main multitool module in sys.modules
    spec = importlib.util.spec_from_file_location("multitool_fallback_test", "multitool.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["multitool_fallback_test"] = module

    # Execute the module with tqdm and tomllib mocked to trigger the ImportErrors
    with patch.dict(sys.modules, {"tqdm": None, "tomllib": None}):
        spec.loader.exec_module(module)

    # Verify fallback tqdm works correctly
    assert hasattr(module, "tqdm")
    fallback_class = module.tqdm
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

    # Verify fallback tomllib works correctly
    assert module._TOMLLIB_AVAILABLE is False

    # Clean up sys.modules
    sys.modules.pop("multitool_fallback_test", None)
