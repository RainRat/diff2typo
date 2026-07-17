import sys
import importlib
import diff2typo

def test_diff2typo_tqdm_fallback_implementation():
    real_tqdm = sys.modules.pop("tqdm", None)
    sys.modules["tqdm"] = None
    try:
        importlib.reload(diff2typo)
        fallback_class = diff2typo.tqdm

        pbar_with_iterable = fallback_class([1, 2, 3])
        assert list(pbar_with_iterable) == [1, 2, 3]

        pbar_no_iterable = fallback_class()
        assert list(pbar_no_iterable) == []

        with fallback_class() as pbar:
            pbar.update(2)
            pbar.close()
            pbar.set_description("Loading")
            pbar.set_postfix(status="done")

    finally:
        if real_tqdm is not None:
            sys.modules["tqdm"] = real_tqdm
        else:
            sys.modules.pop("tqdm", None)
        importlib.reload(diff2typo)
