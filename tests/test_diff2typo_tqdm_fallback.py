import sys
import importlib

def test_diff2typo_tqdm_fallback_behavior():
    original_tqdm = sys.modules.get("tqdm")

    try:
        sys.modules["tqdm"] = None
        import diff2typo
        importlib.reload(diff2typo)

        fallback_tqdm_class = diff2typo.tqdm

        items = [1, 2, 3]
        t = fallback_tqdm_class(items)
        assert list(t) == [1, 2, 3]

        t_none = fallback_tqdm_class(None)
        assert list(t_none) == []

        with fallback_tqdm_class(items) as t_ctx:
            assert t_ctx is t_ctx
            t_ctx.update(2)
            t_ctx.set_description("Processing")
            t_ctx.set_postfix(status="ok")
            t_ctx.close()

    finally:
        if original_tqdm is not None:
            sys.modules["tqdm"] = original_tqdm
        else:
            sys.modules.pop("tqdm", None)

        import diff2typo
        importlib.reload(diff2typo)
