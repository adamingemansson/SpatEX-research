from pathlib import Path

import numpy as np

from spatex.data import SlideCache, SlideSpec


def _slide(path: Path, value: float) -> SlideSpec:
    np.savez_compressed(
        path,
        image_features=np.full((3, 2), value, dtype=np.float32),
        coordinates=np.zeros((3, 2), dtype=np.float32),
        image_available=np.ones(3, dtype=bool),
        expression=np.full((3, 4), value, dtype=np.float32),
        barcodes=np.asarray(["a", "b", "c"]),
    )
    return SlideSpec(path.stem, "train", "organ", path)


def test_slide_cache_is_bounded(tmp_path):
    first = _slide(tmp_path / "first.npz", 1)
    second = _slide(tmp_path / "second.npz", 2)
    cache = SlideCache(4, capacity=1)
    assert cache.get(first) is cache.get(first)
    cache.get(second)
    assert len(cache._records) == 1
