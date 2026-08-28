import numpy as np

from spatex.structured_metrics import structured_panel_metrics


def test_perfect_structured_field_scores_high():
    generator = np.random.default_rng(4)
    coordinates = np.stack(np.meshgrid(np.arange(6), np.arange(5)), axis=-1).reshape(-1, 2)
    target = generator.normal(size=(len(coordinates), 5))
    result = structured_panel_metrics(
        target,
        target,
        coordinates,
        np.ones(5),
        {"panel": range(5)},
        local_k=3,
        wide_k=6,
    )["panel"]
    assert result["coexpression"]["pcc"] > 0.999
    assert result["moran"]["pcc"] > 0.999
    assert result["gradient_local"]["pcc"] > 0.999
    assert result["spatial_ssim"]["mean"] > 0.999


def test_collapsed_field_is_penalized():
    generator = np.random.default_rng(7)
    coordinates = np.stack(np.meshgrid(np.arange(5), np.arange(5)), axis=-1).reshape(-1, 2)
    target = generator.normal(size=(len(coordinates), 4))
    prediction = np.repeat(target.mean(axis=0, keepdims=True), len(target), axis=0)
    result = structured_panel_metrics(
        prediction,
        target,
        coordinates,
        np.ones(4),
        {"panel": range(4)},
        local_k=3,
        wide_k=5,
    )["panel"]
    assert result["moran"]["pcc"] == 0.0
    assert result["gradient_local"]["energy_ratio"] < 1e-10
