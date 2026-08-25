"""Configuration tests for compact TensorBoard diagnostics."""

from spatex.tensorboard import example_map


def test_example_map_skips_unknown_genes_and_deduplicates_indices():
    logging = {
        "spatial_examples": [
            {"sample_id": "slide_a", "genes": ["B", "missing", "B"]},
            {"sample_id": "slide_b", "genes": ["A"]},
        ]
    }
    assert example_map(logging, ("A", "B")) == {
        "slide_a": (1,),
        "slide_b": (0,),
    }
