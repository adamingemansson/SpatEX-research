import json

import pytest

from spatex.presentation import _reports


def test_report_parser_requires_named_v2_report(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"kind": "spatex_evaluation", "version": 2}))
    parsed = _reports([f"model={path}"])
    assert parsed[0][0] == "model"
    with pytest.raises(ValueError):
        _reports([str(path)])
