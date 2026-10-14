"""Runs the paper's collect end-to-end check under pytest where pandas is installed (docs/ops/21 open issue 29 a).

docs/paper/retiseg_paper/paper/scripts/test_collect_runs.py scores a toy dataset with the real
evaluator and checks collect_runs.py, including the E5 toy tree. It lives outside bench/tests and
needs pandas, so nobody ran it on the machine that wrote the code (review A1 was found that way).
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "docs" / "paper" / "retiseg_paper" / "paper" / "scripts"


def test_collect_runs_end_to_end_when_pandas_is_installed():
    pytest.importorskip("pandas")
    sys.path.insert(0, str(SCRIPTS))
    import test_collect_runs  # noqa: E402  (the paper's script; its main() asserts and prints two ok lines)

    test_collect_runs.main()
