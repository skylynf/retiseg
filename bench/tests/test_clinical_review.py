"""The clinical review pipeline tests (clinical/tests/test_clinical_pipeline.py, docs/clinical/28) run with
bench/tests, so GitHub Actions covers them without a change to the workflow. The module is loaded under its own
name, so collecting clinical/tests in the same session does not clash."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "clinical" / "tests" / "test_clinical_pipeline.py"
_spec = importlib.util.spec_from_file_location("clinical_review_pipeline_tests", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if k.startswith("test_") or k == "synthetic"})
