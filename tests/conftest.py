from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "fixtures"

requires_oasdiff = pytest.mark.skipif(
    shutil.which("oasdiff") is None and not os.environ.get("OASDIFF_BIN"),
    reason="oasdiff binary not available",
)


@pytest.fixture
def write_config(tmp_path: Path):
    def _write(text: str) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    return _write
