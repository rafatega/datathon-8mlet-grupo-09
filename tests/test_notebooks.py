import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize("name", ["01_eda.ipynb", "02_bandit_experimento.ipynb"])
def test_notebook_executes(name, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--output-dir", str(tmp_path), "--ExecutePreprocessor.timeout=600",
         str(NOTEBOOKS / name)],
        cwd=NOTEBOOKS, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
