from pathlib import Path

import datathon


def test_package_importable():
    assert datathon.__doc__


def test_raw_data_present():
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "bank-additional-full.csv"
    assert path.exists()
