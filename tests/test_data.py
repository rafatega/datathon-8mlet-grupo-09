import numpy as np
import pandas as pd
import pytest

from datathon.data import (
    LEAKAGE_COLUMNS,
    block_arm_rates,
    clean,
    dataset_sha256,
    evaluation_set,
    load_raw,
)


def _raw(rows):
    cols = ["age", "poutcome", "contact", "month", "duration", "campaign", "y"]
    return pd.DataFrame(rows, columns=cols)


@pytest.fixture
def toy():
    return _raw([
        (25, "nonexistent", "telephone", "may", 100, 1, "no"),
        (40, "success", "telephone", "may", 200, 2, "yes"),
        (40, "success", "telephone", "may", 200, 2, "yes"),   # duplicata exata
        (65, "failure", "cellular", "jun", 50, 1, "yes"),
        (35, "nonexistent", "telephone", "jun", 80, 3, "no"),
        (33, "nonexistent", "cellular", "jun", 90, 1, "no"),
        (50, "nonexistent", "cellular", "may", 60, 1, "no"),  # novo bloco "may"
    ])


def test_clean_removes_leakage_and_duplicates(toy):
    df = clean(toy)
    for col in LEAKAGE_COLUMNS:
        assert col not in df.columns
    assert "y" not in df.columns
    assert len(df) == 6


def test_clean_reward_binary_and_order(toy):
    df = clean(toy)
    assert df["reward"].tolist() == [0, 1, 1, 0, 0, 0]
    assert df["age"].tolist() == [25, 40, 65, 35, 33, 50]


def test_clean_blocks_are_contiguous_months(toy):
    df = clean(toy)
    assert df["block"].tolist() == [1, 1, 2, 2, 2, 3]


def test_clean_arm_and_segment(toy):
    df = clean(toy)
    assert df["arm"].tolist() == [1, 1, 0, 1, 0, 0]
    assert df["segment"].iloc[1] == "adulto_com_sucesso"
    assert df["segment_idx"].iloc[2] == 4  # senior_sem_sucesso


def test_evaluation_set_keeps_only_blocks_with_both_arms(toy):
    ev = evaluation_set(clean(toy))
    assert ev["block"].unique().tolist() == [2]
    assert ev["propensity"].round(4).tolist() == [0.6667, 0.3333, 0.6667]


def test_block_arm_rates(toy):
    ev = evaluation_set(clean(toy))
    rates = block_arm_rates(ev)
    assert rates.shape == (3, 2)
    np.testing.assert_allclose(rates[0], [0.5, 0.0])


def test_real_dataset_contract():
    raw = load_raw()
    df = clean(raw)
    assert len(raw) == 41188
    assert len(df) == 41176
    assert df[["age", "poutcome", "contact", "reward", "block"]].isna().sum().sum() == 0
    assert set(df["reward"].unique()) == {0, 1}
    ev = evaluation_set(df)
    assert ev["block"].min() == 3                      # mai-jun/2008 só telefone: excluídos
    assert ev["propensity"].between(0, 1, inclusive="neither").all()
    assert len(dataset_sha256()) == 64
