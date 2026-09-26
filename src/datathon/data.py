"""Carga, limpeza e preparação da base Bank Marketing."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from datathon.features import ARMS, segment_index, segment_of

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"

# duration: só é conhecida após a ligação. campaign: inclui o contato atual.
LEAKAGE_COLUMNS = ("duration", "campaign")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def dataset_sha256(path: Path = RAW_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas e leakage, cria reward, bloco temporal, braço e segmento."""
    df = raw.drop_duplicates(keep="first").reset_index(drop=True)
    df = df.drop(columns=list(LEAKAGE_COLUMNS))
    df["reward"] = (df["y"] == "yes").astype(int)
    df = df.drop(columns=["y"])
    # A base não tem ano: um bloco é uma sequência contígua do mesmo mês.
    df["block"] = (df["month"] != df["month"].shift()).cumsum().astype(int)
    df["arm"] = df["contact"].map({arm: i for i, arm in enumerate(ARMS)}).astype(int)
    df["segment"] = [segment_of(a, p) for a, p in zip(df["age"], df["poutcome"])]
    df["segment_idx"] = df["segment"].map(segment_index).astype(int)
    return df


def evaluation_set(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém só blocos com os dois canais e calcula a propensão logada por bloco."""
    n_arms = df.groupby("block")["arm"].transform("nunique")
    ev = df[n_arms == len(ARMS)].copy()
    block_size = ev.groupby("block")["arm"].transform("size")
    arm_count = ev.groupby(["block", "arm"])["arm"].transform("size")
    ev["propensity"] = arm_count / block_size
    return ev.reset_index(drop=True)


def block_arm_rates(ev: pd.DataFrame) -> np.ndarray:
    """Conversão logada de cada braço no bloco de cada evento, shape (n, 2)."""
    rates = ev.groupby(["block", "arm"])["reward"].mean().unstack("arm")
    rates = rates.reindex(columns=range(len(ARMS)))
    return rates.loc[ev["block"]].to_numpy()
