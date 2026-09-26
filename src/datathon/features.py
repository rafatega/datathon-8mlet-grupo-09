"""Segmentação do cliente — fonte única usada no treino, na avaliação e na API."""

ARMS = ("cellular", "telephone")

_BANDS = ("jovem", "adulto", "senior")
_HISTORY = ("sem_sucesso", "com_sucesso")
SEGMENTS = tuple(f"{band}_{hist}" for band in _BANDS for hist in _HISTORY)


def age_band(age: int) -> str:
    """Faixa etária: jovem (<=30), adulto (31-59), senior (>=60)."""
    if age <= 30:
        return "jovem"
    if age <= 59:
        return "adulto"
    return "senior"


def segment_of(age: int, poutcome: str) -> str:
    """Segmento do cliente a partir de informações disponíveis antes do contato."""
    history = "com_sucesso" if poutcome == "success" else "sem_sucesso"
    return f"{age_band(age)}_{history}"


def segment_index(segment: str) -> int:
    return SEGMENTS.index(segment)
