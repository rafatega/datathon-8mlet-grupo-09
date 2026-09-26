import pytest

from datathon.features import ARMS, SEGMENTS, age_band, segment_index, segment_of


def test_arms_order():
    assert ARMS == ("cellular", "telephone")


def test_segments_order():
    assert SEGMENTS == (
        "jovem_sem_sucesso", "jovem_com_sucesso",
        "adulto_sem_sucesso", "adulto_com_sucesso",
        "senior_sem_sucesso", "senior_com_sucesso",
    )


@pytest.mark.parametrize("age,band", [(17, "jovem"), (30, "jovem"), (31, "adulto"),
                                      (59, "adulto"), (60, "senior"), (98, "senior")])
def test_age_band_boundaries(age, band):
    assert age_band(age) == band


@pytest.mark.parametrize("poutcome,suffix", [("success", "com_sucesso"),
                                             ("failure", "sem_sucesso"),
                                             ("nonexistent", "sem_sucesso")])
def test_segment_of_poutcome(poutcome, suffix):
    assert segment_of(45, poutcome) == f"adulto_{suffix}"


def test_segment_index_matches_order():
    for i, seg in enumerate(SEGMENTS):
        assert segment_index(seg) == i
