import numpy as np
import pandas as pd
import pytest

from src.clean import map_outcome, parse_dti, parse_ltv, to_num


@pytest.mark.parametrize("raw,band,num", [
    ("<20%", "<20", 15.0),
    ("20%-<30%", "20-30", 25.0),
    ("30%-<36%", "30-36", 33.0),
    ("36", "36-40", 36.0),
    ("40", "36-40", 40.0),
    ("41", "41-45", 41.0),
    ("45", "41-45", 45.0),
    ("46", "46-49", 46.0),
    ("49", "46-49", 49.0),
    ("50%-60%", "50-60", 55.0),
    (">60%", ">60", 65.0),
])
def test_parse_dti_valid(raw, band, num):
    assert parse_dti(raw) == (band, num)


@pytest.mark.parametrize("raw", ["NA", "Exempt", "", "garbage", "12"])
def test_parse_dti_missing(raw):
    band, num = parse_dti(raw)
    assert band == "Missing" and np.isnan(num)


def test_to_num_tokens():
    s = to_num(pd.Series(["100", "NA", "Exempt", "", "-5"]))
    assert s.iloc[0] == 100 and s.iloc[4] == -5
    assert s.iloc[1:4].isna().all()


def test_parse_ltv_bounds():
    s = parse_ltv(pd.Series(["80.0", "96.5", "Exempt", "0", "151", "150"]))
    assert list(s.iloc[[0, 1, 5]]) == [80.0, 96.5, 150.0]
    assert s.iloc[[2, 3, 4]].isna().all()


def test_map_outcome():
    s = map_outcome(pd.Series(["1", "2", "3", "4", "6", "8"]))
    assert list(s.iloc[:3]) == [1.0, 1.0, 0.0]
    assert s.iloc[3:].isna().all()
