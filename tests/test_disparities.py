"""Hand-computed checks of AIR and the two-proportion z-test."""
import math

import pandas as pd
import pytest

from src.disparities import disparity_table


@pytest.fixture
def toy():
    # Reference: 90/100 approved. Group B: 60/80 approved. Group C: 9/10 approved.
    rows = ([("Ref", 1)] * 90 + [("Ref", 0)] * 10 + [("B", 1)] * 60 + [("B", 0)] * 20
            + [("C", 1)] * 9 + [("C", 0)] * 1)
    return pd.DataFrame(rows, columns=["grp", "approved"])


def test_counts_sum(toy):
    t = disparity_table(toy, "grp", "Ref")
    assert t["apps"].sum() == len(toy)
    assert t["approvals"].sum() == toy["approved"].sum()


def test_air_by_hand(toy):
    t = disparity_table(toy, "grp", "Ref").set_index("group")
    assert t.loc["B", "AIR"] == pytest.approx((60 / 80) / (90 / 100))  # 0.8333
    assert not t.loc["B", "flag_4_5ths"]
    assert t.loc["Ref", "AIR"] == 1.0
    assert t.loc["B", "diff_pp"] == pytest.approx(-15.0)


def test_ztest_by_hand(toy):
    t = disparity_table(toy, "grp", "Ref").set_index("group")
    p1, p2, n1, n2 = 60 / 80, 90 / 100, 80, 100
    pooled = (60 + 90) / (n1 + n2)
    z = (p1 - p2) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    p = math.erfc(abs(z) / math.sqrt(2))   # two-sided normal p-value
    assert t.loc["B", "p_ztest"] == pytest.approx(p, rel=1e-6)


def test_small_cell_triggers_fisher(toy):
    t = disparity_table(toy, "grp", "Ref").set_index("group")
    assert not math.isnan(t.loc["C", "p_fisher"])   # C has a cell of 1
    assert t.loc["C", "small_n"]
