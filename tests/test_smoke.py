from counterstrat.constants import BUY_BINS, TICK_RATE


def test_constants():
    assert TICK_RATE == 64
    assert list(BUY_BINS) == ["full_eco", "semi_eco", "semi_buy", "full_buy"]
