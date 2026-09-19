import pytest

from score_benchmark import wilson_interval


def test_wilson_interval_zero_of_twenty_contains_zero() -> None:
    lower, upper = wilson_interval(0, 20)
    assert lower <= 0 <= upper


def test_wilson_interval_twenty_of_twenty_contains_one() -> None:
    lower, upper = wilson_interval(20, 20)
    assert lower <= 1 <= upper


def test_wilson_interval_ten_of_twenty_contains_half() -> None:
    lower, upper = wilson_interval(10, 20)
    assert lower <= 0.5 <= upper


@pytest.mark.parametrize(
    ("successes", "total"),
    [(0, 20), (10, 20), (20, 20), (1, 20), (19, 20)],
)
def test_wilson_interval_bounds_stay_in_unit_interval(
    successes: int,
    total: int,
) -> None:
    lower, upper = wilson_interval(successes, total)
    assert 0 <= lower <= upper <= 1


def test_wilson_interval_rejects_invalid_totals_and_successes() -> None:
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(1, -1)
    with pytest.raises(ValueError):
        wilson_interval(-1, 20)
    with pytest.raises(ValueError):
        wilson_interval(21, 20)
