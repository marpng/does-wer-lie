import numpy as np
import pytest

from analyse_ratings import gwet_ac2, weighted_kappa


def test_perfect_agreement_gives_one():
    a = np.array([0, 1, 2, 2, 1, 0])
    assert weighted_kappa(a, a, power=1) == pytest.approx(1.0)
    assert gwet_ac2(a, a) == pytest.approx(1.0)


def test_complete_disagreement_is_negative():
    a = np.array([0, 2, 0, 2])
    b = np.array([2, 0, 2, 0])
    assert weighted_kappa(a, b, power=1) < 0
    assert gwet_ac2(a, b) < 0


def test_kappa_is_undefined_when_both_raters_give_one_constant_score():
    a = np.array([2, 2, 2, 2])
    assert np.isnan(weighted_kappa(a, a.copy(), power=1))


def test_kappa_is_zero_when_one_rater_always_gives_the_same_score():
    a = np.array([2, 2, 2, 2])
    b = np.array([2, 1, 2, 2])
    assert weighted_kappa(a, b, power=1) == 0.0


def test_kappa_paradox_ac2_stays_high_when_almost_everything_is_fully():
    a = np.array([2] * 9 + [1])
    b = np.array([2] * 8 + [1, 2])
    assert weighted_kappa(a, b, power=1) < 0.1
    assert gwet_ac2(a, b) > 0.7


def test_a_one_point_difference_costs_less_than_a_two_point_difference():
    a = np.array([0, 1, 2, 2, 1, 0, 2, 1])
    near = a.copy(); near[0] = 1
    far = a.copy(); far[0] = 2
    assert weighted_kappa(a, near, power=1) > weighted_kappa(a, far, power=1)
