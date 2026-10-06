import math
import pytest

from qnn_edge_agent.eval.stats import mcnemar_exact, wilson_ci


def test_wilson_ci_known_values():
    t_zero = wilson_ci(0, 10)
    assert t_zero.rate == 0.0
    assert t_zero.ci_low == 0.0
    assert math.isclose(t_zero.ci_high, 0.2775328, rel_tol=1e-4)

    t_all = wilson_ci(10, 10)
    assert t_all.rate == 1.0
    assert math.isclose(t_all.ci_low, 0.7224672, rel_tol=1e-4)
    assert math.isclose(t_all.ci_high, 1.0, rel_tol=1e-4)

    t_half = wilson_ci(50, 100)
    assert t_half.rate == 0.5
    assert math.isclose(t_half.ci_low, 0.4038315, rel_tol=1e-4)
    assert math.isclose(t_half.ci_high, 0.5961685, rel_tol=1e-4)

    t_empty = wilson_ci(0, 0)
    assert t_empty.rate == 0.0
    assert t_empty.n == 0


def test_mcnemar_exact_known_values():
    res_zero = mcnemar_exact([True, False, True], [True, False, True])
    assert res_zero.b == 0
    assert res_zero.c == 0
    assert res_zero.n_discordant == 0
    assert res_zero.p_value == 1.0

    res_skewed = mcnemar_exact([True] * 10 + [False] * 10, [False] * 20)
    assert res_skewed.b == 10
    assert res_skewed.c == 0
    assert res_skewed.n_discordant == 10
    assert math.isclose(res_skewed.p_value, 2.0 / 1024.0, rel_tol=1e-5)

    res_sym = mcnemar_exact(
        [True] * 5 + [False] * 5,
        [False] * 5 + [True] * 5,
    )
    assert res_sym.b == 5
    assert res_sym.c == 5
    assert res_sym.n_discordant == 10
    assert res_sym.p_value == 1.0


def test_mcnemar_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])
