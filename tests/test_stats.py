import numpy as np

from pylluminator.stats import norm_exp_convolution, get_factors_from_formula

def test_norm_exp_convolution():
    signal_val = np.array([2, 3, 5])
    assert (norm_exp_convolution(1.2, None, 2, signal_val, 2) == signal_val).all()
    assert (norm_exp_convolution(1.2, 3, None, signal_val, 2) == signal_val).all()
    assert (norm_exp_convolution(None, 1, 2, signal_val, 2) == signal_val).all()
    assert (norm_exp_convolution(2, -1, 2, signal_val, 2) == signal_val).all()
    assert (norm_exp_convolution(2, 1, -2, signal_val, 2) == signal_val).all()

def test_get_factors_from_formula():
    formula = 'sample_type + sample_number'
    factors = get_factors_from_formula(formula)

    assert len(factors) == 2
    assert 'sample_type' in factors
    assert 'sample_number' in factors