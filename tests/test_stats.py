import numpy as np

from pylluminator.stats import norm_exp_convolution

def test_norm_exp_convolution():
    signal_val = np.array([2, 3, 5])
    assert (norm_exp_convolution(1.2, None, 2, signal_val, 2) == signal_val).all()
    assert (norm_exp_convolution(1.2, 3, None, signal_val, 2) == signal_val).all()
    assert (norm_exp_convolution(None, 1, 2, signal_val, 2) == signal_val).all()
    assert (norm_exp_convolution(2, -1, 2, signal_val, 2) == signal_val).all()
    assert (norm_exp_convolution(2, 1, -2, signal_val, 2) == signal_val).all()
