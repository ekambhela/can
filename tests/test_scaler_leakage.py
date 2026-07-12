"""Regression test for the target-leakage bug fixed in model/train.py.

Before the fix, load_frame() z-scored each drug's target using the mean/std of
*all* cell lines — including the ones train.py later held out for testing. That
let test-set statistics bleed into the training target (inflating R^2). These
tests pin the contract that the scaler is fit on TRAIN indices only.
"""

import numpy as np

from model.train import apply_target_scaler, fit_target_scaler


def test_scaler_never_sees_test_indices():
    # index 4 is a held-out test line with an extreme IC50; 0-3 are train.
    y = np.array([1.0, 1.1, 0.9, 1.0, 1000.0])
    train_idx = [0, 1, 2, 3]

    mu, sd = fit_target_scaler(y, train_idx)

    # the scaler reflects ONLY the train lines ...
    assert abs(mu - float(np.mean(y[train_idx]))) < 1e-9
    # ... and was NOT influenced by the extreme test line (all-lines mean ~200).
    assert abs(mu - float(np.mean(y))) > 100

    # the test line still gets scaled (by train stats) to an extreme sensitivity,
    # but its value never fed back into the mean/std the model was trained on.
    sens = apply_target_scaler(y, mu, sd)
    assert sens[4] < sens[:4].min()  # highest IC50 -> most-resistant


def test_scaler_ignores_nans_in_train():
    y = np.array([1.0, np.nan, 2.0, np.nan])
    mu, _sd = fit_target_scaler(y, [0, 1, 2, 3])
    assert abs(mu - 1.5) < 1e-9  # mean of the two non-NaN train values


def test_scaler_constant_target_is_safe():
    # a drug with no variance in train shouldn't divide by zero.
    y = np.array([5.0, 5.0, 5.0, 5.0])
    mu, sd = fit_target_scaler(y, [0, 1, 2, 3])
    assert sd == 1.0
    assert np.all(np.isfinite(apply_target_scaler(y, mu, sd)))
