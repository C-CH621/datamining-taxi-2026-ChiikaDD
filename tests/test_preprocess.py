import numpy as np

from src.evaluate import regression_metrics


def test_regression_metrics_skips_non_finite_pairs():
    y_true = np.array([1.0, 2.0, np.nan, 4.0])
    y_pred = np.array([1.0, 4.0, 3.0, np.inf])

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["rows"] == 4
    assert metrics["evaluated_rows"] == 2
    assert metrics["skipped_rows"] == 2
    assert metrics["rmse"] == np.sqrt(2.0)
    assert metrics["mae"] == 1.0
