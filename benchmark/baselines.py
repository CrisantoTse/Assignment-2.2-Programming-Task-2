from numcompute.utils import _as_float_array, _validate_integer
import math
import numpy as np

def loop_mean(values):
    data = _as_float_array(values, ndim=1, finite=True)
    
    if data.size == 0:
        raise ValueError("values must not be empty.")

    total = 0.0
    for value in data:
        total += float(value)

    return total / data.size

def loop_standardize(X):
    data = _as_float_array(X, ndim=2, finite=True)
    rows, columns = data.shape

    if rows == 0 or columns == 0:
        raise ValueError("X must have samples and features.")

    result = np.empty_like(data)

    for column in range(columns):
        total = 0.0
        for row in range(rows):
            total += float(data[row, column])
        mean = total / rows

        squared_total = 0.0
        for row in range(rows):
            difference = float(data[row, column]) - mean
            squared_total += difference * difference

        scale = math.sqrt(squared_total / rows) or 1.0

        for row in range(rows):
            result[row, column] = (
                float(data[row, column]) - mean
            ) / scale

    return result

def loop_mse(y_true, y_pred):
    true = _as_float_array(y_true, name="y_true", finite=True)
    pred = _as_float_array(y_pred, name="y_pred", finite=True)

    if true.ndim not in (1, 2) or pred.ndim not in (1, 2):
        raise ValueError("Inputs must be 1D or 2D.")
    if true.shape != pred.shape:
        raise ValueError("Inputs must have identical shapes.")

    if true.size == 0:
        raise ValueError("Inputs must not be empty.")
    total = 0.0
    for actual, predicted in zip(true.flat, pred.flat):
        difference = float(actual) - float(predicted)
        total += difference * difference
    
    return total / true.size

def full_sort_topk(values, k):
    data = _as_float_array(values, ndim=1, finite=True)
    k = _validate_integer(k, name="k", minimum=0)

    if k > data.size:
        raise ValueError("k must not exceed the number of values.")

    indices = np.arange(data.size)

    order = np.lexsort((indices, -data))

    selected = order[:k]
    return data[selected], selected