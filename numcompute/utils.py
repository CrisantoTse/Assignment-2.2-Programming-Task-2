
from numbers import Real
import numpy as np

def _as_float_array(x, *, name="x", ndim=None, finite=False):

    try:
        array = np.asarray(x)
    except ValueError as exc:
        raise ValueError(
            f"{name} must have a regular rectangular shape."
        ) from exc

    if isinstance(x, (list, tuple)):
        try:
            elements = np.asarray(x, dtype=object)
        except ValueError as exc:
            raise ValueError(
                f"{name} must have a regular rectangular shape."
            ) from exc

        for value in elements.flat:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, Real
            ):
                raise TypeError(
                    f"{name} must contain only real numeric values."
                )


    if array.dtype.kind not in "iuf":
        raise TypeError(
            f"{name} must contain real numbers, not boolean, "
            "complex, string or object data."
        )

    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}D; got {array.ndim}D."
        )

    result = array.astype(np.float64, copy=True)

    if finite and not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return result


def _validate_integer(value, *, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}; got {value}."
        )

    return value

def sigmoid(x):
    array = _as_float_array(x)

    result = np.empty_like(array)
    positive = array >= 0

    with np.errstate(under="ignore"):
        result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))

        exp_x = np.exp(array[~positive])
        result[~positive] = exp_x / (1.0 + exp_x)

    return result

def _validate_axis(axis, ndim):
    axis = _validate_integer(axis, name="axis")

    if not -ndim <= axis < ndim:
        raise ValueError(f"axis must satisfy {-ndim} <= axis < {ndim}; got {axis}.")

    return axis % ndim


def softmax(x, axis=-1):
    array = _as_float_array(x)

    if array.ndim == 0:
        raise ValueError("x must have at least one dimension.")

    axis = _validate_axis(axis, array.ndim)

    if array.shape[axis] == 0:
        raise ValueError("The softmax axis must not be empty.")

    has_nan = np.any(np.isnan(array), axis=axis, keepdims=True)
    positive_inf = np.isposinf(array)
    inf_count = np.sum(positive_inf, axis=axis, keepdims=True)
    all_negative_inf = np.all(np.isneginf(array), axis=axis, keepdims=True)

    regular = (~has_nan & (inf_count == 0) & ~all_negative_inf)

    safe_array = np.where(regular, array, 0.0)
    maximum = np.max(safe_array, axis=axis, keepdims=True)

    with np.errstate(over="ignore"):
        shifted = safe_array - maximum

    with np.errstate(under="ignore"):
        exp_values = np.exp(shifted)
        result = exp_values / np.sum(exp_values, axis=axis, keepdims=True)

    inf_probabilities = (positive_inf / np.maximum(inf_count, 1))
    result = np.where(inf_count > 0, inf_probabilities, result)

    result = np.where(has_nan | all_negative_inf, np.nan, result)

    return result

def logsumexp(x, axis=None, keepdims=False):
    array = _as_float_array(x)

    if not isinstance(keepdims, (bool, np.bool_)):
        raise TypeError("keepdims must be a boolean.")

    if axis is not None:
        axis = _validate_axis(axis, array.ndim)

    maximum = np.max(array, axis=axis, keepdims=True, initial=-np.inf,)

    regular = np.isfinite(maximum)
    base = np.where(regular, maximum, 0.0)

    safe_array = np.where(regular, array, -np.inf)

    with np.errstate(over="ignore"):
        shifted = safe_array - base

    with np.errstate(under="ignore"):
        total = np.sum(np.exp(shifted), axis=axis, keepdims=True,)

    log_total = np.full_like(total, -np.inf)
    np.log(total, out=log_total, where=total > 0)

    result = np.where(regular, base + log_total, maximum,)

    if not keepdims:
        result = np.squeeze(result, axis=axis)

    return result[()] if result.ndim == 0 else result

def euclidean_distances(X, Y=None):
    X_array = _as_float_array(X, name="X", ndim=2, finite=True)

    if Y is None:
        Y_array = X_array
    else:
        Y_array = _as_float_array(Y, name="Y", ndim=2, finite=True)

    if X_array.shape[1] == 0 or Y_array.shape[1] == 0:
        raise ValueError("X and Y must have at least one feature.")

    if X_array.shape[1] != Y_array.shape[1]:
        raise ValueError("X and Y must have the same number of features.")

    n_samples = X_array.shape[0]
    m_samples = Y_array.shape[0]

    result = np.empty((n_samples, m_samples), dtype=np.float64)

    block_size = 64

    for i in range(0, n_samples, block_size):
        X_block = X_array[i:i + block_size]

        for j in range(0, m_samples, block_size):
            Y_block = Y_array[j:j + block_size]

            with np.errstate(over="ignore", under="ignore"):
                difference = (X_block[:, None, :]- Y_block[None, :, :])

                result[i:i + block_size,j:j + block_size,] = np.hypot.reduce(difference, axis=2)

    return result

def batch_iter(X,batch_size=32,shuffle=False,random_state=None,drop_last=False,):
    batch_size = _validate_integer(batch_size, name="batch_size", minimum=1)

    for name, value in (("shuffle", shuffle),("drop_last", drop_last),):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean.")

    if random_state is not None:
        random_state = _validate_integer(random_state,name="random_state",minimum=0,)

    array = np.asarray(X)

    if array.ndim == 0:
        raise ValueError("X must have at least one dimension.")

    n_samples = array.shape[0]

    if drop_last:
        stop = n_samples - (n_samples % batch_size)
    else:
        stop = n_samples

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)

    for start in range(0, stop, batch_size):
        end = min(start + batch_size, stop)

        if shuffle:
            yield array[indices[start:end]]
        else:
            yield array[start:end].copy()