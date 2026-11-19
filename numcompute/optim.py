
from numbers import Real
import numpy as np
from .utils import _as_float_array


def _validate_method(method):
    if not isinstance(method, str):
        raise TypeError("method must be a string.")

    if method not in ("central", "forward"):
        raise ValueError("method must be 'central' or 'forward'.")

    return method


def _validate_step(h):
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, Real):
        raise TypeError("h must be a real numeric scalar.")

    try:
        value = float(h)
    except (OverflowError, ValueError) as exc:
        raise ValueError("h must be representable as finite float64.") from exc

    if not np.isfinite(value):
        raise ValueError("h must be finite.")

    if value <= 0.0:
        raise ValueError(f"h must be positive; got {value}.")

    return value



def _prepare(x, method, h):
    method = _validate_method(method)
    h = _validate_step(h)
    array = _as_float_array(x, name="x", ndim=1, finite=True)

    if array.size == 0:
        raise ValueError("x must contain at least one value.")

    with np.errstate(over="ignore"):
        upper = array + h
        lower = array - h if method == "central" else array.copy()

    for point in (upper, lower) if method == "central" else (upper,):
        if not np.all(np.isfinite(point)):
            raise ValueError("x + h and x - h must remain finite.")

        if np.any(point == array):
            raise ValueError(
                "h is too small to change every coordinate of x; "
                "use a larger h."
            )

    with np.errstate(over="ignore"):
        steps = upper - lower

    if not np.all(np.isfinite(steps)):
        raise ValueError("The step between perturbed points must be finite.")

    return array, method, upper, lower


def _perturbed(array, index, value):
    point = array.copy()
    point[index] = value
    return point


def _scalar_output(value):
    result = _as_float_array(value, name="f(x)")

    if result.ndim != 0:
        raise ValueError(
            f"f(x) must return a scalar; got shape {result.shape}."
        )

    if not np.isfinite(result):
        raise ValueError("f(x) must return a finite value.")

    return float(result)


def _vector_output(value, n_outputs=None):
    result = _as_float_array(value, name="f(x)")

    if result.ndim != 1:
        raise ValueError(
            f"f(x) must return a 1D array; got {result.ndim}D."
        )

    if result.size == 0:
        raise ValueError("f(x) must return at least one value.")

    if n_outputs is not None and result.size != n_outputs:
        raise ValueError(
            f"f(x) must return {n_outputs} values on every evaluation; "
            f"got {result.size}."
        )

    if not np.all(np.isfinite(result)):
        raise ValueError("f(x) must return only finite values.")

    return result


def _difference(f_upper, f_lower, step):
    upper = np.asarray(f_upper, dtype=np.float64)
    lower = np.asarray(f_lower, dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore",):
        delta = upper - lower
        result = delta / step

        overflowed = ~np.isfinite(delta)
        if np.any(overflowed):
            fallback = upper / step - lower / step
            result = np.where(overflowed, fallback, result)

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "The finite-difference result exceeds the supported numeric range."
        )

    return result


def grad(f, x, method="central", h=1e-5):
    array, method, upper, lower = _prepare(x, method, h)

    if method == "forward":
        f_lower = _scalar_output(f(array.copy()))

    result = np.empty(array.size, dtype=np.float64)

    for i in range(array.size):
        f_upper = _scalar_output(f(_perturbed(array, i, upper[i])))

        if method == "central":
            f_lower = _scalar_output(f(_perturbed(array, i, lower[i])))

        result[i] = _difference(f_upper, f_lower, upper[i] - lower[i])

    return result


def jacobian(f, x, method="central", h=1e-5):
    array, method, upper, lower = _prepare(x, method, h)

    n_outputs = None

    if method == "forward":
        f_lower = _vector_output(f(array.copy()))
        n_outputs = f_lower.size

    columns = []

    for j in range(array.size):
        f_upper = _vector_output(f(_perturbed(array, j, upper[j])), n_outputs)
        n_outputs = f_upper.size

        if method == "central":
            f_lower = _vector_output(
                f(_perturbed(array, j, lower[j])), n_outputs
            )

        columns.append(_difference(f_upper, f_lower, upper[j] - lower[j]))

    return np.stack(columns, axis=1)
