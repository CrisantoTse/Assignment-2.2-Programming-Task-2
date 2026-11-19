
import numpy as np

from numcompute.io import iter_csv_chunks, load_csv

__all__ = ["iter_csv_chunks", "iter_supervised_csv", "load_csv"]


def iter_supervised_csv(path, chunk_size=1024, *, target_column=-1, **kwargs):
    if isinstance(target_column, (bool, np.bool_)) or not isinstance(target_column, (int, np.integer)):
        raise TypeError("target_column must be an integer.")
    for chunk in iter_csv_chunks(path, chunk_size=chunk_size, **kwargs):
        width = chunk.shape[1]
        if not -width <= target_column < width or width < 2:
            raise ValueError("The selected CSV columns must include one target and at least one feature.")
        index = target_column % width
        yield np.delete(chunk, index, axis=1), chunk[:, index].copy()
