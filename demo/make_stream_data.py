
import csv
from pathlib import Path

import numpy as np


def make_csv(path, seed=2026, n=2048):
    rng = np.random.default_rng(seed)
    x, z, noise = rng.normal(size=(3, n))
    raw = x + 0.52 * z + 0.18 * noise
    raw[n // 2:] = -x[n // 2:] + 0.52 * z[n // 2:] + 0.18 * noise[n // 2:]
    y = raw > 0
    observed_x, observed_z = x.copy(), z.copy()
    observed_x[rng.random(n) < 0.05] = np.nan
    observed_z[rng.random(n) < 0.05] = np.nan
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["signal", "context", "label"])
        for a, b, target in zip(observed_x, observed_z, y):
            writer.writerow(["" if np.isnan(a) else f"{a:.8f}",
                             "" if np.isnan(b) else f"{b:.8f}", int(target)])


if __name__ == "__main__":
    make_csv(Path(__file__).with_name("stream_data.csv"))
