"""Standalone numpy helpers for the honic data pipeline.

Ported verbatim (behaviour-for-behaviour) from delphi/data/utils.py so honic runs
without importing delphi. Each is a plain function -- honic applies them directly
in Dataset.__getitem__ instead of wrapping them in a Transform class.

Everything here is time-in-DAYS: HonicReader converts age(years)->days at load, so
the 36525 (=100*365.25) no_event range and the perturb bounds below are day-space.
"""

import numpy as np


def collate_batch(
    batch_data: list[np.ndarray], fill_value: int | float = 0, pad_left: bool = True
) -> np.ndarray:
    """Pad a list of 1-D arrays to a (B, max_len) block. LEFT-pad by default.

    dtype is taken from batch_data[0], so pass float arrays when fill_value is
    -1e4 or the fill truncates to int and the age-pad sentinel breaks.
    """
    max_len = max(bd.size for bd in batch_data)
    collated = np.full((len(batch_data), max_len), fill_value, dtype=batch_data[0].dtype)
    for i, bd in enumerate(batch_data):
        if bd.size > 0:
            if pad_left:
                collated[i, -bd.size :] = bd
            else:
                collated[i, : bd.size] = bd
    return collated


def append_no_event(
    x: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
    interval: float,
    mode: str = "legacy-random",
    token: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert no_event (`token`) markers so the model sees elapsed disease-free time.

    Only the two modes honic uses are ported; `interval` is in DAYS. legacy-random
    is the delphi-m4/UKB default. The caller sorts afterwards.
    """
    if mode == "legacy-random":
        min_age = np.min(t[t >= 0])
        max_age = np.max(t)
        no_event_t = rng.uniform(1, 36525, size=(int(36525 / interval),))
        no_event_t = no_event_t[(no_event_t >= min_age) & (no_event_t < max_age)]
    elif mode == "random":
        max_age = np.max(t)
        min_age = max(np.min(t[t >= 0]), 0) + 1e-6  # +eps so no_event never co-occurs with the first token
        n = int((max_age - min_age) // interval) - 1
        no_event_t = rng.uniform(min_age, max_age, size=(n,)) if n > 0 else np.array([])
    else:
        raise ValueError(f"unsupported no_event mode {mode!r}")

    no_event_t = no_event_t.astype(np.float32)
    no_event_x = np.full(no_event_t.shape, token)
    return np.concatenate((x, no_event_x)), np.concatenate((t, no_event_t))


def exclude_tokens(x: np.ndarray, t: np.ndarray, blacklist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop every position whose token id is in `blacklist`."""
    keep = ~np.isin(x, blacklist)
    return x[keep], t[keep]


def sort_by_time(t: np.ndarray, *args: np.ndarray, stable: bool = False):
    """Sort `t` ascending and reorder the parallel arrays. Returns (t, *args_sorted).
    Note the arg order: t comes FIRST, in and out. `stable` for reproducible eval."""
    s = np.argsort(t, kind="stable" if stable else "quicksort")
    return t[s], *[a[s] for a in args]


def _crop_slice(mode: str, max_len: int, block_size: int, rng: np.random.Generator) -> slice:
    if mode == "left":
        start = 0
    elif mode == "right":
        start = max_len - block_size
    elif mode == "random":
        start = int(rng.integers(0, max_len - block_size + 1))
    else:
        raise ValueError(f"unsupported crop mode {mode!r}")
    return slice(start, start + block_size)


def crop_contiguous(
    x: np.ndarray, t: np.ndarray, *, block_size: int, rng: np.random.Generator, mode: str = "right"
) -> tuple[np.ndarray, np.ndarray]:
    """Crop the time-sorted (x, t) to a contiguous block_size window. No-op if short enough."""
    L = x.shape[0]
    if L <= block_size:
        return x, t
    cut = _crop_slice(mode, L, block_size, rng)
    return x[cut], t[cut]
