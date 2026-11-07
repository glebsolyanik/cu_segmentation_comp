"""Decode submission masks using one-based column-major RLE."""

import numpy as np


def rle_decode(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    if encoded.strip() == "EMPTY":
        return flat.reshape(shape, order="F")
    runs = [int(value) for value in encoded.split()]
    if len(runs) % 2:
        raise ValueError("RLE must contain start/length pairs")
    for start, length in zip(runs[::2], runs[1::2]):
        if start < 1 or length < 1 or start - 1 + length > flat.size:
            raise ValueError("RLE run is outside the image")
        flat[start - 1:start - 1 + length] = True
    return flat.reshape(shape, order="F")
