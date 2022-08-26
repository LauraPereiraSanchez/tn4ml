import math
import numpy as np
import itertools
from numbers import Number
from typing import Callable
import quimb.tensor as qtn

def trigonometric(x: Number, k: int = 1, dtype=np.float32) -> np.ndarray:
    return 1 / np.sqrt(k) * np.fromiter((f(np.pi * x / 2**i) for f, i in itertools.product([np.cos, np.sin], range(1, k + 1))), dtype=dtype)

def fourier(x: Number, p: int = 2, dtype=np.float32) -> np.ndarray:
    return 1 / p * np.fromiter((np.abs(sum((np.exp(1j * 2 * np.pi * k * ((p - 1) * x - j) / p) for k in range(p)))) for j in range(p)), dtype=dtype)


def embed(x: np.ndarray, phi: Callable, site_ind_id='k{}', site_tag_id='phi{}', **kwargs):
    """Creates a product state from a vector of features `x`."""
    assert x.ndim == 1

    # reshape to expected shape by MPS
    arrays = [phi(xi, **kwargs) for xi in x]
    embd_dim = arrays[0].shape[0]
    arrays[0] = arrays[0].reshape(1, embd_dim)
    for i in range(1, len(arrays) - 1):
        arrays[i] = arrays[i].reshape(1, 1, embd_dim)
    arrays[-1] = arrays[-1].reshape(1, embd_dim)

    return qtn.MatrixProductState(arrays, site_ind_id=site_ind_id, site_tag_id=site_tag_id), embd_dim