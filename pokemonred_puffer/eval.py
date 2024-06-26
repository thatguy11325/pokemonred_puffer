import os

import cv2
import numpy as np
from numba import jit

KANTO_MAP_PATH = os.path.join(os.path.dirname(__file__), "kanto_map_dsv.png")
BACKGROUND = np.array(cv2.imread(KANTO_MAP_PATH))


@jit(nopython=True, nogil=True)
def make_pokemon_red_overlay(counts: np.ndarray):
    # TODO: Rethink how this scaling works
    # Divide by number of elements > 0
    # The clip scaling needs to be re-calibrated since my
    # overlay is from the global map with fading
    scaled = np.ascontiguousarray(np.sum(counts, axis=0).astype(np.float32))
    scaled = scaled / np.max(scaled)
    nonzero = np.ascontiguousarray(np.where(scaled > 0, 1, 0).astype(np.float32))
    # scaled = np.clip(counts, 0, 1000) / 1000.0

    # Convert counts to hue map
    hsv = np.stack((2 * (1 - scaled) / 3, nonzero, nonzero), axis=-1)

    # Convert the HSV image to RGB
    overlay = 255 * hsv_to_rgb(hsv)

    # Upscale to 16x16
    kernel = np.ascontiguousarray(np.ones((16, 16), dtype=np.uint8))
    h = np.ascontiguousarray(overlay[..., 0])
    s = np.ascontiguousarray(overlay[..., 1])
    v = np.ascontiguousarray(overlay[..., 2])
    h = np.kron(h, kernel).astype(np.uint8)
    s = np.kron(s, kernel).astype(np.uint8)
    v = np.kron(v, kernel).astype(np.uint8)
    overlay = np.stack((h, s, v), axis=-1)
    mask = np.kron(nonzero, np.ascontiguousarray(kernel[..., 0])).astype(np.uint8)
    mask = np.stack((mask, mask, mask), axis=-1) != 0

    # Combine with background
    render = BACKGROUND.copy().astype(np.int32)
    render_shape = render.shape
    render = render.ravel()
    render[mask.ravel()] = 0.2 * render[mask.ravel()] + 0.8 * overlay.ravel()[mask.ravel()]
    render = render.reshape(render_shape)
    render = np.clip(render, 0, 255).astype(np.uint8)

    return render


@jit(nopython=True, nogil=True)
def hsv_to_rgb(hsv):
    """
    Copied from matplotlib for numba
    Convert HSV values to RGB.

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    (..., 3) `~numpy.ndarray`
       Colors converted to RGB values in range [0, 1]
    """
    # hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError(
            "Last dimension of input array must be 3; " f"shape {hsv.shape} was found."
        )

    in_shape = hsv.shape
    # hsv = np.array(
    #     hsv, copy=False,
    #     dtype=np.float32,  # Don't work on ints.
    #     ndmin=2,  # In case input was 1D.
    # )
    hsv = hsv.astype(np.float32)

    h = hsv[..., 0].ravel()
    s = hsv[..., 1].ravel()
    v = hsv[..., 2].ravel()

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(np.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    r = r.reshape(hsv[..., 0].shape)
    g = g.reshape(hsv[..., 0].shape)
    b = b.reshape(hsv[..., 0].shape)
    rgb = np.stack((r, g, b), axis=-1)

    return rgb.reshape(in_shape)
