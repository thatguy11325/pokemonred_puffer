import os

import cv2
import matplotlib.colors as mcolors
import numpy as np
from numba import jit

KANTO_MAP_PATH = os.path.join(os.path.dirname(__file__), "kanto_map_dsv.png")
BACKGROUND = np.array(cv2.imread(KANTO_MAP_PATH))


@jit(nopython=True, nogil=True, parallel=True)
def make_pokemon_red_overlay(counts: np.ndarray):
    # TODO: Rethink how this scaling works
    # Divide by number of elements > 0
    # The clip scaling needs to be re-calibrated since my
    # overlay is from the global map with fading
    scaled = np.mean(counts, axis=0) / np.max(counts)
    nonzero = np.where(scaled > 0, 1, 0)
    # scaled = np.clip(counts, 0, 1000) / 1000.0

    # Convert counts to hue map
    hsv = np.stack([2 * (1 - scaled) / 3, nonzero, nonzero], axis=-1)

    # Convert the HSV image to RGB
    overlay = 255 * mcolors.hsv_to_rgb(hsv)

    # Upscale to 16x16
    kernel = np.ones((16, 16, 1), dtype=np.uint8)
    overlay = np.kron(overlay, kernel).astype(np.uint8)
    mask = np.kron(nonzero, kernel[..., 0]).astype(np.uint8)
    mask = np.stack([mask, mask, mask], axis=-1).astype(bool)

    # Combine with background
    render = BACKGROUND.copy().astype(np.int32)
    render[mask] = 0.2 * render[mask] + 0.8 * overlay[mask]
    render = np.clip(render, 0, 255).astype(np.uint8)

    return render
