import os
import cv2
import math
import numpy as np


def mean_shift(img, init_pos, g, kernel=None, target_size=None, q=None, eps=0.01, max_iter=20, use_hsv=False):
    x0, y0 = init_pos
    positions = [(x0, y0)]

    g_height, g_width = g.shape[:2]
    half_height = math.floor(g_height / 2)
    half_width = math.floor(g_width / 2)
    x_coords = np.tile(np.arange(-half_width, half_width+1), g_height).reshape(g_height, g_width)
    y_coords = np.tile(np.arange(-half_height, half_height+1), g_width).reshape(g_width, g_height).T

    num_iter = 0
    while True:
        num_iter += 1
        if kernel is not None:
            cut_out, _ = get_patch(img, (x0, y0), (g_width, g_height))
            num_bins = round(math.pow(len(q), 1/3))
            if use_hsv:
                p = extract_histogram_hsv(cut_out, num_bins, weights=kernel)
            else:
                p = extract_histogram(cut_out, num_bins, weights=kernel)
            p /= np.sum(p)
            V = np.sqrt(np.divide(q, (p + 1e-3)))
            if use_hsv:
                weights = backproject_histogram_hsv(cut_out, V, num_bins)
            else:
                weights = backproject_histogram(cut_out, V, num_bins)
        else:
            weights, _ = get_patch(img, (x0, y0), (g_width, g_height))

        x_nom = np.sum(np.multiply(np.multiply(x_coords, weights), g))
        y_nom = np.sum(np.multiply(np.multiply(y_coords, weights), g))
        denom = np.sum(np.multiply(weights, g))

        x = x0 + x_nom / (denom + 1e-6)
        y = y0 + y_nom / (denom + 1e-6)
        positions.append((x, y))

        if abs(x - x0) < eps and abs(y - y0) < eps or kernel is not None and num_iter > max_iter or num_iter > 5000:
            break

        x0 = x
        y0 = y

    if kernel is None:
        return x, y, positions, num_iter
    return x, y
