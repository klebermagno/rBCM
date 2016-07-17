"""Small functions used throughout the testing to generate sample data"""

import numpy as np


def f1(x):
    x = x / 1000
    return 4 * np.abs(np.sin(x)) + x


def f2(x):
    return 20 * np.exp(-x / 150)


def add_noise(y, noise_level=0.2):
    return y + np.random.normal(0, noise_level, y.shape[0])
