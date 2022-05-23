import numpy as np
import torch
import unittest

from dpe.utils import normalize_pitch, denormalize_pitch


class TestPitchNorm(unittest.TestCase):

    def test_norm_denorm(self):
        pitch = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        pmin = 2
        pmax = 8
        n_channels = 116

        normalized = normalize_pitch(pitch=pitch, pmin=pmin,
                                     pmax=pmax, n_channels=n_channels)
        denormalized = denormalize_pitch(pitch=normalized, pmin=pmin,
                                         pmax=pmax, n_channels=n_channels)

        norm_expected = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 0])
        np.testing.assert_allclose(normalized, norm_expected, atol=1e-8)

        denorm_expected = np.array([0, 0, 2, 3, 4, 5, 6, 7, 8, 0])
        np.testing.assert_allclose(denormalized, denorm_expected, atol=1e-8)