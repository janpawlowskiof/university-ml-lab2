import unittest

import numpy as np

from configs import CONFIGS_ROOT
from src.config import Config


class TestReadConfig(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        Config.from_path(CONFIGS_ROOT / "simple.vrp")

    def test_read_file(self):
        self.assertEqual(Config.capacity, 30)
        np.testing.assert_allclose(Config.cities.coordinates[0], np.array([38, 46]))
        np.testing.assert_allclose(Config.cities.coordinates[1], np.array([59, 46]))
        np.testing.assert_allclose(Config.cities.demand[1], 16)

    def test_distance_matrix(self):
        distance_between_0_2 = ((38-96)**2 + (46-42)**2)**0.5
        np.testing.assert_allclose(Config.distance_matrix[0, 2], distance_between_0_2)
        np.testing.assert_allclose(Config.distance_matrix[2, 0], distance_between_0_2)


if __name__ == '__main__':
    unittest.main()
