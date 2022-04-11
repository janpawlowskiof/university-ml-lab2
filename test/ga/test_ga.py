import time
import unittest

from configs import CONFIGS_ROOT
from src.config import Config
from src.ga.ga import run_ga


class TestGA(unittest.TestCase):
    def test_ga(self):
        Config.from_path(CONFIGS_ROOT / "simple.vrp")
        run_ga(population_size=100, num_iterations=1, tournament_size=5, cross_probability=0.5)

        start_time = time.time()
        run_ga(population_size=100, num_iterations=1000, tournament_size=5, cross_probability=0.5)
        total_time = time.time() - start_time
        print(f"ga took {total_time}")


if __name__ == '__main__':
    unittest.main()
