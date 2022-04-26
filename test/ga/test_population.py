import time
import unittest

from configs import CONFIGS_ROOT
from src.config import Config
from src.cvrp_population import CVRPPopulation


class TestPopulation(unittest.TestCase):
    def setUp(self) -> None:
        Config.from_path(CONFIGS_ROOT / "simple.vrp")
        self.population_size = 100
        self.population = CVRPPopulation(population_size=self.population_size, capacity=30, distance_matrix=Config.distance_matrix, demand=Config.cities.demand)

    def test_create(self):
        for i in range(self.population_size):
            self.assertEqual(len(self.population.genomes[i]), 6)
            for j in range(6):
                self.assertIn(j, self.population.genomes[i])

    def test_calculate_fitness(self):
        self.population.recalculate_fitness()
        self.assertEqual(len(self.population.fitness), 100)

    def test_fitness_time(self):
        num_repeats = 100_000
        self.population.recalculate_fitness()

        start_time = time.time()
        for _ in range(num_repeats):
            self.population.recalculate_fitness()
        total_time = time.time() - start_time
        print(f"Nb 100k fitness took {total_time}")


if __name__ == '__main__':
    unittest.main()
