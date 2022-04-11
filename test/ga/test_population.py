import time
import unittest

from configs import CONFIGS_ROOT
from src.config import Config
from src.ga.population import NbPopulation


class TestPopulation(unittest.TestCase):
    def test_create(self):
        population_size = 100
        p = NbPopulation(population_size=population_size, num_cities=3)

        for i in range(population_size):
            self.assertEqual(len(p.genomes[i]), 3)
            self.assertIn(0, p.genomes[i])
            self.assertIn(1, p.genomes[i])
            self.assertIn(2, p.genomes[i])

    def test_calculate_fitness(self):
        Config.from_path(CONFIGS_ROOT / "simple.vrp")
        p = NbPopulation(population_size=100, num_cities=Config.num_cities())

        p.recalculate_fitness(Config.distance_matrix)
        self.assertEqual(len(p.fitness), 100)

    def test_fitness_time(self):
        Config.from_path(CONFIGS_ROOT / "simple.vrp")
        nb_population = NbPopulation(population_size=100, num_cities=Config.num_cities())

        num_repeats = 100_000
        nb_population.recalculate_fitness(Config.distance_matrix)

        start_time = time.time()
        for _ in range(num_repeats):
            nb_population.recalculate_fitness(Config.distance_matrix)
        total_time = time.time() - start_time
        print(f"Nb 100k fitness took {total_time}")


if __name__ == '__main__':
    unittest.main()
