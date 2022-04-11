import time
import unittest

from configs import CONFIGS_ROOT
from src.config import Config
from src.ga.population import Population


class TestPopulation(unittest.TestCase):
    def setUp(self) -> None:
        Config.from_path(CONFIGS_ROOT / "simple.vrp")

    def test_create(self):
        population_size = 100
        p = Population(population_size=population_size, distance_matrix=Config.distance_matrix)

        for i in range(population_size):
            self.assertEqual(len(p.genomes[i]), 6)
            for j in range(6):
                self.assertIn(j, p.genomes[i])

    def test_calculate_fitness(self):
        Config.from_path(CONFIGS_ROOT / "simple.vrp")
        p = Population(population_size=100, distance_matrix=Config.distance_matrix)

        p.recalculate_fitness()
        self.assertEqual(len(p.fitness), 100)

    def test_fitness_time(self):
        Config.from_path(CONFIGS_ROOT / "simple.vrp")
        nb_population = Population(population_size=100, distance_matrix=Config.distance_matrix)

        num_repeats = 100_000
        nb_population.recalculate_fitness()

        start_time = time.time()
        for _ in range(num_repeats):
            nb_population.recalculate_fitness()
        total_time = time.time() - start_time
        print(f"Nb 100k fitness took {total_time}")


if __name__ == '__main__':
    unittest.main()
