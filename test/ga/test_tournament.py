import unittest

import numpy as np

from src.ga.tournament import tournament_selection_round


class TestTournament(unittest.TestCase):
    def test_run(self):
        population_size = 100
        num_cities = 10

        genotypes = np.empty((population_size, num_cities), dtype=np.int32)
        for individual_index in range(population_size):
            genotypes[individual_index] = np.random.permutation(num_cities)

        fitnesses = np.random.randn(population_size)
        tournament_selection_round(genomes=genotypes, fitnesses=fitnesses, tournament_size=5)

    def test_deterministic(self):
        population_size = 100
        num_cities = 10

        genotypes = np.empty((population_size, num_cities), dtype=np.int32)
        for individual_index in range(population_size):
            genotypes[individual_index] = np.random.permutation(num_cities)

        fitnesses = np.random.randn(population_size)
        fitnesses[0] = -1000000

        winner = tournament_selection_round(genomes=genotypes, fitnesses=fitnesses, tournament_size=population_size)
        np.testing.assert_allclose(winner, genotypes[0])


if __name__ == '__main__':
    unittest.main()
