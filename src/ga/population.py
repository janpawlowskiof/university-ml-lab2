import numpy as np
from numba import int32, float32
from numba.experimental import jitclass


@jitclass([
    ('population_size', int32),
    ('num_cities', int32),
    ('genomes', int32[:, :]),
    ('fitness', float32[:]),
])
class NbPopulation:
    def __init__(self, population_size: int, num_cities: int) -> None:
        self.population_size = population_size
        self.num_cities = num_cities
        self.genomes = self._get_random_genotypes()
        self.fitness = np.ones(self.population_size, dtype=np.float32)

    def _get_random_genotypes(self) -> np.ndarray:
        genotypes = np.empty((self.population_size, self.num_cities), dtype=np.int32)
        for individual_index in range(self.population_size):
            genotypes[individual_index] = np.random.permutation(self.num_cities)
        return genotypes

    def recalculate_fitness(self, distance_matrix: np.ndarray):
        distances = np.zeros(self.population_size, dtype=np.float32)
        for individual_index in range(self.population_size):
            for step_index in range(self.num_cities - 1):
                distances[individual_index] += distance_matrix[
                    self.genomes[individual_index, step_index], self.genomes[individual_index, step_index + 1]
                ]
        self.fitness = distances
