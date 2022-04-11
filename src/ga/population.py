from typing import Tuple

import numpy as np
from numba import int32, float32

from numba.experimental import jitclass


@jitclass([
    ('population_size', int32),
    ('num_cities', int32),
    ('distance_matrix', float32[:, :]),
    ('genomes', int32[:, :]),
    ('fitness', float32[:]),
])
class Population:
    def __init__(self, population_size: int, distance_matrix: np.ndarray) -> None:
        self.population_size = population_size
        self.num_cities = len(distance_matrix)
        self.distance_matrix = distance_matrix
        self.genomes = self._get_random_genotypes()
        self.fitness = np.ones(self.population_size, dtype=np.float32)

        self.recalculate_fitness()

    def _get_random_genotypes(self) -> np.ndarray:
        genotypes = np.empty((self.population_size, self.num_cities), dtype=np.int32)
        for individual_index in range(self.population_size):
            genotypes[individual_index] = np.random.permutation(self.num_cities)
        return genotypes

    def recalculate_fitness(self):
        distances = np.zeros(self.population_size, dtype=np.float32)
        for individual_index in range(self.population_size):
            for step_index in range(self.num_cities - 1):
                distances[individual_index] += self.distance_matrix[
                    self.genomes[individual_index, step_index], self.genomes[individual_index, step_index + 1]
                ]
        self.fitness = distances

    def get_best_genome_and_fitness(self) -> Tuple[np.ndarray, np.ndarray]:
        best_index = np.argmin(self.fitness)
        return self.genomes[best_index], self.fitness[best_index]
