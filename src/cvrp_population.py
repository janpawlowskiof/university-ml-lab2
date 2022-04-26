from typing import Tuple, List

import numpy as np
from numba import int32, float32, prange
from numba.experimental import jitclass


@jitclass([
    ('population_size', int32),
    ('num_cities', int32),
    ('distance_matrix', float32[:, :]),
    ('demand', int32[:]),
    ('capacity', int32),
    ('genomes', int32[:, :]),
    ('fitness', float32[:]),
])
class CVRPPopulation:
    def __init__(self, population_size: int, capacity: int, distance_matrix: np.ndarray, demand: np.ndarray) -> None:
        self.population_size = population_size
        self.num_cities = len(distance_matrix)
        self.capacity = capacity
        self.distance_matrix = distance_matrix
        self.demand = demand
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
        for individual_index in prange(self.population_size):
            # getting from magazine to the first city
            carried_cargo = self.capacity
            first_city_index = self.genomes[individual_index, 0]
            distances[individual_index] += self.distance_matrix[0, first_city_index]
            carried_cargo -= self.demand[first_city_index]

            # going through the route
            num_steps = self.genomes.shape[1] - 1
            for step_index in range(num_steps):
                current_city_index = self.genomes[individual_index, step_index]
                next_city_index = self.genomes[individual_index, step_index + 1]
                demanded_cargo = self.demand[next_city_index]
                if carried_cargo < demanded_cargo:
                    distances[individual_index] += self.distance_matrix[current_city_index, 0]
                    carried_cargo = self.capacity
                    distances[individual_index] += self.distance_matrix[0, next_city_index]
                else:
                    distances[individual_index] += self.distance_matrix[current_city_index, next_city_index]

                carried_cargo -= demanded_cargo

            # getting from last city to magazine
            last_city_index = self.genomes[individual_index, -1]
            distances[individual_index] += self.distance_matrix[last_city_index, 0]

        self.fitness = distances

    def get_best_genome_and_fitness(self) -> Tuple[np.ndarray, np.ndarray]:
        best_index = np.argmin(self.fitness)
        return self.genomes[best_index], self.fitness[best_index]
