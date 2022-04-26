import numpy as np

from src.config import Config
from src.cvrp_population import CVRPPopulation


def run_greedy() -> CVRPPopulation:
    population = CVRPPopulation(population_size=1, capacity=Config.capacity, distance_matrix=Config.distance_matrix, demand=Config.cities.demand)
    population.genomes = _greedy_get_genome(distance_matrix=Config.distance_matrix, demands=Config.cities.demand, capacity=Config.capacity)[np.newaxis, :]
    population.recalculate_fitness()

    return population


def _greedy_get_genome(distance_matrix: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    current_cargo = 0
    num_cities = len(demands)
    unvisited_cities = list(range(num_cities))
    genome = [0]
    unvisited_cities.remove(0)

    while len(unvisited_cities) > 0:
        nearest_city = unvisited_cities[
            np.argmin(distance_matrix[genome[-1], unvisited_cities])
        ]
        if current_cargo < demands[nearest_city]:
            genome.append(0)
            current_cargo = capacity
        else:
            genome.append(nearest_city)
            unvisited_cities.remove(nearest_city)
            current_cargo -= demands[nearest_city]
    return np.array(genome)
