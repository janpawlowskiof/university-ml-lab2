import numpy as np

from src.cvrp_population import CVRPPopulation


def TSPPopulation(population_size: int, demand: np.ndarray, *args, **kwargs):
    return CVRPPopulation(population_size, *args, **kwargs, demand=np.zeros_like(demand))
