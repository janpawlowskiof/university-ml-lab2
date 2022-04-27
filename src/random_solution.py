from src.config import Config
from src.cvrp_population import CVRPPopulation


def run_random(population_class=CVRPPopulation) -> CVRPPopulation:
    return population_class(population_size=1, capacity=Config.capacity, distance_matrix=Config.distance_matrix, demand=Config.cities.demand)
