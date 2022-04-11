from configs import CONFIGS_ROOT
from src.config import Config
from src.ga.population import NbPopulation


def run():
    Config.from_path(CONFIGS_ROOT / "simple.vrp")
    population = NbPopulation(population_size=100, num_cities=Config.num_cities())
    population.recalculate_fitness(Config.distance_matrix)


if __name__ == '__main__':
    run()
