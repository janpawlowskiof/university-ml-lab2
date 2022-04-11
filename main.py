import time

from configs import CONFIGS_ROOT
from src.config import Config
from src.ga.ga import run_ga


def run():
    Config.from_path(CONFIGS_ROOT / "simple.vrp")

    start_time = time.time()
    population = run_ga(population_size=100, num_iterations=100, tournament_size=5, cross_probability=0.8, mutation_probability=0.5)
    total_time = time.time() - start_time
    print(f"ga took {total_time}")
    best_genome, best_fitness = population.get_best_genome_and_fitness()
    print(f"best genome is {best_genome}")
    print(f"best fitness is {best_fitness}")


if __name__ == '__main__':
    run()
