import time

from configs import CONFIGS_ROOT
from src.config import Config
from src.cvrp_population import CVRPPopulation
from src.ga.ga import run_ga
from src.greedy_solution import run_greedy
from src.random_solution import run_random


def main():
    Config.from_path(CONFIGS_ROOT / "tai385.vrp")
    check_random_solution()
    check_greedy_solution()
    check_ga_solution()


def check_ga_solution():
    Config.from_path(CONFIGS_ROOT / "tai385.vrp")
    start_time = time.time()
    population, _ = run_ga(population_size=1000, num_iterations=10000, tournament_size=5, cross_probability=0.8, mutation_probability=0.8)
    total_time = time.time() - start_time
    print(f"ga took {total_time}")
    best_genome, best_fitness = population.get_best_genome_and_fitness()
    print(f"ga best fitness is {best_fitness}")


def check_greedy_solution():
    start_time = time.time()
    population = run_greedy()
    total_time = time.time() - start_time
    print(f"greedy took {total_time}")
    best_genome, best_fitness = population.get_best_genome_and_fitness()
    print(f"greedy best fitness is {best_fitness}")


def check_random_solution():
    start_time = time.time()
    population = run_random()
    total_time = time.time() - start_time
    print(f"random took {total_time}")
    best_genome, best_fitness = population.get_best_genome_and_fitness()
    print(f"random best fitness is {best_fitness}")


if __name__ == '__main__':
    main()
