import numpy as np
from numba import jit, prange

from src.config import Config
from src.ga.cross.cross import cross_genomes
from src.ga.mutation import mutate
from src.ga.population import Population
from src.ga.tournament import tournament_selection_round


def run_ga(population_size: int, num_iterations: int, tournament_size: int, cross_probability: float, mutation_probability: float) -> Population:
    population = Population(population_size=population_size, capacity=Config.capacity, distance_matrix=Config.distance_matrix, demand=Config.cities.demand)

    for iteration in range(num_iterations):
        population.genomes = run_ga_iteration(
            genomes=population.genomes,
            fitnesses=population.fitness,
            tournament_size=tournament_size,
            cross_probability=cross_probability,
            mutation_probability=mutation_probability
        )
        population.recalculate_fitness()
        print(f"iteration {iteration}: fitness: {population.get_best_genome_and_fitness()[1]}")
    return population


@jit(nopython=True)
def run_ga_iteration(genomes: np.ndarray, fitnesses: np.ndarray, tournament_size: int, cross_probability: float, mutation_probability: float) -> np.ndarray:
    new_genomes = np.zeros_like(genomes)
    for individual_index in prange(len(genomes)):
        parent_a = tournament_selection_round(genomes, fitnesses, tournament_size)
        parent_b = tournament_selection_round(genomes, fitnesses, tournament_size)
        new_genome = cross_genomes(parent_a, parent_b, cross_probability)
        new_genome = mutate(new_genome, mutation_probability)
        new_genomes[individual_index] = new_genome
    return new_genomes
