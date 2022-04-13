import numpy as np
from numba import jit, prange
from tqdm import tqdm
import numba as nb

from src.config import Config
from src.ga.cross.cross import cross_genomes
from src.ga.mutation import mutate
from src.population import Population
from src.ga.tournament import tournament_selection_round

nb.config.THREADING_LAYER = 'threadsafe'


def run_ga(population_size: int, num_iterations: int, tournament_size: int, cross_probability: float, mutation_probability: float) -> Population:
    population = Population(population_size=population_size, capacity=Config.capacity, distance_matrix=Config.distance_matrix, demand=Config.cities.demand)

    pbar = tqdm(range(num_iterations))
    for iteration in pbar:
        population.genomes = run_ga_iteration(
            genomes=population.genomes,
            fitnesses=population.fitness,
            tournament_size=tournament_size,
            cross_probability=cross_probability,
            mutation_probability=mutation_probability
        )
        population.recalculate_fitness()
        pbar.set_description(f"iteration {iteration}: fitness: {population.get_best_genome_and_fitness()[1]}")
    return population


@jit(nopython=True, parallel=True)
def run_ga_iteration(genomes: np.ndarray, fitnesses: np.ndarray, tournament_size: int, cross_probability: float, mutation_probability: float) -> np.ndarray:
    new_genomes = np.zeros_like(genomes)
    num_genomes, num_genes = genomes.shape
    for individual_index in prange(num_genomes):
        new_genomes[individual_index] = _ga_get_new_genome(
            genomes=genomes,
            fitnesses=fitnesses,
            tournament_size=tournament_size,
            cross_probability=cross_probability,
            mutation_probability=mutation_probability
        )
    return new_genomes


@jit(nopython=True)
def _ga_get_new_genome(genomes: np.ndarray, fitnesses: np.ndarray, tournament_size: int, cross_probability: float, mutation_probability: float) -> np.ndarray:
    parent_a = tournament_selection_round(genomes, fitnesses, tournament_size)
    parent_b = tournament_selection_round(genomes, fitnesses, tournament_size)
    new_genome = cross_genomes(parent_a, parent_b, cross_probability)
    new_genome = mutate(new_genome, mutation_probability)
    return new_genome
