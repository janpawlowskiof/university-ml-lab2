from typing import Tuple, List

import numba as nb
import numpy as np
import pandas as pd
from numba import jit, prange
from tqdm import tqdm

from src.config import Config
from src.cvrp_population import CVRPPopulation
from src.ga.cross.cross import cross_genomes
from src.ga.mutation import mutate
from src.ga.tournament import tournament_selection_round

nb.config.THREADING_LAYER = 'threadsafe'


def run_ga(population_size: int, num_iterations: int, tournament_size: int, cross_probability: float, mutation_probability: float, population_class=CVRPPopulation) -> Tuple[CVRPPopulation, pd.DataFrame]:
    population: CVRPPopulation = population_class(population_size=population_size, capacity=Config.capacity, distance_matrix=Config.distance_matrix, demand=Config.cities.demand)
    history = []

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

        _add_entries_to_history(population, iteration, history)
        pbar.set_description(f"iteration {iteration}: fitness: {population.get_best_genome_and_fitness()[1]}")

    history = pd.DataFrame.from_records(history, columns=["iteration", "fitness"])
    return population, history


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


def _add_entries_to_history(population: CVRPPopulation, iteration: int, history: List):
    for individual_fitness in population.fitness:
        entry = (iteration, individual_fitness)
        history.append(entry)
