import numpy as np
from numba import jit


@jit
def tournament_selection(genomes: np.ndarray, fitnesses: np.ndarray, tournament_size: int) -> np.ndarray:
    new_genomes = np.empty(genomes)
    num_tournaments = fitnesses.shape[0]

    for tournament_index in range(num_tournaments):
        parent_a = tournament_selection_round(genomes, fitnesses, tournament_size)
        parent_b = tournament_selection_round(genomes, fitnesses, tournament_size)



        parent_b_index = tournament_index * 2 + 1
        new_genomes[parent_a_index] = parent_a
        new_genomes[parent_b_index] = parent_b

    return new_genomes


@jit
def tournament_selection_round(genomes: np.ndarray, fitnesses: np.ndarray, tournament_size: int) -> np.ndarray:
    population_size = fitnesses.shape[0]
    contestants_indexes = np.random.choice(population_size, size=tournament_size, replace=False)
    contestants_fitnesses = fitnesses[contestants_indexes]
    winner_index = contestants_indexes[np.argmax(contestants_fitnesses)]
    return genomes[winner_index]


# @jit
# def cross_genomes(genome_a: np.ndarray, genome_b: np.ndarray, cross_probability: float) -> np.ndarray:
#     if np.random.random() < cross_probability:
#         return genome_a
#     for
