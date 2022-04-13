import numpy as np
from numba import jit


@jit(nopython=True)
def tournament_selection_round(genomes: np.ndarray, fitnesses: np.ndarray, tournament_size: int) -> np.ndarray:
    population_size = fitnesses.shape[0]
    contestants_indexes = np.random.choice(population_size, size=tournament_size, replace=False)
    contestants_fitnesses = fitnesses[contestants_indexes]
    winner_index = contestants_indexes[np.argmin(contestants_fitnesses)]
    return genomes[winner_index]
