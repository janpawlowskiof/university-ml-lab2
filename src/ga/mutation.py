import numpy as np
from numba import jit


@jit
def mutate(genome: np.ndarray, mutation_probability: float) -> np.ndarray:
    num_genes = len(genome)
    new_genome = np.copy(genome)
    if np.random.random() > mutation_probability:
        index_a, index_b = np.random.choice(num_genes, 2, replace=False)
        new_genome[index_a], new_genome[index_b] = genome[index_b], genome[index_a]
    return new_genome
