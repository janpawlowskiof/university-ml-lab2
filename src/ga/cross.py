import numpy as np
from numba import float32, jit
from numba.experimental import jitclass


@jitclass([
    ('cross_probability', float32),
])
class Cross:
    def __init__(self, cross_probability: float) -> None:
        self.cross_probability = cross_probability

    def cross_genome(self, genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
        if np.random.random() > self.cross_probability:
            return genome_a
        return self._ox(genome_a, genome_b)

    def _ox(self, genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
        num_genes = genome_a    .shape[0]
        copy_start_index, copy_end_index = np.random.choice(num_genes, 2, replace=False)
        if copy_start_index > copy_end_index:
            copy_start_index, copy_end_index = copy_end_index, copy_start_index

        new_genome = np.copy(genome_a)
        genome_b_without_copied_cities = remove_values_from_permutation(genome_b, genome_a[copy_start_index:copy_end_index + 1])

        new_genome[:copy_start_index] = genome_b_without_copied_cities[:copy_start_index]
        new_genome[copy_end_index + 1:] = genome_b_without_copied_cities[copy_start_index:]

        return new_genome


@jit
def remove_values_from_permutation(array: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    this works only in very specific usecase
    """
    result = np.empty(len(array) - len(values), dtype=np.int32)
    current_index = 0
    for x in array:
        if x not in values:
            result[current_index] = x
            current_index += 1
    return result
