import numpy as np
from numba import jit


@jit(nopython=True)
def ox(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
    num_genes = genome_a.shape[0]
    copy_start_index, copy_end_index = np.random.choice(num_genes, 2, replace=False)
    if copy_start_index > copy_end_index:
        copy_start_index, copy_end_index = copy_end_index, copy_start_index

    new_genome = np.copy(genome_a)
    genome_b_without_copied_cities = remove_values_from_permutation(genome_b, genome_a[copy_start_index:copy_end_index + 1])

    new_genome[:copy_start_index] = genome_b_without_copied_cities[:copy_start_index]
    new_genome[copy_end_index + 1:] = genome_b_without_copied_cities[copy_start_index:]

    return new_genome


@jit(nopython=True)
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
