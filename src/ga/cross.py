import random
from typing import Dict

import numpy as np
from numba import jit, int32


@jit(nopython=True)
def cross_genomes(genome_a: np.ndarray, genome_b: np.ndarray, cross_probability: float) -> np.ndarray:
    if np.random.random() > cross_probability:
        return genome_a
    cross_method = np.random.choice(3, size=1)
    if cross_method == 0:
        return _ox(genome_a, genome_b)
    if cross_method == 1:
        return _pmx(genome_a, genome_b)
    if cross_method == 2:
        return _ox(genome_a, genome_b)


@jit(nopython=True)
def _ox(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
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
def _pmx(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
    """
    https://www.researchgate.net/profile/Pedro-Larranaga/publication/226665831_Genetic_Algorithms_for_the_Travelling_Salesman_Problem_A_Review_of_Representations_and_Operators/links/55b7b5c808aec0e5f43841d8/Genetic-Algorithms-for-the-Travelling-Salesman-Problem-A-Review-of-Representations-and-Operators.pdf?origin=publication_detail
    """
    num_genes = genome_a.shape[0]
    new_genome_a = np.copy(genome_b)

    copy_start_index, copy_end_index = np.random.choice(num_genes, 2, replace=False)
    if copy_start_index > copy_end_index:
        copy_start_index, copy_end_index = copy_end_index, copy_start_index

    mapping_a_to_b = {
        gene_a: gene_b
        for gene_b, gene_a
        in zip(genome_a[copy_start_index: copy_end_index + 1], genome_b[copy_start_index: copy_end_index + 1])
    }

    for index in range(0, copy_start_index):
        _pmx_fill_non_mapped_gene(genome_a, mapping_a_to_b, new_genome_a, index)

    for index in range(copy_end_index + 1, num_genes):
        _pmx_fill_non_mapped_gene(genome_a, mapping_a_to_b, new_genome_a, index)

    return new_genome_a


@jit(nopython=True)
def _pmx_fill_non_mapped_gene(genome: np.ndarray, mapping_a_to_b: Dict[int32, int32], new_genome: np.ndarray, index: int32) -> None:
    current_gene = genome[index]
    while current_gene in mapping_a_to_b:
        current_gene = mapping_a_to_b[current_gene]
    new_genome[index] = current_gene


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
