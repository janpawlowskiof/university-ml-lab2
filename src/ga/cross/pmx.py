from typing import Dict

import numpy as np
from numba import jit, int32


@jit(nopython=True)
def pmx(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
    """
    https://www.researchgate.net/profile/Pedro-Larranaga/publication/226665831_Genetic_Algorithms_for_the_Travelling_Salesman_Problem_A_Review_of_Representations_and_Operators/links/55b7b5c808aec0e5f43841d8/Genetic-Algorithms-for-the-Travelling-Salesman-Problem-A-Review-of-Representations-and-Operators.pdf?origin=publication_detail
    """
    num_genes = genome_a.shape[0]
    new_genome_a = np.copy(genome_b)

    copy_start_index, copy_end_index = np.random.choice(num_genes, 2, replace=False)
    if copy_start_index > copy_end_index:
        copy_start_index, copy_end_index = copy_end_index, copy_start_index

    mapping_b_to_a = {
        gene_a: gene_b
        for gene_b, gene_a
        in zip(genome_a[copy_start_index: copy_end_index + 1], genome_b[copy_start_index: copy_end_index + 1])
    }

    for index in range(0, copy_start_index):
        _pmx_fill_non_mapped_gene(genome_a, mapping_b_to_a, new_genome_a, index)

    for index in range(copy_end_index + 1, num_genes):
        _pmx_fill_non_mapped_gene(genome_a, mapping_b_to_a, new_genome_a, index)

    return new_genome_a


@jit(nopython=True)
def _pmx_fill_non_mapped_gene(genome: np.ndarray, mapping_b_to_a: Dict[int32, int32], new_genome: np.ndarray, index: int32) -> None:
    current_gene = genome[index]
    while current_gene in mapping_b_to_a:
        current_gene = mapping_b_to_a[current_gene]
    new_genome[index] = current_gene
