from typing import Set, Dict

import numpy as np
from numba import jit


@jit(nopython=True)
def cx(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
    """
    https://www.researchgate.net/profile/Pedro-Larranaga/publication/226665831_Genetic_Algorithms_for_the_Travelling_Salesman_Problem_A_Review_of_Representations_and_Operators/links/55b7b5c808aec0e5f43841d8/Genetic-Algorithms-for-the-Travelling-Salesman-Problem-A-Review-of-Representations-and-Operators.pdf?origin=publication_detail
    """
    new_genome = np.empty_like(genome_a)
    unfilled_indexes: Set[int] = set(range(len(new_genome)))

    gene_value_to_index_a = {
        gene: index
        for index, gene
        in enumerate(genome_a)
    }

    gene_value_to_index_b = {
        gene: index
        for index, gene
        in enumerate(genome_b)
    }

    while len(unfilled_indexes) > 0:
        _cx_fill_with_genes_on_cycle(father_genome=genome_a, mother_gene_value_to_index=gene_value_to_index_b, unfilled_indexes=unfilled_indexes, new_genome=new_genome)
        _cx_fill_with_genes_on_cycle(father_genome=genome_b, mother_gene_value_to_index=gene_value_to_index_a, unfilled_indexes=unfilled_indexes, new_genome=new_genome)
    return new_genome


@jit(nopython=True)
def _cx_fill_with_genes_on_cycle(father_genome: np.ndarray, mother_gene_value_to_index: Dict[int, int], unfilled_indexes: Set[int], new_genome: np.ndarray):
    if len(unfilled_indexes) <= 0:
        return
    current_gene_index = next(iter(unfilled_indexes))
    while current_gene_index in unfilled_indexes:
        current_gene_value = father_genome[current_gene_index]
        new_genome[current_gene_index] = current_gene_value
        unfilled_indexes.remove(current_gene_index)
        current_gene_index = mother_gene_value_to_index[current_gene_value]
