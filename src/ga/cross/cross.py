import numpy as np
from numba import jit

from src.ga.cross.cx import cx
from src.ga.cross.ox import ox
from src.ga.cross.pmx import pmx


@jit(nopython=True)
def cross_genomes(genome_a: np.ndarray, genome_b: np.ndarray, cross_probability: float) -> np.ndarray:
    if np.random.random() > cross_probability:
        return genome_a
    cross_method = np.random.choice(3, size=1)
    if cross_method == 0:
        return ox(genome_a, genome_b)
    if cross_method == 1:
        return pmx(genome_a, genome_b)
    if cross_method == 2:
        return cx(genome_a, genome_b)
