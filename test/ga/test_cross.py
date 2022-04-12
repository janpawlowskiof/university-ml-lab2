import time
import unittest

import numpy as np

from src.ga.cross.cross import cross_genomes
from src.ga.cross.ox import remove_values_from_permutation


class TestCross(unittest.TestCase):

    def setUp(self) -> None:
        self.num_genes = 1000
        self.genome_a = np.random.permutation(self.num_genes)
        self.genome_b = np.random.permutation(self.num_genes)

    def test_create(self):
        num_repeats = 10_000

        for _ in range(1000):
            new_genotype = cross_genomes(self.genome_a, self.genome_b, cross_probability=1.0)
            for i in range(self.num_genes):
                self.assertIn(i, new_genotype)

        start_time = time.time()
        for _ in range(num_repeats):
            cross_genomes(self.genome_a, self.genome_b, cross_probability=1.0)
        total_time = time.time() - start_time
        print(f"cross took {total_time}")

    def test_remove_values_from_permutation(self):
        genome_a_slice = self.genome_a[10:100]
        genome_b_without_copied_cities = remove_values_from_permutation(self.genome_b, genome_a_slice)
        for i in range(self.num_genes):
            self.assertIn(i, self.genome_a)
            self.assertIn(i, self.genome_b)

        for i in range(self.num_genes):
            assert (i in genome_b_without_copied_cities) ^ (i in genome_a_slice)


if __name__ == '__main__':
    unittest.main()
