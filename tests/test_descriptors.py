import unittest
from deepatom.descriptors import calculate_coulomb_matrix
from pymatgen.core import Structure

class TestDescriptors(unittest.TestCase):
    def test_coulomb_matrix(self):
        lattice_vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        atomic_numbers = [1, 1]
        positions = [[0, 0, 0], [0, 0, 1]]
        structure = Structure(lattice_vectors, atomic_numbers, positions)
        cm = calculate_coulomb_matrix(structure)
        self.assertIsNotNone(cm)

if __name__ == "__main__":
    unittest.main()
