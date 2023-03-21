import unittest
from Assignment2 import lj_potential

class TestLJPotential(unittest.TestCase):
    def test_lj_potential(self):
        # Test a distance of 1 (minimum cutoff distance)
        c6 = 1
        c12 = 1
        r = 1
        expected_energy = -1
        calculated_energy = lj_potential(r, c6, c12)
        self.assertAlmostEqual(calculated_energy, expected_energy, places=7)

        # Test a distance of 2
        c6 = 1
        c12 = 1
        r = 2
        expected_energy = -0.0625
        self.assertAlmostEqual(lj_potential(r, c6, c12), expected_energy)

        # Test a distance of 3
        c6 = 1
        c12 = 1
        r = 3
        expected_energy = -0.01388888
        self.assertAlmostEqual(lj_potential(r, c6, c12), expected_energy)

if __name__ == '__main__':
    unittest.main()
