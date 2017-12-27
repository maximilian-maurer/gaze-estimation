import unittest
import itertools
import math

import numpy as np
from src.rotation_matrix import calculate_rotation_matrix_extrinsic, euler_angles_from_rotation_matrix


class TestRotateMatrix(unittest.TestCase):

    def test_calculate_rotation_matrix_extrinsic(self):
        alpha, beta, gamma = np.radians((0, 0, 0))
        expected_value = [[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]]
        rotation_matrix = calculate_rotation_matrix_extrinsic(alpha, beta, gamma)
        np.testing.assert_array_almost_equal(rotation_matrix, expected_value)

        alpha, beta, gamma = np.radians((90, 0, 0))
        expected_value = [[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]]
        rotation_matrix = calculate_rotation_matrix_extrinsic(alpha, beta, gamma)
        np.testing.assert_array_almost_equal(rotation_matrix, expected_value)

        alpha, beta, gamma = np.radians((0, 90, 0))
        expected_value = [[0, 0, 1],
                          [0, 1, 0],
                          [-1, 0, 0]]
        rotation_matrix = calculate_rotation_matrix_extrinsic(alpha, beta, gamma)
        np.testing.assert_array_almost_equal(rotation_matrix, expected_value)

        alpha, beta, gamma = np.radians((0, 0, 90))
        expected_value = [[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]]
        rotation_matrix = calculate_rotation_matrix_extrinsic(alpha, beta, gamma)
        np.testing.assert_array_almost_equal(rotation_matrix, expected_value)

        alpha, beta, gamma = np.radians((90, 90, 0))
        expected_value = [[0, 1, 0],
                          [0, 0, -1],
                          [-1, 0, 0]]
        rotation_matrix = calculate_rotation_matrix_extrinsic(alpha, beta, gamma)
        np.testing.assert_array_almost_equal(rotation_matrix, expected_value)

        alpha, beta, gamma = np.radians((90, 90, 90))
        expected_value = [[0, 0, 1],
                          [0, 1, 0],
                          [-1, 0, 0]]
        rotation_matrix = calculate_rotation_matrix_extrinsic(alpha, beta, gamma)
        np.testing.assert_array_almost_equal(rotation_matrix, expected_value)

    def test_calculate_angles_complies_with_calc_rotation_matrix(self):
        # depends on calculate rotation matrix extrinsic being correct. Ensures that the two match up
        test_data = itertools.product(np.linspace(-np.pi, np.pi, 30), np.linspace(-np.pi, np.pi, 30),
                                      np.linspace(-np.pi, np.pi, 30))

        for angles in test_data:
            R = calculate_rotation_matrix_extrinsic(*angles)
            newAngles = euler_angles_from_rotation_matrix(R)
            newR = calculate_rotation_matrix_extrinsic(*newAngles)

            self.assertTrue(np.isclose(R, newR).all())


if __name__ == '__main__':
    unittest.main(verbosity=1)
