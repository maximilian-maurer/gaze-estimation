import math
import unittest

import numpy as np

from src.generate_eye_data import determine_ecs


class TestGenerateEyeData(unittest.TestCase):

    def test_determine_ecs_basic(self):
        # ecs rotation when at rest position looking straight forward,
        # must match the negative of the deviation of the visual axis
        # (to a reasonable degree of precision)
        args = {
            'alpha': math.radians(-5),
            'beta': math.radians(1.5),
            'D_cm': 0.53
        }
        theta, phi, kappa = determine_ecs(np.array([0, 0, 50]),
                                          np.array([0, 0, 0]),
                                          **args)

        self.assertLess(abs(np.degrees(theta) + np.degrees(args['alpha'])), 1)
        self.assertLess(abs(np.degrees(phi) + np.degrees(args['beta'])), 1)

        args['alpha'] = math.radians(5)
        theta, phi, kappa = determine_ecs(np.array([0, 0, 50]),
                                          np.array([0, 0, 0]),
                                          **args)

        self.assertLess(abs(np.degrees(theta) + np.degrees(args['alpha'])), 1)
        self.assertLess(abs(np.degrees(phi) + np.degrees(args['beta'])), 1)


if __name__ == '__main__':
    unittest.main()
