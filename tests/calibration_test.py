import math
import unittest
from collections import OrderedDict

import numpy as np

from src.calculate_cornea_center import calculate_cornea_center
from src.calculate_optic_axis import calculate_optic_axis_unit_vector
from src.calculate_point_of_interest import transform_to_screen_coordinate_system, \
    calculate_point_of_interest
from src.calculate_visual_axis import calculate_visual_axis_unit_vector
from src.calibration import calibrate_eye_angles
from src.coordinate_system_transformations import transform_2D_to_3D, inverse_transform_3D_to_3D

constants = {
    'light_1_wcs': np.array([-23, 0, 0]),
    'light_2_wcs': np.array([23, 0, 0]),
    'camera_position_wcs': np.array([0, 0, 0]),
    'focal_length_cm': 1.2,
    'pixel_size_cm': (0.00048, 0.00048),
    'principal_point': (400, 300),
    'z_shift': -18,
    'alpha_right': math.radians(-5),
    'beta': math.radians(1.5),
    'R_cm': 0.78,
    'K_cm': 0.42,
    'n1': 1.3375,
    'n2': 1,
    'distance_to_camera_cm': 52,
    'camera_rotation': np.array([math.radians(8), 0, 0])  # rotation only over X axis
}

test_data = OrderedDict(
    {
        1: {
            'glint1': (332, 164),
            'glint2': (322, 169),
            'glint3': (313, 165),
            'pupil': (323, 163),
            'point_on_screen': (490, 65)
        },
        2: {
            'glint1': (333, 164),
            'glint2': (323, 169),
            'glint3': (314, 165),
            'pupil': (324, 163),
            'point_on_screen': (840, 65)
        },
        3: {
            'glint1': (324.5, 164.4),
            'glint2': (314.9, 168.4),
            'glint3': (305.6, 164),
            'pupil': (305.3, 159.5),
            'point_on_screen': (1190, 65)
        },
        4: {
            'glint1': (327.45, 168.75),
            'glint2': (318.3, 172.4),
            'glint3': (309.05, 169.2),
            'pupil': (322.9, 173.3),
            'point_on_screen': (490, 520)
        },
        5: {
            'glint1': (322.711, 168.816),
            'glint2': (313.447, 172.579),
            'glint3': (304.579, 169.211),
            'pupil': (310.421, 171.658),
            'point_on_screen': (840, 520)
        },
        6: {
            'glint1': (317.78, 168.707),
            'glint2': (308.78, 172.561),
            'glint3': (299.561, 169.122),
            'pupil': (300.463, 173.024),
            'point_on_screen': (1190, 520)
        },
        7: {
            'glint1': (330.5, 175.5),
            'glint2': (320.667, 179),
            'glint3': (311, 176),
            'pupil': (320, 185.5),
            'point_on_screen': (490, 975)
        },
        8: {
            'glint1': (321.722, 174.583),
            'glint2': (312.75, 178.389),
            'glint3': (303.389, 174.722),
            'pupil': (310.278, 185.556),
            'point_on_screen': (840, 975)
        },
        9: {
            'glint1': (317.2, 174.933),
            'glint2': (308.267, 178.567),
            'glint3': (298.933, 174.933),
            'pupil': (302.967, 186.1),
            'point_on_screen': (1190, 975)
        }
    }
)


class TestCalibrateSinglePoint(unittest.TestCase):

    def test_calibrate_eye_angles(self):
        # synthetic test. Choose alpha and beta, use those to calculate the visual axis and the point of
        # interest and then assert that the parameters that result from a calibration given this point as
        # truth are in fact the same alpha and beta that were chosen to begin with.
        for index in test_data:
            data = test_data[index]

            center_of_cornea_curvature = calculate_cornea_center(data['glint1'], data['glint3'], **constants)

            pupil_on_image_wgs = \
                transform_2D_to_3D(*data['pupil'], constants['focal_length_cm'], *constants['pixel_size_cm'],
                                   *constants['principal_point'])

            optic_axis_unit_vector = calculate_optic_axis_unit_vector(pupil_on_image_wgs,
                                                                      constants['camera_position_wcs'],
                                                                      center_of_cornea_curvature,
                                                                      constants['R_cm'],
                                                                      constants['K_cm'],
                                                                      constants['n1'],
                                                                      constants['n2'])

            visual_axis_unit_vector = \
                calculate_visual_axis_unit_vector(optic_axis_unit_vector,
                                                  constants['alpha_right'],
                                                  constants['beta'])

            # transform to coordinate system aligned with screen
            center_of_cornea_curvature_scs, visual_axis_unit_vector_scs = \
                transform_to_screen_coordinate_system(center_of_cornea_curvature, visual_axis_unit_vector,
                                                      constants['camera_rotation'])

            poi = \
                calculate_point_of_interest(center_of_cornea_curvature_scs,
                                            visual_axis_unit_vector_scs,
                                            constants['z_shift'])

            poi = inverse_transform_3D_to_3D(poi, *constants['camera_rotation'], np.zeros(3))
            alpha, beta = calibrate_eye_angles(data['glint1'], data['glint3'], optic_axis_unit_vector, poi,
                                                 **constants)

            # print("Alpha {:.4f} vs {:.4f}, Beta {:.4f} vs {:.4f}".format(
            #   math.degrees(alpha), math.degrees(constants['alpha_right']),
            #   math.degrees(beta), math.degrees(constants['beta'])))

            self.assertAlmostEqual(alpha, constants['alpha_right'])
            self.assertAlmostEqual(beta, constants['beta'])


if __name__ == '__main__':
    unittest.main()
