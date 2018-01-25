import itertools
import math
import unittest
from multiprocessing import Pool

import numpy as np

from src.calculate_point_of_interest import get_point_of_interest
from src.calibration import calibrate_multi_point, poi_estimation_error
from src.generate_eye_data import generate_sample_points_1c_2l, calculate_effective_focal_length, \
    calculate_camera_rotation_matrix
from src.rotation_matrix import euler_angles_from_rotation_matrix

import csv


class TestCalibration(unittest.TestCase):
    """
    Note that all the tests in this class depend on test_pog_equality being true.

    If that test fails, then the error lies in the estimation of the point of gaze or
    the generation of the sample data, and not necessarily in the calibration
    """

    @staticmethod
    def fill_test_constants():
        """ Shared initial parameters for all these tests."""
        constants = {
            'alpha': math.radians(-5),
            'beta': math.radians(1.5),
            'R_cm': 0.78,
            'K_cm': 0.42,
            'n1': 1.3375,
            'n2': 1,
            'D_cm': 0.53,
            'principal_point': (695.5, 449.5),
            'pixel_size_cm': (4.65 * 1e-6, 4.65 * 1e-6)
        }
        light_1_position = np.array([-25, 5, 0])
        light_2_position = np.array([25, 5, 0])
        camera_position = np.array([0, -21, 2])
        camera_focal_length = 0.035
        eyeball_center = np.array([-5, 12, 65])
        optic_axis_intersection = np.array([0, 12.5, 65.0])

        effective_focal_length = calculate_effective_focal_length(camera_focal_length, optic_axis_intersection,
                                                                  camera_position)
        constants['focal_length_cm'] = effective_focal_length

        R_cam = calculate_camera_rotation_matrix(camera_position, optic_axis_intersection)
        constants['camera_rotation'] = euler_angles_from_rotation_matrix(R_cam)

        wcs_offset = -1 * camera_position
        constants['camera_position_wcs'] = camera_position + wcs_offset
        constants['light_1_wcs'] = light_1_position + wcs_offset
        constants['light_2_wcs'] = light_2_position + wcs_offset
        constants['z_shift'] = -camera_position[2]

        constants['distance_to_camera_cm'] = abs(camera_position[1] - eyeball_center[1])
        constants['alpha_right'] = constants['alpha']

        return constants, light_1_position, light_2_position, camera_position, camera_focal_length, eyeball_center, optic_axis_intersection, wcs_offset

    def test_pog_equality(self):
        constants, light_1_position, light_2_position, \
        camera_position, camera_focal_length, \
        eyeball_center, optic_axis_intersection, wcs_offset = self.fill_test_constants()

        p_1_pog = np.array([20, 2, 0])
        p_1_pupil, p_1_glint_1, p_1_glint_2 = generate_sample_points_1c_2l(light_1_position=light_1_position,
                                                                           light_2_position=light_2_position,
                                                                           camera_position=camera_position,
                                                                           camera_focal_length=camera_focal_length,
                                                                           eyeball_center=eyeball_center,
                                                                           optic_axis_intersection=optic_axis_intersection,
                                                                           point_of_gaze=p_1_pog,
                                                                           constants=constants)
        estimated_pog = get_point_of_interest(p_1_glint_1, p_1_glint_2, p_1_pupil, **constants)
        estimated_pog = estimated_pog - wcs_offset

        self.assertTrue(np.linalg.norm(estimated_pog - p_1_pog) < 1)

        constants['alpha'] = constants['alpha_right'] = np.radians(-4.0)
        p_1_pupil, p_1_glint_1, p_1_glint_2 = generate_sample_points_1c_2l(light_1_position=light_1_position,
                                                                           light_2_position=light_2_position,
                                                                           camera_position=camera_position,
                                                                           camera_focal_length=camera_focal_length,
                                                                           eyeball_center=eyeball_center,
                                                                           optic_axis_intersection=optic_axis_intersection,
                                                                           point_of_gaze=p_1_pog,
                                                                           constants=constants)
        estimated_pog = get_point_of_interest(p_1_glint_1, p_1_glint_2, p_1_pupil, **constants)
        estimated_pog = estimated_pog - wcs_offset

        self.assertTrue(np.linalg.norm(estimated_pog - p_1_pog) < 1)

        constants['beta'] = np.radians(2.0)
        p_1_pupil, p_1_glint_1, p_1_glint_2 = generate_sample_points_1c_2l(light_1_position=light_1_position,
                                                                           light_2_position=light_2_position,
                                                                           camera_position=camera_position,
                                                                           camera_focal_length=camera_focal_length,
                                                                           eyeball_center=eyeball_center,
                                                                           optic_axis_intersection=optic_axis_intersection,
                                                                           point_of_gaze=p_1_pog,
                                                                           constants=constants)
        estimated_pog = get_point_of_interest(p_1_glint_1, p_1_glint_2, p_1_pupil, **constants)
        estimated_pog = estimated_pog - wcs_offset

        self.assertTrue(np.linalg.norm(estimated_pog - p_1_pog) < 1)

    def test_calculate_poi_error(self):
        self.test_pog_equality()

        constants, light_1_position, light_2_position, \
        camera_position, camera_focal_length, \
        eyeball_center, optic_axis_intersection, wcs_offset = self.fill_test_constants()

        p_1_pog = np.array([20, 2, 0])
        p_1_pupil, p_1_glint_1, p_1_glint_2 = generate_sample_points_1c_2l(light_1_position=light_1_position,
                                                                           light_2_position=light_2_position,
                                                                           camera_position=camera_position,
                                                                           camera_focal_length=camera_focal_length,
                                                                           eyeball_center=eyeball_center,
                                                                           optic_axis_intersection=optic_axis_intersection,
                                                                           point_of_gaze=p_1_pog,
                                                                           constants=constants)
        input_rows = [[p_1_glint_1, p_1_glint_2, p_1_pupil, p_1_pog + wcs_offset]]
        error = poi_estimation_error([constants['alpha'], constants['beta'], constants['R_cm'], constants['K_cm'],
                                      constants['camera_rotation'][1], constants['camera_rotation'][2]],
                                     input_rows, constants)
        self.assertAlmostEqual(0, error)

        # check multi-row input
        p_2_pog = np.array([-20, 5, 0])
        p_2_pupil, p_2_glint_1, p_2_glint_2 = generate_sample_points_1c_2l(light_1_position=light_1_position,
                                                                           light_2_position=light_2_position,
                                                                           camera_position=camera_position,
                                                                           camera_focal_length=camera_focal_length,
                                                                           eyeball_center=eyeball_center,
                                                                           optic_axis_intersection=optic_axis_intersection,
                                                                           point_of_gaze=p_2_pog,
                                                                           constants=constants)
        input_rows = [[p_1_glint_1, p_1_glint_2, p_1_pupil, p_1_pog + wcs_offset],
                      [p_2_glint_1, p_2_glint_2, p_2_pupil, p_2_pog + wcs_offset]]
        error = poi_estimation_error([constants['alpha'], constants['beta'], constants['R_cm'], constants['K_cm'],
                                      constants['camera_rotation'][1], constants['camera_rotation'][2]],
                                     input_rows, constants)
        self.assertAlmostEqual(0, error)

    def test_calibrate_multi_point(self):
        constants, light_1_position, light_2_position, \
        camera_position, camera_focal_length, \
        eyeball_center, optic_axis_intersection, wcs_offset = self.fill_test_constants()

        # add some other tests with all three varying for good measure
        alpha_eyes = np.radians(np.linspace(-4, -6, 5))
        beta_eyes = np.radians(np.linspace(0.5, 2.5, 5))

        R_base = 0.781
        R_range = 0.036 / 2 * 4

        R_eyes = np.linspace(R_base - R_range / 2, R_base + R_range / 2, 10)
        random_tests = list(itertools.product(alpha_eyes, beta_eyes, R_eyes))

        test_sets = [(np.radians(-5), np.radians(1.5), 0.78),  # already correct
                     (np.radians(-4), np.radians(1.5), 0.78),  # vary alpha
                     (np.radians(-5), np.radians(2), 0.78),  # vary beta
                     (np.radians(-5), np.radians(1.5), 0.75),  # vary r
                     *random_tests]

        alpha_accept_margin_degrees = 1
        beta_accept_margin_degrees = 1
        R_accept_margin_cm = 1

        for alpha, beta, R in test_sets:
            constants['alpha'] = constants['alpha_right'] = alpha
            constants['beta'] = beta
            constants['R_cm'] = R

            # generate calibration points
            input_points = []
            true_pogs = []
            samplesX = 3
            samplesY = 3
            for i, j in np.ndindex(samplesX, samplesY):
                point_of_gaze = np.array([-20 + (40 / samplesX) * i, -15 + (30 / samplesY) * j, 0])
                pupil, glint_1, glint_2 = generate_sample_points_1c_2l(light_1_position=light_1_position,
                                                                       light_2_position=light_2_position,
                                                                       camera_position=camera_position,
                                                                       camera_focal_length=camera_focal_length,
                                                                       eyeball_center=eyeball_center,
                                                                       optic_axis_intersection=optic_axis_intersection,
                                                                       point_of_gaze=point_of_gaze,
                                                                       constants=constants)
                input_points.append([glint_1, glint_2, pupil])
                true_pogs.append(point_of_gaze + wcs_offset)

            # set them back to default initial values
            constants['alpha'] = constants['alpha_right'] = math.radians(-5)
            constants['beta'] = math.radians(1.5)
            constants['R_cm'] = 0.78

            calib_alpha, calib_beta, calib_R, calib_K, calib_camera_angle_y, \
                camera_calib_angle_z, solution = calibrate_multi_point(input_points,
                                                                      true_pogs,
                                                                      **constants)

            print("------")
            print(math.degrees(alpha), math.degrees(calib_alpha))
            print(math.degrees(beta), math.degrees(calib_beta))
            print(R, calib_R)

            self.assertTrue(np.isclose(np.degrees(alpha), np.degrees(calib_alpha),
                                       atol=alpha_accept_margin_degrees),
                            msg="Alpha: Should: {:.4f} Is: {:4f}".format(np.degrees(alpha), np.degrees(calib_alpha)))
            self.assertTrue(np.isclose(np.degrees(beta), np.degrees(calib_beta),
                                       atol=beta_accept_margin_degrees),
                            msg="Beta: Should: {:.4f} Is: {:4f}".format(np.degrees(beta), np.degrees(calib_beta)))
            self.assertTrue(np.isclose(R, calib_R, atol=R_accept_margin_cm),
                            msg="R: Should: {:.4f} Is {:4f}".format(R, calib_R))


if __name__ == '__main__':
    unittest.main()
