# All equations are taken from PhD thesis:
# Remote, Non-Contact Gaze Estimation with Minimal Subject Cooperation
# Guestrin, Elias Daniel
# https://tspace.library.utoronto.ca/handle/1807/24349
# Calibration also based on:
# A novel head-free point-of-gaze estimation system
# Guestrin, Elias Daniel
# http://www.individual.utoronto.ca/elias_guestrin/Guestrin_Elias_D_200306_MASc_thesis.pdf
from src.calculate_cornea_center import calculate_cornea_center, normalized
from src.calculate_visual_axis import calculate_eye_angles, calculate_rotation_matrix
from src.calculate_point_of_interest import get_point_of_interest, get_point_of_interest_extended

import numpy as np
import scipy.optimize as opt

from src.coordinate_system_transformations import inverse_transform_3D_to_3D


def poi_estimation_error(variables, *constant_data):
    """
    Calculates the sum of square distance error for the POI calculation
    given the variables (alpha, beta, R_cm), and the provided constant data
    :param variables: (alpha, beta, R_cm)
    :param constant_data: (input_data, constants) with constants as with all other functions here,
                            and input_data as a list of 4-element arrays [glint1, glint2, pupil, pog]
    """
    error = 0

    assert len(constant_data) == 2
    assert len(variables) == 6

    constants = constant_data[1].copy()
    constants['alpha_right'] = constants['alpha'] = variables[0]
    constants['beta'] = variables[1]
    constants['R_cm'] = variables[2]
    constants['K_cm'] = variables[3]
    constants['camera_rotation'] = (constants['camera_rotation'][0], variables[4], variables[5])

    try:
        for row in constant_data[0]:
            # in light of the ease of mixing up input parameters due to the function parameter
            # format minimize forces on us,  some basic validation here
            assert len(row) == 4
            assert len(row[0]) == 2  # glint 1 ics
            assert len(row[1]) == 2  # glint 2 ics
            assert len(row[2]) == 2  # pupil ics
            assert len(row[3]) == 3  # pog

            point_of_interest, center_of_corneal_curvature = get_point_of_interest_extended(row[0], row[1], row[2],
                                                                                            **constants)
            error += np.linalg.norm(point_of_interest - row[3]) ** 2

            coc_camera_distance = np.linalg.norm(center_of_corneal_curvature - constants['camera_position_wcs'])

            # Todo: expose these system specific bounds to the calling code
            # if not (40 < coc_camera_distance < 100):
            #    raise ValueError
    except ValueError:
        # a value error here means that with these parameters calculate_kr did not
        # find a solution to Formula 3.29, so these parameters are no good.
        return 10000000

    # todo: evaluate this scaling factor
    return error


def calibrate_multi_point(input_points, true_pois, **kwargs):
    """ Calibrates for eye parameters based on the input calibration points, imposing only lax restrictions on the parameters to calibrate for"""
    assert len(input_points) == len(true_pois)

    initial_solution = [kwargs['alpha'], kwargs['beta'], kwargs['R_cm'], kwargs['K_cm'],
                        kwargs['camera_rotation'][1], kwargs['camera_rotation'][2]]

    input_rows = [[input_points[i][0], input_points[i][1], input_points[i][2], true_pois[i]] for i
                  in range(len(input_points))]

    input_data = (input_rows, kwargs)

    bounds = np.array([(np.radians(-10), np.radians(10)),
                       (np.radians(-5), np.radians(5)),
                       (0.3, 2.0),
                       (0.2, 1.5),
                       (np.radians(-8), np.radians(8)),
                       (np.radians(-5), np.radians(5))
                       ])

    solution = opt.minimize(fun=poi_estimation_error, x0=initial_solution, args=input_data,
                            method='SLSQP',
                            bounds=bounds,
                            options={'eps': 1e-6})

    alpha, beta, R, K, camera_angle_y, camera_angle_z = solution.x
    return alpha, beta, R, K, camera_angle_y, camera_angle_z, solution
