# All equations are taken from PhD thesis:
# Remote, Non-Contact Gaze Estimation with Minimal Subject Cooperation
# Guestrin, Elias Daniel
# https://tspace.library.utoronto.ca/handle/1807/24349
from src.calculate_cornea_center import calculate_cornea_center, normalized
from src.calculate_visual_axis import calculate_eye_angles, calculate_rotation_matrix
from src.calculate_point_of_interest import get_point_of_interest

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
    assert len(variables) == 3

    constants = constant_data[1].copy()
    constants['alpha_right'] = constants['alpha'] = variables[0]
    constants['beta'] = variables[1]
    constants['R_cm'] = variables[2]

    try:
        for row in constant_data[0]:
            # in light of the ease of mixing up input parameters due to the function parameter
            # format minimize forces on us,  some basic validation here
            assert len(row) == 4
            assert len(row[0]) == 2  # glint 1 ics
            assert len(row[1]) == 2  # glint 2 ics
            assert len(row[2]) == 2  # pupil ics
            assert len(row[3]) == 3  # pog

            point_of_interest = get_point_of_interest(row[0], row[1], row[2], **constants)
            error += np.linalg.norm(point_of_interest - row[3]) ** 2
    except ValueError:
        # a value error here means that with these parameters calculate_kr did not
        # find a solution to Formula 3.29, so these parameters are no good.
        return 10000000

    return error


def calibrate_multi_point(input_points, true_pois, **kwargs):
    assert len(input_points) == len(true_pois)

    initial_solution = [kwargs['alpha'], kwargs['beta'], kwargs['R_cm']]

    input_rows = [[input_points[i][0], input_points[i][1], input_points[i][2], true_pois[i]] for i
                  in range(len(input_points))]

    input_data = (input_rows, kwargs)

    # todo: review of the bounds
    # todo: review of acceptable termination tolerance
    solution = opt.minimize(poi_estimation_error, initial_solution, input_data,
                            bounds=[(np.radians(-10), np.radians(10)),
                                    (np.radians(-10), np.radians(10)),
                                    (0.01, 3)])  # corneal curvature radius can't be negative)


    #if not solution.success:
    #    raise Exception("No solution found.")

    return solution.x
