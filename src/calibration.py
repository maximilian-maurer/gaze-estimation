# All equations are taken from PhD thesis:
# Remote, Non-Contact Gaze Estimation with Minimal Subject Cooperation
# Guestrin, Elias Daniel
# https://tspace.library.utoronto.ca/handle/1807/24349
from src.calculate_cornea_center import calculate_cornea_center, normalized
from src.calculate_visual_axis import calculate_eye_angles, calculate_rotation_matrix
import numpy as np


def calibrate_eye_angles(glint_1_ics, glint_2_ics, optic_axis_unit_vector, true_point_of_interest, **kwargs):
    """
    Calibration for alpha_eye and beta_eye as given on pg. 92 and 93 using a single point

    :param glint_1_ics: The location of the glint from the first light source in the image coordinate system
    :param glint_2_ics: The location of the glint from the second light source in the image coordinate system
    :param optic_axis_unit_vector: The location of the optic axis in WCS
    :param true_point_of_interest: The true point of interest in WCS
    :param kwargs: The constants underlying the calculation (see integration_test.py)
    :return: (alpha_eye, beta_eye), the subject-specific direction of the visual axis in the ECS
    """
    cornea_center = calculate_cornea_center(glint_1_ics, glint_2_ics, **kwargs)
    true_visual_axis = normalized(true_point_of_interest - cornea_center)

    theta, phi, kappa = calculate_eye_angles(optic_axis_unit_vector)

    eye_rotation_matrix = calculate_rotation_matrix(theta, phi, kappa)
    inv_eye_rotation_matrix = np.linalg.inv(eye_rotation_matrix)
    visual_axis_ecs = inv_eye_rotation_matrix.dot(true_visual_axis)

    # eq. 2.1: v_ecs = (-sin(alpha_eye)cos(beta_eye), sin(beta_eye), cos(alpha_eye)cos(beta_eye))
    beta_eye = np.arcsin(visual_axis_ecs[1])
    alpha_eye = -np.arcsin(visual_axis_ecs[0] / np.cos(beta_eye))

    return alpha_eye, beta_eye
