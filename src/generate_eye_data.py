# All equations are taken from PhD thesis:
# Remote, Non-Contact Gaze Estimation with Minimal Subject Cooperation
# Guestrin, Elias Daniel
# https://tspace.library.utoronto.ca/handle/1807/24349
from src.calculate_cornea_center import normalized

import numpy as np
import math
import scipy.optimize as opt

from src.calculate_visual_axis import calculate_rotation_matrix, calculate_nu_ecs


def calculate_optical_axis_unit_vector(theta, phi, kappa):
    """
    Generates the optical axis unit vector in WCS through the description of the ECS
    given by the parameters.

    Caution: Although this function accepts kappa as a parameter, no eye torsion model
    is used at the moment, and the parameter is simply ignored.

    :param theta: ECS rotation angle
    :param phi: ECS rotation angle
    :param kappa: ECS rotation angle
    :return: Unit Vector of the Optical Axis in WCS
    """
    return np.array([np.sin(theta) * np.cos(phi), np.sin(phi), -np.cos(theta) * np.cos(phi)])


def error_ecs(variables, *args):
    theta, phi = variables
    pog_to_eyeball = args[0]
    D_cm = args[1]
    alpha = args[2]
    beta = args[3]
    # Formula 2.26
    optic_axis_uv = calculate_optical_axis_unit_vector(theta, phi, 0)
    left_side = normalized(pog_to_eyeball - D_cm * optic_axis_uv)
    right_side = np.dot(calculate_rotation_matrix(theta, phi, 0), calculate_nu_ecs(alpha, beta))
    return np.linalg.norm(left_side - right_side)


def determine_ecs(eyeball_center, point_of_gaze, **kwargs):
    """
    Numerically solves for a description of the ECS by its rotation angles given the
    center of the eyeball and the point of gaze
    :param eyeball_center: the center of the eyeball
    :param point_of_gaze: the point of gaze
    :param kwargs:
    :return: (theta, phi, kappa)
    """
    # Formula 2.55-2.58
    v_initial = normalized(point_of_gaze - eyeball_center)
    theta_initial = -np.arctan(v_initial[0] / v_initial[2]) - kwargs['alpha']
    phi_initial = np.arcsin(v_initial[1]) - kwargs['beta']

    optimization_constants = (
        point_of_gaze - eyeball_center,
        kwargs['D_cm'],
        kwargs['alpha'],
        kwargs['beta']
    )

    solution = opt.minimize(error_ecs, (theta_initial, phi_initial), optimization_constants)
    return solution.x[0], solution.x[1], 0


def point_of_reflection_from_chi_psi(chi, psi, cornea_center, cornea_curvature_radius, camera_nodal_point,
                                     light_position):
    """ Formula 2.64 """
    factor = cornea_curvature_radius / (1 - np.cos(chi) ** 2)
    light_cornea = (np.cos(chi - psi) - np.cos(psi) * np.cos(chi)) * normalized(light_position - cornea_center)
    camera_cornea = (np.cos(psi) - np.cos(chi - psi) * np.cos(chi)) * normalized(camera_nodal_point - cornea_center)
    return cornea_center + factor * (light_cornea + camera_cornea)


def error_point_of_reflection(variables, *args):
    psi = variables
    chi, cornea_center, cornea_curvature_radius, camera_nodal_point, light_position = args
    point_of_reflection = point_of_reflection_from_chi_psi(chi, psi, cornea_center, cornea_curvature_radius,
                                                           camera_nodal_point, light_position)
    # Eq. 2.39 - R cancelled out
    q_to_c = point_of_reflection - cornea_center
    return abs(np.dot(normalized(light_position - point_of_reflection), q_to_c)
               - np.dot(normalized(camera_nodal_point - point_of_reflection), q_to_c))


def calculate_point_of_reflection(cornea_center, light_position, camera_nodal_point, cornea_curvature_radius):
    """
    Calculates the point of reflection in WCS of the light, camera and eye
    :param cornea_center: The center of corneal curvature
    :param light_position: The position of the light
    :param camera_nodal_point:  the nodal point of the camera
    :param cornea_curvature_radius: The radius of corneal curvature (R) in cm
    :return:
    """
    # if light and camera are coincident, Formula 2.59 applies
    coincident_check = np.cross(light_position - camera_nodal_point, cornea_center - camera_nodal_point)
    if np.isclose(coincident_check, np.array([0, 0, 0])).all():
        return cornea_center + cornea_curvature_radius * normalized(camera_nodal_point - cornea_center)
    # otherwise use 2.63, 2.64
    chi = np.arccos(np.dot(normalized(light_position - cornea_center),
                           normalized(camera_nodal_point - cornea_center)))

    args = (chi, cornea_center, cornea_curvature_radius, camera_nodal_point, light_position)
    initial_guess = (chi / 2)

    # note the bound here is not only to aid calculation, it is essential to prevent
    # an unwanted solution from appearing (see paragraph after initial definition of Eq. 2.64)
    solution = opt.minimize(error_point_of_reflection, initial_guess, args, bounds=[(0, chi)])
    psi = solution.x

    return point_of_reflection_from_chi_psi(chi, psi, cornea_center, cornea_curvature_radius,
                                            camera_nodal_point, light_position)


def validate_point_of_reflection(point_of_reflection, eyeball_center, optical_axis_unit_vector, tau=0.6):
    """ Validates the point of reflection as descripted in 2.6.3 using Eq. 2.82"""
    return math.sqrt(np.linalg.norm(point_of_reflection - eyeball_center) ** 2 -
                     np.dot(point_of_reflection - eyeball_center, optical_axis_unit_vector) ** 2) <= tau


def point_of_refraction_from_chi_psi(chi, psi, cornea_center, cornea_curvature_radius, pupil_center,
                                     camera_nodal_point):
    """
    Formula 2.70
    :param chi:
    :param psi:
    :param cornea_center:
    :param cornea_curvature_radius:
    :param pupil_center:
    :param camera_nodal_point:
    :return:
    """
    factor = cornea_curvature_radius / (1 - np.cos(chi) ** 2)
    pupil_cornea = (np.cos(chi - psi) - np.cos(psi) * np.cos(chi)) * normalized(pupil_center - cornea_center)
    camera_cornea = (np.cos(psi) - np.cos(chi - psi) * np.cos(chi)) * normalized(camera_nodal_point - cornea_center)
    return cornea_center + factor * (pupil_cornea + camera_cornea)


def error_point_of_refraction(variables, *args):
    psi = variables
    chi, cornea_center, cornea_curvature_radius, camera_nodal_point, pupil_center, n1, n2 = args

    point_of_refraction = point_of_refraction_from_chi_psi(chi, psi, cornea_center, cornea_curvature_radius,
                                                           pupil_center, camera_nodal_point)
    # Formula 2.71
    v = np.cross(n1 * normalized(pupil_center - point_of_refraction)
                 + n2 * normalized(camera_nodal_point - point_of_refraction),
                 point_of_refraction - cornea_center)
    eta = np.cross(pupil_center - cornea_center, camera_nodal_point - cornea_center)
    return abs(np.dot(v, eta))  # this must be 0 if psi is correct


def calculate_point_of_refraction(cornea_center, camera_nodal_point, pupil_center_to_nodal_point,
                                  cornea_curvature_radius, optic_axis, n1, n2):
    # Eq. 2.28  (pupil_center_to_nodal_point = K)
    pupil_center = cornea_center + optic_axis * pupil_center_to_nodal_point

    coincident_check = np.cross(cornea_center - camera_nodal_point, pupil_center - camera_nodal_point)
    if np.isclose(coincident_check, np.array([0, 0, 0])).all():
        return cornea_center + cornea_curvature_radius * normalized(camera_nodal_point - cornea_center)

    # Formula 2.69
    chi = np.arccos(np.dot(normalized(pupil_center - cornea_center), normalized(camera_nodal_point - cornea_center)))

    # Formula 2.72 & 2.73
    k_initial = -np.dot((pupil_center - cornea_center), normalized(camera_nodal_point - pupil_center)) \
                + np.sqrt(np.dot(pupil_center - cornea_center, normalized(camera_nodal_point - pupil_center)) ** 2
                          - np.linalg.norm(pupil_center - cornea_center) ** 2 + cornea_curvature_radius ** 2)
    r_initial = pupil_center + k_initial * normalized(camera_nodal_point - pupil_center)
    psi_initial = np.arccos(np.dot((1 / cornea_curvature_radius) * (r_initial - cornea_center),
                                   normalized(camera_nodal_point - cornea_center)))

    assert psi_initial < chi
    assert psi_initial >= 0

    initial_guess = psi_initial
    args = (chi, cornea_center, cornea_curvature_radius, camera_nodal_point, pupil_center, n1, n2)
    solution = opt.minimize(error_point_of_refraction, initial_guess, args, bounds=[(0, chi)])

    return point_of_refraction_from_chi_psi(chi, solution.x, cornea_center, cornea_curvature_radius,
                                            pupil_center, camera_nodal_point)


def projection_in_ccs(point, focal_length):
    # Formula 2.20
    return np.array([-focal_length * point[0] / point[2], -focal_length * (point[1] / point[2])])


def projected_ccs_to_ics(point, pixel_pitch, center_coordinates, skew_coefficient):
    """
    transforms the projected point in ccs into the x,y pixel coordinates in ics
    :param point: The point in CCS
    :param pixel_pitch: The pixel pitch in each direction as 2D vector
    :param center_coordinates: The ics coordinates of the intersection between
                                optic axis and image sensor as 2D vector
    :param skew_coefficient: the skew coefficient
    :return: the coordinates in ics
    """
    return np.array([(point[0] + skew_coefficient * point[1]) / pixel_pitch[0] + center_coordinates[0],
                     point[1] / pixel_pitch[1] + center_coordinates[1]])


# noinspection PyUnusedLocal
def generate_samples_default_debug_callback(*unused):
    pass


# right now, kappa_eye = 0 is assumed!
def generate_camera_light_points(eyeball_center,
                                 point_of_gaze,
                                 camera_position,
                                 camera_focal_length,
                                 R_cam,
                                 light_position,
                                 camera_skew=0,
                                 debug_callback=generate_samples_default_debug_callback,
                                 **kwargs):
    theta, phi, kappa = determine_ecs(eyeball_center, point_of_gaze, **kwargs)
    optical_axis = calculate_optical_axis_unit_vector(theta, phi, kappa)
    debug_callback(optical_axis, 'optical_axis')

    # Formula 2.27
    cornea_center = eyeball_center + kwargs['D_cm'] * optical_axis
    debug_callback(cornea_center, 'cornea_center')

    point_of_reflection = calculate_point_of_reflection(cornea_center, light_position, camera_position, kwargs['R_cm'])
    debug_callback(point_of_reflection, 'point_of_reflection')

    if not validate_point_of_reflection(point_of_reflection, eyeball_center, optical_axis):
        return False

    point_of_refraction = calculate_point_of_refraction(cornea_center, camera_position, kwargs['K_cm'], kwargs['R_cm'],
                                                        optical_axis, kwargs['n1'], kwargs['n2'])
    debug_callback(point_of_refraction, 'point_of_refraction')

    if not validate_point_of_reflection(point_of_refraction, eyeball_center, optical_axis):
        return False

    # translate these into ccs, Formula 2.25 with R^-1 = R^T in this case
    R_cam_inv = np.matrix.transpose(R_cam)
    po_reflection_cam = np.dot(R_cam_inv, (point_of_reflection - camera_position))
    po_refraction_cam = np.dot(R_cam_inv, (point_of_refraction - camera_position))

    debug_callback(po_reflection_cam, 'po_reflection_cam')
    debug_callback(po_refraction_cam, 'po_refraction_cam')

    po_reflection_proj = projection_in_ccs(po_reflection_cam, camera_focal_length)
    po_refraction_proj = projection_in_ccs(po_refraction_cam, camera_focal_length)

    debug_callback(po_reflection_proj, 'po_reflection_proj')
    debug_callback(po_refraction_proj, 'po_refraction_proj')

    # no lens distortion model used at the moment
    po_reflection_ics = projected_ccs_to_ics(po_reflection_proj,
                                             np.array([kwargs['pixel_size_cm'][0], kwargs['pixel_size_cm'][1]]),
                                             np.array(kwargs['principal_point']), camera_skew)
    po_refraction_ics = projected_ccs_to_ics(po_refraction_proj,
                                             np.array([kwargs['pixel_size_cm'][0], kwargs['pixel_size_cm'][1]]),
                                             np.array(kwargs['principal_point']), camera_skew)

    return po_reflection_ics, po_refraction_ics


def calculate_effective_focal_length(focal_length, optic_axis_intersection, camera_position):
    return 1 / (1 / focal_length - 1 / (np.linalg.norm(optic_axis_intersection - camera_position)))


def calculate_camera_rotation_matrix(camera_position, optic_axis_point):
    """ Calculates the rotation matrix of the camera from the given parameters,
    so that CCS Y is up, and CCS +Z goes towards the point

    Based on Eq. 4.1-4.7
    :param camera_position The position of the nodal point of the camera in WCS
    :param optic_axis_point A point on the optic axis of the camera
    """
    camera_optic_axis = normalized(optic_axis_point - camera_position)
    ccs_x_axis_wcs = normalized(np.cross([0, 1, 0], camera_optic_axis))
    ccs_y_axis_wcs = normalized(np.cross(camera_optic_axis, ccs_x_axis_wcs))

    return np.array([ccs_x_axis_wcs, ccs_y_axis_wcs, camera_optic_axis]).T


def generate_sample_points_1c_2l(light_1_position, light_2_position, camera_position,
                                 optic_axis_intersection, eyeball_center,
                                 camera_focal_length, point_of_gaze,
                                 constants):
    """
    End to end method to generate sample points for a one-camera, two light setup

    This assumes a World Coordinate System with the origin independent of the camera.

    This makes no guarantee that the conditions described by the parameters will actually result
    in images on the camera, or valid points of reflection/refraction. It is up to the calling
    function to validate the results to that end. Also no restriction as to the size of the
    imaging sensor is assumed, limiting this is also up to the calling function.

    The requried constants are K_cm, R_cm, n1, n2, pixel_size_cm, principal_point, D_cm, alpha, beta

    :param point_of_gaze: The desired point of gaze in WCS
    :param camera_focal_length: The focal length of the camera
    :param eyeball_center: The center of the eyeball in WCS
    :param light_1_position: The position of the first light in WCS
    :param light_2_position: The position of the second light in WCS
    :param camera_position: The position of the nodal point of the camera in WCS
    :param optic_axis_intersection: The intersection of the optic axes of the cameras, or in this case
                                    the point along the the optic axis that is the most in focus
    :param constants: A set of constants as described above
    :return: A tuple (pupil_center_ics, glint_1_ics, glint_2_ics)
    """
    effective_focal_length = calculate_effective_focal_length(camera_focal_length, optic_axis_intersection,
                                                              camera_position)
    R_cam = calculate_camera_rotation_matrix(camera_position, optic_axis_intersection)
    po_reflection_ics, po_refraction_ics = generate_camera_light_points(eyeball_center=eyeball_center,
                                                                        point_of_gaze=point_of_gaze,
                                                                        camera_position=camera_position,
                                                                        camera_focal_length=effective_focal_length,
                                                                        R_cam=R_cam,
                                                                        light_position=light_1_position,
                                                                        **constants)
    po_reflection_ics_2, po_refraction_ics_2 = generate_camera_light_points(eyeball_center=eyeball_center,
                                                                            point_of_gaze=point_of_gaze,
                                                                            camera_position=camera_position,
                                                                            camera_focal_length=effective_focal_length,
                                                                            R_cam=R_cam,
                                                                            light_position=light_2_position,
                                                                            **constants)
    assert np.isclose(po_refraction_ics, po_refraction_ics_2).all()

    return po_refraction_ics, po_reflection_ics, po_reflection_ics_2
