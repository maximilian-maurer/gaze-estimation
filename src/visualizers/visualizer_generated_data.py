# (Debugging) render of the points involved in the construction of the synthetic data
import math
from _functools import partial

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from src.calculate_optic_axis import normalized
from src.calculate_point_of_interest import get_point_of_interest
from src.calculate_visual_axis import calculate_visual_axis_unit_vector
from src.generate_eye_data import calculate_effective_focal_length, generate_camera_light_points, \
    calculate_camera_rotation_matrix, generate_sample_points_1c_2l
from src.rotation_matrix import euler_angles_from_rotation_matrix, calculate_rotation_matrix_extrinsic


# overview of the scene to provide context
# axes are swapped up to match the drawings in the paper, so to
# plot input the coordinates into transform_coordinates to transform them into
# input for the matplotlib functions
def transform_coordinates(x, y, z):
    return x, z, y


def draw_point(axis, a, color='b', label='', marker='*', **kwargs):
    x, y, z = zip(*[a])
    axis.scatter(*transform_coordinates(x, y, z), label=label, color=color, marker=marker, **kwargs)


def draw_line(axis, a, b, color='b', label='', **kwargs):
    x, y, z = zip(*[a, b])
    axis.plot(*transform_coordinates(x, y, z), label=label, color=color, **kwargs)


def draw_unit_vector(axis, origin, direction, magnitude=1, color='b', label='', marker='*', **kwargs):
    x, y, z = zip(*[origin, np.array(origin) + np.array(direction) * magnitude])
    axis.plot(*transform_coordinates(x, y, z), label=label, color=color, marker=marker, **kwargs)


def configure_coordinate_hints(axis):
    draw_line(axis, [0, 0, 0], [1, 0, 0], color='red')
    draw_line(axis, [0, 0, 0], [0, 1, 0], color='green')
    draw_line(axis, [0, 0, 0], [0, 0, 1], color='blue')

    axis.set_xlabel('X')
    axis.set_ylabel('Z')
    axis.set_zlabel('Y')


def drawing_debug_callback(destination, vector, tag):
    print(tag + " " + str(vector))
    destination[tag] = vector


def conv_array_to_tuple_2d(a):
    return a[0], a[1]


def setup_figures(eyeball_center, wcsEyeRegionSpan):
    # create plots
    figure = plt.figure(figsize=(19, 9))
    wcsAxis = figure.add_subplot(1, 2, 1, projection='3d')

    configure_coordinate_hints(wcsAxis)

    wcsEyeRegion = figure.add_subplot(1, 2, 2, projection='3d')
    configure_coordinate_hints(wcsEyeRegion)
    wcsEyeRegion.set_title('World Coordinate System - Eye Region')
    wcsEyeRegion.set_xlim(eyeball_center[0] - wcsEyeRegionSpan, eyeball_center[0] + wcsEyeRegionSpan)
    wcsEyeRegion.set_zlim(eyeball_center[1] - wcsEyeRegionSpan, eyeball_center[1] + wcsEyeRegionSpan)
    wcsEyeRegion.set_ylim(eyeball_center[2] - wcsEyeRegionSpan, eyeball_center[2] + wcsEyeRegionSpan)

    # setup axes
    wcsAxis.set_xlim(-30, 30)
    wcsAxis.set_zlim(-30, 30)
    wcsAxis.set_ylim(-10, 60)
    wcsAxis.set_title('World Coordinate System')

    return wcsAxis, wcsEyeRegion


if __name__ == '__main__':
    # ---- constants
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

    light_y = 0
    light_1_position = np.array([-25, light_y, 0])
    light_2_position = np.array([25, light_y, 0])

    eyeball_center = np.array([-5, 12, 65])
    optic_axis_intersection = np.array([0, 12.5, 65.0])

    camera_offcenter_x = 0
    camera_position = np.array([0, -5, 10])
    camera_focal_length = 0.035
    wcs_offset = -1 * camera_position
    # ---- end constants
    # ---- constants for the drawing
    detail_viz_pog = np.array([0, 0, 0])
    wcsEyeRegionSpan = 3
    # ---- end constants for the drawing

    effective_focal_length = calculate_effective_focal_length(camera_focal_length, optic_axis_intersection,
                                                              camera_position)
    R_cam = calculate_camera_rotation_matrix(camera_position, optic_axis_intersection)

    constants['focal_length_cm'] = effective_focal_length
    constants['distance_to_camera_cm'] = abs(camera_position[1] - eyeball_center[1])
    constants['camera_position_wcs'] = camera_position + wcs_offset
    constants['light_1_wcs'] = light_1_position + wcs_offset
    constants['light_2_wcs'] = light_2_position + wcs_offset
    constants['alpha_right'] = constants['alpha']
    constants['z_shift'] = -camera_position[2]
    constants['camera_rotation'] = euler_angles_from_rotation_matrix(R_cam)
    constants['camera_position_wcs'] = camera_position + wcs_offset
    constants['light_1_wcs'] = light_1_position + wcs_offset
    constants['light_2_wcs'] = light_2_position + wcs_offset

    wcsMain, wcsEyeRegion = setup_figures(eyeball_center, wcsEyeRegionSpan)

    # draw constant positions, camera orientation
    draw_point(wcsMain, light_1_position, label='light 1')
    draw_point(wcsMain, light_2_position, label='light 2')
    draw_point(wcsMain, camera_position, label="camera", marker='8')

    draw_unit_vector(wcsMain, camera_position, normalized(np.dot(R_cam, np.array([0, 0, 1]))),
                     magnitude=10, color='b',
                     label="camera forward in wcs")
    draw_unit_vector(wcsMain, camera_position, normalized(np.dot(R_cam, np.array([1, 0, 0]))),
                     magnitude=1, color='r',
                     label="camera x in wcs")
    draw_unit_vector(wcsMain, camera_position, normalized(np.dot(R_cam, np.array([0, 1, 0]))),
                     magnitude=1, color='g',
                     label="camera y in wcs")
    draw_point(wcsMain, optic_axis_intersection, label="camera oa intersection point", color='b', marker='8')

    draw_point(wcsMain, eyeball_center, label="eyeball center", marker='D', color='r')
    draw_point(wcsMain, detail_viz_pog, label="point of gaze", marker="D", color='g')

    draw_point(wcsEyeRegion, eyeball_center, label="eyeball center", marker='D', color='r')

    # generate sample data for one camera, two light sources
    collected_debug_data_1 = {}
    collected_debug_data_2 = {}
    debug_callback = partial(drawing_debug_callback, collected_debug_data_1)
    po_reflection_ics, po_refraction_ics = generate_camera_light_points(eyeball_center=eyeball_center,
                                                                        point_of_gaze=detail_viz_pog,
                                                                        camera_position=camera_position,
                                                                        camera_focal_length=effective_focal_length,
                                                                        R_cam=R_cam,
                                                                        light_position=light_1_position,
                                                                        debug_callback=debug_callback, **constants)
    debug_callback = partial(drawing_debug_callback, collected_debug_data_2)
    po_reflection_ics_2, po_refraction_ics_2 = generate_camera_light_points(eyeball_center=eyeball_center,
                                                                            point_of_gaze=detail_viz_pog,
                                                                            camera_position=camera_position,
                                                                            camera_focal_length=effective_focal_length,
                                                                            R_cam=R_cam,
                                                                            light_position=light_2_position,
                                                                            debug_callback=debug_callback, **constants)
    assert np.isclose(po_refraction_ics, po_refraction_ics_2).all()

    found_poi = get_point_of_interest(conv_array_to_tuple_2d(po_reflection_ics),
                                      conv_array_to_tuple_2d(po_reflection_ics_2),
                                      conv_array_to_tuple_2d(po_refraction_ics), **constants)
    found_poi = found_poi - wcs_offset
    print("Actual poi {}".format(detail_viz_pog))
    print("Found poi {}".format(found_poi))

    draw_point(wcsMain, found_poi, label="Found poi", marker='v', color='r')

    draw_point(wcsEyeRegion, collected_debug_data_1['point_of_reflection'], label='point_of_reflection', color=None)
    draw_point(wcsEyeRegion, collected_debug_data_2['point_of_reflection'], label='point_of_reflection2', color=None)
    draw_point(wcsEyeRegion, collected_debug_data_1['point_of_refraction'], label='point_of_refraction', color=None)

    por = collected_debug_data_1['point_of_reflection']
    por_to_camera = normalized(camera_position - por)
    por_to_light1 = normalized(light_1_position - por)
    draw_unit_vector(wcsEyeRegion, por, por_to_camera, label='por to camera', color='m')
    draw_unit_vector(wcsEyeRegion, por, por_to_light1, label='por to light1', color='c')

    por2 = collected_debug_data_2['point_of_reflection']
    por2_to_camera = normalized(camera_position - por2)
    por2_to_light2 = normalized(light_2_position - por2)
    draw_unit_vector(wcsEyeRegion, por2, por2_to_camera, label='por 2 to camera', color='m')
    draw_unit_vector(wcsEyeRegion, por2, por2_to_light2, label='por 2 to light2', color='c')

    draw_point(wcsEyeRegion, collected_debug_data_1['cornea_center'], label='cornea center', marker='h')

    draw_line(wcsEyeRegion, eyeball_center, eyeball_center + collected_debug_data_1['optical_axis'])
    draw_unit_vector(wcsEyeRegion, eyeball_center, collected_debug_data_1['optical_axis'], magnitude=4)
    visual = calculate_visual_axis_unit_vector(collected_debug_data_1['optical_axis'], constants['alpha'],
                                               constants['beta'])
    draw_unit_vector(wcsMain, collected_debug_data_1['cornea_center'], visual, magnitude=70)
    draw_unit_vector(wcsEyeRegion, collected_debug_data_1['cornea_center'], visual, magnitude=5, label="visual axis",
                     color='r')

    print("Distance: {:.4f}".format(np.linalg.norm(detail_viz_pog - found_poi)))

    # run a set of test points
    colors = ['#590000', '#ff4040', '#f2beb6', '#b23000', '#ffb380', '#ff8800', '#664400', '#cc9933', '#5f6600',
              '#a3cc00', '#afbf8f', '#33663a', '#3df285', '#00b38f', '#00e2f2', '#005359', '#acdae6', '#0080bf',
              '#1d3473', '#bfd0ff', '#4059ff', '#0f0073', '#524359', '#b300bf', '#d9a3ce', '#731d56', '#d90074',
              '#a6535e']
    distances = []
    screen_offset = np.array([0, 15, 0])
    for y in range(-2, 2):
        for x in range(-2, 2):
            detail_viz_pog = np.array([5 * x, 5 * y, 0]) + screen_offset
            pupil_center_ics, glint_1_ics, glint_2_ics = generate_sample_points_1c_2l(light_1_position,
                                                                                      light_2_position,
                                                                                      camera_position,
                                                                                      optic_axis_intersection,
                                                                                      eyeball_center,
                                                                                      camera_focal_length,
                                                                                      detail_viz_pog,
                                                                                      constants)

            found_poi = get_point_of_interest(conv_array_to_tuple_2d(glint_1_ics),
                                              conv_array_to_tuple_2d(glint_2_ics),
                                              conv_array_to_tuple_2d(pupil_center_ics), **constants)
            found_poi = found_poi - wcs_offset
            draw_point(wcsMain, detail_viz_pog, color=colors[y * 5 + x], marker='s')
            draw_point(wcsMain, found_poi, color=colors[y * 5 + x], marker='*')

            distances.append(np.linalg.norm(detail_viz_pog - found_poi))

    distances = np.array(distances)
    print("count: {} min: {:.8f}, max {:.8f}, avg {:.8f}".format(len(distances), distances.min(), distances.max(),
                                                                 distances.mean()))

    wcsMain.view_init(elev=30, azim=25)

    wcsMain.legend()
    wcsEyeRegion.legend()
    plt.show()
