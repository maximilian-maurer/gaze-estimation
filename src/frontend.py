# Minimal Temporary rontend that takes as its command line parameters
# frontentd.py FileWithCalibrationData FileWithTestData
# and outputs them to a CSV file as prompted
import csv
import math
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import itertools

from src.calculate_point_of_interest import get_point_of_interest
from src.calibration import calibrate_multi_point, poi_estimation_error

eye_assumptions = {
    'alpha': math.radians(-5),
    'beta': math.radians(1.5),
    'R_cm': 0.78,
    'K_cm': 0.42,
    'n1': 1.3375,
    'n2': 1,
    'D_cm': 0.53,
}

camera_intrinsic = {
    'principal_point': (299.5, 399.5),
    'pixel_size_cm': (2.4 * 1e-6, 2.4 * 1e-6),
    'focal_length_cm': 0.0119144
}

screen_size_cm = (48.7, 27.4)
screen_resolution = (1680, 1050)

camera_position = np.array([24.5, -35, 10])

light_1_position = camera_position + np.array([+13, 0, 0])
light_2_position = camera_position + np.array([-13, 0, 0])

camera_rotation = (np.radians(8.0), np.radians(0), 0)
distance_to_camera_cm = 71  # initial guess for the distance to the camera

plot = True


def setup_environment():
    constants = {
        **eye_assumptions,
        **camera_intrinsic
    }

    screen_pixel_size_cm = (screen_size_cm[0] / screen_resolution[0], screen_size_cm[1] / screen_resolution[1])

    constants['camera_rotation'] = camera_rotation
    constants['distance_to_camera_cm'] = distance_to_camera_cm

    wcs_offset = -camera_position  # move the coordinate system to the camera
    constants['camera_position_wcs'] = camera_position + wcs_offset
    constants['light_1_wcs'] = light_1_position + wcs_offset
    constants['light_2_wcs'] = light_2_position + wcs_offset
    constants['z_shift'] = -camera_position[2]
    constants['alpha_right'] = constants['alpha']

    return constants, screen_pixel_size_cm, wcs_offset


def read_input_file(filename):
    """
    Read all non empty rows from the CSV file, and remove its last column
    :param filename:
    :return:
    """
    data = []
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=',', quotechar='|')
    for row in reader:
        if len(row) != 0:
            data.append(row[:len(row) - 1])

    file.close()
    return data


if __name__ == "__main__":
    constants, screen_pixel_size_cm, wcs_offset = setup_environment()

    # read input data
    print(sys.argv[1])
    print(sys.argv[2])

    calibration_data = read_input_file(sys.argv[1])
    test_data = read_input_file(sys.argv[2])

    print(calibration_data)
    print(test_data)

    calibration_data = np.array(calibration_data).astype(np.int)
    test_data = np.array(test_data).astype(np.int)

    # calibrate
    input_points = []
    true_pogs = []

    for row in calibration_data:
        screen_x, screen_y, pupil_right_x, pupil_right_y, glint_r1_x, glint_r1_y, \
        glint_r2_x, glint_r2_y, glint_r3_x, glint_r3_y = row

        input_points.append([np.array([glint_r1_x, glint_r1_y]),
                             np.array([glint_r3_x, glint_r3_y]),
                             np.array([pupil_right_x, pupil_right_y])])

        screen_point_wcs = np.array([screen_x * screen_pixel_size_cm[0], -screen_y * screen_pixel_size_cm[1]])
        true_pogs.append(np.array([screen_point_wcs[0], screen_point_wcs[1], 0]) + wcs_offset)

    calib_alpha, calib_beta, calib_R, calib_K, calib_camera_angle_y, \
        calib_camera_angle_z, solution = calibrate_multi_point(input_points,
                                                                    true_pogs,
                                                                    **constants)

    print("Calibration result:")
    print("---------------------")
    print("alpha", np.degrees(calib_alpha))
    print("beta", np.degrees(calib_beta))
    print("R", calib_R)
    print("K", calib_K)
    print("camera angle y", np.degrees(calib_camera_angle_y))
    print("camera_angle_z", np.degrees(calib_camera_angle_z))
    print(solution)

    # use the calibrated constants
    constants['alpha'] = constants['alpha_right'] = calib_alpha
    constants['beta'] = calib_beta
    constants['R_cm'] = calib_R
    constants['K_cm'] = calib_K
    constants['camera_rotation'] = (constants['camera_rotation'][0], calib_camera_angle_y, calib_camera_angle_z)

    true_pogs = []
    estimated_pogs = []
    errors_pixels = []
    errors_cm = []
    for row in test_data:
        screen_x, screen_y, pupil_right_x, pupil_right_y, glint_r1_x, glint_r1_y, \
        glint_r2_x, glint_r2_y, glint_r3_x, glint_r3_y = row

        point_of_interest = get_point_of_interest(np.array([glint_r1_x, glint_r1_y]),
                                                  np.array([glint_r3_x, glint_r3_y]),
                                                  np.array([pupil_right_x, pupil_right_y]),
                                                  **constants)
        point_of_interest = point_of_interest - wcs_offset

        estimated_screen_point_ics = np.array(
            [point_of_interest[0] / screen_pixel_size_cm[0],
             -point_of_interest[1] / screen_pixel_size_cm[1]])

        print(screen_x, screen_y, estimated_screen_point_ics)

        true_pogs.append([screen_x, screen_y])
        estimated_pogs.append(estimated_screen_point_ics)
        errors_pixels.append(np.linalg.norm(np.array([screen_x, screen_y]) - estimated_screen_point_ics))
        errors_cm.append(np.linalg.norm(
            np.array([screen_x * screen_pixel_size_cm[0], -screen_y * screen_pixel_size_cm[1]])
            - np.array([point_of_interest[0], point_of_interest[1]])
        ))

    print(errors_cm)
    print(errors_pixels)
    print("Errors pixel: avg {}".format(np.mean(errors_pixels)))
    print("Errors cm: avg {}".format(np.mean(errors_cm)))

    print(test_data)

    output_filename = input("Output filenamename: ")
    with open(output_filename+".csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(test_data)):
            csvwriter.writerow([
                *test_data[i], *estimated_pogs[i], errors_pixels[i], errors_cm[i]
            ])
        csvwriter.writerow(["-------"])
        csvwriter.writerow(["avg error_pixels, avg error_cm, calibrated_alpha, calibrated_beta, calibrated_R"])
        csvwriter.writerow([np.mean(errors_pixels), np.mean(errors_cm),
                            calib_alpha, calib_beta, calib_R])

    if plot:
        colors = ['#590000', '#ff4040', '#f2beb6', '#b23000', '#ffb380', '#ff8800', '#664400', '#cc9933', '#5f6600',
                  '#a3cc00', '#afbf8f', '#33663a', '#3df285', '#00b38f', '#00e2f2', '#005359', '#acdae6', '#0080bf',
                  '#1d3473', '#bfd0ff', '#4059ff', '#0f0073', '#524359', '#b300bf', '#d9a3ce', '#731d56', '#d90074',
                  '#a6535e']

        figure = plt.figure(figsize=(19, 9))
        figure.set_figheight(15)
        figure.set_figwidth(15 * 0.625)
        wcsAxis = figure.add_subplot(1, 1, 1)
        wcsAxis.invert_yaxis()
        wcsAxis.set_xlim(0, 48.7)
        wcsAxis.set_ylim(0, 27.4)
        i = 0
        for p in true_pogs:
            wcsAxis.scatter(p[0] * screen_pixel_size_cm[0], p[1] * screen_pixel_size_cm[1], color=colors[i], marker='^')
            i += 1
        i = 0
        for p in estimated_pogs:
            wcsAxis.scatter(p[0] * screen_pixel_size_cm[0], p[1] * screen_pixel_size_cm[1], color=colors[i], marker='*')
            i += 1
        #plt.show()
        plt.savefig(output_filename)
