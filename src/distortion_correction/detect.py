import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import os
import math
import copy
from skimage.transform import PiecewiseAffineTransform, warp
import discorpy.losa.loadersaver as io
import discorpy.prep.preprocessing as prep
import discorpy.prep.linepattern as lprep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post

def intensity_compress(src, min_num=2.5, max_num=97.5):
    max_threshold = np.percentile(src, max_num)
    min_threshold = np.percentile(src, min_num)
    img = np.clip(src, min_threshold, max_threshold)

    return img

def find_x_candidates_from_pivot(pivot, centroids, x_step, y_step, x_max, y_max):
    output = [pivot]
    fixed_x, fixed_y = pivot
    while fixed_x + x_step < x_max:
        next_x_min = fixed_x
        next_x_max = fixed_x + x_step
        next_y_min = max(0, fixed_y - y_step)
        next_y_max = min(y_max, fixed_y + y_step)
        candidates = [point for point in centroids if next_x_min < point[0] < next_x_max and next_y_min < point[1] < next_y_max]
        # print(candidates)
        if len(candidates) == 0:
            break
        candidates = sorted(candidates, key=lambda x: math.fabs(x[0] - fixed_x) + math.fabs(x[1] - fixed_y))
        output.append(candidates[0])
        fixed_x, fixed_y = candidates[0]
    
    return output

def find_y_candidates_from_pivot(pivot, centroids, x_step, y_step, x_max, y_max):
    output = [pivot]
    fixed_x, fixed_y = pivot
    while fixed_y + y_step < y_max:
        next_x_min = max(0, fixed_x - x_step)
        next_x_max = min(x_max, fixed_x + x_step)
        next_y_min = fixed_y
        next_y_max = fixed_y + y_step
        candidates = [point for point in centroids if next_x_min < point[0] < next_x_max and next_y_min < point[1] < next_y_max]
        # print(candidates)
        if len(candidates) == 0:
            break
        candidates = sorted(candidates, key=lambda x: math.fabs(x[0] - fixed_x) + math.fabs(x[1] - fixed_y))
        output.append(candidates[0])
        fixed_x, fixed_y = candidates[0]
    
    return output

if __name__ == "__main__":
    base_path = r"D:/MyCode/Light Field Correction/Data/2023.3.23-stitching/2-U-1X-square/CellVideo1/CellVideo"

    input_path = os.path.join(base_path, "CellVideo 0.tif")
    output_path = os.path.join(base_path, "normalized.tif")
    binary_path = os.path.join(base_path, "binary.tif")
    eroted_path = os.path.join(base_path, "eroted.tif")

    ret, image_stack = cv2.imreadmulti(input_path, flags=cv2.IMREAD_UNCHANGED)
    image = np.mean(image_stack, axis=0)

    image = intensity_compress(image, min_num=1, max_num=99)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imwrite(output_path, image)
    blur_image = cv2.medianBlur(image, 5)
    
    ret, thresh = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(thresh, cmap="gray")
    plt.show()
    
    binary_image = (blur_image >= 12).astype(np.uint8)

    dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroted_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 膨胀加侵蚀
    dilated_image = cv2.dilate(binary_image, dilated_kernel, iterations=1)
    eroted_image = cv2.erode(dilated_image, eroted_kernel, iterations=1)
    
    plt.subplot(131)
    plt.imshow(binary_image, cmap="gray")
    plt.subplot(132)
    plt.imshow(dilated_image, cmap="gray")
    plt.subplot(133)
    plt.imshow(eroted_image, cmap="gray")
    plt.show()
    
    reversed_eroted_image = 1 - eroted_image
    print(np.unique(reversed_eroted_image))
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(reversed_eroted_image, connectivity=8)

    reversed_eroted_image *= 255
    
    # --------------------------------------------- #
    
    y_scale, x_scale = image.shape
    x_ratio = 0.15 
    y_ratio = 0.15
    print(reversed_eroted_image.shape)
    available_pivot = [point for point in centroids if point[0] < x_scale * x_ratio and point[1] < y_scale * y_ratio]
    available_pivot = sorted(available_pivot, key=lambda x: x[1] + x[0], reverse=True)
    pivot = available_pivot[0]

    # print(pivot)
    # cv2.circle(reversed_eroted_image, (int(pivot[0]), int(pivot[1])), 3, (0, 0, 255), -1)
    # cv2.imshow('image', reversed_eroted_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(reversed_eroted_image)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(reversed_eroted_image)
    
    std_x_step = dist_ver
    std_y_step = dist_hor
    
    starting_candidates = find_y_candidates_from_pivot(pivot, centroids, x_step=0.5 * std_x_step, 
                                                       y_step=1.5 * std_y_step, x_max=x_scale, y_max=y_scale)
    all_moving_points = [find_x_candidates_from_pivot(point, centroids, x_step=std_x_step * 1.5, 
                            y_step=0.75 * std_y_step, x_max=x_scale, y_max=y_scale) for point in starting_candidates]
    
    all_length = np.array([len(x_line) for x_line in all_moving_points])
    shorest_length = np.min(all_length)
    all_moving_points = np.array([x_line[:shorest_length] for x_line in all_moving_points])
    
    
    
    
    
    x_point_num, y_point_num = all_moving_points.shape[:2]
    
    # 将central_pivot视作reference坐标的原点
    central_pivot = all_moving_points[x_point_num // 2, y_point_num // 2]
    central_pivot_xindex, central_pivot_yindex = x_point_num // 2, y_point_num // 2
    shifted_moving_points = all_moving_points - central_pivot
    
    fixed_points = np.zeros_like(all_moving_points)
    for j in range(y_point_num):
        for i in range(x_point_num):
            fixed_points[i, j] = [(j - central_pivot_yindex) * std_x_step, (i - central_pivot_xindex) * std_y_step]
    
    fixed_points = fixed_points + central_pivot
    
    fixed_points = fixed_points.astype(np.float32).reshape(-1, 2)
    all_moving_points = all_moving_points.astype(np.float32).reshape(-1, 2)
    
    # swaped_move_points = np.zeros_like(all_moving_points)
    # swaped_move_points[:, 0] = all_moving_points[:, 1]
    # swaped_move_points[:, 1] = all_moving_points[:, 0]
    # list_points_hor_lines = list_points_ver_lines = swaped_move_points
    # slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(reversed_eroted_image)
    # slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(reversed_eroted_image)
    
    # print("    Horizontal slope: ", slope_hor, " Distance: ", dist_hor)
    # print("    Vertical slope: ", slope_ver, " Distance: ", dist_ver)
    
    # print("4-> Group points into lines !!!!")
    # list_hor_lines = prep.group_dots_hor_lines(list_points_hor_lines, slope_hor, dist_hor)
    # list_ver_lines = prep.group_dots_ver_lines(list_points_ver_lines, slope_ver, dist_ver)
    
    # height, width = reversed_eroted_image.shape
    # output_base = r"D:/MyCode/Light Field Correction/Output/2021.4.13-sitching/1-1X-circle-X17Y14/CellVideo1/CellVideo"
    
    # # Optional: remove residual dots
    # list_hor_lines = prep.remove_residual_dots_hor(list_hor_lines, slope_hor, 2.0)
    # list_ver_lines = prep.remove_residual_dots_ver(list_ver_lines, slope_ver, 2.0)
    # io.save_plot_image(output_base + "/grouped_hor_lines_new.png", list_hor_lines, height, width)
    # io.save_plot_image(output_base + "/grouped_ver_lines_new.png", list_ver_lines, height, width)
    
    # print("5-> Correct perspective effect !!!!")
    # # Optional: correct perspective effect.
    # list_hor_lines, list_ver_lines = proc.regenerate_grid_points_parabola(
    #     list_hor_lines, list_ver_lines, perspective=True)

    # # Check if the distortion is significant.
    # list_hor_data = post.calc_residual_hor(list_hor_lines, 0.0, 0.0)
    # io.save_residual_plot(output_base + "/residual_horizontal_points_before.png",
    #                     list_hor_data, height, width)
    # list_ver_data = post.calc_residual_ver(list_ver_lines, 0.0, 0.0)
    # io.save_residual_plot(output_base + "/residual_vertical_points_before.png",
    #                     list_ver_data, height, width)

    # print("6-> Calculate the centre of distortion !!!!")
    # (xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines, list_ver_lines)
    # print("   X-center: {0}, Y-center: {1}".format(xcenter, ycenter))

    # num_coef = 5
    # print("7-> Calculate radial distortion coefficients !!!!")
    # list_fact = proc.calc_coef_backward(list_hor_lines, list_ver_lines, xcenter,
    #                                     ycenter, num_coef)

    # # Check the correction results
    # list_uhor_lines = post.unwarp_line_backward(list_hor_lines, xcenter, ycenter, list_fact)
    # list_uver_lines = post.unwarp_line_backward(list_ver_lines, xcenter, ycenter, list_fact)
    # list_hor_data = post.calc_residual_hor(list_uhor_lines, xcenter, ycenter)
    # list_ver_data = post.calc_residual_ver(list_uver_lines, xcenter, ycenter)
    # io.save_residual_plot(output_base + "/residual_horizontal_points_after.png",
    #                     list_hor_data, height, width)
    # io.save_residual_plot(output_base + "/residual_vertical_points_after.png",
    #                     list_ver_data, height, width)
    # # Output
    # print("8-> Apply correction to image !!!!")
    # corrected_mat = post.unwarp_image_backward(reversed_eroted_image, xcenter, ycenter, list_fact)
    # io.save_image(output_base + "/corrected_image.tif", corrected_mat)
    # io.save_metadata_txt(output_base + "/coefficients.txt", xcenter, ycenter, list_fact)
    # io.save_image(output_base + "/difference.tif", reversed_eroted_image - corrected_mat)
    # print("!!! Done !!!!")
    
    
    """ Test Old Method """
    tform = PiecewiseAffineTransform()
    tform.estimate(fixed_points, all_moving_points)

    test_path = r"D:\MyCode\Light Field Correction\Data\2023.3.23-stitching\1-U-1X-circle\CellVideo1\CellVideo"
    test_file = "CellVideo 0.tif"
    ret, test_circle = cv2.imreadmulti(os.path.join(test_path, test_file), flags=cv2.IMREAD_UNCHANGED)
    test_circle = np.mean(test_circle, axis=0)
    
    test_circle = intensity_compress(test_circle, min_num=1, max_num=99)
    test_circle = cv2.normalize(test_circle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    warped_image = warp(test_circle, tform, output_shape=(y_scale, x_scale))
    
    plt.imshow(warped_image, cmap="gray")
    plt.show()
    # plt.plot(fixed_points[:, 0], fixed_points[:, 1], "r.")
    plt.plot(all_moving_points[:, 0], all_moving_points[:, 1], "r.")
    plt.imshow(reversed_eroted_image, cmap="gray")
    plt.show()



    # print(all_moving_points.shape)
    # test show out
    # reshaped_moving_points = all_moving_points[0].reshape(-1, 2)
    # for point in reshaped_moving_points:
    #     cv2.circle(reversed_eroted_image, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
    # cv2.imshow('image', reversed_eroted_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(all_moving_points)
    