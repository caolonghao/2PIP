import numpy as np
import cv2
import json
import logging
import os

def create_shift_from_image_stack(image_stack, row, col, row_shift, row_y_shift, col_shift, col_x_shift, z_order):
    """
    Create a shift list from an image stack.
    
    Parameters
    ----------
    image_stack : list
        A list of images.
    row : int
        The number of rows.
    col : int
        The number of columns.
    row_shift : int
        The shift in a single row.
    col_shift : int
        The shift in a single column.
    z_order : str
        The order of the path.
    """
    
    assert image_stack.shape[0] == row * col, "The number of images must match row * col"
    
    
    index = 0
    image_shift = []
    for r in range(row):
        if z_order == 'left_to_right_top_to_bottom':
            if r % 2 == 0:
                for c in range(col):
                    image_shift.append((index, c * row_shift + r * col_x_shift, r * col_shift + c * row_y_shift))
                    index += 1
            else:
                for c in range(col - 1, -1, -1):
                    # stitched_image[r * height:(r + 1) * height, c * width:(c + 1) * width] = image_stack[index]
                    image_shift.append((index, c * row_shift + r * col_x_shift, r * col_shift + c * row_y_shift))
                    index += 1
        elif z_order == 'right_to_left_top_to_bottom':
            if r % 2 == 0:
                for c in range(col - 1, -1, -1):
                    # stitched_image[r * height:(r + 1) * height, c * width:(c + 1) * width] = image_stack[index]
                    image_shift.append((index, c * row_shift + r * col_x_shift, r * col_shift + c * row_y_shift))
                    index += 1
            else:
                for c in range(col):
                    # stitched_image[r * height:(r + 1) * height, c * width:(c + 1) * width] = image_stack[index]
                    image_shift.append((index, c * row_shift + r * col_x_shift, r * col_shift + c * row_y_shift))
                    index += 1
        elif z_order == 'left_to_right_bottom_to_top':
            if (row - 1 - r) % 2 == 0:
                for c in range(col):
                    # stitched_image[(row - 1 - r) * height:(row - r) * height, c * width:(c + 1) * width] = image_stack[index]
                    image_shift.append((index, c * row_shift + (row - 1 - r) * col_x_shift, (row - 1 - r) * col_shift + c * row_y_shift))
                    index += 1
            else:
                for c in range(col - 1, -1, -1):
                    # stitched_image[(row - 1 - r) * height:(row - r) * height, c * width:(c + 1) * width] = image_stack[index]
                    image_shift.append((index, c * row_shift + (row - 1 - r) * col_x_shift, (row - 1 - r) * col_shift + c * row_y_shift))
                    index += 1
        elif z_order == 'right_to_left_bottom_to_top':
            if (row - 1 - r) % 2 == 0:
                for c in range(col - 1, -1, -1):
                    # stitched_image[(row - 1 - r) * height:(row - r) * height, c * width:(c + 1) * width] = image_stack[index]
                    image_shift.append((index, c * row_shift + (row - 1 - r) * col_x_shift, (row - 1 - r) * col_shift + c * row_y_shift))
                    index += 1
            else:
                for c in range(col):
                    # stitched_image[(row - 1 - r) * height:(row - r) * height, c * width:(c + 1) * width] = image_stack[index]
                    image_shift.append((index, c * row_shift + (row - 1 - r) * col_x_shift, (row - 1 - r) * col_shift + c * row_y_shift))
                    index += 1
        else:
            raise ValueError("Invalid path_type. Must be one of the following: 'left_to_right_top_to_bottom', 'right_to_left_top_to_bottom', 'left_to_right_bottom_to_top', or 'right_to_left_bottom_to_top'.")
    
    # sort the image_shift list by row and column
    image_shift.sort(key=lambda x: (x[2], x[1]))
    return image_shift

    