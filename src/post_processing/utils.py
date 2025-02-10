import numpy as np
import cv2

def stitch_image_stack(image_stack, row, col, z_order):
    assert image_stack.shape[0] == row * col, "The number of images must match row * col"
    
    height, width = image_stack.shape[1], image_stack.shape[2]
    stitched_image = np.zeros((row * height, col * width), dtype=image_stack.dtype)
    
    index = 0
    for r in range(row):
        if z_order == 'left_to_right_top_to_bottom':
            if r % 2 == 0:
                for c in range(col):
                    stitched_image[r * height:(r + 1) * height, c * width:(c + 1) * width] = image_stack[index]
                    index += 1
            else:
                for c in range(col - 1, -1, -1):
                    stitched_image[r * height:(r + 1) * height, c * width:(c + 1) * width] = image_stack[index]
                    index += 1
        elif z_order == 'right_to_left_top_to_bottom':
            if r % 2 == 0:
                for c in range(col - 1, -1, -1):
                    stitched_image[r * height:(r + 1) * height, c * width:(c + 1) * width] = image_stack[index]
                    index += 1
            else:
                for c in range(col):
                    stitched_image[r * height:(r + 1) * height, c * width:(c + 1) * width] = image_stack[index]
                    index += 1
        elif z_order == 'left_to_right_bottom_to_top':
            if (row - 1 - r) % 2 == 0:
                for c in range(col):
                    stitched_image[(row - 1 - r) * height:(row - r) * height, c * width:(c + 1) * width] = image_stack[index]
                    index += 1
            else:
                for c in range(col - 1, -1, -1):
                    stitched_image[(row - 1 - r) * height:(row - r) * height, c * width:(c + 1) * width] = image_stack[index]
                    index += 1
        elif z_order == 'right_to_left_bottom_to_top':
            if (row - 1 - r) % 2 == 0:
                for c in range(col - 1, -1, -1):
                    stitched_image[(row - 1 - r) * height:(row - r) * height, c * width:(c + 1) * width] = image_stack[index]
                    index += 1
            else:
                for c in range(col):
                    stitched_image[(row - 1 - r) * height:(row - r) * height, c * width:(c + 1) * width] = image_stack[index]
                    index += 1
        else:
            raise ValueError("Invalid path_type. Must be one of the following: 'left_to_right_top_to_bottom', 'right_to_left_top_to_bottom', 'left_to_right_bottom_to_top', or 'right_to_left_bottom_to_top'.")
    
    return stitched_image

def merge_images(images, num_rows, num_cols):
    image_height, image_width = images[0].shape
    
    # create a new image for the final stitched image
    final_image = np.zeros((image_height * num_rows, image_width * num_cols), dtype=np.float32)

    # loop through the images and paste them into the final image
    for i, image in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        if row % 2 == 0: # even rows are left to right
            x = col * image_width
        else: # odd rows are right to left
            x = (num_cols - col - 1) * image_width
        y = row * image_height
        final_image[y:y+image_height, x:x+image_width] = image

    return final_image.astype(np.uint16)


def crop_wise_clahe(img_stack, clipLimit, gridSize):
    img_stack = img_stack.astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=gridSize)
    
    output = []
    for index in range(img_stack.shape[0]):
        output.append(clahe.apply(img_stack[index]))
    
    return np.array(output)
    
def select_nonzero(img_stack):
    img_ = img_stack.reshape(img_stack.shape[0], -1)
    img_ = np.sum(img_, axis=1)
    valid_pos = img_ > 0
    return img_stack[valid_pos]

def image_padding(images, grid_shape):
    if images.shape[0] != grid_shape[0] * grid_shape[1]:
        new_img = np.zeros(
            (
                grid_shape[0] * grid_shape[1],
                images.shape[1],
                images.shape[2],
            ),
            dtype=np.uint16,
        )
        new_img[: images.shape[0]] = images
        images = new_img
    
    return images