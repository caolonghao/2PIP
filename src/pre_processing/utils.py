import numpy as np

def image_padding(images, average_num, col, row):
    applied_padding = False
    if images.shape[0] != col * row * average_num:
        applied_padding = True
        new_img = np.zeros(
            (
                col * row * average_num,
                images.shape[1],
                images.shape[2],
            ),
            dtype=np.uint16,
        )
        new_img[: images.shape[0]] = images
        images = new_img
    
    return images, applied_padding

def average_img(images, average_num):
    img_avg = np.zeros(
        (images.shape[0] // average_num, images.shape[1], images.shape[2]),
        dtype=np.uint16,
    )
    for i in range(images.shape[0] // average_num):
        img_stack = images[i * average_num : (i + 1) * average_num]
        img_avg[i] = np.mean(img_stack, axis=0)
    return img_avg

