import numpy as np
import cv2

def reverse_with_flat_bg(src, flat, bg, max_value=4096.0):
    print("---- bg max/min: {:.2f}, {:.2f}".format(bg.max(), bg.min()))
    print("---- flat max/min: {:.2f}, {:.2f}".format(flat.max(), flat.min()))
    flat = flat.astype(np.float32)
    src = src.astype(np.float32)
    bg = bg.astype(np.float32)

    src = ((src - bg) / (flat))
    src = cv2.normalize(src, None, 0, max_value, cv2.NORM_MINMAX)
    
    return src